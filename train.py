import time
from pathlib import Path
import argparse
import torch
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from data_module import MusicDataModule, DataCfg
from tokenizer import MusicTokenizerWithStyle
from model import TransEncoder
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from pytorch_lightning.loggers import WandbLogger
from lit_gpt.utils import get_default_supported_precision, num_parameters, step_csv_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to train .pkl file')
    parser.add_argument('--val_data_dir', type=str, default=None, help='Path to validation .pkl file')
    parser.add_argument('--batch_size', type=int, default=8, help='Global batch size')
    parser.add_argument('--max_steps', type=int, default=100000, help='Max training steps')
    parser.add_argument('--devices', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--out_dir', type=str, default='workdir', help='Output directory')
    return parser.parse_args()

args = parse_args()
out_dir = Path(args.out_dir) / 'music_diffusion'
learning_rate = 2e-4
micro_batch_size = 8
warmup_steps = 1000
log_step_interval = 10
eval_step_interval = 1000
save_step_interval = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 2e-5
batch_size = args.batch_size
gradient_accumulation_steps = batch_size // micro_batch_size
warmup_iters = warmup_steps * gradient_accumulation_steps
max_iters = args.max_steps * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

def forward_process(batch, total_dim=32001, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim-1, batch)
    return noisy_batch, mask_indices, p_mask

def setup(devices: int = args.devices, precision: str = None, resume: bool = False):
    precision = precision or get_default_supported_precision(training=True)
    strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", limit_all_gathers=True, cpu_offload=False) if devices > 1 else "auto"
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[
        step_csv_logger(out_dir, "MusicDiffTransformer_113M", flush_logs_every_n_steps=log_iter_interval),
        WandbLogger(name='music_diffusion_113M', save_dir=out_dir, project='music_diffusion')
    ])
    fabric.print(f"Training with {devices} GPUs, batch_size={batch_size}, max_steps={args.max_steps}")
    main(fabric, resume)

def main(fabric, resume):
    tokenizer = MusicTokenizerWithStyle()
    data_cfg = DataCfg(
        data_dir=args.data_dir,
        val_data_dir=args.val_data_dir,
        max_len=2048,
        augment=True
    )
    data_module = MusicDataModule(data_cfg, tokenizer, batch_size=micro_batch_size, num_workers=4)
    data_module.setup(stage='fit')
    
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    
    fabric.seed_everything(3407)
    model = TransEncoder(vocab_size=tokenizer.vocab_size)
    fabric.print(f"Model initialized with {num_parameters(model):,} parameters")
    
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2)
    )
    optimizer = fabric.setup_optimizers(optimizer)
    
    state = {"model": model, "optimizer": optimizer, "iter_num": 0, "step_count": 0}
    if resume:
        checkpoint_path = sorted(out_dir.glob("*.pth"), key=lambda x: int(x.stem.split('-')[1]) if x.stem.startswith('iter-') else 0)
        if checkpoint_path:
            fabric.load(checkpoint_path[-1], state)
            fabric.print(f"Resuming from {checkpoint_path[-1]}")
    
    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def train(fabric, state, train_dataloader, val_dataloader):
    model = state["model"]
    optimizer = state["optimizer"]
    loss_func = CrossEntropyLoss(reduction='none')
    
    total_lengths = 0
    total_t0 = time.perf_counter()
    initial_iter = state["iter_num"]
    
    for train_data in train_dataloader:
        if state["iter_num"] >= max_iters:
            break
        
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        iter_t0 = time.perf_counter()
        input_ids = train_data["input_ids"][:, :2048].contiguous()
        prompt_ids = train_data["prompt_ids"][:, :2048].contiguous()
        attention_mask = train_data["attention_mask"][:, :2048].contiguous()
        
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            noisy_input, mask_indices, p_mask = forward_process(input_ids)
            # Combine prompt and noisy input for conditioning
            batch_size, seq_len = noisy_input.shape
            max_prompt_len = prompt_ids.shape[1]
            combined_input = torch.full((batch_size, seq_len), model.vocab_size-1, dtype=torch.long, device=fabric.device)
            combined_input[:, :max_prompt_len] = prompt_ids
            combined_input[:, max_prompt_len:] = noisy_input[:, max_prompt_len:]
            
            logits = model(combined_input)
            loss = loss_func(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            fabric.backward(loss / gradient_accumulation_steps)
        
        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        
        state["iter_num"] += 1
        total_lengths += input_ids.size(1)
        
        if state["iter_num"] % log_iter_interval == 0:
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, "
                f"iter time: {(time.perf_counter() - iter_t0) * 1000:.2f}ms, "
                f"remaining time: {(time.perf_counter() - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours"
            )
        
        if val_dataloader and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            val_loss = validate(fabric, model, val_dataloader)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}")
            fabric.log_dict({"val_loss": val_loss, "step": state["step_count"]})
        
        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {checkpoint_path}")
            fabric.save(checkpoint_path, state)

@torch.no_grad()
def validate(fabric, model, val_dataloader):
    fabric.print("Validating...")
    model.eval()
    losses = []
    for k, val_data in enumerate(val_dataloader):
        if k >= 100:  # Limit validation iterations
            break
        input_ids = val_data["input_ids"][:, :2048].contiguous()
        noisy_input, mask_indices, p_mask = forward_process(input_ids)
        logits = model(noisy_input)
        loss = torch.nn.functional.cross_entropy(
            logits[mask_indices], input_ids[mask_indices], reduction='none'
        ) / p_mask[mask_indices]
        loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        losses.append(loss.item())
    model.train()
    return torch.tensor(losses).mean()

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    setup()