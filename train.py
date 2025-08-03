import argparse
import math
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch.nn.functional as F
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datamodule import DataCfg, MusicDataModule
from lit_gpt.diffmodel import Block, Config, TransEncoder
from lit_gpt.utils import get_default_supported_precision, num_parameters
from tokenizer import MusicTokenizerWithStyle
from torch.nn.utils.rnn import pad_sequence

class DiffusionLightningModule(L.LightningModule):
    def __init__(self, config: Config, learning_rate: float, weight_decay: float, beta1: float, beta2: float,
                 warmup_iters: int, min_lr: float, decay_lr: bool, grad_clip: float,
                 tokenizer: MusicTokenizerWithStyle, eval_iters: int = 100, mc_samples: int = 128):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.config = config
        self.model = TransEncoder(config)
        self.model.apply(lambda module: self.model._init_weights(module, config.n_layer))
        self.tokenizer = tokenizer
        self.automatic_optimization = False  # Manual optimization for gradient accumulation and custom logic
        self.eval_iters = eval_iters
        self.mc_samples = mc_samples
        self.total_lengths = 0

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        opt = self.optimizers()
        full_input, full_attn, prompt_lengths = self.process_batch(batch, self.device, self.config.block_size)
        noisy_input, mask_indices, p_mask = self.forward_process(full_input, prompt_lengths, attn_mask=full_attn)
        
        logits = self(noisy_input, attention_mask=full_attn)
        ce_losses = torch.nn.functional.cross_entropy(
            logits[mask_indices], full_input[mask_indices], reduction='none'
        )
        loss = (ce_losses / p_mask[mask_indices]).sum() / (full_input.shape[0] * full_input.shape[1])
        
        self.manual_backward(loss)
        
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self.clip_gradients(opt, gradient_clip_val=self.hparams.grad_clip)
            opt.step()
            opt.zero_grad()
            self.log('train/loss', loss, prog_bar=True)
            self.total_lengths += full_input.size(1)
            self.log('train/total_lengths', self.total_lengths, prog_bar=False)
        
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        full_input, full_attn, prompt_lengths = self.process_batch(batch, self.device, self.config.block_size, is_val=True)
        input_ids = full_input
        mc_loss = torch.zeros(self.mc_samples, device=self.device)
        
        for i in range(self.mc_samples):
            noisy_input, mask_indices, p_mask = self.forward_process(input_ids, prompt_lengths, attn_mask=full_attn)
            logits = self(noisy_input, attention_mask=full_attn)
            ce_losses = torch.nn.functional.cross_entropy(
                logits[mask_indices], input_ids[mask_indices], reduction='none'
            )
            loss = (ce_losses / p_mask[mask_indices]).sum() / (input_ids.shape[0] * input_ids.shape[1])
            mc_loss[i] = loss
        
        val_loss = mc_loss.mean()
        self.log('val/loss', val_loss, prog_bar=True)
        self.log('val/ppl', math.exp(val_loss), prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,
                          betas=(self.hparams.beta1, self.hparams.beta2), foreach=False)
        return optimizer

    def forward_process(self, batch: torch.Tensor, prompt_lengths: torch.Tensor, total_dim: int = 32000,
                        eps: float = 1e-3, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l = batch.shape
        t = torch.rand((b,), device=batch.device)
        p_mask = (1 - eps) * t + eps
        p_mask_full = torch.zeros((b, l), dtype=torch.float, device=batch.device)
        for i in range(b):
            resp_start = prompt_lengths[i]
            resp_end = attn_mask[i].sum() if attn_mask is not None else l
            p_mask_full[i, resp_start:resp_end] = p_mask[i]
        mask_indices = torch.rand((b, l), device=batch.device) < p_mask_full
        noisy_batch = torch.where(mask_indices, total_dim, batch)
        return noisy_batch, mask_indices, p_mask_full

    def process_batch(self, batch: dict, device: torch.device, block_size: int, is_val: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_ids = batch['prompt_ids'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        full_inputs_list = []
        full_attns_list = []
        prompt_lens = []
        b = prompt_ids.shape[0]
        for i in range(b):
            p_mask = (prompt_ids[i] != self.tokenizer.pad_id)
            i_mask = attention_mask[i].bool()
            p_len = p_mask.sum().item()
            i_len = i_mask.sum().item()
            prompt_trim = prompt_ids[i][p_mask]
            inp_trim = input_ids[i][i_mask]
            full = torch.cat([prompt_trim, inp_trim])
            f_len = full.shape[0]
            if f_len > block_size:
                inp_trim = inp_trim[:block_size - p_len]
                full = torch.cat([prompt_trim, inp_trim])
                f_len = full.shape[0]
            full_inputs_list.append(full)
            full_attns_list.append(torch.ones(f_len, dtype=torch.long, device=device))
            prompt_lens.append(p_len)
        full_input = pad_sequence(full_inputs_list, batch_first=True, padding_value=self.tokenizer.pad_id)
        full_attn = pad_sequence(full_attns_list, batch_first=True, padding_value=0)
        return full_input, full_attn, torch.tensor(prompt_lens, device=device)

@torch.no_grad()
def generate_reconstruction(model: TransEncoder, tokenizer: MusicTokenizerWithStyle, prompt_ids: List[int], target_length: int, K: int = 1000, device: str = 'cuda') -> List[int]:
    model.eval()
    print(f"Generating reconstruction on device: {device}")

    mask_id = tokenizer.mask_id
    x = torch.tensor(prompt_ids + [mask_id] * target_length, dtype=torch.long, device=device).unsqueeze(0)
    
    b, seq_len = x.shape
    
    for i in range(1, K + 1):
        t = 1 - i / K
        alpha_t = 1 - t
        
        mask_pos = (x == mask_id)
        if mask_pos.sum() == 0:
            break
        
        x= x.to(device)
        
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        
        # Stochastic keep
        keep = torch.rand_like(mask_pos.float()) < alpha_t
        update_pos = mask_pos & keep
        x[update_pos] = pred[update_pos]
    
    generated = x[0, len(prompt_ids):].cpu().tolist()
    return generated

def main():
    args = parse_args()
    config = Config.from_name(f'Diff_LLaMA_{args.model}M')
    tokenizer = MusicTokenizerWithStyle()
    config.block_size = 6144
    config.vocab_size = tokenizer.vocab_size
    data_cfg = DataCfg(
        train_pkl=args.train_pkl,
        val_pkl=args.val_pkl,
        max_len=config.block_size,
        seq_limit=None,
        shuffle_records=True,
        skip_long_after_tokenization=True,
        augment=True,
        block_size=config.block_size,
        overfit_single_batch=args.overfit_single_batch
    )
    datamodule = MusicDataModule(cfg=data_cfg, tokenizer=tokenizer, batch_size=args.batch_size, num_workers=args.num_workers)
    
    model = DiffusionLightningModule(
        config=config,
        learning_rate=1e-4,
        weight_decay=1e-1,
        beta1=0.9,
        beta2=0.95,
        warmup_iters=0,  # Set to 0 to disable warmup
        min_lr=1e-3,  # Set same as learning_rate since we're not using decay
        decay_lr=False,  # Disable learning rate decay
        grad_clip=1.0,
        tokenizer=tokenizer,
        eval_iters=int(100 * 1024 / args.batch_size)
    )
    
    out_dir = Path('workdir') / f'mdm-{args.model}M-{args.max_epochs}'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_logger = CSVLogger(save_dir=out_dir, name='csv_logs')
    wandb_logger = WandbLogger(name=f'mdm-{args.model}M-{args.max_epochs}-mc', save_dir=out_dir, project='scaling')
    
    callbacks = [
        ModelCheckpoint(dirpath=out_dir, filename='epoch-{epoch:02d}-ckpt', save_top_k=-1, every_n_epochs=100),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        # accumulate_grad_batches=0,  # Adjust if needed
        precision=get_default_supported_precision(training=True),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        num_nodes=args.nodes_num,
        strategy='fsdp' if args.nodes_num > 1 or args.devices > 1 else 'auto',
        logger=[csv_logger, wandb_logger],
        callbacks=callbacks,
        log_every_n_steps=10,
        limit_val_batches=0,  # Run val at end of each epoch
    )
    
    print(f"Total parameters: {num_parameters(model.model):,}")
    
    # Resume from checkpoint if specified
    ckpt_path = None
    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        print(f"Resuming training from checkpoint: {ckpt_path}")
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    if args.overfit_single_batch:
        print("Performing reconstruction on overfit batch...")
        model.eval()
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        # Take first sample for simplicity
        prompt_ids = batch['prompt_ids'][0].cpu().tolist()
        original_ids = batch['input_ids'][0].cpu().tolist()
        target_length = len(original_ids)
        
        # Ensure model is properly moved to GPU after training
        # After Lightning training, we need to explicitly move the model
        if torch.cuda.is_available():
            model.model = model.model.cuda()
            # Ensure model is in bfloat16 to match training precision
            model.model = model.model.to(dtype=torch.bfloat16)
            device_str = 'cuda'
        else:
            model.model = model.model.cpu()
            device_str = 'cpu'
        
        generated = generate_reconstruction(model.model, tokenizer, prompt_ids, target_length, K=1000, device=device_str)
        
        reconstruction_accuracy = (torch.tensor(generated) == torch.tensor(original_ids)).float().mean().item()
        print(f"Reconstruction accuracy: {reconstruction_accuracy * 100:.2f}%")
        
        # Print decoded tokens for analysis
        print("\nDecoded tokens:")
        print(f"Prompt: {tokenizer.decode_tokens(prompt_ids)}")
        print(f"Original: {tokenizer.decode_tokens(original_ids)}")
        print(f"Generated: {tokenizer.decode_tokens(generated)}")
        # # Save MIDIs
        # tokenizer.save_tokens_as_midi(prompt_ids + original_ids, 'original.mid')
        # tokenizer.save_tokens_as_midi(prompt_ids + generated, 'reconstructed.mid')
        # print("Saved original.mid and reconstructed.mid")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=6, help='model parameters')
    parser.add_argument('--nodes_num', type=int, default=1, help='number of nodes')
    parser.add_argument('--devices', type=int, default=1, help='number of devices')
    parser.add_argument('--batch_size', type=int, default=4, help='global_batch_size')
    parser.add_argument('--train_pkl', type=str, default="cache/dataset_paths_synthetic_structured-aria-unique_limitNone_37776ff2_train.pkl", help='Path to training PKL file')
    parser.add_argument('--val_pkl', type=str, default="cache/dataset_paths_synthetic_structured-aria-unique_limitNone_37776ff2_val.pkl", help='Path to validation PKL file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--overfit_single_batch', action='store_true', help='Overfit on single batch')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from')
    return parser.parse_args()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()