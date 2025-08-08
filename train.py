import argparse
import math
import random
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import json
from dataclasses import asdict
import hashlib

import torch.nn.functional as F
import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datamodule import DataCfg, MusicDataModule
from lit_gpt.diffmodel import Block, Config, TransEncoder
from lit_gpt.utils import get_default_supported_precision, num_parameters
from tokenizer import MusicTokenizerWithStyle
from torch.nn.utils.rnn import pad_sequence

class DiffusionLightningModule(L.LightningModule):
    def __init__(self, config: Config, learning_rate: float, weight_decay: float, beta1: float, beta2: float,
                 warmup_iters: int, min_lr: float, decay_lr: bool, grad_clip: float,
                 tokenizer: MusicTokenizerWithStyle, eval_iters: int = 100, mc_samples: int = 128,
                 mask_prob_min: float = 0.05, mask_prob_max: float = 0.30, mask_schedule: str = 'linear',
                 deterministic_mask: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.config = config
        self.model = TransEncoder(config)
        self.model.apply(lambda module: self.model._init_weights(module, config.n_layer))
        self.tokenizer = tokenizer
        self.eval_iters = eval_iters
        self.mc_samples = mc_samples
        self.total_lengths = 0
        self.mask_prob_min = mask_prob_min
        self.mask_prob_max = mask_prob_max
        self.mask_schedule = mask_schedule
        self.deterministic_mask = deterministic_mask

    def _current_mask_prob(self) -> float:
        # Linear ramp from min->max across epochs; constant uses max as target
        min_p = float(self.mask_prob_min)
        max_p = float(self.mask_prob_max)
        if self.mask_schedule == 'constant' or self.trainer is None:
            return max_p
        # avoid division by zero when max_epochs==1
        total_epochs = max(1, int(getattr(self.trainer, 'max_epochs', 1)))
        # progress in [0,1]
        progress = 0.0
        try:
            progress = min(1.0, float(self.current_epoch) / float(max(1, total_epochs - 1)))
        except Exception:
            progress = 0.0
        return min_p + (max_p - min_p) * progress

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x, attention_mask=attention_mask)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        if batch is None:
            return None  # skip empty batch
        full_input, full_attn, prompt_lengths = self.process_batch(batch, self.device, self.config.block_size)
        noisy_input, mask_indices = self.forward_process(full_input, prompt_lengths, attn_mask=full_attn)
        
        logits = self(noisy_input, attention_mask=full_attn)
        ce_losses = torch.nn.functional.cross_entropy(
            logits[mask_indices], full_input[mask_indices], reduction='none'
        )
        # Average only over masked positions
        loss = ce_losses.mean()
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            masked_acc = (preds[mask_indices] == full_input[mask_indices]).float().mean() if mask_indices.any() else torch.tensor(0.0, device=loss.device)
            self.log('train/masked_acc', masked_acc, prog_bar=True)
            # Log current learning rate on progress bar
            try:
                if self.trainer is not None and self.trainer.optimizers:
                    lr = self.trainer.optimizers[0].param_groups[0]['lr']
                    self.log('lr', lr, prog_bar=True, on_step=True, on_epoch=False)
                    # Log current masking probability
                    curr_mp = float(self._current_mask_prob())
                    self.log('mask_prob', curr_mp, prog_bar=True, on_step=True, on_epoch=False)
            except Exception:
                pass
        
        self.log('train/loss', loss, prog_bar=True)
        self.total_lengths += full_input.size(1)
        self.log('train/total_lengths', self.total_lengths, prog_bar=False)
        
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        if batch is None:
            return None
        full_input, full_attn, prompt_lengths = self.process_batch(batch, self.device, self.config.block_size, is_val=True)
        input_ids = full_input
        mc_loss = torch.zeros(self.mc_samples, device=self.device)
        
        for i in range(self.mc_samples):
            noisy_input, mask_indices = self.forward_process(input_ids, prompt_lengths, attn_mask=full_attn)
            logits = self(noisy_input, attention_mask=full_attn)
            ce_losses = torch.nn.functional.cross_entropy(
                logits[mask_indices], input_ids[mask_indices], reduction='none'
            )
            # Average only over masked positions
            loss = ce_losses.mean()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                masked_acc = (preds[mask_indices] == input_ids[mask_indices]).float().mean() if mask_indices.any() else torch.tensor(0.0, device=loss.device)
                self.log('val/masked_acc_iter', masked_acc, prog_bar=False)
            mc_loss[i] = loss
        
        val_loss = mc_loss.mean()
        self.log('val/loss', val_loss, prog_bar=True)
        self.log('val/ppl', math.exp(val_loss), prog_bar=True)
        return val_loss
    def configure_optimizers(self):
        # Exclude biases and norm weights from weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_bias = name.endswith('.bias')
            is_norm = ('.norm' in name.lower()) or (param.ndim == 1)
            if is_bias or is_norm:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use regular AdamW without fused operations
        optimizer = AdamW(
            param_groups,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            foreach=False,
        )

        if not self.hparams.decay_lr:
            return optimizer

        # Cosine annealing with warmup, scheduled per step
        warmup_steps = int(self.hparams.warmup_iters)
        min_lr = float(self.hparams.min_lr)
        base_lr = float(self.hparams.learning_rate)
        # Lightning populates this after setup
        total_steps = int(getattr(self.trainer, 'estimated_stepping_batches', 0))
        total_steps = max(total_steps, 1)

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # progress after warmup in [0, 1]
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            # map to [min_lr, base_lr] and return as a factor
            target_lr = min_lr + (base_lr - min_lr) * cosine
            return max(target_lr / base_lr, 0.0)

        from torch.optim.lr_scheduler import LambdaLR

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward_process(self, batch: torch.Tensor, prompt_lengths: torch.Tensor,
                        eps: float = 1e-3, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, l = batch.shape
        mask_indices = torch.zeros((b, l), dtype=torch.bool, device=batch.device)

        # Sample per-sample masking probability from [min, current_or_max], then mask independent random tokens
        lower = float(self.mask_prob_min)
        upper = float(self._current_mask_prob())
        lower = max(0.0, min(1.0, lower))
        upper = max(lower, min(1.0, upper))
        p_mask = lower + (upper - lower) * torch.rand((b,), device=batch.device)

        for i in range(b):
            resp_start = int(prompt_lengths[i].item())
            resp_end = int(attn_mask[i].sum().item()) if attn_mask is not None else l
            if resp_end <= resp_start:
                continue
            resp_len = resp_end - resp_start

            if self.deterministic_mask:
                # Deterministic Bernoulli per token with probability p_mask[i]
                sample_ids = batch[i, :resp_end].detach().to('cpu').numpy().tobytes()
                seed = int.from_bytes(hashlib.sha256(sample_ids).digest()[:8], 'little') & 0x7FFFFFFF
                gen = torch.Generator(device=batch.device)
                gen.manual_seed(seed)
                coin = torch.rand((resp_len,), generator=gen, device=batch.device)
                mask_slice = coin < p_mask[i]
            else:
                mask_slice = torch.rand((resp_len,), device=batch.device) < p_mask[i]

            # Ensure at least one token masked
            if not mask_slice.any():
                mask_slice[resp_len // 2] = True

            mask_indices[i, resp_start:resp_end] = mask_slice
        # Use tokenizer's mask token id consistently across train/val/gen
        noisy_value = torch.tensor(self.tokenizer.mask_id, device=batch.device, dtype=batch.dtype)
        noisy_batch = torch.where(mask_indices, noisy_value, batch)
        return noisy_batch, mask_indices

    def process_batch(self, batch: dict, device: torch.device, block_size: int, is_val: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Handle missing keys gracefully (pure records may omit prompts)
        prompt_ids = batch.get('prompt_ids', torch.empty((len(batch['input_ids']), 0), dtype=torch.long)).to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids, dtype=torch.long)).to(device)
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
def generate_reconstruction(model: TransEncoder, tokenizer: MusicTokenizerWithStyle, prompt_ids: List[int], target_length: int, K: int = 1000, device: str = 'cuda', initial_ids: Optional[List[int]] = None) -> List[int]:
    model.eval()
    print(f"Generating reconstruction on device: {device}")

    mask_id = tokenizer.mask_id
    if initial_ids is None:
        inpaint_ids = [mask_id] * target_length
    else:
        # ensure length matches target_length
        if len(initial_ids) != target_length:
            raise ValueError("initial_ids length must match target_length")
        inpaint_ids = initial_ids
    x = torch.tensor(prompt_ids + inpaint_ids, dtype=torch.long, device=device).unsqueeze(0)
    
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

class InpaintingCallback(Callback):
    def __init__(self, out_dir: Path, tokenizer: MusicTokenizerWithStyle, device: str, max_samples: int = 64):
        self.out_dir = out_dir
        self.tokenizer = tokenizer
        self.device = device
        self.max_samples = max_samples
        
        # Define test tiers: (percentage_to_mask, num_gaps, tier_name)
        self.test_tiers = [
            (0.10, 3, "easy"),     # 10% of sequence masked across 3 gaps
            (0.20, 2, "medium"),   # 20% of sequence masked across 2 gaps  
            (0.40, 1, "hard"),     # 40% of sequence masked in 1 large gap
        ]

    def on_validation_end(self, trainer: L.Trainer, pl_module: DiffusionLightningModule) -> None:
        if trainer.sanity_checking:
            return
        
        # Only run inpainting evaluation on the main process (rank 0) to avoid duplicates
        if trainer.global_rank != 0:
            return

        print(f"\nPerforming multi-tier inpainting reconstruction on validation set after epoch {trainer.current_epoch}...")
        pl_module.eval()
        
        # Model is already on the correct device and dtype through Lightning's trainer
        
        val_loader = trainer.datamodule.val_dataloader()
        if not val_loader:
            print("No validation dataloader available. Skipping inpainting test.")
            return
        
        # Collect up to max_samples from val set
        all_samples = []
        collected = 0
        for batch in val_loader:
            batch_size = batch['input_ids'].shape[0]
            for i in range(batch_size):
                if collected >= self.max_samples:
                    break
                sample = {
                    'prompt_ids': batch['prompt_ids'][i].cpu().tolist(),
                    'input_ids': batch['input_ids'][i].cpu().tolist()
                }
                all_samples.append(sample)
                collected += 1
            if collected >= self.max_samples:
                break
        
        if not all_samples:
            print("No validation samples available. Skipping.")
            return
        
        print(f"Processing {len(all_samples)} validation samples...")
        
        epoch_dir = self.out_dir / f'inpainting_epoch_{trainer.current_epoch:03d}'
        epoch_dir.mkdir(exist_ok=True)
        
        # Store results for all tiers
        all_results = {}
        
        for tier_pct, tier_gaps, tier_name in self.test_tiers:
            print(f"\n{'='*60}")
            print(f"TIER: {tier_name.upper()} - {tier_pct*100:.0f}% masked in {tier_gaps} gap(s)")
            print(f"{'='*60}")
            
            tier_dir = epoch_dir / tier_name
            tier_dir.mkdir(exist_ok=True)
            
            tier_total_accuracy = 0.0
            tier_inpainting_accuracy = 0.0
            num_samples = len(all_samples)
            
            for i, sample in enumerate(all_samples):
                prompt_ids = sample['prompt_ids']
                original_ids = sample['input_ids']
                seq_len = len(original_ids)
                
                # Find non-padding tokens (avoid masking padding)
                non_pad_mask = [token_id != self.tokenizer.pad_id for token_id in original_ids]
                non_pad_indices = [idx for idx, is_non_pad in enumerate(non_pad_mask) if is_non_pad]
                
                if not non_pad_indices:
                    print(f"  Sample {i}: All tokens are padding, skipping...")
                    continue
                
                effective_seq_len = len(non_pad_indices)
                
                # Calculate total tokens to mask for this tier (only from non-padding tokens)
                total_tokens_to_mask = max(1, int(effective_seq_len * tier_pct))
                
                # Create gaps based on tier specifications
                masked_ids = original_ids.copy()
                mask_positions = []
                
                if tier_gaps == 1:
                    # Single large gap - select contiguous non-padding tokens
                    gap_len = total_tokens_to_mask
                    if gap_len <= effective_seq_len:
                        # Choose starting position within non-padding tokens
                        max_start_idx = max(0, effective_seq_len - gap_len)
                        start_idx = random.randint(0, max_start_idx)
                        end_idx = start_idx + gap_len
                        
                        # Convert to actual sequence positions
                        gap_positions = non_pad_indices[start_idx:end_idx]
                        for pos in gap_positions:
                            masked_ids[pos] = self.tokenizer.mask_id
                        mask_positions.extend(gap_positions)
                else:
                    # Multiple gaps - only mask non-padding tokens
                    tokens_per_gap = total_tokens_to_mask // tier_gaps
                    remaining_tokens = total_tokens_to_mask % tier_gaps
                    
                    # Randomly sample non-overlapping positions from non-padding indices
                    available_indices = list(range(effective_seq_len))
                    random.shuffle(available_indices)
                    
                    current_pos = 0
                    for gap_idx in range(tier_gaps):
                        gap_len = tokens_per_gap + (1 if gap_idx < remaining_tokens else 0)
                        
                        if current_pos + gap_len <= len(available_indices):
                            # Get gap positions from shuffled non-padding indices
                            gap_indices = available_indices[current_pos:current_pos + gap_len]
                            # Convert back to actual sequence positions
                            gap_positions = [non_pad_indices[idx] for idx in gap_indices]
                            
                            for pos in gap_positions:
                                masked_ids[pos] = self.tokenizer.mask_id
                            mask_positions.extend(gap_positions)
                            current_pos += gap_len
                        else:
                            print(f"  Warning: Not enough non-padding tokens for gap {gap_idx + 1} in sample {i}")
                            break
                
                # Generate reconstruction
                generated = generate_reconstruction(
                    pl_module.model,
                    self.tokenizer,
                    prompt_ids,
                    len(original_ids),
                    K=1000,
                    device=self.device,
                    initial_ids=masked_ids
                )
                
                # Calculate accuracies
                full_accuracy = (torch.tensor(generated) == torch.tensor(original_ids)).float().mean().item()
                tier_total_accuracy += full_accuracy
                
                # Inpainting accuracy (only on masked positions)
                if mask_positions:
                    masked_original = [original_ids[pos] for pos in mask_positions]
                    masked_generated = [generated[pos] for pos in mask_positions]
                    inpainting_accuracy = (torch.tensor(masked_generated) == torch.tensor(masked_original)).float().mean().item()
                else:
                    inpainting_accuracy = 1.0
                tier_inpainting_accuracy += inpainting_accuracy
                
                # Save detailed results for this tier
                output_file = tier_dir / f'sample_{i:03d}_{tier_name}.txt'
                with open(output_file, 'w') as f:
                    f.write(f"Sample {i} - {tier_name.upper()} Tier Inpainting Results\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Tier: {tier_name} ({tier_pct*100:.0f}% masked in {tier_gaps} gap(s))\n")
                    f.write(f"Total sequence length: {seq_len}\n")
                    f.write(f"Non-padding sequence length: {effective_seq_len}\n")
                    f.write(f"Target tokens to mask: {total_tokens_to_mask}\n")
                    f.write(f"Actual tokens masked: {len(mask_positions)}\n")
                    f.write(f"Masking efficiency: {len(mask_positions)/total_tokens_to_mask*100:.1f}%\n")
                    f.write(f"Percentage of non-padding masked: {len(mask_positions)/effective_seq_len*100:.1f}%\n\n")
                    f.write(f"Full sequence accuracy: {full_accuracy * 100:.2f}%\n")
                    f.write(f"Inpainting accuracy: {inpainting_accuracy * 100:.2f}%\n\n")
                    f.write(f"Masked positions: {mask_positions[:30]}{'...' if len(mask_positions) > 30 else ''}\n\n")
                    f.write("Decoded tokens:\n")
                    f.write(f"Prompt: {self.tokenizer.decode_tokens(prompt_ids)}\n\n")
                    f.write(f"Original: {self.tokenizer.decode_tokens(original_ids)}\n\n")
                    f.write(f"Generated: {self.tokenizer.decode_tokens(generated)}\n\n")
                    f.write(f"Masked input: {self.tokenizer.decode_tokens(prompt_ids + masked_ids)}\n")
                
                print(f"  Sample {i}: Full: {full_accuracy*100:.1f}%, Inpainting: {inpainting_accuracy*100:.1f}%")
            
            # Calculate tier averages
            avg_full = tier_total_accuracy / num_samples
            avg_inpainting = tier_inpainting_accuracy / num_samples
            
            # Store results for overall summary
            all_results[tier_name] = {
                'avg_full_accuracy': avg_full,
                'avg_inpainting_accuracy': avg_inpainting,
                'percentage_masked': tier_pct,
                'num_gaps': tier_gaps
            }
            
            print(f"\n{tier_name.upper()} TIER SUMMARY:")
            print(f"  Average full sequence accuracy: {avg_full * 100:.2f}%")
            print(f"  Average inpainting accuracy: {avg_inpainting * 100:.2f}%")
            
            # Save tier summary
            tier_summary_file = tier_dir / f'{tier_name}_summary.txt'
            with open(tier_summary_file, 'w') as f:
                f.write(f"{tier_name.upper()} Tier Inpainting Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Tier configuration:\n")
                f.write(f"  - Percentage masked: {tier_pct*100:.0f}%\n")
                f.write(f"  - Number of gaps: {tier_gaps}\n")
                f.write(f"  - Difficulty: {tier_name}\n\n")
                f.write(f"Results:\n")
                f.write(f"  - Total samples: {num_samples}\n")
                f.write(f"  - Average full sequence accuracy: {avg_full * 100:.2f}%\n")
                f.write(f"  - Average inpainting accuracy: {avg_inpainting * 100:.2f}%\n")
        
        # Save overall summary comparing all tiers for this epoch
        overall_summary_file = epoch_dir / 'overall_summary.txt'
        with open(overall_summary_file, 'w') as f:
            f.write(f"Epoch {trainer.current_epoch} Multi-Tier Inpainting Test Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total samples per tier: {num_samples}\n")
            f.write(f"Reconstruction steps (K): 1000\n\n")
            
            f.write("TIER COMPARISON:\n")
            f.write("-" * 40 + "\n")
            for tier_name, results in all_results.items():
                f.write(f"{tier_name.upper()} ({results['percentage_masked']*100:.0f}% in {results['num_gaps']} gap(s)):\n")
                f.write(f"  Full accuracy:      {results['avg_full_accuracy']*100:.2f}%\n")
                f.write(f"  Inpainting accuracy: {results['avg_inpainting_accuracy']*100:.2f}%\n")
                f.write("\n")
            
            f.write("DIFFICULTY PROGRESSION:\n")
            f.write("-" * 40 + "\n")
            sorted_tiers = sorted(all_results.items(), key=lambda x: x[1]['avg_inpainting_accuracy'], reverse=True)
            for i, (tier_name, results) in enumerate(sorted_tiers, 1):
                f.write(f"{i}. {tier_name.upper()}: {results['avg_inpainting_accuracy']*100:.2f}% inpainting accuracy\n")
        
        print(f"\n{'='*60}")
        print(f"EPOCH {trainer.current_epoch} OVERALL SUMMARY:")
        print(f"{'='*60}")
        for tier_name, results in all_results.items():
            print(f"{tier_name.upper()}: {results['avg_inpainting_accuracy']*100:.2f}% inpainting accuracy")
        print(f"\nDetailed results saved to: {epoch_dir}")
        print(f"Overall summary saved to: {overall_summary_file}")

def main():
    args = parse_args()
    tokenizer = MusicTokenizerWithStyle()
    # Construct config with correct vocab and block size so padded_vocab_size is consistent
    config = Config.from_name(f'Diff_LLaMA_{args.model}M', vocab_size=tokenizer.vocab_size, block_size=6144)
    data_cfg = DataCfg(
        train_pkl=args.train_pkl,
        val_pkl=args.val_pkl,
        max_len=config.block_size,
        seq_limit=None,
        shuffle_records=True,
        skip_long_after_tokenization=True,
        augment=False,
        block_size=config.block_size,
        overfit_single_batch=args.overfit_single_batch,
        objective=args.objective,
        span_mask_prob=args.mask_prob_max,
        print_sample_batch=args.print_sample_batch,
    )
    datamodule = MusicDataModule(cfg=data_cfg, tokenizer=tokenizer, batch_size=args.batch_size, num_workers=args.num_workers)
    
    model = DiffusionLightningModule(
        config=config,
        learning_rate=args.lr,
        weight_decay=0.0 if args.overfit_single_batch else args.weight_decay,
        beta1=0.9,
        beta2=0.95,
        warmup_iters=args.warmup_iters,
        min_lr=args.min_lr,
        decay_lr=args.decay_lr,
        grad_clip=1.0,
        tokenizer=tokenizer,
        eval_iters=1,
        mask_prob_min=args.mask_prob_min,
        mask_prob_max=args.mask_prob_max,
        mask_schedule=args.mask_schedule,
        deterministic_mask=args.deterministic_mask,
    )
    
    out_dir = Path('workdir') / f'mdm-{args.model}M-{args.max_epochs}'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_logger = CSVLogger(save_dir=out_dir, name='csv_logs')
    wandb_logger = WandbLogger(name=f'mdm-{args.model}M-{args.max_epochs}-mc', save_dir=out_dir, project=args.wandb_project)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    callbacks = [
        ModelCheckpoint(dirpath=out_dir, filename='epoch-{epoch:02d}-ckpt', save_top_k=-1, every_n_epochs=1),
        LearningRateMonitor(logging_interval='step'),
        InpaintingCallback(out_dir, tokenizer, device, max_samples=4),
    ]
    # Optionally add SWA towards the end
    if args.use_swa:
        callbacks.append(StochasticWeightAveraging(swa_lrs=args.swa_lr, swa_epoch_start=args.swa_epoch_start))
    
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=('32-true' if args.overfit_single_batch else get_default_supported_precision(training=True)),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        num_nodes=args.nodes_num,
        strategy='ddp_find_unused_parameters_true' if args.nodes_num > 1 or args.devices > 1 else 'auto',
        logger=[csv_logger, wandb_logger],
        callbacks=callbacks,
        log_every_n_steps=10,
        limit_val_batches=4,
    )
    
    print(f"Total parameters: {num_parameters(model.model):,}")

    # Print full training configuration before starting
    try:
        config_dict = asdict(config)
    except Exception:
        config_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith('_') and not callable(getattr(config, k))}
    try:
        data_cfg_dict = asdict(data_cfg)
    except Exception:
        data_cfg_dict = data_cfg.__dict__
    try:
        hparams_dict = dict(model.hparams)
    except Exception:
        hparams_dict = {}
    trainer_cfg = {
        "max_epochs": args.max_epochs,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "precision": ('32-true' if args.overfit_single_batch else get_default_supported_precision(training=True)),
        "accelerator": ('gpu' if torch.cuda.is_available() else 'cpu'),
        "devices": args.devices,
        "num_nodes": args.nodes_num,
        "strategy": ('ddp_find_unused_parameters_true' if args.nodes_num > 1 or args.devices > 1 else 'auto'),
        "use_swa": args.use_swa,
        "swa_lr": args.swa_lr,
        "swa_epoch_start": args.swa_epoch_start,
        "wandb_project": args.wandb_project,
    }
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print("Model Config:")
    print(json.dumps(config_dict, indent=2, sort_keys=True))
    print("\nData Config:")
    print(json.dumps(data_cfg_dict, indent=2, sort_keys=True))
    print("\nOptim/Mask/Module Hyperparams:")
    print(json.dumps(hparams_dict, indent=2, sort_keys=True, default=str))
    print("\nTrainer Config:")
    print(json.dumps(trainer_cfg, indent=2, sort_keys=True))
    print("="*80 + "\n")
    
    # Resume from checkpoint if specified
    ckpt_path = None
    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        print(f"Resuming training from checkpoint: {ckpt_path}")
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=6, help='model parameters')
    parser.add_argument('--nodes_num', type=int, default=1, help='number of nodes')
    parser.add_argument('--devices', type=int, default=1, help='number of devices')
    parser.add_argument('--batch_size', type=int, default=1, help='global_batch_size')
    parser.add_argument('--train_pkl', type=str, default="cache/dataset_paths_synthetic_structured-aria-unique_limitNone_37776ff2_train.pkl", help='Path to training PKL file')
    parser.add_argument('--val_pkl', type=str, default="cache/dataset_paths_synthetic_structured-aria-unique_limitNone_37776ff2_val.pkl", help='Path to validation PKL file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--max_epochs', type=int, default=1, help='Maximum training epochs')
    
    parser.add_argument('--objective', type=str, default='pure', choices=['prompt', 'pure'], help='Dataset objective mode ("prompt" or "pure")')
    parser.add_argument('--mask_prob_min', type=float, default=0.05, help='Minimum mask prob (lower bound)')
    parser.add_argument('--mask_prob_max', type=float, default=0.30, help='Maximum mask prob (upper bound)')
    parser.add_argument('--mask_schedule', type=str, default='linear', choices=['linear', 'constant'], help='How masking probability evolves with epoch (linear or constant)')
    parser.add_argument('--deterministic_mask', action='store_true', help='Use deterministic span masking per sample for sanity checks')
    parser.add_argument('--print_sample_batch', action='store_true', help='Print sample batch info for debugging')
    parser.add_argument('--overfit_single_batch', action='store_true', help='Overfit on single batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    # Scheduler & training system
    parser.add_argument('--warmup_iters', type=int, default=400, help='Warmup steps for LR scheduler')
    parser.add_argument('--decay_lr', action='store_true', help='Enable cosine LR decay with warmup')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum LR for cosine schedule')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--wandb_project', type=str, default='scaling', help='Weights & Biases project name')
    # SWA
    parser.add_argument('--use_swa', action='store_true', help='Enable stochastic weight averaging')
    parser.add_argument('--swa_lr', type=float, default=None, help='SWA learning rate (defaults to scheduler LR)')
    parser.add_argument('--swa_epoch_start', type=int, default=None, help='Epoch to start SWA (defaults to 75% of max_epochs)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from')
    return parser.parse_args()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()