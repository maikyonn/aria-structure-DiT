
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any
import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torch.nn.utils.rnn import pad_sequence
from tokenizer import (
    MusicTokenizerWithStyle,
    PROMPT_START_TOKEN, PROMPT_END_TOKEN,
    A_SECTION_TOKEN, B_SECTION_TOKEN, C_SECTION_TOKEN, D_SECTION_TOKEN,
    SECTION_TOKENS
)
from utils import music_style_from_labels, get_style_labels

logger = logging.getLogger(__name__)

@dataclass
class DataCfg:
    train_pkl: str
    val_pkl: Optional[str] = None
    max_len: int = 2048
    seq_limit: Optional[int] = None
    shuffle_records: bool = True
    skip_long_after_tokenization: bool = True
    augment: bool = True
    block_size: int = 2048
    overfit_single_batch: bool = False

_SNIPPET_LEN = 256

class LazyMidiDataset(Dataset):
    def __init__(self, cfg: DataCfg, tokenizer: MusicTokenizerWithStyle, path_pkl_file: str, is_validation_set: bool = False):
        self.cfg = cfg
        self.tok = tokenizer
        self.path_pkl_file = path_pkl_file
        self.is_validation_set = is_validation_set
        self.pad_id = tokenizer.pad_id
        self.mask_id = tokenizer.mask_id
        logger.info(f"Initializing LazyMidiDataset with path_pkl_file: {path_pkl_file}")
        
        try:
            with open(path_pkl_file, "rb") as f:
                loaded_data = pickle.load(f)
            self._records = loaded_data["data_records"]
            self.metadata = loaded_data.get("metadata", {})
            logger.info(f"Loaded {len(self._records)} path records. Metadata: {self.metadata}")
        except Exception as e:
            logger.error(f"Failed to load PKL file {path_pkl_file}: {e}", exc_info=True)
            self._records = []
            self.metadata = {}

        if self.cfg.seq_limit and self.cfg.seq_limit < len(self._records):
            logger.info(f"Using seq_limit: {self.cfg.seq_limit} samples out of {len(self._records)}")
            self._records = self._records[:self.cfg.seq_limit]

        self.aug_fns = []
        if self.cfg.augment and not self.is_validation_set:
            logger.info("Augmentation ENABLED: pitch=5, velocity=10, tempo=0.1")
            base = tokenizer._base_tokenizer
            self.aug_fns.append(base.export_pitch_aug(5))
            self.aug_fns.append(base.export_velocity_aug(10))
            self.aug_fns.append(base.export_tempo_aug(0.1, mixup=False))
        else:
            logger.info("Augmentation DISABLED")

    def _apply_aug(self, tokens: List[str]) -> List[str]:
        if not self.aug_fns:
            return tokens
        special_indices = [i for i, token in enumerate(tokens) if token in SECTION_TOKENS.values()]
        special_tokens = [tokens[i] for i in special_indices]
        regular_tokens = [token for i, token in enumerate(tokens) if i not in special_indices]
        augmented = regular_tokens
        for fn in self.aug_fns:
            augmented = fn(augmented)
        result = list(augmented)
        for idx, token in zip(special_indices, special_tokens):
            if idx <= len(result):
                result.insert(idx, token)
            else:
                result.append(token)
        return result

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx >= len(self._records):
            raise IndexError("Index out of bounds")
        
        record = self._records[idx]
        midi_file_path = record.get("midi_file_path")
        style_file_path = record.get("style_file_path")
        
        if not midi_file_path or not os.path.exists(midi_file_path):
            logger.warning(f"MIDI file not found: {midi_file_path} at index {idx}")
            return None
        
        tokens = self.tok.tokenize_from_file(midi_file_path)
        if not tokens:
            return None
        
        style_labels = get_style_labels(style_file_path)
        if not style_labels or len(tokens) != len(style_labels):
            logger.warning(f"Style labels invalid or mismatched for {midi_file_path}")
            return None

        if self.cfg.augment and not self.is_validation_set:
            tokens = self._apply_aug(tokens)

        original_tokens = self.tok.tokenize_from_file(midi_file_path)
        if not original_tokens:
            return None
        
        tokens_no_prefix = self.tok.remove_instrument_prefix(tokens)
        music_style = music_style_from_labels(style_labels)
        if not music_style:
            return None

        enc_prompt_tokens = [PROMPT_START_TOKEN]
        runs = []
        start_idx = 0
        for i in range(1, len(style_labels)):
            if style_labels[i] != style_labels[i-1]:
                runs.append((style_labels[i-1], start_idx, i-1))
                start_idx = i
        if style_labels:
            runs.append((style_labels[-1], start_idx, len(style_labels)-1))

        for style_char in music_style:
            for lab, s, e in runs:
                if lab == style_char:
                    enc_prompt_tokens.append(SECTION_TOKENS[lab])
                    seg_end = min(e + 1, s + _SNIPPET_LEN)
                    if s < len(tokens_no_prefix):
                        seg = tokens_no_prefix[s:seg_end]
                        enc_prompt_tokens.extend(seg)
                    break
        enc_prompt_tokens.append(PROMPT_END_TOKEN)

        enc_ids = self.tok.encode_tokens(enc_prompt_tokens)
        dec_ids = self.tok.encode_tokens(original_tokens)
        
        if len(enc_ids) > self.cfg.max_len or len(dec_ids) > self.cfg.max_len:
            if self.cfg.skip_long_after_tokenization:
                return None
            enc_ids = enc_ids[:self.cfg.max_len]
            dec_ids = dec_ids[:self.cfg.max_len]
        
        if not enc_ids or not dec_ids:
            return None

        return {
            "input_ids": torch.tensor(dec_ids, dtype=torch.long),
            "prompt_ids": torch.tensor(enc_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(dec_ids), dtype=torch.long)
        }

def midi_collate_fn(batch: List[Optional[Dict[str, Any]]], tokenizer: MusicTokenizerWithStyle) -> Optional[Dict[str, Any]]:
    batch = [b for b in batch if b is not None and isinstance(b, dict) and b]
    if not batch:
        logger.warning("Collate: empty batch after filtering")
        return None
    
    result = {}
    for key in ['input_ids', 'prompt_ids']:
        tensor_list = [b[key] for b in batch if key in b and b[key].numel() > 0]
        result[key] = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_id) if tensor_list else torch.empty((len(batch), 0), dtype=torch.long)
    
    tensor_list = [b['attention_mask'] for b in batch if 'attention_mask' in b and b['attention_mask'].numel() > 0]
    result['attention_mask'] = pad_sequence(tensor_list, batch_first=True, padding_value=0) if tensor_list else torch.empty((len(batch), 0), dtype=torch.long)
    
    if result['input_ids'].shape[1] == 0:
        logger.warning("Collate: all input_ids empty")
        return None
    return result

class MusicDataModule(L.LightningDataModule):
    def __init__(self, cfg: DataCfg, tokenizer: MusicTokenizerWithStyle, batch_size: int = 8, num_workers: int = 0):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            # Debug: Check if train_pkl file exists
            logger.info(f"Attempting to load training dataset from: {self.cfg.train_pkl}")
            if not os.path.exists(self.cfg.train_pkl):
                logger.error(f"Training PKL file does not exist: {self.cfg.train_pkl}")
                return
            
            # Load training dataset from PKL file
            self.train_dataset = LazyMidiDataset(self.cfg, self.tokenizer, self.cfg.train_pkl, is_validation_set=False)
            
            # Debug: Check if dataset was created successfully
            if self.train_dataset is None:
                logger.error("Failed to create training dataset")
                return
            
            logger.info(f"Training dataset created with {len(self.train_dataset)} samples")
            
            # Shuffle records if needed
            if self.cfg.shuffle_records:
                random.shuffle(self.train_dataset._records)
            
            if self.cfg.overfit_single_batch:
                self.train_dataset._records = self.train_dataset._records[:self.batch_size]
                self.val_dataset = self.train_dataset  # Use same for validation
                logger.info(f"Overfitting on single batch: limited to {self.batch_size} samples")
            else:
                # Load validation dataset if specified
                if self.cfg.val_pkl:
                    logger.info(f"Attempting to load validation dataset from: {self.cfg.val_pkl}")
                    if not os.path.exists(self.cfg.val_pkl):
                        logger.error(f"Validation PKL file does not exist: {self.cfg.val_pkl}")
                    else:
                        self.val_dataset = LazyMidiDataset(self.cfg, self.tokenizer, self.cfg.val_pkl, is_validation_set=True)
                        if self.val_dataset:
                            logger.info(f"Validation dataset created with {len(self.val_dataset)} samples")

    def train_dataloader(self):
        if not self.train_dataset:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: midi_collate_fn(batch, self.tokenizer),
            shuffle=not self.cfg.overfit_single_batch,  # No shuffle for overfit
            pin_memory=True
        )

    def val_dataloader(self):
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: midi_collate_fn(batch, self.tokenizer),
            shuffle=False,
            pin_memory=True
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s %(name)s %(process)d: %(message)s')
    import argparse
    parser = argparse.ArgumentParser(description="Test MusicDataModule")
    parser.add_argument("--train_pkl", type=str, default="cache/dataset_paths_synthetic_structured-aria-unique_limitNone_37776ff2_train.pkl", help="Path to training PKL file")
    parser.add_argument("--val_pkl", type=str, default="cache/dataset_paths_synthetic_structured-aria-unique_limitNone_37776ff2_val.pkl", help="Path to validation PKL file")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--seq_limit", type=int, default=None)
    args = parser.parse_args()

    tok = MusicTokenizerWithStyle()
    data_cfg = DataCfg(
        train_pkl=args.train_pkl,
        val_pkl=args.val_pkl,
        max_len=args.max_len,
        seq_limit=args.seq_limit,
        shuffle_records=True,
        augment=True
    )
    dm = MusicDataModule(cfg=data_cfg, tokenizer=tok, batch_size=args.batch_size, num_workers=args.num_workers)
    dm.setup(stage='fit')
    for split in ['train', 'val']:
        logger.info(f"\n--- Checking {split} dataloader ---")
        loader = getattr(dm, f"{split}_dataloader")()
        if loader:
            logger.info(f"DataLoader for {split} created. Length: {len(loader.dataset)}")
            for i, batch in enumerate(loader):
                if batch is None:
                    logger.warning(f"{split.capitalize()} Batch {i+1} is None")
                    continue
                logger.info(f"{split.capitalize()} Batch {i+1} keys: {batch.keys()}")
                for k, v in batch.items():
                    logger.info(f" {k}: shape={v.shape}, dtype={v.dtype}")
                
                # Print detokenized prompt_ids and input_ids for first batch
                if i == 0:
                    if 'prompt_ids' in batch and 'input_ids' in batch:
                        prompt_tokens = tok.decode_tokens(batch['prompt_ids'][0].tolist())
                        input_tokens = tok.decode_tokens(batch['input_ids'][0].tolist())
                        logger.info(f"\n{split.capitalize()} Sample 0 - Detokenized Prompt: {prompt_tokens[:5000]}...")
                        logger.info(f"{split.capitalize()} Sample 0 - Detokenized Input: {input_tokens[:5000]}...")
                
                if i >= 1:
                    break
        else:
            logger.info(f"{split.capitalize()} dataloader is None")