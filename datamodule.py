import logging
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tokenizer import (
    MusicTokenizerWithStyle,
    PROMPT_END_TOKEN,
    PROMPT_START_TOKEN,
    SECTION_TOKENS,
)
from utils import get_style_labels, music_style_from_labels

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
    objective: str = "prompt"  # 'prompt' or 'pure'
    span_mask_prob: float = 0.3
    print_sample_batch: bool = False

_SNIPPET_LEN = 256

class MidiDataset(Dataset):
    def __init__(
        self,
        cfg: DataCfg,
        tokenizer: MusicTokenizerWithStyle,
        pkl_path: str,
        is_val: bool = False,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.is_val = is_val
        self.pad_id = tokenizer.pad_id
        self.mask_id = tokenizer.mask_id

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            self.records = data["data_records"]
            self.metadata = data.get("metadata", {})

        if cfg.seq_limit:
            self.records = self.records[: cfg.seq_limit]

        self.aug_fns = []
        if cfg.augment and not is_val:
            base = tokenizer._base_tokenizer
            self.aug_fns = [
                base.export_pitch_aug(5),
                base.export_velocity_aug(10),
                base.export_tempo_aug(0.1, mixup=False),
            ]

    def __len__(self) -> int:
        return len(self.records)

    def _augment_tokens(self, tokens: List[str]) -> List[str]:
        if not self.aug_fns:
            return tokens
        special_indices = [
            i for i, token in enumerate(tokens) if token in SECTION_TOKENS.values()
        ]
        special_tokens = [tokens[i] for i in special_indices]
        regular_tokens = [
            token for i, token in enumerate(tokens) if i not in special_indices
        ]
        augmented = regular_tokens
        for fn in self.aug_fns:
            augmented = fn(augmented)
        result = list(augmented)
        for idx, token in zip(special_indices, special_tokens):
            result.insert(idx, token)
        return result

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        record = self.records[idx]
        # 1) Support pre-tokenized records (pure sequences): {'input_ids': List[int]} or {'tokens': List[str]}
        if self.cfg.objective == "pure" and ("input_ids" in record or "tokens" in record):
            if "input_ids" in record and isinstance(record["input_ids"], list) and len(record["input_ids"]) > 0:
                input_ids = record["input_ids"]
                return self._truncate_and_return([], input_ids)
            if "tokens" in record and isinstance(record["tokens"], list) and len(record["tokens"]) > 0:
                input_ids = self.tokenizer.encode_tokens(record["tokens"])  # type: ignore[arg-type]
                return self._truncate_and_return([], input_ids)

        # 2) Default: tokenize from MIDI file path
        midi_path = record.get("midi_file_path")
        if not midi_path or not os.path.exists(midi_path):
            return None

        tokens = self.tokenizer.tokenize_from_file(midi_path)
        if not tokens:
            return None

        if self.cfg.augment and not self.is_val:
            tokens = self._augment_tokens(tokens)

        if self.cfg.objective == "prompt":
            return self._get_prompt_item(tokens, record, midi_path)
        elif self.cfg.objective == "pure":
            return self._get_pure_item(tokens)
        else:
            raise ValueError(f"Unknown objective: {self.cfg.objective}")

    def _get_prompt_item(
        self, tokens: List[str], record: Dict[str, Any], midi_path: str
    ) -> Optional[Dict[str, Any]]:
        style_path = record.get("style_file_path")
        style_labels = get_style_labels(style_path)
        if not style_labels or len(tokens) != len(style_labels):
            return None

        music_style = music_style_from_labels(style_labels)
        if not music_style:
            return None

        tokens_no_prefix = self.tokenizer.remove_instrument_prefix(tokens)
        prompt_tokens = [PROMPT_START_TOKEN]

        runs = []
        start_idx = 0
        for i in range(1, len(style_labels)):
            if style_labels[i] != style_labels[i - 1]:
                runs.append((style_labels[i - 1], start_idx, i - 1))
                start_idx = i
        if style_labels:
            runs.append((style_labels[-1], start_idx, len(style_labels) - 1))

        for style_char in music_style:
            for lab, s, e in runs:
                if lab == style_char:
                    prompt_tokens.append(SECTION_TOKENS[lab])
                    seg_end = min(e + 1, s + _SNIPPET_LEN)
                    if s < len(tokens_no_prefix):
                        seg = tokens_no_prefix[s:seg_end]
                        prompt_tokens.extend(seg)
                    break
        prompt_tokens.append(PROMPT_END_TOKEN)

        prompt_ids = self.tokenizer.encode_tokens(prompt_tokens)
        input_ids = self.tokenizer.encode_tokens(tokens)

        return self._truncate_and_return(prompt_ids, input_ids)

    def _get_pure_item(self, tokens: List[str]) -> Optional[Dict[str, Any]]:
        input_ids = self.tokenizer.encode_tokens(tokens)

        # REMOVED: Data-level masking - now only happens at model level
        # if not self.is_val and random.random() < self.cfg.span_mask_prob:
        #     start = random.randint(0, len(input_ids) - 2)
        #     end = min(start + random.randint(100, 500), len(input_ids))
        #     input_ids[start:end] = [self.mask_id] * (end - start)

        return self._truncate_and_return([], input_ids)

    def _truncate_and_return(
        self, prompt_ids: List[int], input_ids: List[int]
    ) -> Optional[Dict[str, Any]]:
        if len(prompt_ids) > self.cfg.max_len or len(input_ids) > self.cfg.max_len:
            if self.cfg.skip_long_after_tokenization:
                return None
            prompt_ids = prompt_ids[: self.cfg.max_len]
            input_ids = input_ids[: self.cfg.max_len]

        if not input_ids:
            return None

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }

def midi_collate_fn(
    batch: List[Optional[Dict[str, Any]]], tokenizer: MusicTokenizerWithStyle
) -> Optional[Dict[str, Any]]:
    batch = [item for item in batch if item]
    if not batch:
        return None

    result: Dict[str, Any] = {}
    # Some pure-mode records may not include prompt_ids; build safely
    input_tensors = [item["input_ids"] for item in batch if item.get("input_ids") is not None and item["input_ids"].numel() > 0]
    result["input_ids"] = (
        pad_sequence(input_tensors, batch_first=True, padding_value=tokenizer.pad_id)
        if input_tensors else torch.empty((len(batch), 0), dtype=torch.long)
    )
    prompt_tensors = [item.get("prompt_ids", torch.empty(0, dtype=torch.long)) for item in batch]
    prompt_tensors = [t for t in prompt_tensors if t is not None and t.numel() > 0]
    result["prompt_ids"] = (
        pad_sequence(prompt_tensors, batch_first=True, padding_value=tokenizer.pad_id)
        if prompt_tensors else torch.empty((len(batch), 0), dtype=torch.long)
    )

    masks = [item.get("attention_mask", torch.empty(0, dtype=torch.long)) for item in batch]
    masks = [m for m in masks if m is not None and m.numel() > 0]
    result["attention_mask"] = (
        pad_sequence(masks, batch_first=True, padding_value=0)
        if masks
        else torch.empty((len(batch), 0), dtype=torch.long)
    )

    if result["input_ids"].shape[1] == 0:
        return None
    return result

class MusicDataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DataCfg,
        tokenizer: Optional[MusicTokenizerWithStyle] = None,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer or MusicTokenizerWithStyle()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset: Optional[MidiDataset] = None
        self.val_dataset: Optional[MidiDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if not Path(self.cfg.train_pkl).exists():
                raise FileNotFoundError(f"Training PKL not found: {self.cfg.train_pkl}")

            self.train_dataset = MidiDataset(
                self.cfg, self.tokenizer, self.cfg.train_pkl
            )

            if self.cfg.shuffle_records:
                random.shuffle(self.train_dataset.records)
            if self.cfg.overfit_single_batch:
                # Take first batch_size records and duplicate until we have 800 total
                base_records = self.train_dataset.records[: self.batch_size]
                target_count = 2000
                multiplier = (target_count + len(base_records) - 1) // len(base_records)  # Ceiling division
                records = base_records * multiplier
                self.train_dataset.records = records[:target_count]  # Trim to exactly 800
                self.val_dataset = self.train_dataset
            elif self.cfg.val_pkl:
                if not Path(self.cfg.val_pkl).exists():
                    raise FileNotFoundError(
                        f"Validation PKL not found: {self.cfg.val_pkl}"
                    )
                self.val_dataset = MidiDataset(
                    self.cfg, self.tokenizer, self.cfg.val_pkl, is_val=True
                )

            if self.cfg.print_sample_batch:
                self._print_sample_batch()

    def train_dataloader(self) -> Optional[DataLoader]:
        if not self.train_dataset:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda b: midi_collate_fn(b, self.tokenizer),
            shuffle=not self.cfg.overfit_single_batch,
            pin_memory=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda b: midi_collate_fn(b, self.tokenizer),
            shuffle=False,
            pin_memory=True,
        )

    def _print_sample_batch(self) -> None:
        print("\n" + "=" * 80)
        print("SAMPLE BATCH DEBUG INFO")
        print("=" * 80)

        if not self.train_dataset:
            print("No training dataset available.")
            return

        loader = DataLoader(
            self.train_dataset,
            batch_size=min(2, self.batch_size),
            num_workers=0,
            collate_fn=lambda b: midi_collate_fn(b, self.tokenizer),
            shuffle=False,
        )

        try:
            batch = next(iter(loader))
            if batch is None:
                print("Batch is None.")
                return

            print(f"Batch keys: {list(batch.keys())}")

            for key, tensor in batch.items():
                print(f"\n{key.upper()}:")
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                if tensor.numel() == 0:
                    continue

                print(f"  Min value: {tensor.min().item()}")
                print(f"  Max value: {tensor.max().item()}")

                for i in range(min(tensor.shape[0], 2)):
                    sample = tensor[i]
                    print(f"  Sample {i} first 10 values: {sample[:10].tolist()}")
                    print(f"  Sample {i} last 10 values: {sample[-10:].tolist()}")

                    if key in ["input_ids", "prompt_ids"]:
                        non_pad = sample[sample != self.tokenizer.pad_id]
                        if non_pad.numel() > 0:
                            decoded = self.tokenizer.decode_tokens(non_pad.tolist())
                            decoded_preview = (
                                " ".join(map(str, decoded))[:500] + "..."
                                if len(decoded) > 500
                                else " ".join(map(str, decoded))
                            )
                            print(f"  Sample {i} decoded: {decoded_preview}")

                            first_10_dec = self.tokenizer.decode_tokens(
                                sample[:10].tolist()
                            )
                            print(
                                f"  Sample {i} first 10 decoded: {' '.join(map(str, first_10_dec))}"
                            )

                            last_10_dec = self.tokenizer.decode_tokens(
                                sample[-10:].tolist()
                            )
                            print(
                                f"  Sample {i} last 10 decoded: {' '.join(map(str, last_10_dec))}"
                            )

        except Exception as e:
            print(f"Error creating sample batch: {e}")
            import traceback

            traceback.print_exc()

        print("=" * 80)
        print("END SAMPLE BATCH DEBUG INFO")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )
    import argparse

    parser = argparse.ArgumentParser(description="Test MusicDataModule")
    parser.add_argument(
        "--train_pkl",
        type=str,
        default="cache/dataset_paths_synthetic_structured-aria-unique_limitNone_37776ff2_train.pkl",
    )
    parser.add_argument(
        "--val_pkl",
        type=str,
        default="cache/dataset_paths_synthetic_structured-aria-unique_limitNone_37776ff2_val.pkl",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--seq_limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer = MusicTokenizerWithStyle()
    cfg = DataCfg(
        train_pkl=args.train_pkl,
        val_pkl=args.val_pkl,
        max_len=args.max_len,
        seq_limit=args.seq_limit,
        shuffle_records=False,
        augment=True,
        objective='pure',
        span_mask_prob=0.3,
        print_sample_batch=True
    )
    dm = MusicDataModule(cfg=cfg, tokenizer=tokenizer, batch_size=args.batch_size, num_workers=args.num_workers)
    dm.setup(stage="fit")