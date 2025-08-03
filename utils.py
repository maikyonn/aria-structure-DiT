import math
import re
from itertools import groupby
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional, Union
import logging
import random
from tokenizer import (
    MusicTokenizerWithStyle, STYLE_LABEL_MAP, ID_TO_STYLE_MAP,
    PROMPT_START_TOKEN, PROMPT_END_TOKEN, A_SECTION_TOKEN, B_SECTION_TOKEN,
    C_SECTION_TOKEN, D_SECTION_TOKEN, SECTION_TOKENS, STYLE_TO_SNIPPET_IDX
)
import torch

logger = logging.getLogger(__name__)

def condense_style_sequence(labels: Sequence[str]) -> str:
    if not labels:
        return ""
    parts = [f"{label}x{sum(1 for _ in group)}" for label, group in groupby(labels)]
    return ", ".join(parts)

def filter_significant_styles(labels: Sequence[str], min_prop: float = 0.05) -> List[str]:
    if len(labels) <= 1:
        return list(labels)
    N = len(labels)
    threshold = math.ceil(min_prop * N)
    runs = [(label, sum(1 for _ in grp)) for label, grp in groupby(labels)]
    kept_runs = [(lab, ln) for lab, ln in runs if ln >= threshold]
    if not kept_runs:
        return []
    kept_len = sum(ln for _, ln in kept_runs)
    if kept_len == N:
        expanded = [lab for lab, ln in kept_runs for _ in range(ln)]
    else:
        scale = N / kept_len
        new_lengths = [max(1, round(ln * scale)) for _, ln in kept_runs]
        diff = N - sum(new_lengths)
        for i in range(abs(diff)):
            j = i % len(new_lengths)
            new_lengths[j] += 1 if diff > 0 else -1
        expanded = [lab for (lab, _), ln_new in zip(kept_runs, new_lengths) for _ in range(ln_new)]
    return relative_label_sequence(expanded)

def relative_label_sequence(labels: Iterable[str]) -> List[str]:
    mapping = {}
    next_ord = ord("A")
    rel = []
    for lab in labels:
        if lab not in mapping:
            mapping[lab] = chr(next_ord)
            next_ord += 1
        rel.append(mapping[lab])
    return rel

def get_music_style_from_condensed(condensed: str) -> str:
    return "".join(re.findall(r"([A-D])x\d+", condensed))

def music_style_from_labels(labels: Sequence[str], min_prop: float = 0.05) -> str:
    if not labels:
        return ""
    filtered_labels = filter_significant_styles(labels, min_prop=min_prop)
    condensed = condense_style_sequence(filtered_labels)
    return get_music_style_from_condensed(condensed)

def extract_style_change_timestamps(
    midi_tokens: Sequence[int],
    style_labels: Sequence[str],
    tokenizer: MusicTokenizerWithStyle,
    min_prop: float = 0.05
) -> List[Tuple[str, Union[str, None]]]:
    if len(style_labels) <= 1:
        return []
    N = len(style_labels)
    threshold = math.ceil(min_prop * N)
    runs = []
    start = 0
    for i in range(1, N):
        if style_labels[i] != style_labels[i - 1]:
            runs.append((style_labels[i - 1], start, i - 1))
            start = i
    runs.append((style_labels[-1], start, N - 1))
    significant = [run for run in runs if run[2] - run[1] + 1 >= threshold]
    if len(significant) <= 1:
        return []
    change_indices = [seg[1] for seg in significant[1:]]
    out = []
    for idx in change_indices:
        try:
            ts_ms = tokenizer.calculate_duration_ms(
                tokenizer.decode_tokens(midi_tokens[: idx + 1]), onset=False
            )
            mm, ss = divmod(ts_ms / 1000, 60)
            timestamp = f"{int(mm)}:{ss:06.3f}" if mm else f"{ss:.3f}s"
        except Exception:
            timestamp = None
        out.append((style_labels[idx], timestamp))
    return out

def get_prompt_from_midi_snippets(
    snippets: List[List[str]],
    music_style: str,
    tokenizer: MusicTokenizerWithStyle,
    max_prompt_length: int = 256
) -> List[str]:
    structure_form_token = f"<{music_style}>"
    prompt_tokens = [PROMPT_START_TOKEN, structure_form_token]
    overhead_tokens = len(prompt_tokens) + len(set(music_style)) + 1
    available_space = max_prompt_length - overhead_tokens
    if available_space <= 0:
        return prompt_tokens + [PROMPT_END_TOKEN]
    unique_sections = list(dict.fromkeys(music_style))
    space_per_section = available_space // len(unique_sections)
    for section_char in unique_sections:
        if section_char in STYLE_TO_SNIPPET_IDX and section_char in SECTION_TOKENS:
            snippet_idx = STYLE_TO_SNIPPET_IDX[section_char]
            if snippet_idx < len(snippets):
                snippet = snippets[snippet_idx][:space_per_section]
                prompt_tokens.append(SECTION_TOKENS[section_char])
                prompt_tokens.extend(snippet)
    prompt_tokens.append(PROMPT_END_TOKEN)
    return prompt_tokens[:max_prompt_length]

def get_prompt_from_midi_style_ids(
    input_tokens: Sequence[int],
    style_ids: Sequence[int],
    tokenizer: MusicTokenizerWithStyle,
    max_prompt_length: int = 256
) -> Tuple[List[int], List[str]]:
    style_labels = [ID_TO_STYLE_MAP.get(sid, 'A') for sid in style_ids]
    music_style = music_style_from_labels(style_labels)
    if not music_style:
        music_style = 'A'
    snippets = []
    current_snippet = []
    current_style = None
    for token, style_label in zip(input_tokens, style_labels):
        if style_label != current_style:
            if current_snippet:
                snippets.append(current_snippet)
            current_snippet = [tokenizer.decode_tokens([token])[0]]
            current_style = style_label
        else:
            current_snippet.append(tokenizer.decode_tokens([token])[0])
    if current_snippet:
        snippets.append(current_snippet)
    prompt_tokens = get_prompt_from_midi_snippets(snippets, music_style, tokenizer, max_prompt_length)
    prompt_ids = tokenizer.encode_tokens(prompt_tokens)
    return prompt_ids, prompt_tokens

def get_batch_prompts_from_midi_style_ids(
    input_tokens_batch: torch.Tensor,
    style_ids_batch: torch.Tensor,
    tokenizer: MusicTokenizerWithStyle,
    max_prompt_length: int = 256
) -> Tuple[List[List[int]], List[List[str]]]:
    batch_size = input_tokens_batch.shape[0]
    batch_prompts = []
    batch_prompt_tokens = []
    for batch_idx in range(batch_size):
        prompt, prompt_tokens = get_prompt_from_midi_style_ids(
            input_tokens_batch[batch_idx].tolist(),
            style_ids_batch[batch_idx].tolist(),
            tokenizer,
            max_prompt_length
        )
        batch_prompts.append(prompt)
        batch_prompt_tokens.append(prompt_tokens)
    return batch_prompts, batch_prompt_tokens

def get_random_midi(data_dir: str) -> str:
    midi_files = list(Path(data_dir).glob('**/*.mid'))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {data_dir}")
    return str(random.choice(midi_files))

def get_style_labels(label_path: str) -> Optional[List[str]]:
    try:
        with open(label_path, 'r') as f:
            content = f.read().strip()
        if not content:
            return None
        return list(content)
    except Exception as e:
        logger.error(f"Error processing style file {label_path}: {e}")
        return None

def log_music_style(style_str: str, logfile: str = "music_styles_log.txt") -> None:
    if style_str:
        try:
            Path(logfile).expanduser().resolve().write_text(
                style_str + "\n", encoding="utf-8"
            )
        except Exception:
            pass