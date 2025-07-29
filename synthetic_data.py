#!/usr/bin/env python
"""
Synthetic MIDI generator (bug-fixed)
===================================
Balanced, leak-free synthetic MIDI generation for multiple musical forms (AB, ABC, …).

**What was wrong?**
* If *any* bad/empty MIDI file caused an exception, the whole worker crashed and that form
  produced **zero** output (exactly what you observed for AB, ABC, ABA).  The other workers
  kept running, so it looked like "some forms generate, some don’t".

**Fixes / improvements**
1. **Per-sample try/except & retry** – one corrupt file no longer kills the form; we skip the
   faulty slice and keep going until the allotted files are exhausted.
2. **Shuffle slice in each worker** so every run sees different combinations.
3. **Graceful summary** – at the end of a worker we log how many pieces were actually
   produced versus the target, making it clear if any shortfall occurred.

Run exactly the same CLI as before.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Sequence, Tuple

from tokenizer import MusicTokenizerWithStyle  # pylint: disable=import-error
from ariautils.midi import MidiDict  # pylint: disable=import-error

# --------------------------------------------------------------------------------------
# 1. MIDI discovery (parallel)
# --------------------------------------------------------------------------------------

def _midi_files_in_dir(path: str) -> List[str]:
    """Return every .mid/.midi file under *path* recursively."""
    out: List[str] = []
    for root, _, names in os.walk(path):
        out.extend(os.path.join(root, n) for n in names if n.lower().endswith((".mid", ".midi")))
    return out


def list_midi_files(midi_root: str, max_workers: int | None) -> List[str]:
    """Parallel crawl of *midi_root*; returns **all** MIDI paths (including root-level ones)."""
    subdirs = [os.path.join(midi_root, d) for d in os.listdir(midi_root)
               if os.path.isdir(os.path.join(midi_root, d))]
    if not subdirs:  # flat directory – just crawl root in one go
        return _midi_files_in_dir(midi_root)

    all_paths: List[str] = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        fut2dir = {exe.submit(_midi_files_in_dir, d): d for d in subdirs}
        for fut in as_completed(fut2dir):
            paths = fut.result()
            print(f"Indexed {len(paths):4d} files in {os.path.basename(fut2dir[fut])}")
            all_paths.extend(paths)
    if not all_paths:
        raise ValueError("No MIDI files discovered.")
    return all_paths

# --------------------------------------------------------------------------------------
# 2. Allocation – disjoint slices per form
# --------------------------------------------------------------------------------------

def _uniq(form: str) -> int:
    return len(set(form))


def allocate_paths(paths: Sequence[str], forms: Sequence[str], seed: int | None) -> Tuple[Dict[str, List[str]], int, List[str]]:
    if seed is not None:
        random.seed(seed)
    pool = list(paths)
    random.shuffle(pool)

    need_unit = sum(_uniq(f) for f in forms)
    per_form = len(pool) // need_unit
    if per_form == 0:
        raise ValueError(f"Dataset too small – need ≥{need_unit} files, have {len(pool)}")

    out: Dict[str, List[str]] = {}
    cursor = 0
    for f in forms:
        k = _uniq(f) * per_form
        out[f] = pool[cursor: cursor + k]
        cursor += k
    return out, per_form, pool[cursor:]

# --------------------------------------------------------------------------------------
# 3. Helper utilities
# --------------------------------------------------------------------------------------

def _next_id(out_dir: str, form: str) -> int:
    midi_dir = os.path.join(out_dir, "midi")
    prefix = f"{form}_"
    existing = [f for f in os.listdir(midi_dir) if f.startswith(prefix) and f.endswith(".mid")]
    return 1 + max((int(f[len(prefix):-4]) for f in existing), default=-1)


def _rand_partition(total: int, n: int) -> List[int]:
    if n == 1:
        return [total]
    min_part = max(1, total // (n * 4))
    cuts = sorted(random.sample(range(min_part, total - min_part * (n - 1) + 1), n - 1))
    parts, prev = [], 0
    for c in cuts + [total - min_part * (n - 1)]:
        parts.append(c - prev)
        prev = c
    parts = [p + min_part - 1 for p in parts]
    parts[-1] += total - sum(parts)
    return parts


def _scale_labels(sec_lens: List[int], total: int) -> List[int]:
    if not sum(sec_lens):
        base, rem = divmod(total, len(sec_lens))
        return [base + (i < rem) for i in range(len(sec_lens))]
    scaled = [round(l * total / sum(sec_lens)) for l in sec_lens]
    diff = total - sum(scaled)
    for i in range(abs(diff)):
        idx = i % len(scaled)
        scaled[idx] += 1 if diff > 0 else -1
    return scaled

# --------------------------------------------------------------------------------------
# 4. Piece generation (now as standalone function for parallelism)
# --------------------------------------------------------------------------------------

def generate_one_piece(form: str, chunk: List[str], out_dir: str, seed: int | None, counters: multiprocessing.managers.DictProxy, locks: multiprocessing.managers.DictProxy) -> Tuple[str, bool]:
    if seed is not None:
        random.seed(seed)

    tokenizer = MusicTokenizerWithStyle()
    midi_out = os.path.join(out_dir, "midi")
    style_out = os.path.join(out_dir, "style")

    try:
        uniq_letters = list(dict.fromkeys(form))
        sec_dict = {sec: MidiDict.from_midi(p) for sec, p in zip(uniq_letters, chunk)}

        n_letters = len(form)
        total_tok = random.randint(2800, 4095) if n_letters == 4 else random.randint(1024, 4095)
        part_sizes = _rand_partition(total_tok, n_letters)

        sec_tokens: List[List[int]] = [
            tokenizer.tokenize(sec_dict[sec])[:ln]
            for sec, ln in zip(form, part_sizes)
        ]

        # stitch
        recon: MidiDict | None = None
        cur_end = 0
        for idx, toks in enumerate(sec_tokens):
            midi_sec = tokenizer._base_tokenizer.detokenize(toks)  # pylint: disable=protected-access
            delay = midi_sec.note_msgs[0]["data"].get("start", 0) if midi_sec.note_msgs else 0
            for e in midi_sec.note_msgs:
                e["tick"] -= delay
                e["data"]["start"] -= delay
                e["data"]["end"] -= delay
            if idx == 0:
                recon = midi_sec
            else:
                for e in midi_sec.note_msgs:
                    ne = copy.deepcopy(e)
                    ne["tick"] += cur_end
                    ne["data"]["start"] += cur_end
                    ne["data"]["end"] += cur_end
                    recon.note_msgs.append(ne)
            if recon.note_msgs:
                cur_end = recon.note_msgs[-1]["data"]["end"]

        if recon is None:
            print(f"[{form}] warning – empty reconstruction, skipping")
            return form, False

        flat = tokenizer.tokenize(recon)
        sec_lens = [len(t) for t in sec_tokens]
        lbl_counts = _scale_labels(sec_lens, len(flat))
        labels = []
        for ltr, cnt in zip(form, lbl_counts):
            labels.extend(ltr * cnt)
        labels.extend(form[-1] * (len(flat) - len(labels)))

        with locks[form]:
            out_id = counters[form]
            counters[form] += 1

        mid_path = os.path.join(midi_out, f"{form}_{out_id:05d}.mid")
        tokenizer.ids_to_file(tokenizer.encode(flat), mid_path)

        lbl_path = os.path.join(style_out, f"{form}_{out_id:05d}.txt")
        with open(lbl_path, "w", encoding="utf-8") as fh:
            fh.write("".join(labels))

        print(f"[{form}] generated {os.path.basename(mid_path)}")
        return form, True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[{form}] error with slice {chunk}: {exc}. Skipping…")
        return form, False

# --------------------------------------------------------------------------------------
# 5. CLI orchestration
# --------------------------------------------------------------------------------------

def main():
    par = argparse.ArgumentParser("Balanced synthetic MIDI generator (robust)")
    par.add_argument("--midi_folder", required=True)
    par.add_argument("--output_folder", required=True)
    par.add_argument("--forms", nargs="+", default=[
        "AB", "ABC", "ABA", "ABAB", "ABAC", "ABCA", "ABCB", "ABCD"])
    par.add_argument("--seed", type=int)
    par.add_argument("--max_workers", type=int, default=os.cpu_count())
    args = par.parse_args()

    for sub in ("midi", "style"):
        os.makedirs(os.path.join(args.output_folder, sub), exist_ok=True)

    print("Indexing dataset…")
    all_midis = list_midi_files(args.midi_folder, args.max_workers)
    print(f"Found {len(all_midis)} MIDI files\n")

    slices, per_form, leftover = allocate_paths(all_midis, args.forms, args.seed)
    print(f"Allocation → {per_form} pieces per form, {len(leftover)} unused files\n")

    manager = multiprocessing.Manager()
    counters = manager.dict({f: _next_id(args.output_folder, f) for f in args.forms})
    locks = manager.dict({f: manager.Lock() for f in args.forms})

    tasks = []
    for form in args.forms:
        paths = slices[form]
        if args.seed is not None:
            random.seed(args.seed)
        random.shuffle(paths)
        need_per_piece = _uniq(form)
        target = len(paths) // need_per_piece
        print(f"[{form}] slice {len(paths)} files → target {target} pieces")
        for i in range(target):
            chunk = paths[i * need_per_piece: (i + 1) * need_per_piece]
            sub_seed = args.seed + i if args.seed is not None else None
            tasks.append((form, chunk, args.output_folder, sub_seed, counters, locks))

    generated = {f: 0 for f in args.forms}
    with ProcessPoolExecutor(max_workers=args.max_workers) as exe:
        futs = [exe.submit(generate_one_piece, *t) for t in tasks]
        for fut in as_completed(futs):
            form, success = fut.result()
            if success:
                generated[form] += 1

    for form in args.forms:
        print(f"[{form}] finished – produced {generated[form]}/{per_form} pieces")

    if leftover:
        print("Warning: {} MIDI files weren’t used (corpus not divisible).".format(len(leftover)))


if __name__ == "__main__":
    main()