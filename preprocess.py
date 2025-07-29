"""
MIDI Preprocessing Module - Path Indexer

This module finds MIDI files and their style labels (for synthetic mode)
and saves their paths to a .pkl file. Tokenization is deferred to runtime.

Output PKL File Structure:
- A list of dictionaries. Each dictionary represents a sample:
  - {"midi_file_path": "/path/to/song1.mid", "style_file_path": "/path/to/song1_style.txt"} (for synthetic)
  - {"midi_file_path": "/path/to/song_real.mid"} (for real)
- Saved with metadata about the preprocessing run.
"""
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path
from multiprocessing import cpu_count
import random
import hashlib
import pickle
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# No StariaTokenizer needed here as we are not tokenizing

logger_preprocess = logging.getLogger(__name__) # Renamed to avoid conflict

def process_midi_subdir(subdir_path: str) -> List[str]:
    """Finds all MIDI files in a subdirectory."""
    midi_files = []
    for root, _, files in os.walk(subdir_path):
        for f_name in files:
            if f_name.lower().endswith((".mid", ".midi")):
                midi_files.append(os.path.join(root, f_name))
    return midi_files

def find_all_midi_files(data_dir: str, num_workers: int) -> List[str]:
    """Finds all MIDI files in the given directory, using parallel processing for subdirectories."""
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    all_midis_found = []

    if subdirs:
        logger_preprocess.info(f"Found {len(subdirs)} subdirectories in {data_dir}, processing each in parallel.")
        subdir_paths = [os.path.join(data_dir, subdir) for subdir in subdirs]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_midi_subdir, sp) for sp in subdir_paths]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning subdirs"):
                try:
                    midi_files_in_subdir = future.result()
                    all_midis_found.extend(midi_files_in_subdir)
                except Exception as e:
                    logger_preprocess.error(f"Error processing subdirectory: {e}")
    else:
        logger_preprocess.info(f"No subdirectories found in {data_dir}. Scanning root directory for MIDI files.")
        for ext in ("*.mid", "*.midi"):
            all_midis_found.extend([str(p) for p in Path(data_dir).rglob(f"**/{ext}")]) # rglob for recursive
    
    logger_preprocess.info(f"Found {len(all_midis_found)} MIDI files in total from {data_dir}.")
    return all_midis_found

def create_path_index(
    data_dir: str,
    mode: str = "synthetic",
    dataset_seq_limit: Optional[int] = None,
    cache_dir: str = "./cache",
    shuffle_files: bool = True,
    output_filename_prefix: str = "dataset_paths",
    split_train_val: bool = False,
    val_ratio: float = 0.1
) -> Tuple[List[Dict], str]:
    """
    Scans data directory, creates a list of file path records, and saves to a .pkl file.
    Tokenization is NOT performed here.
    
    If split_train_val is True, creates separate train/val PKL files with a 90/10 split.
    """
    num_scan_workers = max(1, int(cpu_count() / 2)) # Fewer workers for I/O bound scanning
    
    os.makedirs(cache_dir, exist_ok=True)
    data_dir_name = os.path.basename(os.path.normpath(data_dir))
    
    # Simplified cache naming for path files
    # Max_len is not relevant here, tokenizer vocab size also not relevant yet
    params_str = f"{data_dir_name}_{mode}_{dataset_seq_limit}_{shuffle_files}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    # Filename clearly indicates it contains paths
    fname = f"{output_filename_prefix}_{mode}_{data_dir_name}_limit{dataset_seq_limit}_{params_hash}.pkl"
    output_pkl_path = os.path.join(cache_dir, fname)
    
    # Define train/val split filenames if needed
    train_pkl_path = None
    val_pkl_path = None
    if split_train_val:
        train_fname = f"{output_filename_prefix}_{mode}_{data_dir_name}_limit{dataset_seq_limit}_{params_hash}_train.pkl"
        val_fname = f"{output_filename_prefix}_{mode}_{data_dir_name}_limit{dataset_seq_limit}_{params_hash}_val.pkl"
        train_pkl_path = os.path.join(cache_dir, train_fname)
        val_pkl_path = os.path.join(cache_dir, val_fname)

    if os.path.exists(output_pkl_path) and (not split_train_val or 
                                           (os.path.exists(train_pkl_path) and os.path.exists(val_pkl_path))):
        logger_preprocess.info(f"Path index file already exists: {output_pkl_path}. Skipping generation.")
        try:
            with open(output_pkl_path, "rb") as f:
                # The structure is a dict: {"metadata": ..., "data_records": ...}
                loaded_data = pickle.load(f) 
                return loaded_data["data_records"], output_pkl_path
        except Exception as e:
            logger_preprocess.warning(f"Could not load existing cache file {output_pkl_path}: {e}. Regenerating.")
            os.remove(output_pkl_path)
            if split_train_val:
                if os.path.exists(train_pkl_path):
                    os.remove(train_pkl_path)
                if os.path.exists(val_pkl_path):
                    os.remove(val_pkl_path)

    logger_preprocess.info(f"Creating path index for {mode} data from {data_dir} using {num_scan_workers} scan workers.")
    
    path_records: List[Dict] = []

    if mode == "synthetic":
        midi_dir = os.path.join(data_dir, "midi")
        style_dir = os.path.join(data_dir, "style")
        if not os.path.isdir(midi_dir) or not os.path.isdir(style_dir):
            raise FileNotFoundError(f"For synthetic mode, 'midi' and 'style' subdirectories are required in {data_dir}")

        midi_files_map = {p.stem: str(p.resolve()) for p in Path(midi_dir).glob("*.mid")} # Use resolve for absolute paths
        style_files_map = {p.stem: str(p.resolve()) for p in Path(style_dir).glob("*.txt")}
        
        shared_stems = sorted(list(set(midi_files_map.keys()) & set(style_files_map.keys())))
        
        logger_preprocess.info(f"Found {len(shared_stems)} shared stems for synthetic data.")

        for stem in tqdm(shared_stems, desc="Gathering synthetic file paths"):
            path_records.append({
                "midi_file_path": midi_files_map[stem],
                "style_file_path": style_files_map[stem]
            })
            if dataset_seq_limit and len(path_records) >= dataset_seq_limit:
                break
    
    elif mode == "real":
        all_midi_paths = find_all_midi_files(data_dir, num_scan_workers)
        for midi_path in tqdm(all_midi_paths, desc="Gathering real MIDI file paths"):
            path_records.append({
                "midi_file_path": str(Path(midi_path).resolve()) # Store absolute path
            })
            if dataset_seq_limit and len(path_records) >= dataset_seq_limit:
                break
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'synthetic' or 'real'.")

    if shuffle_files:
        random.shuffle(path_records)

    # Metadata for the PKL file
    metadata = {
        "source_data_dir": data_dir,
        "mode": mode,
        "dataset_seq_limit": dataset_seq_limit, # The limit applied during this path gathering
        "num_records": len(path_records),
        "shuffled_during_creation": shuffle_files,
        "creation_timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0_paths_only"
    }
    
    data_to_save = {
        "metadata": metadata,
        "data_records": path_records # This is the List[Dict]
    }

    # Save the full dataset
    with open(output_pkl_path, "wb") as f:
        pickle.dump(data_to_save, f)
    
    logger_preprocess.info(f"Saved {len(path_records)} path records to {output_pkl_path}")
    
    # Create train/val split if requested
    if split_train_val:
        # Calculate split indices
        val_size = int(len(path_records) * val_ratio)
        train_size = len(path_records) - val_size
        
        # Split the records
        train_records = path_records[:train_size]
        val_records = path_records[train_size:]
        
        # Create and save train data
        train_metadata = metadata.copy()
        train_metadata["split"] = "train"
        train_metadata["num_records"] = len(train_records)
        train_metadata["val_ratio"] = val_ratio
        
        train_data = {
            "metadata": train_metadata,
            "data_records": train_records
        }
        
        with open(train_pkl_path, "wb") as f:
            pickle.dump(train_data, f)
        
        logger_preprocess.info(f"Saved {len(train_records)} train records to {train_pkl_path}")
        
        # Create and save val data
        val_metadata = metadata.copy()
        val_metadata["split"] = "val"
        val_metadata["num_records"] = len(val_records)
        val_metadata["val_ratio"] = val_ratio
        
        val_data = {
            "metadata": val_metadata,
            "data_records": val_records
        }
        
        with open(val_pkl_path, "wb") as f:
            pickle.dump(val_data, f)
        
        logger_preprocess.info(f"Saved {len(val_records)} validation records to {val_pkl_path}")
    
    return path_records, output_pkl_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    parser = argparse.ArgumentParser(description="Preprocess MIDI dataset to create a PKL file of paths.")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing source data (e.g., 'midi' and 'style' subdirs for synthetic).")
    parser.add_argument("--mode", type=str, choices=["synthetic", "real"], default="synthetic",
                        help="Processing mode: 'synthetic' for paired MIDI+style, 'real' for MIDI only.")
    parser.add_argument("--dataset_seq_limit", type=int, default=None, 
                        help="Limit the number of file path records to include in the PKL.")
    parser.add_argument("--cache_dir", type=str, default="./cache", 
                        help="Directory for saving the output .pkl path index file.")
    parser.add_argument("--output_filename_prefix", type=str, default="dataset_paths",
                        help="Prefix for the output PKL filename.")
    parser.add_argument("--no_shuffle", action="store_true", 
                        help="Disable shuffling of the gathered file paths before saving.")
    parser.add_argument("--split_train_val", action="store_true",
                        help="Create separate train/val PKL files with a 90/10 split.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1 for 90/10 split).")
    
    args = parser.parse_args()

    logger_preprocess.info(f"Starting path indexing in '{args.mode}' mode for data in '{args.data_dir}'.")
    
    records, output_file = create_path_index(
        data_dir=args.data_dir,
        mode=args.mode,
        dataset_seq_limit=args.dataset_seq_limit,
        cache_dir=args.cache_dir,
        shuffle_files=not args.no_shuffle,
        output_filename_prefix=args.output_filename_prefix,
        split_train_val=args.split_train_val,
        val_ratio=args.val_ratio
    )
    
    logger_preprocess.info(f"Path indexing complete. Saved {len(records)} records to {output_file}")

    # Verify by loading and printing some info
    if os.path.exists(output_file):
        logger_preprocess.info(f"\nVerifying content of {output_file}:")
        with open(output_file, "rb") as f:
            loaded_content = pickle.load(f)
        logger_preprocess.info(f"  Metadata: {loaded_content['metadata']}")
        if loaded_content['data_records']:
            logger_preprocess.info(f"  First 3 records: {loaded_content['data_records'][:3]}")
        else:
            logger_preprocess.info("  No data records found in the PKL file.")