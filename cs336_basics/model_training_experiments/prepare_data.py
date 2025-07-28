import json, numpy as np
import os
from pathlib import Path

from cs336_basics.bpe_tokenizer.bpe import BPETokenizer
from cs336_basics.bpe_tokenizer.prepare_data import process_data

def prepare_data():
    
    # Get paths using pathlib (more modern approach)
    current_dir = Path(__file__).parent
    tokenizer_dir = current_dir / "tinystories_tokenizer"
    project_root = current_dir.parent

    # Load special tokens
    special_tokens = ["<|endoftext|>"]

    # Set up tokenizer paths
    vocab_file = tokenizer_dir / "vocab.json"
    merges_file = tokenizer_dir / "merges.jsonl"

    tokenizer = BPETokenizer.from_files(str(vocab_file), str(merges_file), special_tokens=special_tokens)
    
    tokenizer.split_special_token = "<|endoftext|>" 

    # Set up data directory path
    data_train_path = project_root / "data" / "TinyStoriesV2-GPT4-train.txt"
    data_val_path = project_root / "data" / "TinyStoriesV2-GPT4-valid.txt"
    
    # Prepare output paths
    train_ids_path = tokenizer_dir / "train_ids.npy"
    val_ids_path = tokenizer_dir / "val_ids.npy"
    
    # Process training data
    print("Processing training data...")
    
    process_data(
        tokenizer=tokenizer,
        input_file=str(data_train_path),
        output_file=str(train_ids_path),
        num_process=os.cpu_count()
    )
    
    # Process validation data
    print("Processing validation data...")  
    process_data(
        tokenizer=tokenizer,
        input_file=str(data_val_path),
        output_file=str(val_ids_path),
        num_process=os.cpu_count()
    )
    
    


if __name__ == "__main__":
    prepare_data()
    print("Data preparation complete. Train and validation IDs saved as numpy arrays.")