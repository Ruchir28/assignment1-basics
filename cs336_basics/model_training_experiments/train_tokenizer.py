from cs336_basics.bpe_tokenizer.bpe import train_bpe

import os

from pathlib import Path
import json

def main():
    current_dir = Path(__file__).parent
    tokenizer_dir = current_dir / "tinystories_tokenizer"
    project_root = current_dir.parent
    # Set up data directory path
    data_train_path = project_root / "data" / "TinyStoriesV2-GPT4-train.txt"
    
    vocab, merges = train_bpe(
        input_path=str(data_train_path),
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    inverted_vocab = {token.hex(): i for i, token in vocab.items()}
    vocab_filepath = tokenizer_dir / "vocab.json"
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(inverted_vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to {vocab_filepath}")
    
    merges_jsonl = tokenizer_dir / "merges.jsonl"
    with open(merges_jsonl, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            f.write(json.dumps([p1.hex(), p2.hex()]) + "\n")
    print(f"Merges saved to {merges_jsonl}")

    
if __name__ == "__main__":
    main()