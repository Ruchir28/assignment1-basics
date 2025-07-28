import os
import json
from cs336_basics.bpe_tokenizer.bpe import train_bpe

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "..", "data", "TinyStoriesV2-GPT4-train.txt")
    output_dir = os.path.join(script_dir, "tinystories_tokenizer")
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    os.makedirs(output_dir, exist_ok=True)

    print("Training BPE tokenizer...")
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print("Training complete.")

    inverted_vocab = {token.hex(): i for i, token in vocab.items()}
    vocab_filepath = os.path.join(output_dir, "vocab.json")
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(inverted_vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to {vocab_filepath}")

    merges_jsonl = os.path.join(output_dir, "merges.jsonl")
    with open(merges_jsonl, "w", encoding="utf-8") as f:
        for p1, p2 in merges:            # p1, p2 are bytes objects
            f.write(json.dumps([p1.hex(), p2.hex()]) + "\n")
    print(f"Merges saved to {merges_jsonl}")
    
    special_tokens_filepath = os.path.join(output_dir, "special_tokens.json")
    with open(special_tokens_filepath, "w", encoding="utf-8") as f:
        json.dump(special_tokens, f, ensure_ascii=False, indent=2)
    print(f"Special tokens saved to {special_tokens_filepath}")


if __name__ == "__main__":
    main() 