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

    inverted_vocab = {token.decode("utf-8", "replace"): i for i, token in vocab.items()}
    vocab_filepath = os.path.join(output_dir, "vocab.json")
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(inverted_vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to {vocab_filepath}")

    merges_filepath = os.path.join(output_dir, "merges.txt")
    with open(merges_filepath, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            s1 = p1.decode('utf-8', 'replace')
            s2 = p2.decode('utf-8', 'replace')
            f.write(f"{s1} {s2}\n")
    print(f"Merges saved to {merges_filepath}")
    
    special_tokens_filepath = os.path.join(output_dir, "special_tokens.json")
    with open(special_tokens_filepath, "w", encoding="utf-8") as f:
        json.dump(special_tokens, f, ensure_ascii=False, indent=2)
    print(f"Special tokens saved to {special_tokens_filepath}")


if __name__ == "__main__":
    main() 