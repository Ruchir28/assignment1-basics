import json
import regex as re
from typing import BinaryIO, Iterable, Iterator, List, Tuple, Dict, Optional
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _process_chunk_from_boundaries(
    input_path: str, 
    start: int, 
    end: int, 
    special_tokens: List[str]
) -> Dict[Tuple[int, ...], int]:
    """
    Worker function that reads a specific byte range (a chunk) from a file
    and returns pre-token counts.
    """
    word_counts = {}
    pre_tok_pattern = re.compile(PAT)
    
    special_pattern = "|".join(re.escape(s) for s in special_tokens)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_str = chunk_bytes.decode("utf-8", errors="ignore")

    text_chunks = re.split(special_pattern, chunk_str)
    for sub_chunk in text_chunks:
        if not sub_chunk:
            continue
        for match in pre_tok_pattern.finditer(sub_chunk):
            word_bytes = match.group(0).encode("utf-8")
            word_tuple = tuple(word_bytes)
            word_counts[word_tuple] = word_counts.get(word_tuple, 0) + 1
            
    return word_counts

from multiprocessing import Pool, cpu_count
from collections import Counter

def get_pre_token_counts(input_path: str, special_tokens: List[str]) -> Dict[Tuple[int, ...], int]:
    """
    Reads a file in parallel and returns counts of pre-tokenized words.
    """
    num_processes = cpu_count()
    
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
    
    worker_args = [
        (input_path, start, end, special_tokens) 
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
    ]
    
    with Pool(len(worker_args)) as pool:
        list_of_counts = pool.starmap(_process_chunk_from_boundaries, worker_args)

    # Aggregate the results from all processes
    total_counts = Counter()
    for counts in list_of_counts:
        total_counts.update(counts)

    return dict(total_counts)


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def get_pair_counts(
    word_counts: Dict[Tuple[int,...],int]
) -> Dict[Tuple[int,int], int]:
    pair_counts = {}
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts

def merge_pair(
    best_pair: Tuple[int, int],
    new_token_id: int,
    word_counts: Dict[Tuple[int,...], int],
    pair_counts: Dict[Tuple[int, int], int],
) -> None:
    """
    Merges the best pair of tokens into a new token and updates the word and pair counts.

    Args:
        best_pair: The pair of tokens to merge.
        new_token_id: The ID of the new token.
        word_counts: A dictionary mapping token sequences to their counts.
        pair_counts: A dictionary mapping token pairs to their counts.
    """
    # Find all words that contain the best pair
    affected_words = [
        word for word in word_counts 
        if len(word) > 1 and best_pair[0] in word and best_pair[1] in word
    ]

    for word in affected_words:
        if word not in word_counts:
            continue

        count = word_counts[word]
        
        has_pair = False
        i = 0
        while i < len(word) - 1:
            if (word[i], word[i+1]) == best_pair:
                has_pair = True
                break
            i += 1

        if not has_pair:
            continue

        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] -= count
            if pair_counts[pair] == 0:
                del pair_counts[pair]

        new_word_list = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                new_word_list.append(new_token_id)
                i += 2
            else:
                new_word_list.append(word[i])
                i += 1
        new_word = tuple(new_word_list)
        
        del word_counts[word]
        word_counts[new_word] = word_counts.get(new_word, 0) + count

        # Increment counts for all pairs in the new word
        for i in range(len(new_word) - 1):
            pair = (new_word[i], new_word[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Trains a BPE tokenizer from a text file.

    Args:
        input_path: Path to the training text file.
        vocab_size: The desired final vocabulary size.
        special_tokens: A list of special tokens to include in the vocabulary.

    Returns:
        A tuple containing the vocabulary (mapping from ID to bytes) and the list of merges.
    """
    vocab = {i: bytes([i]) for i in range(256)}  
    
    for i, token_str in enumerate(special_tokens):
        vocab[256 + i] = token_str.encode("utf-8")
        
    special_token_ids = {token.encode("utf-8"): 256 + i for i, token in enumerate(special_tokens)}
    
    counts = get_pre_token_counts(input_path, special_tokens)
    
    pair_counts = get_pair_counts(counts)
        
    merges = []
    
    num_merges = vocab_size - len(vocab)
    
    for i in range(num_merges):
        
        if not pair_counts:
            break
        
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))
        
        new_token_id = 256 + len(special_tokens) + i

        merge_pair(best_pair, new_token_id, counts, pair_counts)

        left_bytes = vocab[best_pair[0]]
        
        right_bytes = vocab[best_pair[1]]

        merges.append((left_bytes, right_bytes))
        
        vocab[new_token_id] = left_bytes + right_bytes

    return vocab, merges


class BPETokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Initializes the tokenizer with a vocabulary and merge rules.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.token_to_ids = { token: idx for idx, token in vocab.items() }
        self.merge_pair_rank = { (left, right): i for i, (left, right) in enumerate(merges) }
        self.special_token_map = {token: self.token_to_ids.get(token.encode("utf-8"), 256 + i) for i, token in enumerate(self.special_tokens)}
        self.inverse_special_token_map = {v: k for k, v in self.special_token_map.items()}
        self.pre_tok_pattern = re.compile(PAT)
        

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        """
        Loads a tokenizer from vocabulary and merges files.
        """
        vocab = {}
        with open(vocab_filepath,"r",encoding="utf-8") as vf:
            inverted_vocab = json.load(vf)
            vocab = {int(v): k.encode("utf-8") for k, v in inverted_vocab.items()}

        special_tokens = special_tokens or []
        merges = []
        with open(merges_filepath,"r",encoding="utf-8") as mf:
            for line in mf:
                if line.strip():
                    left, right = line.strip().split()
                    merges.append((left.encode("utf-8"), right.encode("utf-8")))
        return cls(vocab, merges, special_tokens)
                
            

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        """
        encoded_ids = []
        
        if self.special_tokens:
            # Only do special token splitting if special tokens exist
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = f"({'|'.join(re.escape(s) for s in sorted_special_tokens)})"
            special_tokens_re = re.compile(special_pattern)
            chunks = re.split(special_tokens_re, text)
            
            for chunk in chunks:
                if not chunk:
                    continue
                
                if chunk in self.special_token_map:
                    encoded_ids.append(self.special_token_map[chunk])
                    continue
                
                # Process regular text chunk
                for match in self.pre_tok_pattern.finditer(chunk):
                    word_bytes = match.group(0).encode("utf-8")
                    tokens = self._bpe_merge(word_bytes)
                    encoded_ids.extend(tokens)
        else:
            # No special tokens, process entire text directly
            for match in self.pre_tok_pattern.finditer(text):
                word_bytes = match.group(0).encode("utf-8")
                tokens = self._bpe_merge(word_bytes)
                encoded_ids.extend(tokens)
                    
        return encoded_ids

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encodes an iterable of strings into a generator of token IDs.
        """
        for text in iterable:
            yield from self.encode(text)
    
    
    def _bpe_merge(
        self,
        word_bytes: bytes,
    ) -> List[int]:
        if not word_bytes:
            return []

        tokens = [bytes([byte]) for byte in word_bytes]
        
        while len(tokens) > 1:
            
            pairs = [((tokens[i], tokens[i+1]), i) for i in range(len(tokens) - 1)]

            if not pairs:
                break

            best_pair_to_merge, idx = min(
                pairs, 
                key=lambda p: self.merge_pair_rank.get(p[0], float('inf'))
            )
            
            if best_pair_to_merge not in self.merge_pair_rank:
                break
            
            new_token = best_pair_to_merge[0] + best_pair_to_merge[1]
            
            tokens = tokens[:idx] + [new_token] + tokens[idx + 2:]
            
        final_token_ids = [self.token_to_ids[token] for token in tokens]

        return final_token_ids


    def decode(self, ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        curr_byte_string = b""
        result = []
        for token_id in ids:
            if token_id in self.inverse_special_token_map:
                curr_string = curr_byte_string.decode("utf-8", "replace")
                result.append(curr_string)
                result.append(self.inverse_special_token_map[token_id])
                curr_byte_string = b""
            else:
                curr_byte_string += self.vocab[token_id]
                
        if curr_byte_string:    
            curr_string = curr_byte_string.decode("utf-8", "replace")
            result.append(curr_string)
        
        return "".join(result)
    
    
# if __name__  == "__main__":
    
#     from pathlib import Path
#     FIXTURES_PATH = Path(__file__).parent.parent.parent / "tests" / "fixtures"

    
#     VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
#     MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"
#     bpe_tokenizer = BPETokenizer.from_files(
#         vocab_filepath=VOCAB_PATH,
#         merges_filepath=MERGES_PATH,
#         special_tokens=["<|endoftext|>"]
#     )
    
#     test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    
#     encoded = bpe_tokenizer.encode(test_string)
#     print("Encoded:", encoded)