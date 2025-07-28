from typing import BinaryIO
from cs336_basics.bpe_tokenizer.bpe import BPETokenizer, find_chunk_boundaries

import os

import numpy as np

cpu_count = os.cpu_count()

def process_data(tokenizer: BPETokenizer,input_file: str, output_file: str,num_process = cpu_count):
    
    with open(input_file, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f,num_process,tokenizer.split_special_token.encode("utf-8"))
        
    # Tokenize each chunk in parallel 
    from multiprocessing import Pool
    with Pool(processes=num_process) as pool:
        chunk_token_counts = pool.starmap(
            get_token_count_for_chunk,
            [(input_file, tokenizer, start, end) for (start, end) in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]
        )
        
    total_tokens = sum(chunk_token_counts)
    print(f"Total tokens in the file: {total_tokens}")
    
    dtype = np.uint16
    memmap_array = np.memmap(output_file, dtype=dtype, mode='w+',
                             shape=(total_tokens,))
    
    memmap_array.flush()
    
    del memmap_array  # Close the memmap to allow writing in parallel
    
    offsets = [0] * len(chunk_token_counts)
    
    for i in range(1, len(chunk_token_counts)):
        offsets[i] = offsets[i - 1] + chunk_token_counts[i - 1]
    
    
    worker_args = [
        (input_file, tokenizer, start, end, output_file, offset)
        for (start, end), offset in zip(zip(chunk_boundaries[:-1], chunk_boundaries[1:]), offsets)
    ]
    
    with Pool(processes=num_process) as pool:
        pool.starmap(tokenize_and_write_chunk, worker_args)
        
    print(f"Tokenization complete. Data saved to {output_file}.")


def tokenize_and_write_chunk(input_file: str, tokenizer: BPETokenizer, start: int, end: int, output_file: str, offset: int):
    """
    Tokenizes a chunk of data from the file.
    """
    with open(input_file, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        chunk_str = chunk.decode("utf-8", errors="ignore")
        tokens = tokenizer.encode(chunk_str)

    memmap_array = np.memmap(output_file, dtype=np.uint16, mode='r+', offset=np.dtype('uint16').itemsize * offset, shape=(len(tokens),))
    
    memmap_array[:] = tokens

def get_token_count_for_chunk(input_file: str, tokenizer: BPETokenizer, start: int, end: int) -> int:
    """
    Returns the number of tokens in a chunk of data.
    """
    with open(input_file, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        tokens = tokenizer.encode(chunk.decode("utf-8"))
        return len(tokens)