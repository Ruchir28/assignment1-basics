The BPE tokenizer training procedure consists of three main steps.
Vocabulary initialization The tokenizer vocabulary is a one-to-one mapping from bytestring token to
integer ID. Since we’re training a byte-level BPE tokenizer, our initial vocabulary is simply the set of all
bytes. Since there are 256 possible byte values, our initial vocabulary is of size 256.
Pre-tokenization Once you have a vocabulary, you could, in principle, count how often bytes occur next
to each other in your text and begin merging them starting with the most frequent pair of bytes. However,
this is quite computationally expensive, since we’d have to go take a full pass over the corpus each time
we merge. In addition, directly merging bytes across the corpus may result in tokens that differ only in
punctuation (e.g., dog! vs. dog.). These tokens would get completely different token IDs, even though they
are likely to have high semantic similarity (since they differ only in punctuation).
To avoid this, we pre-tokenize the corpus. You can think of this as a coarse-grained tokenization over the
corpus that helps us count how often pairs of characters appear. For example, the word 'text' might be
a pre-token that appears 10 times. In this case, when we count how often the characters ‘t’ and ‘e’ appear
next to each other, we will see that the word ‘text’ has ‘t’ and ‘e’ adjacent and we can increment their count
by 10 instead of looking through the corpus. Since we’re training a byte-level BPE model, each pre-token is
represented as a sequence of UTF-8 bytes.
The original BPE implementation of Sennrich et al. [2016] pre-tokenizes by simply splitting on whitespace
(i.e., s.split(" ")). In contrast, we’ll use a regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019)
from github.com/openai/tiktoken/pull/234/files:
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
It may be useful to interactively split some text with this pre-tokenizer to get a better sense of its
behavior:
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
6
When using it in your code, however, you should use re.finditer to avoid storing the pre-tokenized words
as you construct your mapping from pre-tokens to their counts.
Compute BPE merges Now that we’ve converted our input text into pre-tokens and represented each
pre-token as a sequence of UTF-8 bytes, we can compute the BPE merges (i.e., train the BPE tokenizer).
At a high level, the BPE algorithm iteratively counts every pair of bytes and identifies the pair with the
highest frequency (“A”, “B”). Every occurrence of this most frequent pair (“A”, “B”) is then merged, i.e.,
replaced with a new token “AB”. This new merged token is added to our vocabulary; as a result, the final
vocabulary after BPE training is the size of the initial vocabulary (256 in our case), plus the number of BPE
merge operations performed during training. For eﬀiciency during BPE training, we do not consider pairs
that cross pre-token boundaries.2 When computing merges, deterministically break ties in pair frequency by
preferring the lexicographically greater pair. For example, if the pairs (“A”, “B”), (“A”, “C”), (“B”, “ZZ”),
and (“BA”, “A”) all have the highest frequency, we’d merge (“BA”, “A”):
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
Special tokens Often, some strings (e.g., <|endoftext|>) are used to encode metadata (e.g., boundaries
between documents). When encoding text, it’s often desirable to treat some strings as “special tokens” that
should never be split into multiple tokens (i.e., will always be preserved as a single token). For example,
the end-of-sequence string <|endoftext|> should always be preserved as a single token (i.e., a single integer
ID), so we know when to stop generating from the language model. These special tokens must be added to
the vocabulary, so they have a corresponding fixed token ID.
Algorithm 1 of Sennrich et al. [2016] contains an ineﬀicient implementation of BPE tokenizer training
(essentially following the steps that we outlined above). As a first exercise, it may be useful to implement
and test this function to test your understanding.
Example (bpe_example): BPE training example
Here is a stylized example from Sennrich et al. [2016]. Consider a corpus consisting of the following text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
and the vocabulary has a special token <|endoftext|>.
Vocabulary values.
We initialize our vocabulary with our special token <|endoftext|> and the 256 byte
Pre-tokenization For simplicity and to focus on the merge procedure, we assume in this example
that pretokenization simply splits on whitespace. When we pretokenize and count, we end up with the
frequency table.
{low: 5, lower: 2, widest: 3, newest: 6}
2Note that the original BPE formulation [Sennrich et al., 2016] specifies the inclusion of an end-of-word token. We do not
add an end-of-word-token when training byte-level BPE models because all bytes (including whitespace and punctuation) are
included in the model’s vocabulary. Since we’re explicitly representing spaces and punctuation, the learned BPE merges will
naturally reflect these word boundaries.
7
It is convenient to represent this as a dict[tuple[bytes], int], e.g. {(l,o,w): 5 …}. Note that even
a single byte is a bytes object in Python. There is no byte type in Python to represent a single byte,
just as there is no char type in Python to represent a single character.
Merges We first look at every successive pair of bytes and sum the frequency of the words where they
appear {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}. The pair ('es')
and ('st') are tied, so we take the lexicographically greater pair, ('st'). We would then merge the
pre-tokens so that we end up with {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}.
In the second round, we see that (e, st) is the most common pair (with a count of 9) and we would
merge into {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}. Continuing this, the
sequence of merges we get in the end will be ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e',
'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r'].
If we take 6 merges, we have ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e'] and our vocab-
ulary elements would be [<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne].
With this vocabulary and set of merges, the word newest would tokenize as [ne, west].
2.5 Experimenting with BPE Tokenizer Training
Let’s train a byte-level BPE tokenizer on the TinyStories dataset. Instructions to find / download the dataset
can be found in Section 1. Before you start, we recommend taking a look at the TinyStories dataset to get
a sense of what’s in the data.
Parallelizing pre-tokenization You will find that a major bottleneck is the pre-tokenization step. You
can speed up pre-tokenization by parallelizing your code with the built-in library multiprocessing. Con-
cretely, we recommend that in parallel implementations of pre-tokenization, you chunk the corpus while
ensuring your chunk boundaries occur at the beginning of a special token. You are free to use the starter
code at the following link verbatim to obtain chunk boundaries, which you can then use to distribute work
across your processes:
https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py
This chunking will always be valid, since we never want to merge across document boundaries. For the
purposes of the assignment, you can always split in this way. Don’t worry about the edge case of receiving
a very large corpus that does not contain <|endoftext|>.
Removing special tokens before pre-tokenization Before running pre-tokenization with the regex
pattern (using re.finditer), you should strip out all special tokens from your corpus (or your chunk, if using
a parallel implementation). Make sure that you split on your special tokens, so that no merging can occur
across the text they delimit. For example, if you have a corpus (or chunk) like [Doc 1]<|endoftext|>[Doc
2], you should split on the special token <|endoftext|>, and pre-tokenize [Doc 1] and [Doc 2] separately,
so that no merging can occur across the document boundary. This can be done using re.split with "|"
⌋
.join(special_tokens) as the delimiter (with careful use of re.escape since | may occur in the special
tokens). The test test_train_bpe_special_tokens will test for this.
Optimizing the merging step The naïve implementation of BPE training in the stylized example above
is slow because for every merge, it iterates over all byte pairs to identify the most frequent pair. However,
the only pair counts that change after each merge are those that overlap with the merged pair. Thus,
BPE training speed can be improved by indexing the counts of all pairs and incrementally updating these
counts, rather than explicitly iterating over each pair of bytes to count pair frequencies. You can get
significant speedups with this caching procedure, though we note that the merging part of BPE training is
not parallelizable in Python.
8
Low-Resource/Downscaling Tip: Profiling
You should use profiling tools like cProfile or scalene to identify the bottlenecks in your imple-
mentation, and focus on optimizing those.
Low-Resource/Downscaling Tip: “Downscaling”
Instead of jumping to training your tokenizer on the full TinyStories dataset, we recommend you
first train on a small subset of the data: a “debug dataset”. For example, you could train your tokenizer
on the TinyStories validation set instead, which is 22K documents instead of 2.12M. This illustrates a
general strategy of downscaling whenever possible to speed up development: for example, using smaller
datasets, smaller model sizes, etc. Choosing the size of the debug dataset or hyperparameter config
requires careful consideration: you want your debug set to be large enough to have the same bottlenecks
as the full configuration (so that the optimizations you make will generalize), but not so big that it
takes forever to run.
Problem (train_bpe): BPE Tokenizer Training (15 points)
Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE
tokenizer. Your BPE training function should handle (at least) the following input parameters:
input_path: str Path to a text file with BPE tokenizer training data.
vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
otherwise affect BPE training.
Your BPE training function should return the resulting vocabulary and merges:
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
lary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation.
To test your BPE training function against our provided tests, you will first need to implement the
test adapter at [adapters.run_train_bpe]. Then, run uv run pytest tests/test_train_bpe.py.
Your implementation should be able to pass all tests. Optionally (this could be a large time-investment),
you can implement the key parts of your training method using some systems language, for instance
C++ (consider cppyy for this) or Rust (using PyO3). If you do this, be aware of which operations
require copying vs reading directly from Python memory, and make sure to leave build instructions, or
make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported
in most regex engines and will be too slow in most that do. We have verified that Oniguruma is
reasonably fast and supports negative lookahead, but the regex package in Python is, if anything,
even faster.
9
Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size
of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary.
Serialize the resulting vocabulary and merges to disk for further inspection. How many hours
and memory did training take? What is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤30 minutes (no GPUs), ≤30GB RAM
Hint You should be able to get under 2 minutes for BPE training using multiprocessing during
pretokenization and the following two facts:
(a) The <|endoftext|> token delimits documents in the data files.
(b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.
Deliverable: A one-to-two sentence response.
(b) Profile your code. What part of the tokenizer training process takes the most time?
Deliverable: A one-to-two sentence response.
Next, we’ll try training a byte-level BPE tokenizer on the OpenWebText dataset. As before, we recom-
mend taking a look at the dataset to better understand its contents.
Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary
size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What
is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤12 hours (no GPUs), ≤100GB RAM
Deliverable: A one-to-two sentence response.
(b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
Deliverable: A one-to-two sentence response.
2.6 BPE Tokenizer: Encoding and Decoding
In the previous part of the assignment, we implemented a function to train a BPE tokenizer on input text
to obtain a tokenizer vocabulary and a list of BPE merges. Now, we will implement a BPE tokenizer that
loads a provided vocabulary and list of merges and uses them to encode and decode text to/from token IDs.
2.6.1 Encoding text
The process of encoding text by BPE mirrors how we train the BPE vocabulary. There are a few major
steps.
Step 1: Pre-tokenize. We first pre-tokenize the sequence and represent each pre-token as a sequence of
UTF-8 bytes, just as we did in BPE training. We will be merging these bytes within each pre-token into
vocabulary elements, handling each pre-token independently (no merges across pre-token boundaries).
Step 2: Apply the merges. We then take the sequence of vocabulary element merges created during BPE
training, and apply it to our pre-tokens in the same order of creation.
10
Example (bpe_encoding): BPE encoding example
For example, suppose our input string is 'the cat ate', our vocabulary is {0: b' ', 1: b'a', 2:
b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b'
at'}, and our learned merges are [(b't', b'h'), (b' ', b'c'), (b' ', 'a'), (b'th', b'e'),
(b' a', b't')]. First, our pre-tokenizer would split this string into ['the', ' cat', ' ate'].
Then, we’ll look at each pre-token and apply the BPE merges.
The first pre-token 'the' is initially represented as [b't', b'h', b'e']. Looking at our list of
merges, we identify the first applicable merge to be (b't', b'h'), and use that to transform the
pre-token into [b'th', b'e']. Then, we go back to the list of merges and identify the next applicable
merge to be (b'th', b'e'), which transforms the pre-token into [b'the']. Finally, looking back at
the list of merges, we see that there are no more that apply to the string (since the entire pre-token
has been merged into a single token), so we are done applying the BPE merges. The corresponding
integer sequence is [9].
Repeating this process for the remaining pre-tokens, we see that the pre-token ' cat' is represented
as [b' c', b'a', b't'] after applying the BPE merges, which becomes the integer sequence [7, 1,
5]. The final pre-token ' ate' is [b' at', b'e'] after applying the BPE merges, which becomes the
integer sequence [10, 3]. Thus, the final result of encoding our input string is [9, 7, 1, 5, 10,
3].
Special tokens. Your tokenizer should be able to properly handle user-defined special tokens when encod-
ing text (provided when constructing the tokenizer).
Memory considerations. Suppose we want to tokenize a large text file that we cannot fit in memory.
To eﬀiciently tokenize this large file (or any other stream of data), we need to break it up into manageable
chunks and process each chunk in-turn, so that the memory complexity is constant as opposed to linear in
the size of the text. In doing so, we need to make sure that a token doesn’t cross chunk boundaries, else
we’ll get a different tokenization than the naïve method of tokenizing the entire sequence in-memory.
2.6.2 Decoding text
To decode a sequence of integer token IDs back to raw text, we can simply look up each ID’s corresponding
entries in the vocabulary (a byte sequence), concatenate them together, and then decode the bytes to a
Unicode string. Note that input IDs are not guaranteed to map to valid Unicode strings (since a user
could input any sequence of integer IDs). In the case that the input token IDs do not produce a valid
Unicode string, you should replace the malformed bytes with the oﬀicial Unicode replacement character
U+FFFD.
3 The errors argument of bytes.decode controls how Unicode decoding errors are handled, and
using errors='replace' will automatically replace malformed data with the replacement marker.
Problem (tokenizer): Implementing the tokenizer (15 points)
Deliverable: Implement a Tokenizer class that, given a vocabulary and a list of merges, encodes
text into integer IDs and decodes integer IDs into text. Your tokenizer should also support user-provided
special tokens (appending them to the vocabulary if they aren’t already there). We recommend the
following interface:
def __init__(self, vocab, merges, special_tokens=None) Construct a tokenizer from a given
vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
3See en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character for more information about the Unicode
replacement character.
11
the following parameters:
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
special_tokens: list[str] | None = None
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) Class
method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
(in the same format that your BPE training code output) and (optionally) a list of special
tokens. This method should accept the following additional parameters:
vocab_filepath: str
merges_filepath: str
special_tokens: list[str] | None = None
def encode(self, text: str) -> list[int] Encode an input text into a sequence of token IDs.
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int] Given an iterable of
strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
required for memory-eﬀicient tokenization of large files that we cannot directly load into
memory.
def decode(self, ids: list[int]) -> str Decode a sequence of token IDs into text.
To test your Tokenizer against our provided tests, you will first need to implement the test adapter
at [adapters.get_tokenizer]. Then, run uv run pytest tests/test_tokenizer.py. Your imple-
mentation should be able to pass all tests.
2.7 Experiments
Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyS-
tories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these
sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
Deliverable: A one-to-two sentence response.
(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Com-
pare the compression ratio and/or qualitatively describe what happens.
Deliverable: A one-to-two sentence response.
(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to
tokenize the Pile dataset (825GB of text)?
Deliverable: A one-to-two sentence response.
(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and devel-
opment datasets into a sequence of integer token IDs. We’ll use this later to train our language
model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is
uint16 an appropriate choice?
12
Deliverable: A one-to-two sentence response.