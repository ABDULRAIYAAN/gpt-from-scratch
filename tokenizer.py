"""
Tokenizers: turn text into a list of integer ids and back.

CharTokenizer: every unique character in the text is one token. This is what
the trained checkpoint and the browser demo use.

BPETokenizer: byte-level byte-pair encoding, written from scratch. It starts
from the 256 possible bytes and learns merges: the most frequent adjacent pair
of tokens in the training text becomes a new token, again and again, until the
vocabulary reaches the size asked for. Common pieces such as " the" or "ing"
end up as single tokens, so the same text needs fewer tokens and the model's
context window covers more of it.
"""

import re
from collections import Counter


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        unknown = set(s) - self.stoi.keys()
        if unknown:
            raise ValueError(f"Characters not in vocabulary: {sorted(unknown)}")
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def state(self) -> dict:
        return {"type": "char", "chars": self.chars}


# Text is first split into chunks (words with their leading space, runs of digits,
# runs of punctuation, runs of whitespace) and merges never cross a chunk boundary,
# so a token is never half of one word and half of the next. The alternatives cover
# every character, so joining the chunks always gives back the original text.
CHUNK_PATTERN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of pair in ids, left to right, with new_id."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE. Ids 0-255 are raw bytes; id 256 + i is the i-th learned merge."""

    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}  # earlier merge = applied first
        self.vocab = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (one token per byte)")
        # Every distinct chunk only needs to be processed once, weighted by how often it occurs.
        chunk_counts = Counter(CHUNK_PATTERN.findall(text))
        words = [[list(chunk.encode("utf-8")), n] for chunk, n in chunk_counts.items()]

        merges = []
        for new_id in range(256, vocab_size):
            pair_counts = Counter()
            for ids, n in words:
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += n
            if not pair_counts:
                break  # every chunk is already a single token
            # Most frequent pair; ties go to the smallest pair so training is deterministic.
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            for w in words:
                if len(w[0]) > 1:
                    w[0] = merge_pair(w[0], best, new_id)
            merges.append(best)
            if verbose and len(merges) % 64 == 0:
                tok = cls(merges)
                print(f"  merge {len(merges):4d}: {tok.vocab[new_id]!r} ({pair_counts[best]:,} times)")
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # Apply the earliest-learned merge present in the chunk, the same order as training.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = merge_pair(ids, pair, 256 + self.ranks[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, s: str) -> list[int]:
        out = []
        for chunk in CHUNK_PATTERN.findall(s):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        # A slice of tokens can end partway through a multi-byte character; show U+FFFD for it.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def state(self) -> dict:
        return {"type": "bpe", "merges": [list(m) for m in self.merges]}


def load_tokenizer(ckpt: dict):
    """Rebuild the tokenizer a checkpoint was trained with."""
    state = ckpt.get("tokenizer")
    if state is None or state["type"] == "char":
        # Checkpoints from before BPE support only stored the character list.
        return CharTokenizer(state["chars"] if state else ckpt["chars"])
    if state["type"] == "bpe":
        return BPETokenizer(state["merges"])
    raise ValueError(f"unknown tokenizer type {state['type']!r}")
