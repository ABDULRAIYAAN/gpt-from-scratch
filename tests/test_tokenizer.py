import os

import pytest
import torch

from tokenizer import CHUNK_PATTERN, BPETokenizer, CharTokenizer, load_tokenizer, merge_pair

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "ckpt.pt")

TEXT = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n\n"
    "All:\nResolved. resolved.\n\nFirst Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n"
)
ODD = "naïve café, 'tis o'er!  Ünïcödé 日本語 🙂\ttabs\r\n  2026 -- end   "


def test_merge_pair():
    assert merge_pair([1, 2, 1, 2, 2], (1, 2), 9) == [9, 9, 2]
    assert merge_pair([5, 5, 5], (5, 5), 7) == [7, 5]  # non-overlapping, left to right
    assert merge_pair([], (1, 2), 9) == []


@pytest.mark.parametrize("text", [TEXT, ODD, "", " ", "\n\n\n", "a"])
def test_chunks_cover_the_whole_text(text):
    assert "".join(CHUNK_PATTERN.findall(text)) == text


def test_bpe_round_trips_any_text():
    tok = BPETokenizer.train(TEXT, 300)
    for s in [TEXT, ODD, "", "zzz", "Speak, speak!"]:
        assert tok.decode(tok.encode(s)) == s
    assert all(0 <= i < tok.vocab_size for i in tok.encode(ODD))


def test_bpe_learns_frequent_pieces_and_compresses():
    tok = BPETokenizer.train(TEXT, 320)
    assert tok.vocab_size == 320
    pieces = {tok.vocab[i] for i in range(256, tok.vocab_size)}
    assert b" Citizen" in pieces or b"Citizen" in pieces
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8")) / 2


def test_bpe_first_merge_is_the_most_frequent_pair():
    tok = BPETokenizer.train("aaab aaab aaab", 257)
    assert tok.merges == [(ord("a"), ord("a"))]
    assert tok.encode("aaab") == [256, ord("a"), ord("b")]


def test_bpe_training_is_deterministic_and_stops_when_nothing_is_left():
    assert BPETokenizer.train(TEXT, 290).merges == BPETokenizer.train(TEXT, 290).merges
    tok = BPETokenizer.train("ab ab", 1000)  # only " ab" and "ab" can be built
    assert tok.vocab_size < 1000
    assert tok.encode("ab ab") == [tok.vocab.index(b"ab"), tok.vocab.index(b" ab")]


def test_bpe_encode_matches_training():
    # Encoding the training text must reproduce exactly what training built, merge by merge.
    tok = BPETokenizer.train(TEXT, 300)
    for chunk in set(CHUNK_PATTERN.findall(TEXT)):
        ids = list(chunk.encode("utf-8"))
        for i, pair in enumerate(tok.merges):
            ids = merge_pair(ids, pair, 256 + i)
        assert tok.encode(chunk) == ids, chunk


def test_bpe_never_merges_across_words():
    tok = BPETokenizer.train("the cat the cat the cat", 400)
    assert all(b"t t" not in piece and b"e c" not in piece for piece in tok.vocab)


def test_bpe_rejects_tiny_vocab():
    with pytest.raises(ValueError):
        BPETokenizer.train(TEXT, 100)


def test_char_tokenizer_unchanged():
    tok = CharTokenizer.from_text(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT
    with pytest.raises(ValueError):
        tok.encode("~")


def test_load_tokenizer_from_state():
    bpe = BPETokenizer.train(TEXT, 280)
    again = load_tokenizer({"tokenizer": bpe.state()})
    assert isinstance(again, BPETokenizer) and again.encode(ODD) == bpe.encode(ODD)
    char = CharTokenizer.from_text(TEXT)
    assert load_tokenizer({"tokenizer": char.state()}).chars == char.chars
    assert load_tokenizer({"chars": char.chars}).chars == char.chars  # older checkpoints


@pytest.mark.skipif(not os.path.exists(CKPT), reason="trained checkpoint not present")
def test_committed_checkpoint_tokenizer():
    ckpt = torch.load(CKPT, map_location="cpu")
    tok = load_tokenizer(ckpt)
    assert isinstance(tok, CharTokenizer) and tok.vocab_size == 65
