"""Check web/tokenizer.js against tokenizer.py: same chunks, same ids, same decoded text."""

import json
import os
import random
import shutil
import subprocess

import pytest
import torch

from data import DEFAULT_PATH, split_text
from tokenizer import CHUNK_PATTERN, BPETokenizer, CharTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(ROOT, "checkpoints", "ckpt.pt")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

TEXT = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n\n"
    "All:\nResolved. resolved.\n\nFirst Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n"
)
ODD = "naïve café, 'tis o'er!  Ünïcödé 日本語 🙂\ttabs\r\n  2026 -- end   "

# Every character Python's \s matches, plus U+FEFF, which only JavaScript's \s matches.
PY_SPACES = "".join(chr(i) for i in range(0x110000) if chr(i).isspace())
EDGE = [
    "", " ", "a", "\n\n\n", "  hi", "a  \n  b", "x   ", "it's we're they'll I'd", "''s",
    "1,000.50 -- 42x", "🙂🙂 🙂a", "﻿hello", "﻿ ﻿", "a\x1cb\x85c　 d e",
    "word" + PY_SPACES + "word", PY_SPACES, " \t  x",
]


def run_js(state, texts, ids=(), tmp_path=None):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"state": state, "texts": texts, "ids": list(ids)}))
    out = subprocess.run(
        ["node", os.path.join(HERE, "run_tokenizer.js"), str(job)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def bpe():
    return BPETokenizer.train(TEXT + ODD, 330)


def test_chunks_match_python(bpe, tmp_path):
    texts = [TEXT, ODD] + EDGE
    js = run_js(bpe.state(), texts, tmp_path=tmp_path)
    for text, chunks in zip(texts, js["chunks"]):
        assert chunks == CHUNK_PATTERN.findall(text), repr(text)


def test_bpe_encodes_and_decodes_like_python(bpe, tmp_path):
    texts = [TEXT, ODD, TEXT.upper(), "Speak, speak!", "zzz qqq"] + EDGE
    js = run_js(bpe.state(), texts, ids=[bpe.encode(t) for t in texts], tmp_path=tmp_path)
    assert js["vocab_size"] == bpe.vocab_size == 330
    assert js["encoded"] == [bpe.encode(t) for t in texts]
    assert js["decoded"] == texts  # round trip, including the leading U+FEFF
    assert js["pieces"] == [v.decode("utf-8", errors="replace") for v in bpe.vocab]


def test_bpe_decodes_partial_characters_like_python(bpe, tmp_path):
    # Slices that end inside a multi-byte character, and random byte soup, must show
    # U+FFFD in the same places as Python's errors="replace".
    ids = bpe.encode(ODD)
    rng = random.Random(0)
    seqs = [ids[:n] for n in range(len(ids) + 1)] + [ids[n:] for n in range(len(ids))]
    seqs += [[rng.randrange(bpe.vocab_size) for _ in range(rng.randrange(1, 12))] for _ in range(300)]
    seqs += [[rng.randrange(128, 256) for _ in range(rng.randrange(1, 8))] for _ in range(300)]
    js = run_js(bpe.state(), [], ids=seqs, tmp_path=tmp_path)
    assert js["decoded"] == [bpe.decode(s) for s in seqs]


def test_char_tokenizer_matches_python(tmp_path):
    tok = CharTokenizer.from_text(TEXT)
    texts = [TEXT, "Speak", "", "naïve 🙂 Z"]
    js = run_js(tok.state(), texts, ids=[tok.encode(TEXT), [0, 1, 2]], tmp_path=tmp_path)
    assert js["encoded"][:3] == [tok.encode(t) for t in texts[:3]]
    assert js["encoded"][3] is None  # unknown characters raise, like the Python version
    assert js["errors"][3] == 'Characters not in vocabulary: ["Z","ï","🙂"]'
    assert js["decoded"] == [TEXT, tok.decode([0, 1, 2])]
    assert js["pieces"] == tok.chars


@pytest.mark.skipif(not os.path.exists(CKPT), reason="trained checkpoint not present")
def test_char_tokenizer_from_committed_checkpoint(tmp_path):
    ckpt = torch.load(CKPT, map_location="cpu")
    tok = CharTokenizer(ckpt["chars"])
    js = run_js(tok.state(), ["ROMEO:\nWhat light"], tmp_path=tmp_path)
    assert js["encoded"] == [tok.encode("ROMEO:\nWhat light")]


@pytest.mark.skipif(not os.path.exists(DEFAULT_PATH), reason="dataset not downloaded")
def test_bpe_matches_python_on_tiny_shakespeare(tmp_path):
    # The tokenizer train.py --tokenizer bpe builds, applied to the whole dataset.
    with open(DEFAULT_PATH, encoding="utf-8") as f:
        text = f.read()
    train_text, _ = split_text(text)
    tok = BPETokenizer.train(train_text, 512)
    js = run_js(tok.state(), [text], tmp_path=tmp_path)
    assert js["encoded"][0] == tok.encode(text)
