"""Dataset loading and batching."""

import os
import urllib.request

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "input.txt")


def load_dataset(path: str = DEFAULT_PATH) -> str:
    """Read the training text, downloading Tiny Shakespeare if it is the default and missing."""
    if not os.path.exists(path):
        if path != DEFAULT_PATH:
            raise FileNotFoundError(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Sample batch_size random chunks; targets are the inputs shifted one step right."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def pick_device(choice: str = "auto") -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
