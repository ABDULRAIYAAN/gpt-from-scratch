"""
The original per-head attention, kept only as a reference for the tests.

This is how model.py computed attention before the heads were batched into one
matrix multiply: a Python loop over Head modules, each with its own key, query
and value Linear. Checkpoints saved from it use keys like
blocks.0.sa.heads.2.query.weight.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

import model as current


class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(head_size * config.n_head, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class LegacyGPT(current.GPT):
    """The current GPT with every attention layer swapped for the per-head version."""

    def __init__(self, config):
        super().__init__(config)
        for block in self.blocks:
            block.sa = MultiHeadAttention(config)
        self.apply(self._init_weights)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Plain nn.Module loading: no conversion, the keys must already be per-head.
        return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)


def split_state_dict(state_dict, n_head, n_layer):
    """Inverse of model.upgrade_state_dict: turn fused qkv weights back into per-head ones."""
    sd = dict(state_dict)
    for l in range(n_layer):
        p = f"blocks.{l}.sa."
        w = sd.pop(f"{p}qkv.weight")
        C = w.shape[1]
        hs = C // n_head
        for part, name in enumerate(("query", "key", "value")):
            for h in range(n_head):
                start = part * C + h * hs
                sd[f"{p}heads.{h}.{name}.weight"] = w[start : start + hs].clone()
    return sd
