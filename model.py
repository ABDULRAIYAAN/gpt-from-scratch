"""
A decoder-only Transformer (GPT) language model, written from scratch in PyTorch.

Architecture follows "Attention Is All You Need" (Vaswani et al., 2017),
decoder half only, with the pre-LayerNorm arrangement used by GPT-2.

Shape notation used in comments:
    B = batch size
    T = time / sequence length (number of tokens in context)
    C = embedding size (n_embd)
    nh = number of attention heads (n_head)
    H = head size (n_embd // n_head)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256  # maximum context length
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2


def upgrade_state_dict(state_dict: dict, n_head: int, n_layer: int) -> dict:
    """Return a copy of state_dict with old per-head attention weights fused.

    Older checkpoints stored one key/query/value Linear per head
    (blocks.{l}.sa.heads.{h}.{key,query,value}.weight, each (H, C)) plus a
    tril mask per head. The model now keeps them in a single (3C, C) weight,
    blocks.{l}.sa.qkv.weight, laid out as [query; key; value] with the heads
    stacked inside each part. A state dict that is already fused comes back
    unchanged (as a copy).
    """
    sd = dict(state_dict)
    for l in range(n_layer):
        p = f"blocks.{l}.sa."
        if f"{p}heads.0.query.weight" not in sd:
            continue
        parts = []
        for name in ("query", "key", "value"):
            parts += [sd.pop(f"{p}heads.{h}.{name}.weight") for h in range(n_head)]
        sd[f"{p}qkv.weight"] = torch.cat(parts, dim=0)
        for h in range(n_head):
            sd.pop(f"{p}heads.{h}.tril", None)
    return sd


class KVCache:
    """Keys and values already computed for the tokens in the context window.

    During generation every step used to run the whole window through the model again,
    even though only the newest token is new. Keys and values for earlier tokens never
    change (a token only attends to itself and the past), so they can be kept and reused:
    each step then costs one position of compute instead of up to block_size.

    Per layer the cache holds k and v of shape (B, nh, t, H), where t is the number of
    positions in the window so far. It cannot slide: positions are learned absolute
    embeddings, so dropping the oldest token would shift every other token's position.
    GPT.next_logits handles a full window by starting a new cache instead.
    """

    def __init__(self, n_layer: int):
        self.k = [None] * n_layer
        self.v = [None] * n_layer
        self.seen = 0  # length of the sequence the cache has been fed, including tokens since dropped

    def __len__(self) -> int:
        """Number of positions cached (the same for every layer between forward passes)."""
        return self.positions(0)

    def positions(self, layer: int) -> int:
        return 0 if self.k[layer] is None else self.k[layer].size(2)

    def clear(self):
        """Drop every cached position (seen is left alone)."""
        self.k = [None] * len(self.k)
        self.v = [None] * len(self.v)

    def update(self, layer: int, k, v):
        """Append this step's k, v (B, nh, T, H) for one layer and return all of them (B, nh, t + T, H)."""
        if self.k[layer] is not None:
            k = torch.cat((self.k[layer], k), dim=2)
            v = torch.cat((self.v[layer], v), dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v


class MultiHeadAttention(nn.Module):
    """Causal (masked) self-attention with all heads computed together.

    One Linear produces the queries, keys and values for every head at once,
    then the heads become an extra batch axis, so the whole layer is a few
    large matrix multiplies instead of a Python loop over heads. Each head
    still attends independently over its own H-wide slice.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        # Rows [0:C) are queries, [C:2C) keys, [2C:3C) values; head h owns rows h*H:(h+1)*H of each.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        # Lower-triangular mask: token t may only look at tokens 0..t.
        # A buffer so it moves with the model, but not saved: it is rebuilt from block_size.
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size)), persistent=False
        )

    def forward(self, x, cache: KVCache | None = None, layer: int = 0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)  # each (B, T, C)
        # Split C into heads and move the head axis next to the batch axis.
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, H)  what each token is looking for
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, H)  what each token contains
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, H)  what each token shares

        # With a cache, the T new tokens sit at positions P..P+T-1 after P cached ones,
        # and attend to the cached keys and values as well as their own.
        P = 0
        if cache is not None:
            P = cache.positions(layer)
            k, v = cache.update(layer, k, v)  # each (B, nh, P + T, H)

        # Attention scores for every head at once.
        # Scaled by 1/sqrt(head_size) so softmax does not saturate at init.
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, nh, T, P + T)
        # Rows P..P+T-1 of the mask: new token i sees positions 0..P+i. Without a cache P = 0.
        wei = wei.masked_fill(self.tril[P:P + T, :P + T] == 0, float("-inf"))  # no peeking at the future
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v  # (B, nh, T, H)  weighted sum of values (over P + T positions)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # concatenate the heads
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-token MLP: lets each token 'think' about what it gathered from attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (MLP)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.sa = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x, cache: KVCache | None = None, layer: int = 0):
        # Residual connections: x + f(x) keeps a clean gradient path through deep stacks.
        x = x + self.sa(self.ln1(x), cache, layer)
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load weights, converting checkpoints saved with the old per-head attention layout."""
        state_dict = upgrade_state_dict(state_dict, self.config.n_head, self.config.n_layer)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, cache: KVCache | None = None):
        """Logits (B, T, vocab_size) for idx (B, T), and the loss if targets are given.

        With a cache, idx holds only tokens the cache has not seen yet. They are placed
        after the cached positions, and their keys and values are added to the cache.
        """
        B, T = idx.shape
        P = len(cache) if cache is not None else 0  # positions already in the cache
        assert P + T <= self.config.block_size, f"{P + T} positions do not fit in block_size {self.config.block_size}"
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(P, P + T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb   # (B, T, C)
        for l, block in enumerate(self.blocks):
            x = block(x, cache, l)  # (B, T, C)
        x = self.ln_f(x)        # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def next_logits(self, idx, cache: KVCache):
        """Logits (B, vocab_size) for the token after idx (B, T), reusing cached keys and values.

        Call it once per new token with the whole sequence so far: the first call fills the
        cache from the last block_size tokens, later calls feed only idx's newest token.
        When the window is full there is no free position for the next token, so the cache
        is rebuilt from the most recent half of the window. The model then sees between
        block_size/2 and block_size tokens, the same rule the browser engine
        (web/engine.js) uses. Rebuilding costs one forward pass over block_size/2 tokens,
        once every block_size/2 tokens.
        """
        block, keep = self.config.block_size, self.config.block_size // 2
        if len(cache) == 0:
            new = idx[:, -block:]  # first call: everything that fits
        else:
            assert idx.size(1) == cache.seen + 1, "next_logits expects exactly one new token per call"
            if len(cache) >= block:
                cache.clear()      # no room left: start over on the recent half
                new = idx[:, -keep:]
            else:
                new = idx[:, -1:]  # (B, 1)
        logits, _ = self(new, cache=cache)  # (B, len(new), vocab_size)
        cache.seen = idx.size(1)
        return logits[:, -1, :]

    @staticmethod
    def _sample(logits, temperature: float, top_k: int | None):
        """Draw one token per row from logits (B, vocab_size), returning (B, 1)."""
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)  # (B, 1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None,
                 use_cache: bool = False):
        """Autoregressively sample max_new_tokens tokens, extending idx of shape (B, T).

        Without the cache every step reruns the last block_size tokens, so the context is
        always as long as it can be. With use_cache=True each step only computes the newest
        token; the results are the same until the window fills, after which the context
        drops back to half the window (see next_logits). That is much faster for long
        samples, but not token-for-token identical past the first window.
        """
        cache = KVCache(self.config.n_layer) if use_cache else None
        for _ in range(max_new_tokens):
            if use_cache:
                logits = self.next_logits(idx, cache)
            else:
                idx_cond = idx[:, -self.config.block_size:]  # crop to the context window
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]                    # only the last time step matters
            idx_next = self._sample(logits, temperature, top_k)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
