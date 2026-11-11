"""
Baseline: a bigram language model.

Each character predicts the next one using only itself, no context at all.
This is the starting point the GPT is compared against.

Usage:
    python bigram.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from data import load_dataset, get_batch, pick_device
from tokenizer import CharTokenizer

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2

torch.manual_seed(1337)
device = pick_device("auto")

text = load_dataset()
tok = CharTokenizer.from_text(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Row i of this table holds the logits for "what comes after token i".
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for name, split in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


if __name__ == "__main__":
    model = BigramLanguageModel(tok.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- sample ---")
    print(tok.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
