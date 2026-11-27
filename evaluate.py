"""
Evaluate checkpoints on the whole validation split and report bits per character.

Loss per token cannot be compared between tokenizers: a BPE token covers about two
characters, so guessing the next token is a harder job than guessing the next character,
and its loss is bigger even when the model is better. Bits per character puts every
tokenizer on the same scale: add up how surprised the model is by each token of the
validation text (its negative log-likelihood), convert from nats to bits, and divide by
the number of characters those tokens spell out.

Unlike the loss printed during training, which is an average over random batches, this
scores every token of the validation text exactly once, so the number is exact and the
same every time it is run.

Usage:
    python evaluate.py                                        # checkpoints/ckpt.pt
    python evaluate.py checkpoints/ckpt.pt checkpoints/bpe/ckpt.pt
    python evaluate.py --stride 32                            # overlapping windows: more context per token
    python evaluate.py --by-context                           # loss by how many tokens of context it had
"""

import argparse
import math

import torch
from torch.nn import functional as F

from data import DEFAULT_PATH, load_dataset, pick_device, split_text
from model import GPT, GPTConfig
from tokenizer import CharTokenizer, load_tokenizer


def check_stride(stride: int | None, T: int) -> int:
    stride = T if stride is None else stride
    if not 1 <= stride <= T:
        raise ValueError(f"stride must be between 1 and block_size ({T}), got {stride}")
    return stride


def score_windows(N: int, T: int, stride: int) -> list[tuple[int, int, int]]:
    """The windows total_nll runs over a sequence of N ids, as (begin, end, n_new).

    Window w reads ids[begin:end] and predicts ids[begin+1:end+1]; only its last n_new
    targets are new (the earlier ones were scored by the window before it).
    """
    windows, prev_end = [], 0
    for begin in range(0, N - 1, stride):
        end = min(begin + T, N - 1)
        windows.append((begin, end, end - prev_end))
        prev_end = end
        if end == N - 1:
            break
    return windows


@torch.no_grad()
def total_nll(model: GPT, ids: torch.Tensor, stride: int | None = None, batch_size: int = 64) -> tuple[float, int]:
    """Sum of -log p(token) in nats over ids[1:], each token predicted from the ones before it.

    The model only sees block_size tokens at a time, so the sequence is cut into windows
    starting every `stride` tokens. Each window predicts its targets from the tokens inside
    it, and only the targets no earlier window has scored are counted, so every token is
    scored once. stride = block_size (the default) gives back-to-back windows, the same
    conditions as training, where a token's context is anywhere from 1 to block_size tokens.
    A smaller stride gives every token after the first window at least block_size - stride
    tokens of context, which lowers the loss but costs block_size / stride times more compute.

    Returns (total nats, number of tokens scored).
    """
    T = model.config.block_size
    stride = check_stride(stride, T)
    N = len(ids)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()  # no dropout
    windows = score_windows(N, T, stride)

    nll, count = 0.0, 0
    # Every window is T long except possibly the last one, so batch windows of equal length.
    for length in sorted({end - begin for begin, end, _ in windows}, reverse=True):
        group = [w for w in windows if w[1] - w[0] == length]
        for i in range(0, len(group), batch_size):
            chunk = group[i : i + batch_size]
            x = torch.stack([ids[b:e] for b, e, _ in chunk]).to(device)          # (b, L)
            y = torch.stack([ids[b + 1 : e + 1] for b, e, _ in chunk]).to(device)  # (b, L)
            logits, _ = model(x)                                                 # (b, L, vocab)
            logp = F.log_softmax(logits.float(), dim=-1)
            tok_nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)              # (b, L)
            for row, (_, _, n_new) in enumerate(chunk):
                nll += tok_nll[row, length - n_new :].sum().item()  # only the targets not scored yet
                count += n_new

    model.train(was_training)
    assert count == N - 1, "every token after the first should be scored exactly once"
    return nll, count


def evaluate_text(model: GPT, tok, text: str, stride: int | None = None, batch_size: int = 64) -> dict:
    """Score text with model and return loss per token and bits per character."""
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    nll, n_tokens = total_nll(model, ids, stride, batch_size)
    # The first token is never predicted (nothing comes before it), so its characters don't count.
    n_chars = len(text) - len(tok.decode(ids[:1].tolist()))
    return {
        "tokens": n_tokens,
        "chars": n_chars,
        "chars_per_token": n_chars / n_tokens,
        "loss": nll / n_tokens,                          # nats per token, comparable to the training log
        "bits_per_token": nll / n_tokens / math.log(2),
        "bits_per_char": nll / n_chars / math.log(2),    # comparable across tokenizers
    }


@torch.no_grad()
def nll_by_context(model: GPT, ids: torch.Tensor, stride: int | None = None,
                   batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """total_nll split up by how much context each prediction had.

    Runs exactly the same windows as total_nll. Inside a window that starts at `begin`, the
    target at position p is predicted from ids[begin : begin+p+1], so it had p+1 tokens of
    context. Returns two (block_size,) tensors: entry c-1 holds the summed nats and the
    number of targets that were predicted from c tokens of context. Their sums are total_nll.
    """
    T = model.config.block_size
    stride = check_stride(stride, T)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    nll = torch.zeros(T, dtype=torch.float64)    # (T,) nats, indexed by context length - 1
    count = torch.zeros(T, dtype=torch.long)     # (T,) targets, indexed the same way
    windows = score_windows(len(ids), T, stride)
    for length in sorted({end - begin for begin, end, _ in windows}, reverse=True):
        group = [w for w in windows if w[1] - w[0] == length]
        for i in range(0, len(group), batch_size):
            chunk = group[i : i + batch_size]
            x = torch.stack([ids[b:e] for b, e, _ in chunk]).to(device)          # (b, L)
            y = torch.stack([ids[b + 1 : e + 1] for b, e, _ in chunk]).to(device)  # (b, L)
            logits, _ = model(x)                                                 # (b, L, vocab)
            logp = F.log_softmax(logits.float(), dim=-1)
            tok_nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).double().cpu()  # (b, L)
            for row, (_, _, n_new) in enumerate(chunk):
                first = length - n_new                      # positions first..L-1 are new targets
                nll[first:length] += tok_nll[row, first:]   # position p had p+1 tokens of context
                count[first:length] += 1

    model.train(was_training)
    assert count.sum().item() == len(ids) - 1, "every token after the first should be scored exactly once"
    return nll, count


def context_buckets(T: int) -> list[tuple[int, int]]:
    """Context lengths 1..T grouped by powers of two: (1, 1), (2, 2), (3, 4), (5, 8), ... (.., T)."""
    buckets, lo = [(1, 1)], 2
    while lo <= T:
        hi = min(2 * (lo - 1), T)
        buckets.append((lo, hi))
        lo = hi + 1
    return buckets


def loss_by_context(model: GPT, tok, text: str, stride: int | None = None, batch_size: int = 64) -> list[dict]:
    """Score text and report the loss for each bucket of context lengths (see context_buckets).

    Each row has the context range in tokens, the number of targets in it, their share of all
    targets, the loss in nats per token and in bits per token. For the character tokenizer a
    token is a character, so bits per token is bits per character.
    """
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    nll, count = nll_by_context(model, ids, stride, batch_size)
    total = count.sum().item()
    rows = []
    for lo, hi in context_buckets(model.config.block_size):
        n = count[lo - 1 : hi].sum().item()
        if n == 0:
            continue  # with a small stride, short contexts only happen in the first window
        loss = nll[lo - 1 : hi].sum().item() / n
        rows.append({"context": (lo, hi), "targets": n, "share": n / total,
                     "loss": loss, "bits_per_token": loss / math.log(2)})
    return rows


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    tok = load_tokenizer(ckpt)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, tok


def main():
    p = argparse.ArgumentParser(description="Bits per character on the full validation split.")
    p.add_argument("ckpts", nargs="*", default=["checkpoints/ckpt.pt"], help="checkpoint files to compare")
    p.add_argument("--data", default=DEFAULT_PATH, help="the text the checkpoints were trained on")
    p.add_argument("--stride", type=int, default=None, help="window step in tokens (default: block_size)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--by-context", action="store_true",
                   help="also break each checkpoint's loss down by how many tokens of context each prediction had")
    args = p.parse_args()

    device = pick_device(args.device)
    _, val_text = split_text(load_dataset(args.data))  # the same characters train.py validates on
    print(f"validation split: {len(val_text):,} characters\n")

    rows, breakdowns = [], []
    for path in args.ckpts:
        model, tok = load_checkpoint(path, device)
        r = evaluate_text(model, tok, val_text, args.stride, args.batch_size)
        if args.by_context:
            breakdowns.append((path, tok, loss_by_context(model, tok, val_text, args.stride, args.batch_size)))
        kind = type(tok).__name__.replace("Tokenizer", "").lower()
        context = model.config.block_size * r["chars_per_token"]
        rows.append(
            f"| `{path}` | {kind} | {tok.vocab_size} | {model.num_params():,} | {r['tokens']:,} | "
            f"{r['chars_per_token']:.2f} | {context:.0f} | {r['loss']:.4f} | {r['bits_per_token']:.3f} | "
            f"**{r['bits_per_char']:.3f}** |"
        )

    print("| Checkpoint | Tokenizer | Vocab | Parameters | Val tokens | Chars/token | Context (chars) "
          "| Loss/token | Bits/token | Bits/char |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))

    for path, tok, buckets in breakdowns:
        unit = "chars" if isinstance(tok, CharTokenizer) else "tokens"
        print(f"\n`{path}` by context length ({unit} the prediction could see):\n")
        print(f"| Context ({unit}) | Targets | Share | Loss/token | Bits/token |")
        print("|---|---|---|---|---|")
        for b in buckets:
            lo, hi = b["context"]
            span = str(lo) if lo == hi else f"{lo}-{hi}"
            print(f"| {span} | {b['targets']:,} | {b['share']:.1%} | {b['loss']:.4f} | {b['bits_per_token']:.3f} |")


if __name__ == "__main__":
    main()
