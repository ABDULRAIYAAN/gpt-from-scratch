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
    python evaluate.py --by-context chars ckpt.pt bpe/ckpt.pt # ... counted in characters, side by side
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
def per_target_nll(model: GPT, ids: torch.Tensor, stride: int | None = None,
                   batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """-log p of every target, with how many tokens of context it was predicted from.

    Runs exactly the same windows as total_nll. Inside a window that starts at `begin`, the
    target at position p is ids[begin+p+1] and is predicted from ids[begin : begin+p+1], so
    it had p+1 tokens of context. Returns two (N-1,) tensors indexed by target: entry j-1 is
    the nats for ids[j] and the number of tokens before it that the model could see.
    """
    T = model.config.block_size
    stride = check_stride(stride, T)
    N = len(ids)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    nll = torch.full((N - 1,), float("nan"), dtype=torch.float64)  # (N-1,) nats per target
    context = torch.zeros(N - 1, dtype=torch.long)                  # (N-1,) tokens of context
    windows = score_windows(N, T, stride)
    for length in sorted({end - begin for begin, end, _ in windows}, reverse=True):
        group = [w for w in windows if w[1] - w[0] == length]
        for i in range(0, len(group), batch_size):
            chunk = group[i : i + batch_size]
            x = torch.stack([ids[b:e] for b, e, _ in chunk]).to(device)          # (b, L)
            y = torch.stack([ids[b + 1 : e + 1] for b, e, _ in chunk]).to(device)  # (b, L)
            logits, _ = model(x)                                                 # (b, L, vocab)
            logp = F.log_softmax(logits.float(), dim=-1)
            tok_nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).double().cpu()  # (b, L)
            for row, (b, e, n_new) in enumerate(chunk):
                first = length - n_new                     # positions first..L-1 are new targets
                # position p predicts ids[b+p+1], which is target index b+p
                nll[b + first : e] = tok_nll[row, first:]
                context[b + first : e] = torch.arange(first + 1, length + 1)

    model.train(was_training)
    assert not nll.isnan().any() and (context > 0).all(), "every token after the first should be scored exactly once"
    return nll, context


def nll_by_context(model: GPT, ids: torch.Tensor, stride: int | None = None,
                   batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """total_nll split up by how many tokens of context each prediction had.

    Returns two (block_size,) tensors: entry c-1 holds the summed nats and the number of
    targets that were predicted from c tokens of context. Their sums are total_nll.
    """
    T = model.config.block_size
    nll, context = per_target_nll(model, ids, stride, batch_size)
    sums = torch.zeros(T, dtype=torch.float64).index_add_(0, context - 1, nll)       # (T,)
    count = torch.bincount(context - 1, minlength=T)                                   # (T,)
    return sums, count


def token_char_counts(tok, ids: list[int] | torch.Tensor) -> torch.Tensor:
    """How many characters each token adds to the text, as a (N,) tensor.

    For the character tokenizer that is 1 per token. A byte-level BPE token can cover a few
    characters, or only part of one (an accented letter is 2 bytes in UTF-8, and a merge can
    split them), so a token is credited with the characters that *start* inside it: every
    byte except the 10xxxxxx continuation bytes. The counts add up to len(text) exactly.
    """
    ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    if isinstance(tok, CharTokenizer):
        return torch.ones(len(ids), dtype=torch.long)
    starts = [sum(1 for byte in tok.vocab[i] if byte & 0xC0 != 0x80) for i in range(tok.vocab_size)]
    return torch.tensor([starts[i] for i in ids], dtype=torch.long)


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


def loss_by_char_context(model: GPT, tok, text: str, stride: int | None = None,
                         batch_size: int = 64) -> list[dict]:
    """Like loss_by_context, but measures context in characters and loss in bits per character.

    A BPE model's 64-token window holds about twice as many characters as the character
    model's, so "16 tokens of context" means different things for the two. Counting the
    characters the prediction could see instead puts both models on the same axis. Each
    target is put in a power-of-two bucket by the characters its context tokens spell out
    (see token_char_counts), and each row reports the bucket's bits per character: its nats
    over the characters its target tokens spell out, so the rows add back up to
    evaluate_text's bits per character.
    """
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    nll, context = per_target_nll(model, ids, stride, batch_size)  # (N-1,), (N-1,)
    chars = token_char_counts(tok, ids)                              # (N,) characters per token
    cum = torch.cat([torch.zeros(1, dtype=torch.long), chars.cumsum(0)])  # (N+1,) chars before each id
    j = torch.arange(1, len(ids))                                     # (N-1,) id index of each target
    # Target ids[j] was predicted from ids[j-c : j]. A context that is only the tail of one
    # multi-byte character starts no character; count it as 1 so it lands in the first bucket.
    ctx_chars = (cum[j] - cum[j - context]).clamp(min=1)              # (N-1,)
    tgt_chars = chars[1:]                                             # (N-1,) characters each target adds

    top = 1 << (int(ctx_chars.max()) - 1).bit_length()  # round up to a power of two, so buckets line up across models
    total = len(j)
    rows = []
    for lo, hi in context_buckets(top):
        mask = (ctx_chars >= lo) & (ctx_chars <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        nats, n_chars = nll[mask].sum().item(), int(tgt_chars[mask].sum())
        rows.append({"context": (lo, hi), "targets": n, "share": n / total, "chars": n_chars,
                     "tokens_of_context": context[mask].double().mean().item(),
                     "loss": nats / n, "bits_per_char": nats / max(n_chars, 1) / math.log(2)})
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
    p.add_argument("--by-context", nargs="?", const="tokens", choices=["tokens", "chars"], default=None,
                   help="also break each checkpoint's loss down by how much context each prediction had, "
                        "counted in tokens (the default) or in characters, which lines BPE and character models up")
    args = p.parse_args()

    device = pick_device(args.device)
    _, val_text = split_text(load_dataset(args.data))  # the same characters train.py validates on
    print(f"validation split: {len(val_text):,} characters\n")

    rows, breakdowns = [], []
    for path in args.ckpts:
        model, tok = load_checkpoint(path, device)
        r = evaluate_text(model, tok, val_text, args.stride, args.batch_size)
        if args.by_context == "tokens":
            breakdowns.append((path, tok, loss_by_context(model, tok, val_text, args.stride, args.batch_size)))
        elif args.by_context == "chars":
            breakdowns.append((path, tok, loss_by_char_context(model, tok, val_text, args.stride, args.batch_size)))
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

    def span(b):
        lo, hi = b["context"]
        return str(lo) if lo == hi else f"{lo}-{hi}"

    for path, tok, buckets in breakdowns:
        if args.by_context == "chars":
            print(f"\n`{path}` by context length (chars the prediction could see):\n")
            print("| Context (chars) | Targets | Share | Tokens of context | Bits/char |")
            print("|---|---|---|---|---|")
            for b in buckets:
                print(f"| {span(b)} | {b['targets']:,} | {b['share']:.1%} | "
                      f"{b['tokens_of_context']:.1f} | {b['bits_per_char']:.3f} |")
            continue
        unit = "chars" if isinstance(tok, CharTokenizer) else "tokens"
        print(f"\n`{path}` by context length ({unit} the prediction could see):\n")
        print(f"| Context ({unit}) | Targets | Share | Loss/token | Bits/token |")
        print("|---|---|---|---|---|")
        for b in buckets:
            print(f"| {span(b)} | {b['targets']:,} | {b['share']:.1%} | {b['loss']:.4f} | {b['bits_per_token']:.3f} |")

    if args.by_context == "chars" and len(breakdowns) > 1:
        # Every checkpoint uses the same power-of-two buckets, so their rows can sit side by side.
        spans = sorted({b["context"] for _, _, buckets in breakdowns for b in buckets})
        tables = [{b["context"]: b for b in buckets} for _, _, buckets in breakdowns]
        print("\nBits per character by context length in characters, side by side:\n")
        print("| Context (chars) | " + " | ".join(f"`{path}`" for path, _, _ in breakdowns) + " |")
        print("|---|" + "---|" * len(breakdowns))
        for s in spans:
            cells = [f"{t[s]['bits_per_char']:.3f}" if s in t else "" for t in tables]
            print(f"| {span({'context': s})} | " + " | ".join(cells) + " |")

if __name__ == "__main__":
    main()
