# Devlog

## 2026-11-15: Characters vs BPE, measured in bits per character

**What changed.** New `evaluate.py` scores a checkpoint on the whole validation split instead of random batches. It slides the 64-token window over the text, adds up -log p for every token exactly once, and reports loss per token, bits per token and bits per character. Bits per character is the number that matters here: a BPE token is about two characters, so its loss per token is naturally higher and can't be put next to the character model's. Dividing the total bits by the number of characters puts both on the same scale. Give it several checkpoints and it prints one table. `--stride` makes the windows overlap so every token gets more context, at the cost of more compute.

**Fixing the split first.** `train.py` used to tokenize the whole text and then take the first 90% of the tokens for training. For characters that is fine, but for BPE the boundary landed at a different character than for the character model, and the BPE merges were learned from the full text, validation part included. Now the text itself is split (new `split_text` in `data.py`) and each part is encoded separately, and BPE learns its merges from the training text only. For the character tokenizer the tensors are exactly the same as before, so `checkpoints/ckpt.pt` is still a fair baseline. Because the merges no longer see the validation text, they compress it a bit less: 1.88 characters per token on the validation split, against 1.94 over the whole dataset.

**The comparison.** I trained the `small` config with `--tokenizer bpe --max-iters 5000 --eval-interval 500`, the same number of steps as the committed character model, on the 2-core CPU. It took 11.1 minutes, though that is a little inflated because I ran the tests alongside it for part of the run. Best validation loss was 2.838 nats per token at step 5000. On the full validation split (111,540 characters):

| | Characters | BPE, 512 tokens |
|---|---|---|
| Parameters | 816,705 | 931,584 |
| Validation tokens | 111,539 | 59,400 |
| Context window | 64 characters | ~120 characters |
| Loss per token | 1.6126 | 2.8339 |
| Bits per character | 2.327 | 2.177 |
| Bits per character, `--stride 16` | 2.272 | 2.151 |

So BPE comes out 6.4% better per character. It isn't a perfectly controlled result: the BPE model has 14% more parameters (512-row embedding and output layers instead of 65), and with the same 64-token window it sees about twice as much text, which is half the reason to use BPE in the first place. It also overfits sooner. At step 5000 the gap between train and validation loss was 0.47 nats per token for BPE (about 0.25 per character), against 0.21 for characters, because the same text gives it half as many training tokens. The BPE sample in `results/train_log_bpe.txt` still has made-up words and no real sense.

**Tests.** Added `tests/test_evaluate.py`: every token is scored exactly once for different lengths and strides; the default stride matches a plain back-to-back window loop; `--stride 1` matches predicting each token one forward pass at a time; a model with a zeroed output layer scores exactly log2(65) bits per character; bits per character divides by characters, not tokens, for BPE; the evaluation doesn't leave the model in eval mode; splitting the text gives the same character tokens as the old token split; and the committed checkpoint's full-split loss is within 0.05 of the loss saved at training time (it is 1.6126 against 1.6017).

**Next.** The browser demo is still character-level. Porting the BPE encoder to `web/engine.js` would let the demo use the better model.

## 2026-11-14: A byte-pair encoding tokenizer, written from scratch

**What changed.** `tokenizer.py` now has a `BPETokenizer` next to the `CharTokenizer`. It is byte-level: the starting vocabulary is the 256 possible bytes, so it can encode any text, including characters it never saw in training. Training counts every adjacent pair of tokens in the text, turns the most frequent pair into a new token, and repeats until the vocabulary reaches the size asked for. Before counting, the text is split into chunks (a word with its leading space, a run of digits, a run of punctuation, a run of whitespace), and merges never cross a chunk boundary, so no token is half of one word and half of the next. Each distinct chunk is processed once and weighted by how often it appears, which is why training is quick. Encoding applies the learned merges in the order they were learned, which gives exactly the tokens training built.

**Using it.** `python train.py --config small --tokenizer bpe` learns the tokenizer from the training text and saves it inside the checkpoint, and `sample.py` now rebuilds whichever tokenizer a checkpoint was trained with. Character-level stays the default, so `checkpoints/ckpt.pt`, the browser demo and `app.py` are unchanged. `export_web.py` now refuses a BPE checkpoint with a clear message, because the JavaScript demo only has a character tokenizer.

**Why not tiktoken.** The roadmap item said `tiktoken`, but its GPT-2 vocabulary has 50,257 tokens. At this model's embedding size of 128, the token embedding and output layer alone would come to about 12.9M parameters, more than 15 times the 0.82M of the whole current model, trained on only 1 MB of text. A small vocabulary learned from Shakespeare itself fits this model much better, and writing it by hand keeps to the spirit of the project.

**Numbers (measured on a 2-core CPU).** With 512 tokens (256 merges), learning the tokenizer from Tiny Shakespeare takes about 7 seconds and encoding the whole dataset about 0.2 seconds. The 1,115,394 characters become 575,345 tokens, 1.94 characters per token, so the 64-token context window now covers about twice as much text. The `small` model grows from 0.82M to 0.93M parameters because of the larger embedding and output layers. The first learned pieces are ` t`, `he`, ` a`, `ou` and ` s`; ` the` is the 12th merge, and whole words such as ` good` follow later.

**Tests.** Added `tests/test_tokenizer.py`: lossless round trips on text with accents, CJK characters, emoji, tabs and Windows line endings; the chunk pattern always covers the whole text; the first merge is the most frequent pair; training is deterministic and stops when nothing is left to merge; encoding reproduces training merge by merge; no token crosses a word boundary; and tokenizers load back from a checkpoint, including the old character-only format.

**Next.** Train the `small` model with BPE and compare it with the character model. Loss per token is not comparable between the two, so the comparison has to be in bits per character.

## 2026-11-13: All attention heads in one matrix multiply

**What changed.** Attention used to run a Python loop over four `Head` modules, each with its own key, query and value `Linear`, and then concatenate the results. `MultiHeadAttention` now has a single `Linear(C, 3C)` that produces the queries, keys and values for every head in one product. The output is reshaped to `(B, n_head, T, head_size)` so the heads become a batch axis, and the scores, mask, softmax and weighted sum run for all heads at once. The math is the same, and the parameter count is still 816,705.

**Old checkpoints.** `checkpoints/ckpt.pt` was saved with the per-head keys (`blocks.{l}.sa.heads.{h}.query.weight` and so on). `GPT.load_state_dict` now runs `upgrade_state_dict`, which stacks those into `blocks.{l}.sa.qkv.weight` in the order `[query; key; value]`, with head `h` owning rows `h*32:(h+1)*32` of each part. It also drops the per-head `tril` masks, which are no longer saved: the mask is a non-persistent buffer rebuilt from `block_size`. The committed checkpoint was left as it is and loads with strict key checking.

**Browser demo.** `web/engine.js` now does one `3C x C` matrix-vector product per layer and slices each head's q, k and v out of it. If it is given an older export with per-head weights, it packs them into the same fused layout first. `export_web.py` now exports from the loaded model instead of the raw checkpoint, so the page always gets the fused layout, and it seeds before building the model so the sample shown on load stays the same. The rebuilt `docs/index.html` has the same weights (bit for bit, just regrouped), the same sample, and generates exactly the same text as before.

**Tests.** Added `tests/` with pytest:

- the batched model matches the original per-head model (kept in `tests/legacy.py` as a reference) to within 1e-5
- converting old to new and back is lossless, and loading does not modify the caller's dict
- the committed checkpoint loads and gives the same logits as the per-head model
- the causal mask still hides future tokens
- the JavaScript engine matches PyTorch for both weight layouts, and on the real checkpoint

**Speed.** Measured on a 2-core CPU, one AdamW training step: the `small` config went from 143.5 ms to 124.8 ms (1.15x), and the `full` config at batch size 8 from 1,166 ms to 1,122 ms (1.04x). The gain should be larger on a GPU, where one large kernel replaces many small ones, but that is not measured yet.
