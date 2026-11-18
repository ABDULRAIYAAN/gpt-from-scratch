# Devlog

## 2026-11-18: A BPE version of the demo page

**What changed.** The page's text used to be written for the character model, and yesterday's BPE page still said "65 characters", "next character" and "the 64 characters it can see". Now `export_web.py` writes the wording from the checkpoint. `web/template.html` has `{{placeholders}}` for every sentence that depends on the model, and `page_text()` fills them: the lede, the "Next letter" button (now "Next token" for BPE), the guesses and attention captions, the about section and the whole spec table. The parameter count, layers, heads, width and the "N times smaller than GPT-3" figure are all worked out from the model, so they can't go stale. The placeholders are filled before the scripts and weights go in, and the export fails if one is left unfilled.

**Numbers that need the text.** A BPE page gives its context as "64 tokens, ~120 characters" and shows bits per character, which is the number that can be compared with the character model. Both come from running `evaluate_text` over the validation split while exporting; the whole BPE export takes about 4 seconds. If the dataset can't be read, the page simply leaves those numbers out. The training row is built from three new fields `train.py` now saves (`max_iters`, `minutes`, `hardware`). The minutes are only shown if the saved step is the last one, because otherwise they don't cover the whole run. The committed checkpoint predates these fields, so its row is kept as a constant.

**Attention strip.** For the character model the strip is a 16-column grid of square cells. A BPE token can be ` Gloucester`, so on a BPE page the strip now flows like text: each cell is as wide as its token, and only the tokens actually in the window are shown, not empty placeholders. The guesses column is also a little wider. All of this is behind a `tokens` class on `<main>`, so the character page's layout doesn't change.

**Checking the character page didn't change.** The rebuilt `docs/index.html` differs from yesterday's only in the six new CSS lines that apply to BPE pages. The weights, sample and every word of text are byte for byte the same. `test_export_builds_a_fused_page` now also checks that the exporter rebuilds the committed `docs/index.html` exactly, so the page and the code can't drift apart.

**Publishing it.** I retrained the BPE model (`--tokenizer bpe --max-iters 5000 --eval-interval 500`). It gave exactly the same losses as before at every eval step (best validation loss 2.838), since the seed is fixed. It took 13.9 minutes this time, against 11.1 before, because I was editing and running checks alongside it. The page records the 13.9, rounded to 14 min. The export measured 1.88 characters per token and 2.177 bits per character, the same as the comparison table. `docs/bpe.html` is 2.5 MB. In headless Chromium with a seeded `Math.random`, it wrote 600 characters with no errors and no U+FFFD. The strip showed 42 tokens, none of them cut off, and guesses like `What` were shown whole. The character page gave the same results as before.

**Tests.** New `tests/test_export.py` checks that `page_text` fills exactly the placeholders the template has, for both tokenizers; that the character wording is unchanged sentence by sentence, with no "token" anywhere in its body; that a BPE page talks about tokens and has none of the character text left; that the BPE page without validation stats drops those numbers cleanly; and that `training_summary` leaves out the minutes when the best step wasn't the last. The BPE export test now passes `--data` and checks for the bits per character row. 76 tests pass, up from 71.

**Next.** `app.py` is the last part of the project that only runs the character model.

## 2026-11-17: The browser engine runs BPE checkpoints

**What changed.** `export_web.py` used to refuse anything but a character-level checkpoint. Now it stores the checkpoint's tokenizer state in the page next to the weights and inlines `web/tokenizer.js` before `web/engine.js`. `createGPT` builds the tokenizer from that state, so the same engine runs either model. It still reads older exports that only stored the character list. `python export_web.py --ckpt checkpoints/bpe/ckpt.pt --out docs/bpe.html` now builds a working BPE page. `docs/index.html` stays the character model.

**The page, token by token.** The page used to look up `gpt.chars[id]` everywhere. Now it asks for `gpt.piece(id)`, the text of a token. A BPE token can be several characters, so inside a longer token a space shows as `·` and a newline as `↵` (` the` becomes `·the`). The length setting still counts characters, not tokens, so "600 characters" means the same for both models. An empty prompt used to start from id 0, which is the newline for the character model but the byte 0x00 for BPE. It now starts from whatever ids a newline encodes to, which is still id 0 for the character model.

**Characters split across tokens.** BPE here is byte-level, so an emoji (4 bytes in UTF-8) can come out as several tokens, and decoding each token on its own gives U+FFFD for every partial byte. I added `decoder()` to both JavaScript tokenizers. Its `push(id)` uses a streaming `TextDecoder` that holds back unfinished bytes and only returns whole characters, and the page writes generated text through it. Joined together, what it returns is the same as `decode(ids)`.

**Checking nothing changed for the character model.** The rebuilt `docs/index.html` has the same weights, bit for bit, and the same sample on load. I loaded the old and new pages in headless Chromium with the same seeded `Math.random` and ran the same steps on both: write 600 characters, 5 single steps, clicking a guess, a blank prompt, and a prompt with `@` and an emoji. The written text, the status line, the guess labels and the attention strip were identical every time.

**Trying it with a real BPE model.** The BPE checkpoint isn't in git, so I retrained the `small` config with `--tokenizer bpe` for 5,000 steps (best validation loss 2.838, the same as last time; 14.3 minutes because the tests were running alongside it). Its page loaded, wrote 600 characters, and showed guesses like `·that` and `'s`. A prompt with an emoji came back whole on the page, with no U+FFFD in any generated text. In Node, generating 1,000 tokens took 1.54 ms per token for the character model and 2.17 ms for BPE. BPE is slower per token because its output layer is 512 wide instead of 65, but each token is 1.86 characters, so it writes about 860 characters a second against 650.

**Tests.** `tests/test_engine.py` now checks the engine on a small BPE model: the logits match PyTorch, `encode` gives the same ids as `tokenizer.py`, an empty prompt starts from the newline's ids, and streaming ids that split multi-byte characters gives exactly `decode(ids)` with no U+FFFD. I checked that last test catches a broken decoder by swapping in the per-token one. The character tests now also cover old exports with only a character list, dropping unknown characters from prompts, and the committed checkpoint starting from id 0. A new test exports a small BPE checkpoint and checks the page's tokenizer state and sample. 71 tests pass, up from 68.

**Next.** The page's text still describes the character model ("65 characters", "next character", "the 64 characters it can see"), and the attention strip's cells are sized for one character, so longer tokens are cut off. Fix both, then publish a BPE page next to the character one.

## 2026-11-16: The tokenizers in JavaScript

**What changed.** New `web/tokenizer.js` has both tokenizers in plain JavaScript. `createTokenizer(state)` takes the same state object `tokenizer.py` saves in a checkpoint (`{type: "char", chars}` or `{type: "bpe", merges}`) and gives back `encode`, `decode`, `vocabSize` and `piece(id)`, the text of a single token, which the demo will need for its labels. The BPE encoder works like the Python one: split the text into chunks, turn each chunk into UTF-8 bytes, then keep applying the earliest-learned merge among the adjacent pairs until none is left. Chunks are cached, since the same words come up again and again. This is the first half of putting the BPE model in the browser demo. The page itself is unchanged for now.

**Getting the ids to match exactly.** Two small differences between Python and JavaScript would have given different tokens:

- The chunk pattern uses `\s`, and the two languages disagree about what counts as whitespace. Python's `\s` matches U+001C to U+001F and U+0085, JavaScript's doesn't, and JavaScript's matches U+FEFF (the byte-order mark), Python's doesn't. So `tokenizer.js` spells out Python's set by hand. It also uses the `u` flag so an emoji counts as one character, not two UTF-16 halves.
- `TextDecoder` drops a byte-order mark at the start of the text by default, and Python's `decode` keeps it. Passing `ignoreBOM: true` keeps it.

Leaving out either fix makes the new tests fail, which I checked by undoing each one in turn.

**Tests.** Added `tests/test_web_tokenizer.py` (with a small Node helper, `tests/run_tokenizer.js`). It checks that the JavaScript chunks are the same as `CHUNK_PATTERN.findall` on awkward text, including every whitespace character Python knows about; that BPE ids, decoded text and the text of every token match Python; that decoding cut-off multi-byte characters and 600 random id sequences puts U+FFFD in the same places as `errors="replace"`; that the character tokenizer matches, including the error for unknown characters; and that a 512-token BPE learned from the training split gives exactly the same 575,806 ids as Python over all 1,115,394 characters of Tiny Shakespeare. 68 tests pass.

**Speed (Node 22, 2-core CPU).** Encoding the whole dataset takes 127 ms the first time and 74 ms once the chunk cache is warm, against 175 ms in Python. Decoding it back takes 21 ms. A prompt-sized string is under a millisecond, so encoding won't be noticeable in the page.

**Next.** Export a BPE checkpoint with its merges into `docs/index.html` and change the demo to show tokens instead of characters in its probability list and attention map.

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
