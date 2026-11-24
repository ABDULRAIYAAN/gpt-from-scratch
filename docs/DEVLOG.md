# Devlog

## 2026-11-24: Keeping the last step as well as the best-estimated one

**Why.** Yesterday's sweep showed that `train.py` sometimes saves step 4,500 instead of step 5,000. It keeps whichever step had the lowest *estimated* validation loss, and that estimate is only 50 random batches, too noisy to tell apart two evals a few thousandths of a nat apart. With the cosine schedule the last 500 steps run at the lowest learning rate, so throwing them away on a noisy estimate seemed wrong, and it adds noise to every comparison in the README.

**What changed.** `train.py` now writes `final.pt`, the weights after the last step, next to `ckpt.pt`. Both checkpoints are built by one `save_checkpoint` helper, so they carry the same config, tokenizer and run details, and `final.pt` loads anywhere `ckpt.pt` does. `ckpt.pt` is chosen exactly as before; I checked that a short run with the old and new `train.py` gives identical `ckpt.pt` weights and the same sample. The summary line now says which step was best. `seeds.py` has a new `--pick best|final|both`: `final` scores the last steps, and `both` scores the two checkpoints of every run and prints a table of the difference per run, with its mean, spread and how often each side wins. Old checkpoints have no `final.pt`, so they only work with `--pick best` and anything else gives a clear error. `final.pt` and the new run folder are in `.gitignore`.

**The runs.** I retrained the four runs that had picked step 4,500: constant rate and warmup + cosine, seeds 1 and 1337, `small` config, 5,000 steps, two at a time on the 2-core CPU (16.5-17.7 minutes each). The seed-1 runs came out bit-for-bit the same as yesterday (2.318 and 2.283 bits per character for the picked step). Seed 1337 differs a little from the README's older numbers (2.332 against 2.341 for constant), because those were trained on an earlier PyTorch version. On the full validation split:

| Run | Picked step | Bits/char, picked | Bits/char, last step | Last - picked |
|---|---|---|---|---|
| Constant, seed 1 | 4500 | 2.318 | 2.305 | -0.013 |
| Constant, seed 1337 | 4500 | 2.332 | 2.319 | -0.014 |
| Cosine, seed 1 | 4500 | 2.283 | 2.273 | -0.010 |
| Cosine, seed 1337 | 4500 | 2.272 | 2.264 | -0.008 |

**What it says.** The last step won all four times, by 0.011 bits per character on average (std 0.003), which is about one seed-to-seed standard deviation. The estimate can be badly off: for cosine seed 1337 it said step 5,000 was 0.018 nats *worse* than step 4,500, and on the full split it is 0.008 bits better. With the constant rate I expected the pick to go either way, but in both runs the last step was still better. Seeds 2 and 3 already picked step 5,000, so with the last step for seeds 1-3 the constant rate averages 2.307 (std 0.008) and warmup + cosine 2.269 (std 0.005). The schedule gain becomes 0.038 bits (1.6%), close to the 0.039 the picked checkpoints give for the same three seeds, and the spread between seeds shrinks, so part of what I measured as seed noise yesterday was really checkpoint-picking noise. The logs and the full `seeds.py --pick both` output are in `results/pick_final.txt`.

**Tests.** In `tests/test_train.py`, a tiny run at two learning rates checks that `ckpt.pt` holds the step with the lowest estimate and `final.pt` the last step with its own estimate, that they are the same weights exactly when the last step was the best and different otherwise, and that both have the config, tokenizer and seed. In `tests/test_seeds.py`: `compare` pairs each run with its last step and gets the differences, spread and win counts right, and refuses lists of different lengths; `final_path` errors for checkpoints without a `final.pt`, including the committed one; and `seeds.py --pick both` trains two seeds and prints all three tables, and `--pick final` scores only `final.pt`. 121 tests pass.

**Next.** Make `--pick final` the default for `seeds.py`, or drop the 50-batch pick altogether for schedules that decay, and rerun the earlier comparisons on the last step. The committed `checkpoints/ckpt.pt` and the demo stay as they are for now.

## 2026-11-23: How much does the seed move the result?

**Why.** The last three entries compared single runs: warmup + cosine decay against a constant rate, tied against separate weights, BPE against characters. Each run is one draw. The seed sets the starting weights, the order of the random batches and the dropout masks, so I had no idea whether a 0.010 bits-per-character gap was a real effect or just a different draw. Yesterday's entry ended on exactly that question.

**What changed.** New `seeds.py` runs `train.py` once per seed (everything after `--` is passed through), saves each run into `<out>/seed<N>/` with its log, scores every checkpoint on the full validation split with `evaluate.py`, and prints each run plus the mean, sample standard deviation and range. A seed that already has a checkpoint is skipped, so a sweep that gets interrupted can be restarted with the same command, and `--ckpts` folds in runs trained earlier. It refuses `--seed`, `--out` and `--data` after the `--`, because it sets those itself. `train.py` now saves `seed` in the checkpoint so runs can be told apart; it doesn't change anything else, and `checkpoints/ckpt.pt` loads as before (its seed shows as `?`).

**The runs.** The character model, `small` config, 5,000 steps, seeds 1, 2 and 3, once with the constant rate and once with warmup + cosine decay. Six runs on the 2-core CPU, two at a time with one thread each, about 18.5 minutes per run. Bits per character on the full validation split:

| | Seed 1337 (earlier) | Seed 1 | Seed 2 | Seed 3 | Mean of 4 | Std of 4 | Std of the 3 new |
|---|---|---|---|---|---|---|---|
| Constant | 2.341 | 2.318 | 2.300 | 2.315 | 2.319 | 0.017 | 0.010 |
| Warmup + cosine | 2.273 | 2.283 | 2.264 | 2.270 | 2.273 | 0.008 | 0.010 |

**What it says.** One standard deviation is about 0.01 bits per character and the range over three seeds is 0.018, for both schedules. So:

- The schedule is a real gain, just smaller than I reported. The cosine run beats the constant one for every seed, and the means differ by 0.046 bits (2.0%), about five standard errors. The single-run comparison said 0.068 (2.9%), because seed 1337 turned out to be the worst constant run of the four.
- Weight tying (2.283 against 2.273) is inside the noise. Seed 1 without tying scored 2.283 too. So "tying doesn't help at this size" stands, but "it's slightly worse" isn't supported.
- The BPE gain over characters (2.177 against 2.327, single runs) is about 15 standard deviations, so it is safe even before running BPE seeds.

The seed-1337 runs came from earlier sessions, possibly on another PyTorch version, so I keep the three-seed numbers separate in the table.

**Something I noticed.** Seed 1 picked its checkpoint at step 4,500 in both schedules, because `train.py` keeps the step with the best *estimated* validation loss, and that estimate is only 50 random batches. With the cosine schedule the learning rate is at its lowest at the end, so step 5,000 should be the better model, and the 50-batch estimate can't tell 0.005 apart reliably. That adds a little noise to every result in the README.

**Tests.** New `tests/test_seeds.py`: `spread` matches Python's `statistics` module, with no standard deviation for one run; the report lists every run (old checkpoints with `?` for the seed); two `train.py` runs with the same seed give exactly the same weights and a different seed doesn't, which is what makes seed comparisons mean anything; `seeds.py` trains two seeds, writes their logs, prints the summary and skips them on a second run; it refuses the arguments it sets itself; and `--ckpts` on the committed checkpoint reproduces the README's 2.327. 116 tests pass.

**Next.** Score the final checkpoint as well as the best-estimated one (or estimate the validation loss with more batches), so checkpoint picking stops adding noise to the comparisons.

## 2026-11-22: Weight tying between the embedding and the output layer

**What changed.** The token embedding (`token_embedding_table.weight`) and the output layer (`lm_head.weight`) are both `(vocab_size, C)` matrices: one turns token i into a vector, the other scores a vector against row i to predict token i. New `tie_weights` in `GPTConfig`, set with `train.py --tie-weights`, makes them the same tensor, the way GPT-2 does it. The output bias stays separate. Because the two uses share one tensor, backprop adds both gradients into it, so every row gets updated on every step through the output layer, not only the rows of tokens that appeared in the input. It saves `vocab_size * C` parameters: 8,320 for the character model (816,705 -> 808,385) and 65,536 for the 512-token BPE model (931,584 -> 866,048). `num_params()` already counted a shared tensor once, and AdamW gets it once too.

**Everything else that loads a model.** The flag lives in the model config, so `sample.py`, `evaluate.py`, `app.py` and `export_web.py` rebuild a tied model from its checkpoint with no extra code, and `torch.save` stores the shared matrix once. The exporter leaves `lm_head.weight` out of the page for a tied model, and `web/engine.js` uses the embedding matrix when the output weight is missing, which made the page for the tied character model about 22 KB smaller (2.20 MB). It is off by default and old checkpoints have no `tie_weights` key, so `checkpoints/ckpt.pt` loads untied as before. The rebuilt `docs/index.html` changes in two places only, the new engine lines and `"tie_weights": false` in its config; the weights and the sample shown on load are identical.

**Does it help? Not here.** I trained both tokenizers with tying and compared them against the untied runs already in the README, with the same seed and settings (characters with warmup + cosine decay, BPE with a constant rate), 5,000 steps on the 2-core CPU, then ran `evaluate.py` over the full validation split:

| | Characters, separate | Characters, tied | BPE, separate | BPE, tied |
|---|---|---|---|---|
| Parameters | 816,705 | 808,385 | 931,584 | 866,048 |
| Best estimated val loss | 1.5650 | 1.5711 | 2.8381 | 2.8476 |
| Train loss at step 5000 | 1.3630 | 1.3705 | 2.3695 | 2.3886 |
| Bits per character, full split | 2.273 | 2.283 | 2.177 | 2.184 |

Both tied models are slightly worse, by 0.010 and 0.007 bits per character. The train loss is higher as well, so the tie isn't acting as a regulariser that closes the train/val gap; it just takes away a little capacity. That fits the usual explanation for why tying helps big models: with a 50k vocabulary the embedding is a large share of the parameters and most rows are rarely seen as inputs, so the extra gradient through the output layer matters. With 65 characters, every row is seen constantly and the matrix is 1% of the model. Even at 512 BPE tokens (7% of the model) it didn't pay off at 5,000 steps. Early in training the tied runs were clearly behind (val 2.052 against 1.985 at step 500 for characters) and then mostly caught up. These are single runs and I don't know the seed-to-seed spread yet, so the gaps could be noise, but there is no sign of a gain. The defaults stay untied. The logs are in `results/train_log_tied.txt` and `results/train_log_bpe_tied.txt`.

**Tests.** New `tests/test_weight_tying.py`: a tied model holds one tensor for both layers (with its own bias); parameter counts drop by exactly `vocab_size * C` for both vocabularies; a tied model gives exactly the same logits as an untied one with a copied matrix; the shared gradient is the sum of the embedding and output-layer gradients, and tokens missing from the input still get one; an AdamW step keeps the tie; a saved tied model loads back tied with identical outputs; the committed checkpoint stays untied; the export ships the matrix once; the JavaScript engine matches PyTorch on tied character and BPE models without `lm_head.weight`; and `train.py --tie-weights` runs end to end and saves a tied checkpoint. 109 tests pass.

**Next.** Measure the seed-to-seed spread of the `small` config (a few runs with different `--seed`), so differences like these, and the earlier schedule and BPE results, can be judged against it.

## 2026-11-21: Learning-rate warmup, cosine decay and gradient clipping

**What changed.** `train.py` has always used the same learning rate for every step. New `get_lr()` gives the rate for each step, and `--lr-schedule cosine` turns on a linear warmup over `--warmup-iters` steps (100 by default, starting at 1/100 of the peak rather than 0) followed by half a cosine down to `--min-lr` (a tenth of the peak by default) at the last step. `--grad-clip N` calls `clip_grad_norm_` before each AdamW step, which scales every gradient by the same factor so their combined norm is at most N. Both are off by default, and I checked the default path is unchanged: 500 steps of the `small` config give exactly the same losses before and after the change (val 1.9976 at step 500). The progress lines now also show the average gradient norm since the last eval, plus the learning rate when a schedule is on, and the checkpoint records the settings it was trained with.

**Measuring it.** The committed checkpoint isn't a fair baseline any more: batching the attention heads changed the random starting weights, so I retrained with a constant rate first. Three runs of the `small` config, 5,000 steps, eval every 500, same seed, one after another on the 2-core CPU, then `evaluate.py` over the full validation split:

| | Constant 1e-3 | Warmup + cosine | Warmup + cosine, clip 1.0 |
|---|---|---|---|
| Best estimated val loss (step 4500) | 1.6101 | 1.5650 | 1.5635 |
| Loss per character, full split | 1.6225 | 1.5756 | 1.5755 |
| Bits per character | 2.341 | 2.273 | 2.273 |
| Time | 12.0 min | 11.7 min | 11.7 min |

So the schedule is worth 2.9% fewer bits per character at no extra cost. For comparison, that is a bit under half of what switching to BPE gained (6.4%). The cosine run is already a little ahead at step 500 (estimated val loss 1.985 against 1.998), and the gap widens as the rate falls, to 0.045 nats at step 4500. The train loss drops faster too, so the train/val gap at step 5000 is 0.22 nats against 0.21, about the same.

Clipping did nothing measurable. The average gradient norm per 500 steps stayed between 0.96 and 1.14 in that run, so a limit of 1.0 only trims ordinary steps a little, and the estimated val losses stay within 0.013 of the unclipped run at every eval and end up the same on the full split. It is still there as an option for bigger configs or higher learning rates, where it is more likely to matter, but I have no measurement of that yet.

**Tests.** New `tests/test_train.py`: the constant schedule returns the same rate for all 5,000 steps; warmup starts at `lr / warmup_iters` and climbs in equal steps to the peak; the decay has no jump where warmup ends, never goes up, is halfway between peak and floor at its midpoint, matches the formula at an arbitrary step and ends exactly at `min_lr`; the default floor is a tenth of the peak, steps past the end stay at the floor, and an unknown schedule is an error; and a tiny `train.py` run with `--lr-schedule cosine --warmup-iters 2 --grad-clip 0.5` finishes, prints the rate and gradient norm, and saves the settings in its checkpoint. 98 tests pass.

**Next.** Retrain the committed small model with the cosine schedule and rebuild `docs/index.html` from it. That will change the demo's sample and spec table, so it is its own change.

## 2026-11-20: A key/value cache for generate()

**What changed.** `generate()` used to run the whole context window through the model for every new token, even though only the last token is new. A token's key and value never change once computed, because it only attends to itself and the tokens before it, so they can be kept. New `KVCache` in `model.py` stores every layer's keys and values, shape `(B, nh, t, H)`, and `forward()` takes an optional `cache`: the new tokens get positions after the cached ones, their keys and values are appended, and the causal mask uses rows `P..P+T-1` so they can see the cached positions. `generate(use_cache=True)` then feeds one token per step through `next_logits()`. `sample.py --kv-cache` turns it on. It is off by default, so `train.py`, `export_web.py` and `app.py` sample exactly as before, and the rebuilt `docs/index.html` is byte for byte the same.

**When the window fills.** The position embeddings are absolute, so the cache can't slide: dropping the oldest token would move every other token to a new position and change all their keys. Rerunning the full window each step would bring back the cost the cache is meant to remove. So when the cache is full, `next_logits()` rebuilds it from the newest half of the window, the same rule `web/engine.js` already used. The model then sees between 32 and 64 tokens instead of always 64, and samples past the first window are different from the uncached ones. I checked what that costs with `evaluate.py`: the half-window rule is the same as windows that start every 33 tokens, and `--stride 33` gives 2.272 bits per character on the validation split, against 2.281 for `--stride 1` (always a full 64-token window). So the shorter context doesn't hurt this model at all, and is even slightly better.

**Speed (2-core CPU, 1,000 tokens from a newline, best of 3).**

| | No cache | Cache | Speedup |
|---|---|---|---|
| `small` (committed checkpoint, 64-token window) | 1.70 ms/token | 0.78 ms/token | 2.2x |
| `full` architecture (random weights, 256-token window) | 28.9 ms/token | 3.2 ms/token | 9.2x |

The `full` numbers use random weights, since there is no trained full checkpoint yet, but the speed doesn't depend on the weights. The longer window is why the gain is so much bigger there.

**Tests.** New `tests/test_kv_cache.py`: feeding a window in pieces through the cache gives the same logits as one forward pass; the cache refuses to grow past `block_size`; `generate(use_cache=True)` gives exactly the same tokens as before while the window has room, with and without top-k; past the window, every step's logits equal a plain forward pass over the context the half-window rule picks, including a prompt longer than the window; long cached samples equal sampling from that rule by hand; the trained checkpoint writes the same first window of text with and without the cache; and `sample.py --kv-cache` runs. One test streams 30 tokens through both the PyTorch cache and a browser engine session and checks the probabilities match after every token, through two rebuilds of the window, so the two implementations can't drift apart. Getting the mask offset wrong, or keeping one token more than half the window, makes several of them fail, which I checked by making each change. 93 tests pass.

**Next.** `app.py` still calls `generate(idx, 1)` once per token, which reruns the whole window every time. Moving it onto `next_logits()` would make the Gradio app faster too, though its samples past the first window would change.

## 2026-11-19: BPE checkpoints in the Gradio app

**What changed.** `app.py` was the last part of the project that only worked with the character model: it built a `CharTokenizer` from the checkpoint by hand and looked up `tok.itos` for each new token. Now it takes `--ckpt` (or the `GPT_CKPT` environment variable, which is easier to set on a Hugging Face Space), rebuilds the tokenizer with `load_tokenizer`, and works the same with either kind of checkpoint. The page's blurb is worked out from the loaded model, so a BPE model says "one token at a time" and gives its own parameter count.

**Streaming text a token at a time.** A byte-level BPE token can be half of a character. An emoji is 4 bytes, and with only 512 tokens its bytes usually land in separate tokens, so decoding each token on its own would show U+FFFD for a moment. Both tokenizers now have `decoder()`, the Python version of the streaming decoder `web/tokenizer.js` already had. `push(id)` returns only the characters that are complete, holding back any unfinished bytes (it uses Python's incremental UTF-8 decoder), and `flush()` returns what is left at the end. Joined together, the output is exactly `decode(ids)`, even for broken byte sequences.

**Length in characters.** The slider still says "Length (characters)", and it now means that for both models: the loop keeps sampling until enough characters have been written, rather than a fixed number of tokens. The page gets an update every 8 new characters, as before. Prompt cleaning depends on the tokenizer: the character model drops characters it doesn't know, the BPE model reads bytes and keeps the prompt as typed.

**Checking it.** The BPE checkpoint behind `docs/bpe.html` isn't committed, so I trained a short one (`--tokenizer bpe --max-iters 600`, 1.4 minutes, validation loss 3.28 per token) just to run the app. Served through Gradio and called with `gradio_client`, a 300-character request with a rose emoji in the prompt came back in 34 updates, with exactly 300 new characters and no U+FFFD. For 600 characters from `ROMEO:\n`, that BPE model took 316 sampling steps against 600 for the character model (0.6 s against 1.1 s on the 2-core CPU, though the two models aren't the same size). With the default checkpoint the app still writes exactly what the old loop wrote for the same seed.

**Tests.** New `tests/test_app.py`: the app module imports without Gradio, since Gradio is only imported when the page is built; prompt cleaning for both tokenizers; the streaming decoder gives exactly `decode(ids)` for 300 random id sequences per tokenizer; an emoji pushed as four separate byte tokens comes out as three empty strings and then the whole emoji; the character app gives the same text as the old loop for a fixed seed, with an update every 8 characters; a BPE model's output is the prompt plus `decode()` of the sampled tokens; and the blurb matches the model. 84 tests pass.

**Next.** Every BPE sub-step on the roadmap is now done. The next roadmap item that fits a CPU is training on a different dataset.

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
