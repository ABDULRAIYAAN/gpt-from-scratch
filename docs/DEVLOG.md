# Devlog

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
