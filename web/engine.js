// Browser inference engine for the GPT checkpoint. Plain JS, no dependencies.
// Mirrors model.py: token + position embeddings, n_layer pre-LN blocks of causal
// multi-head attention and a ReLU MLP, final LayerNorm and linear head.
// Uses a key/value cache so each new character costs one position of compute.

function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

function decodeTensor(t) {
  const bin = typeof atob === 'function' ? atob(t.b64) : Buffer.from(t.b64, 'base64').toString('binary');
  const n = bin.length / 2, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = halfToFloat(bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8));
  return out;
}

function createGPT(blob) {
  const cfg = blob.config, C = cfg.n_embd, NH = cfg.n_head, HS = C / NH, L = cfg.n_layer, V = cfg.vocab_size;
  const W = {};
  for (const k in blob.tensors) W[k] = decodeTensor(blob.tensors[k]);
  const chars = blob.chars, stoi = new Map(chars.map((c, i) => [c, i]));

  // y = W x + b, W stored row-major (out, in) like nn.Linear
  function linear(x, w, b, nOut, nIn, y) {
    for (let o = 0; o < nOut; o++) {
      let s = b ? b[o] : 0; const row = o * nIn;
      for (let i = 0; i < nIn; i++) s += w[row + i] * x[i];
      y[o] = s;
    }
    return y;
  }
  function layerNorm(x, g, b, y) {
    let mean = 0; for (let i = 0; i < C; i++) mean += x[i]; mean /= C;
    let v = 0; for (let i = 0; i < C; i++) { const d = x[i] - mean; v += d * d; } v /= C;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < C; i++) y[i] = (x[i] - mean) * inv * g[i] + b[i];
    return y;
  }

  // pre-pack per-layer weights
  const layers = [];
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`;
    const heads = [];
    for (let h = 0; h < NH; h++) {
      const hp = `${p}sa.heads.${h}.`;
      heads.push({ q: W[hp + 'query.weight'], k: W[hp + 'key.weight'], v: W[hp + 'value.weight'] });
    }
    layers.push({
      ln1g: W[p + 'ln1.weight'], ln1b: W[p + 'ln1.bias'], heads,
      projW: W[p + 'sa.proj.weight'], projB: W[p + 'sa.proj.bias'],
      ln2g: W[p + 'ln2.weight'], ln2b: W[p + 'ln2.bias'],
      f1W: W[p + 'ffwd.net.0.weight'], f1B: W[p + 'ffwd.net.0.bias'],
      f2W: W[p + 'ffwd.net.2.weight'], f2B: W[p + 'ffwd.net.2.bias'],
    });
  }
  const tokEmb = W['token_embedding_table.weight'], posEmb = W['position_embedding_table.weight'];
  const lnfG = W['ln_f.weight'], lnfB = W['ln_f.bias'], headW = W['lm_head.weight'], headB = W['lm_head.bias'];

  // scratch buffers
  const x = new Float32Array(C), h = new Float32Array(C), cat = new Float32Array(C), tmp = new Float32Array(C);
  const q = new Float32Array(HS), ff = new Float32Array(4 * C), logits = new Float32Array(V);
  const scale = 1 / Math.sqrt(HS);

  let cacheK, cacheV, pos, lastAtt = new Float32Array(0);
  function reset() {
    cacheK = layers.map(() => Array.from({ length: NH }, () => []));
    cacheV = layers.map(() => Array.from({ length: NH }, () => []));
    pos = 0;
  }
  reset();

  // feed one token at the next position, return logits for what follows it
  function step(token) {
    for (let i = 0; i < C; i++) x[i] = tokEmb[token * C + i] + posEmb[pos * C + i];
    const attAcc = new Float32Array(pos + 1);
    for (let l = 0; l < L; l++) {
      const ly = layers[l];
      layerNorm(x, ly.ln1g, ly.ln1b, h);
      for (let hd = 0; hd < NH; hd++) {
        const w = ly.heads[hd];
        linear(h, w.q, null, HS, C, q);
        const k = linear(h, w.k, null, HS, C, new Float32Array(HS));
        const v = linear(h, w.v, null, HS, C, new Float32Array(HS));
        const Ks = cacheK[l][hd], Vs = cacheV[l][hd];
        Ks.push(k); Vs.push(v);
        const T = Ks.length, att = new Float32Array(T);
        let mx = -Infinity;
        for (let t = 0; t < T; t++) {
          let s = 0; const kt = Ks[t];
          for (let i = 0; i < HS; i++) s += q[i] * kt[i];
          att[t] = s * scale; if (att[t] > mx) mx = att[t];
        }
        let sum = 0; for (let t = 0; t < T; t++) { att[t] = Math.exp(att[t] - mx); sum += att[t]; }
        const off = hd * HS;
        for (let i = 0; i < HS; i++) cat[off + i] = 0;
        for (let t = 0; t < T; t++) {
          const a = att[t] / sum, vt = Vs[t];
          attAcc[t] += a / (L * NH);
          for (let i = 0; i < HS; i++) cat[off + i] += a * vt[i];
        }
      }
      linear(cat, ly.projW, ly.projB, C, C, tmp);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
      layerNorm(x, ly.ln2g, ly.ln2b, h);
      linear(h, ly.f1W, ly.f1B, 4 * C, C, ff);
      for (let i = 0; i < 4 * C; i++) if (ff[i] < 0) ff[i] = 0;
      linear(ff, ly.f2W, ly.f2B, C, 4 * C, tmp);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
    }
    pos++;
    lastAtt = attAcc;
    layerNorm(x, lnfG, lnfB, h);
    return linear(h, headW, headB, V, C, logits);
  }

  // run a whole context from position 0, return logits after its last token
  function prime(ids) {
    reset();
    let out = null;
    for (const t of ids) out = step(t);
    return out;
  }

  function sample(lg, temperature, topK, rand) {
    const p = new Float32Array(V);
    let mx = -Infinity;
    for (let i = 0; i < V; i++) { p[i] = lg[i] / temperature; if (p[i] > mx) mx = p[i]; }
    if (topK && topK < V) {
      const cut = Array.from(p).sort((a, b) => b - a)[topK - 1];
      for (let i = 0; i < V; i++) if (p[i] < cut) p[i] = -Infinity;
    }
    let sum = 0; for (let i = 0; i < V; i++) { p[i] = Math.exp(p[i] - mx); sum += p[i]; }
    let r = rand() * sum;
    for (let i = 0; i < V; i++) { r -= p[i]; if (r <= 0) return i; }
    return V - 1;
  }

  function softmax(lg, temperature) {
    const p = new Float32Array(V); let mx = -Infinity;
    for (let i = 0; i < V; i++) { p[i] = lg[i] / temperature; if (p[i] > mx) mx = p[i]; }
    let sum = 0; for (let i = 0; i < V; i++) { p[i] = Math.exp(p[i] - mx); sum += p[i]; }
    for (let i = 0; i < V; i++) p[i] /= sum;
    return p;
  }

  // A writing session. Only one session is live at a time (they share the key/value cache).
  // When the context window fills, it re-primes on the most recent half of the window,
  // so the model always sees between block/2 and block characters.
  function session(promptIds) {
    const block = cfg.block_size, keep = Math.floor(block / 2);
    let ctx = promptIds.length ? promptIds.slice(-block) : [0];
    let lg = Float32Array.from(prime(ctx)), att = lastAtt;
    return {
      probs: (temperature = 1.0) => softmax(lg, temperature),
      attention: () => ({ ids: ctx.slice(), weights: att }),
      pick: (temperature = 1.0, topK = 0, rand = Math.random) => sample(lg, temperature, topK, rand),
      push(t) {
        ctx.push(t);
        if (pos >= block) { ctx = ctx.slice(-keep); lg = Float32Array.from(prime(ctx)); }
        else lg = Float32Array.from(step(t));
        att = lastAtt;
      },
    };
  }

  // Streaming generator. When the context window fills, it re-primes on the most
  // recent half of the window, so the model always sees between block/2 and block chars.
  function generator(promptIds, { temperature = 1.0, topK = 0, rand = Math.random } = {}) {
    const s = session(promptIds);
    return { next() { const t = s.pick(temperature, topK, rand); s.push(t); return t; } };
  }

  return {
    config: cfg, chars, step: blob.step, valLoss: blob.val_loss,
    encode: s => Array.from(s).filter(c => stoi.has(c)).map(c => stoi.get(c)),
    unknown: s => [...new Set(Array.from(s).filter(c => !stoi.has(c)))],
    decode: ids => ids.map(i => chars[i]).join(''),
    prime: ids => Float32Array.from(prime(ids)),
    generator, session,
  };
}

if (typeof module !== 'undefined') module.exports = { createGPT };
