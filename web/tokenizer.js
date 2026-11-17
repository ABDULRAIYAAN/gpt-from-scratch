// Tokenizers for the browser, ported from tokenizer.py. Plain JS, no dependencies.
// Both give exactly the same ids as the Python versions, so a page can encode a
// prompt for any checkpoint and decode what the model writes back into text.
//
//   createTokenizer({ type: 'char', chars: [...] })   one id per character
//   createTokenizer({ type: 'bpe', merges: [[a, b], ...] })
//                                                      byte-level BPE: ids 0-255 are bytes,
//                                                      id 256 + i is the i-th learned merge
//
// The state objects are the ones tokenizer.py saves with .state(), so a checkpoint's
// "tokenizer" entry can be passed in as it is.

// Python's \s for str patterns matches every character for which str.isspace() is true.
// JavaScript's \s is a slightly different set (it has U+FEFF but not U+001C-U+001F or
// U+0085), so the whitespace class is spelled out to make the chunks match exactly.
const WS = '\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';

// Same alternatives, in the same order, as CHUNK_PATTERN in tokenizer.py: contractions,
// a word with its leading space, a run of digits, a run of punctuation, then whitespace
// (a run that leaves its last space for the next word, or a run at the end of the text).
// The 'u' flag makes [^...] step over whole code points, so an emoji is one character.
const CHUNK_PATTERN = new RegExp(
  `'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^${WS}A-Za-z0-9]+|[${WS}]+(?![^${WS}])|[${WS}]+`,
  'gu',
);

function splitChunks(s) {
  return Array.from(s.matchAll(CHUNK_PATTERN), m => m[0]);
}

// Replace every non-overlapping occurrence of (a, b) in ids, left to right, with newId.
function mergePair(ids, a, b, newId) {
  const out = [];
  for (let i = 0; i < ids.length; i++) {
    if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) { out.push(newId); i++; }
    else out.push(ids[i]);
  }
  return out;
}

function createCharTokenizer(chars) {
  const stoi = new Map(chars.map((c, i) => [c, i]));
  // Array.from splits by code point, like iterating over a Python str.
  const unknown = s => [...new Set(Array.from(s).filter(c => !stoi.has(c)))]
    .sort((x, y) => x.codePointAt(0) - y.codePointAt(0));  // code point order, like sorted()
  return {
    type: 'char',
    vocabSize: chars.length,
    unknown,
    encode(s) {
      const bad = unknown(s);
      if (bad.length) throw new Error('Characters not in vocabulary: ' + JSON.stringify(bad));
      return Array.from(s, c => stoi.get(c));
    },
    decode: ids => ids.map(i => chars[i]).join(''),
    piece: id => chars[id],
    // Every id is a whole character, so streaming text needs no buffering.
    decoder: () => ({ push: id => chars[id] }),
  };
}

function createBPETokenizer(merges) {
  // A pair (a, b) is stored as one number a * 2^26 + b, exact for vocabularies up to 2^26.
  const SHIFT = 2 ** 26, key = (a, b) => a * SHIFT + b;
  const ranks = new Map(merges.map(([a, b], i) => [key(a, b), i]));  // earlier merge = applied first

  // vocab[id] is the bytes that token stands for: a single byte, or two earlier tokens joined.
  const vocab = Array.from({ length: 256 }, (_, i) => Uint8Array.of(i));
  for (const [a, b] of merges) {
    const t = new Uint8Array(vocab[a].length + vocab[b].length);
    t.set(vocab[a]); t.set(vocab[b], vocab[a].length);
    vocab.push(t);
  }

  const utf8 = new TextEncoder();
  // ignoreBOM keeps a leading U+FEFF in the text, as Python's decode does. Without
  // fatal, a slice that ends partway through a character shows U+FFFD, like errors="replace".
  const fromUtf8 = new TextDecoder('utf-8', { ignoreBOM: true });
  const cache = new Map();

  function encodeChunk(chunk) {
    const cached = cache.get(chunk);
    if (cached) return cached;
    let ids = Array.from(utf8.encode(chunk));
    while (ids.length > 1) {
      // Find the earliest-learned merge among the adjacent pairs, the same order as training.
      let best = Infinity, a = 0, b = 0;
      for (let i = 0; i + 1 < ids.length; i++) {
        const r = ranks.get(key(ids[i], ids[i + 1]));
        if (r !== undefined && r < best) { best = r; a = ids[i]; b = ids[i + 1]; }
      }
      if (best === Infinity) break;  // nothing left that training ever merged
      ids = mergePair(ids, a, b, 256 + best);
    }
    cache.set(chunk, ids);
    return ids;
  }

  function bytesOf(ids) {
    let n = 0;
    for (const i of ids) n += vocab[i].length;
    const out = new Uint8Array(n);
    let o = 0;
    for (const i of ids) { out.set(vocab[i], o); o += vocab[i].length; }
    return out;
  }

  return {
    type: 'bpe',
    vocabSize: vocab.length,
    unknown: () => [],  // every string is bytes, so nothing is ever out of vocabulary
    encode(s) {
      const out = [];
      for (const chunk of splitChunks(s)) for (const id of encodeChunk(chunk)) out.push(id);
      return out;
    },
    decode: ids => fromUtf8.decode(bytesOf(ids)),
    piece: id => fromUtf8.decode(vocab[id]),  // text of one token, for labels in the page
    // Decode a stream of ids one at a time, for text that is written token by token.
    // A byte-level token can be part of a multi-byte character (an emoji is 4 bytes, and
    // its bytes may land in different tokens), so push() holds back unfinished bytes and
    // returns only the characters that are complete. The text it returns, joined
    // together, is decode(ids) for any ids that form valid UTF-8.
    decoder() {
      const d = new TextDecoder('utf-8', { ignoreBOM: true });
      return { push: id => d.decode(vocab[id], { stream: true }) };
    },
  };
}

function createTokenizer(state) {
  if (state.type === 'char') return createCharTokenizer(state.chars);
  if (state.type === 'bpe') return createBPETokenizer(state.merges);
  throw new Error('unknown tokenizer type ' + JSON.stringify(state.type));
}

if (typeof module !== 'undefined') module.exports = { createTokenizer, splitChunks, mergePair };
