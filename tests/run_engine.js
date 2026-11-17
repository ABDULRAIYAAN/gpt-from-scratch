// Test helper: node run_engine.js weights.json '[[ids...], ...]' ['["text", ...]']
// Prints JSON: the engine's logits after the last token of each id sequence, the ids
// the engine's encode() gives for each text, the ids a session starts from when the
// prompt is empty, and the text each id sequence decodes to through the stream decoder.
const fs = require('fs');
const path = require('path');
const { createGPT } = require(path.join(__dirname, '..', 'web', 'engine.js'));

const blob = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const seqs = JSON.parse(process.argv[3]);
const texts = JSON.parse(process.argv[4] || '[]');
const gpt = createGPT(blob);
const streamed = ids => { const d = gpt.decoder(); return ids.map(i => d.push(i)).join(''); };
console.log(JSON.stringify({
  logits: seqs.map(ids => Array.from(gpt.prime(ids))),
  encoded: texts.map(s => gpt.encode(s)),
  start: gpt.session([]).attention().ids,
  streamed: seqs.map(streamed),
  tokenizer: gpt.tokenizer,
}));
