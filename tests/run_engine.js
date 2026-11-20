// Test helper: node run_engine.js weights.json '[[ids...], ...]' ['["text", ...]'] ['[[ids...], ...]']
// Prints JSON: the engine's logits after the last token of each id sequence, the ids
// the engine's encode() gives for each text, the ids a session starts from when the
// prompt is empty, and the text each id sequence decodes to through the stream decoder.
// For each sequence in the optional fourth argument, a session starts on its first id
// and is pushed the rest one at a time; sessionProbs holds its next-token
// probabilities after every token, through any re-priming when the window fills.
const fs = require('fs');
const path = require('path');
const { createGPT } = require(path.join(__dirname, '..', 'web', 'engine.js'));

const blob = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const seqs = JSON.parse(process.argv[3]);
const texts = JSON.parse(process.argv[4] || '[]');
const streams = JSON.parse(process.argv[5] || '[]');
const gpt = createGPT(blob);
const streamed = ids => { const d = gpt.decoder(); return ids.map(i => d.push(i)).join(''); };
const sessionProbs = ids => {
  const s = gpt.session(ids.slice(0, 1)), out = [Array.from(s.probs())];
  for (const t of ids.slice(1)) { s.push(t); out.push(Array.from(s.probs())); }
  return out;
};
console.log(JSON.stringify({
  logits: seqs.map(ids => Array.from(gpt.prime(ids))),
  encoded: texts.map(s => gpt.encode(s)),
  start: gpt.session([]).attention().ids,
  streamed: seqs.map(streamed),
  tokenizer: gpt.tokenizer,
  sessionProbs: streams.map(sessionProbs),
}));
