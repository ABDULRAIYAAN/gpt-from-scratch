// Test helper: node run_engine.js weights.json '[[ids...], ...]'
// Prints the engine's logits after the last token of each id sequence, as JSON.
const fs = require('fs');
const path = require('path');
const { createGPT } = require(path.join(__dirname, '..', 'web', 'engine.js'));

const blob = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const seqs = JSON.parse(process.argv[3]);
const gpt = createGPT(blob);
console.log(JSON.stringify(seqs.map(ids => Array.from(gpt.prime(ids)))));
