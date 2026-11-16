// Test helper: node run_tokenizer.js job.json
// job.json holds {"state": tokenizer state, "texts": [...], "ids": [[...], ...]}.
// Prints as JSON: the chunks and ids for each text (ids is null and error holds the
// message when encoding throws), the text for each id list, and every token's text.
const fs = require('fs');
const path = require('path');
const { createTokenizer, splitChunks } = require(path.join(__dirname, '..', 'web', 'tokenizer.js'));

const job = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const tok = createTokenizer(job.state);
const tryEncode = s => { try { return [tok.encode(s), null]; } catch (e) { return [null, e.message]; } };
const results = job.texts.map(tryEncode);
console.log(JSON.stringify({
  vocab_size: tok.vocabSize,
  chunks: job.texts.map(splitChunks),
  encoded: results.map(r => r[0]),
  errors: results.map(r => r[1]),
  decoded: (job.ids || []).map(ids => tok.decode(ids)),
  pieces: Array.from({ length: tok.vocabSize }, (_, i) => tok.piece(i)),
}));
