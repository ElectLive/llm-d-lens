#!/usr/bin/env node
import { existsSync, readFileSync, mkdirSync, writeFileSync, renameSync } from 'node:fs';
import { registrationCandidates } from './registration.mjs';
import { resolve, dirname } from 'node:path';
import { parseArgs } from 'node:util';
import { git, sources, chunks, digest, validateCatalog, renderMap, duplicates } from './repository.mjs';
import { searchDocuments, updateVectors, rank, rankFiles, localEmbedder } from './search.mjs';
import { readDecisions, evidenceChanged } from './decisions.mjs';

function write(path, content) {
  mkdirSync(dirname(path), { recursive: true });
  const temp = `${path}.${process.pid}.tmp`;
  writeFileSync(temp, content);
  renameSync(temp, path);
}
const writeJson = (path, data) => write(path, JSON.stringify(data, null, 2) + '\n');
const readJson = path => JSON.parse(readFileSync(path, 'utf8'));

async function main() {
  const { values, positionals } = parseArgs({ allowPositionals: true, options: {
    base: { type: 'string' }, purpose: { type: 'string', default: 'audit' }, json: { type: 'boolean' },
    semantic: { type: 'boolean' }, offline: { type: 'boolean' }, lexical: { type: 'boolean' },
    limit: { type: 'string', default: '8' }, 'file-limit': { type: 'string', default: '6' }, help: { type: 'boolean' },
  } });
  const command = positionals[0];
  if (values.help || !command) {
    console.log(`Reuse tools:
  map                          Regenerate the capability map
  check [--base REF] [--purpose audit|startup] [--json]
                               Check catalog, map and existing pending decisions;
                               audit also finds duplicate/registration candidates
  search QUERY [--lexical|--semantic] [--offline] [--limit 8] [--file-limit 6]
  index [--offline]             Build the optional local semantic index
Exit codes: 0 checked, 1 invalid input/check failure, 2 existing pending decision.
Similarity/registration candidates are advisory. Run tests directly; no task ledger.`);
    return;
  }
  if (!['map', 'check', 'search', 'index'].includes(command)) throw new Error(`Unknown command: ${command}`);
  if (values.lexical && values.semantic) throw new Error('Choose --lexical or --semantic, not both');
  if (!['audit', 'startup'].includes(values.purpose)) throw new Error('Unknown check purpose');
  const root = git(process.cwd(), ['rev-parse', '--show-toplevel']).trim();
  const catalog = readJson(resolve(root, '.reuse/catalog.json'));
  const errors = validateCatalog(root, catalog);
  if (command === 'map') {
    if (errors.length) throw new Error(errors.join('\n'));
    write(resolve(root, '.reuse/map.md'), renderMap(catalog));
    console.log('Updated .reuse/map.md.');
    return;
  }
  if (command === 'check') {
    const mapPath = resolve(root, '.reuse/map.md');
    if (!errors.length && (!existsSync(mapPath) || readFileSync(mapPath, 'utf8') !== renderMap(catalog))) errors.push('Capability map is stale/missing: run npm run reuse:map');
    const ledger = readDecisions(root);
    const pending = ledger.requests.filter(row => row.status === 'pending');
    const staleDecisionIds = ledger.requests.filter(row => row.status === 'resolved' && evidenceChanged(root, row)).map(row => row.id);
    const base = values.base ? git(root, ['rev-parse', '--verify', `${values.base}^{commit}`]).trim() : null;
    const files = values.purpose === 'startup' ? [] : sources(root);
    const changed = new Set(base ? [
      ...git(root, ['diff', '--name-only', '-z', base, '--']).split('\0'),
      ...git(root, ['ls-files', '--others', '--exclude-standard', '-z']).split('\0'),
    ].filter(path => files.some(file => file.path === path)) : files.map(file => file.path));
    const registration = registrationCandidates(files.filter(file => changed.has(file.path)), base && changed.size ? sources(root, base, changed) : [], catalog);
    const cachePath = resolve(root, '.cache/reuse/duplicates.json');
    let cache = {};
    if (changed.size && existsSync(cachePath)) {
      try { cache = readJson(cachePath); } catch { /* Rebuild derived cache. */ }
    }
    const candidates = duplicates(files, changed, { cache });
    if (changed.size) writeJson(cachePath, cache);
    const status = errors.length ? 'INVALID' : pending.length ? 'WAITING_FOR_USER' : 'CHECKED';
    if (values.json) console.log(JSON.stringify({ status, purpose: values.purpose, base,
      filesScanned: files.length, changedFiles: changed.size, errors, pending, staleDecisionIds,
      registrationCandidates: registration, candidates }, null, 2));
    else {
      console.log(`${values.purpose} check: ${status}; ${files.length} files, ${changed.size} changed, ${candidates.length} duplicate candidates`);
      for (const error of errors) console.log(`ERROR: ${error}`);
      for (const row of pending) console.log(`WAITING_FOR_USER ${row.id}: ${row.summary}`);
      for (const row of candidates.slice(0, 30)) console.log(`CANDIDATE ${row.left.path}:${row.left.line} <-> ${row.right.path}:${row.right.line}`);
      console.log(`${registration.filter(row => row.disposition === 'unreviewed').length} registration candidates; use --json for details. Candidates are advisory.`);
      if (staleDecisionIds.length) console.log(`${staleDecisionIds.length} historical decisions have changed evidence; do not reuse them as approval.`);
    }
    process.exitCode = errors.length ? 1 : pending.length ? 2 : 0;
    return;
  }
  const files = sources(root);
  if (errors.length) throw new Error(errors.join('\n'));
  const query = positionals.slice(1).join(' ').trim();
  const limit = Number(values.limit);
  const fileLimit = Number(values['file-limit']);
  if (!Number.isInteger(fileLimit) || fileLimit < 1 || fileLimit > 100) throw new Error('--file-limit must be an integer from 1 to 100');
  const outputSearch = result => console.log(JSON.stringify(result, null, 2));
  if (!Number.isInteger(limit) || limit < 1 || limit > 50) throw new Error('--limit must be an integer from 1 to 50');
  if (command === 'search' && !query) throw new Error('search requires a query');
  if (values.lexical || (command === 'search' && !values.semantic && !values.offline)) {
    if (command !== 'search') throw new Error('--lexical is only valid for search');
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    const rows = files.flatMap(file => chunks(file.path, file.text)).map(row => ({ ...row,
      score: terms.filter(term => `${row.path}\n${row.text}`.toLowerCase().includes(term)).length,
    })).filter(row => row.score).sort((a, b) => b.score - a.score).slice(0, limit);
    outputSearch({ mode: 'lexical-not-semantic', fileLimit, scope: 'Whitespace substring matching across all source chunks; file-limit only affects semantic search.', results: rows });
    return;
  }
  console.error(values.offline ? 'Loading cached local embedding model (offline).' : 'Loading local embedding model; first run downloads pinned model assets. Source stays local.');
  const model = await localEmbedder(root, { offline: values.offline });
  try {
    const cachePath = resolve(root, '.cache/reuse/index.json');
    const cache = existsSync(cachePath) ? readJson(cachePath) : null;
    const index = await updateVectors(searchDocuments(files, catalog), cache, model.modelKey, model.embed,
      (done, total) => { if (done % 80 === 0 || done === total) console.error(`Indexed ${done}/${total} changed summaries`); });
    writeJson(cachePath, index);
    if (command === 'index') { console.log(`Indexed ${files.length} source files and ${catalog.entries.length} capability descriptions; ${model.modelKey}`); return; }
    const [vector] = await model.embed([query]);
    const fileResults = rankFiles(index.rows, vector, fileLimit);
    const selected = new Set(fileResults.map(row => row.path));
    const detailDocs = files.filter(file => selected.has(file.path)).flatMap(file => chunks(file.path, file.text))
      .map(row => ({ ...row, sourceHash: digest(files.find(file => file.path === row.path).text), id: `${row.path}:${row.line}`, text: `${row.path} ${row.symbol}\n${row.text.slice(0, 1600)}` }));
    const detailPath = resolve(root, '.cache/reuse/details.json');
    const details = await updateVectors(detailDocs, existsSync(detailPath) ? readJson(detailPath) : null, model.modelKey, model.embed, undefined, new Map(files.map(file => [file.path, digest(file.text)])));
    writeJson(detailPath, details);
    outputSearch({ mode: 'local-semantic', model: model.modelKey, fileLimit,
      scope: `All source summaries plus capability descriptions; chunks within the top ${fileLimit} distinct files. Scores are similarity, not proof of equivalence.`,
      capabilities: rank(index.rows.filter(row => row.id.startsWith('capability:')), vector, Math.min(limit, 3)),
      files: fileResults.map(({ text: _text, ...row }) => row), results: rank(details.rows.filter(row => selected.has(row.path)), vector, limit) });
  } finally { await model.dispose(); }
}

main().catch(error => { console.error(`reuse: ${error.message}`); process.exitCode = 1; });
