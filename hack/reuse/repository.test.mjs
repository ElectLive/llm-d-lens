import assert from 'node:assert/strict';
import { test } from 'node:test';
import { registrationCandidates } from './registration.mjs';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { discover, chunks, validateCatalog, renderMap, duplicates, sources, revisionFiles } from './repository.mjs';

export function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'reuse-test-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  execFileSync('git', ['init', '-q', root]);
  const put = (path, text) => {
    mkdirSync(join(root, path, '..'), { recursive: true });
    writeFileSync(join(root, path), text);
  };
  return { root, put };
}

test('discovery includes new source but excludes ignored files, tests and generated output', t => {
  const { root, put } = fixture(t);
  put('.gitignore', 'src/private.js\n');
  for (const name of ['src/live.js', 'server/new.ts', 'llm_d_lens/store.py', 'src/private.js', 'src/a.test.js', 'hack/reuse/semantic.smoke.mjs', 'llm_d_lens/test_a.py', 'dist/a.js', 'server/mcp/tools.ts']) put(name, '');
  assert.deepEqual(discover(root), ['llm_d_lens/store.py', 'server/new.ts', 'src/live.js']);
});

test('chunk labels identify the containing function rather than a local variable', () => {
  const text = `export function requestJson() {\n  const response = fetch(url);\n${'  handle(response);\n'.repeat(50)}  return response;\n}`;
  const entries = chunks('src/api.js', text);
  assert.ok(entries.every(row => row.symbol === 'requestJson'));
});

test('chunks preserve symbol locations and cover the end of large functions', () => {
  const text = `export function load() {\n${'  consume(value);\n'.repeat(90)}}\nexport const save = () => 1;`;
  const entries = chunks('src/api.js', text);
  assert.ok(entries.some(x => x.symbol === 'load' && x.line === 1));
  assert.ok(entries.some(x => x.text.includes('export const save')));
  assert.ok(entries.some(x => x.endLine >= 90));
  assert.ok(chunks('llm_d_lens/store.py', 'class Store:\n    def load(self):\n        return 1\n').some(x => x.symbol === 'load'));
});

test('catalog rejects missing declarations, missing references and paths outside source roots', t => {
  const { root, put } = fixture(t);
  put('src/api.js', '// missing is mentioned but not declared\nexport function load() {}');
  const entry = { id: 'json', path: 'src/api.js', symbols: ['load'], purpose: 'Read JSON', boundaries: 'No streaming', examples: ['src/api.js'], tests: [], testNote: 'No dedicated tests yet' };
  assert.deepEqual(validateCatalog(root, { version: 1, entries: [entry] }), []);
  assert.match(validateCatalog(root, { version: 1, entries: [{ ...entry, symbols: ['missing'] }] }).join(), /missing/);
  assert.match(validateCatalog(root, { version: 1, entries: [{ ...entry, examples: ['src/gone.js'] }] }).join(), /gone/);
  assert.ok(validateCatalog(root, { version: 1, entries: [{ ...entry, path: '../outside.js' }] }).length);
  assert.ok(validateCatalog(root, { version: 1, entries: [entry, entry] }).length);
});

test('Python catalog validation does not accept declarations inside docstrings', t => {
  const { root, put } = fixture(t);
  put('llm_d_lens/store.py', '"""Example:\ndef missing():\n    pass\n"""\ndef real():\n    return 1\n');
  const entry = { id: 'python', path: 'llm_d_lens/store.py', symbols: ['missing'], purpose: 'Load', boundaries: 'Local', examples: ['llm_d_lens/store.py'], tests: [], testNote: 'None' };
  assert.match(validateCatalog(root, { version: 1, entries: [entry] }).join(), /missing/);
});

test('duplicate evidence catches identifier renaming and reports source locations', () => {
  const a = 'export function load(input) { const result = []; for (const item of input) { if (item.active) { result.push(item.value); } } return result; }';
  const b = a.replaceAll('load', 'collect').replaceAll('input', 'items').replaceAll('result', 'output');
  const files = [{ path: 'src/a.js', text: a }, { path: 'src/b.js', text: b }, { path: 'src/c.js', text: 'export const answer = 42;' }];
  const hits = duplicates(files, new Set(['src/b.js']), { minTokens: 30 });
  assert.equal(hits.length, 1);
  assert.deepEqual(new Set([hits[0].left.path, hits[0].right.path]), new Set(['src/a.js', 'src/b.js']));
  assert.equal(hits[0].left.line, 1);
  assert.equal(duplicates(files, new Set(['src/c.js']), { minTokens: 30 }).length, 0);
});

test('map rendering is deterministic and tracks changes to capability boundaries', () => {
  const entry = { id: 'json', path: 'src/a.js', symbols: ['load'], purpose: 'JSON', boundaries: 'No streams', examples: ['src/b.js'], tests: [], testNote: 'None' };
  const first = renderMap({ version: 1, entries: [entry] });
  assert.ok(first.includes('[reuse development guide](../docs/design/reuse-workflow.md)'));
  assert.ok(first.includes('[src/a.js](../src/a.js)'));
  assert.ok(first.includes('[catalog.json](catalog.json)'));
  assert.ok(!first.includes('[src/b.js](../src/b.js)'));
  assert.equal(first, renderMap({ version: 1, entries: [entry] }));
  assert.equal(first, renderMap({ version: 1, entries: [{ ...entry, boundaries: 'Streams too' }] }));
});

test('batch history preserves binary data, odd filenames, and symlink modes', t => {
  const { root, put } = fixture(t);
  put('src/a\todd.js', 'export const a = 1;');
  put('binary.dat', Buffer.from([0, 255, 10, 128]));
  put('script.sh', '#!/bin/sh\n');
  symlinkSync('script.sh', join(root, 'link'));
  execFileSync('git', ['add', '.'], { cwd: root });
  execFileSync('git', ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'base'], { cwd: root });
  assert.deepEqual(revisionFiles(root, 'HEAD').find(row => row.path === 'binary.dat').contents, Buffer.from([0, 255, 10, 128]));
  assert.deepEqual(sources(root, 'HEAD').map(row => row.path), ['src/a\todd.js']);
  assert.deepEqual(sources(root, 'HEAD', new Set()), []);
  assert.equal(revisionFiles(root, 'HEAD').find(row => row.path === 'link').mode, '120000');
});

test('no changed sources bypasses tokenization entirely', () => {
  const file = { path: 'src/a.js', get text() { throw new Error('Should not read source'); } };
  assert.deepEqual(duplicates([file], new Set()), []);
});

test('duplicate cache preserves cold results and invalidates modified/deleted source', () => {
  const body = 'export function load(input) { const result = []; for (const item of input) { if (item.active) { result.push(item.value); } } return result; }';
  const files = [{ path: 'src/a.js', text: body }, { path: 'src/b.js', text: body.replace('load', 'collect') }];
  const changed = new Set(['src/b.js']);
  const cache = {};
  const first = duplicates(files, changed, { minTokens: 30, cache });
  assert.equal(first.length, 1);
  assert.deepEqual(duplicates(files, changed, { minTokens: 30, cache }), first);
  const modified = [files[0], { path: 'src/b.js', text: 'export const value = 1;' }];
  assert.deepEqual(duplicates(modified, changed, { minTokens: 30, cache }), []);
  duplicates([files[0]], new Set(['src/a.js']), { minTokens: 30, cache });
  assert.deepEqual(Object.keys(cache.files), ['src/a.js']);
  assert.deepEqual(duplicates(files, changed, { minTokens: 10, cache }), duplicates(files, changed, { minTokens: 10 }));
});

test('registration discovery covers aliases, defaults, reexports and public Python classes', () => {
  const files = [
    { path: 'src/a.ts', text: 'const impl = () => 1; export { impl as shared }; export default function() { return 2; } export { x } from "./b"; export * from "./c";' },
    { path: 'llm_d_lens/a.py', text: 'def _private(): pass\ndef public(): pass\nclass Shared: pass\n' },
  ];
  const found = registrationCandidates(files, [], { entries: [] });
  assert.deepEqual(found.map(x => x.symbol), ['shared', 'default', 'x', '*', 'public', 'Shared']);
  assert.equal(registrationCandidates(files, files, { entries: [] }).length, 0);

});

test('default identifier implementation and reexport module changes require review', () => {
  for (const [before, after] of [
    ['const impl = () => 1; export default impl;', 'const impl = () => 2; export default impl;'],
    ['const impl = () => 1; export { impl } from "./old.js";', 'const impl = () => 1; export { impl } from "./new.js";'],
  ]) {
    assert.equal(registrationCandidates([{ path: 'src/a.js', text: after }], [{ path: 'src/a.js', text: before }], { entries: [] }).length, 1);
  }
});
