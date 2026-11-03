import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';

const cli = resolve('hack/reuse/cli.mjs');
function setup(t) {
  const root = mkdtempSync(join(tmpdir(), 'reuse-cli-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  execFileSync('git', ['init', '-q', root]);
  const put = (name, text) => { mkdirSync(resolve(root, name, '..'), { recursive: true }); writeFileSync(join(root, name), text); };
  put('src/a.js', 'export function load() { return 1; }');
  put('.reuse/catalog.json', JSON.stringify({ version: 1, entries: [{ id: 'load', path: 'src/a.js', symbols: ['load'], purpose: 'Load value', boundaries: 'No IO', examples: ['src/a.js'], tests: [], testNote: 'No dedicated test' }] }));
  const run = (...args) => spawnSync(process.execPath, [cli, ...args], { cwd: root, encoding: 'utf8' });
  return { root, put, run };
}

test('map stays under .reuse and check rejects missing or stale maps and invalid bases', t => {
  const { run, put, root } = setup(t);
  assert.equal(run('map').status, 0);
  assert.match(readFileSync(join(root, '.reuse/map.md'), 'utf8'), /\[src\/a.js\]\(\.\.\/src\/a.js\)/);
  assert.equal(existsSync(join(root, 'docs')), false);
  assert.equal(run('check').status, 0);
  rmSync(join(root, '.reuse/map.md'));
  const missing = run('check');
  assert.equal(missing.status, 1, missing.stderr);
  assert.match(missing.stdout, /stale\/missing/);
  assert.equal(run('map').status, 0);
  put('.reuse/map.md', 'old map');
  const stale = run('check');
  assert.equal(stale.status, 1, stale.stderr);
  assert.match(stale.stdout, /map/);
  assert.equal(run('check', '--base', 'does-not-exist').status, 1);
});

test('malformed approval ledger cannot silently bypass a pending review', t => {
  const { run, put } = setup(t);
  run('map');
  put('.reuse/decisions.json', JSON.stringify({ version: 1, requests: [{ status: 'resolved' }] }));
  assert.equal(run('check').status, 1);
});

test('base comparison includes untracked implementations without flooding unchanged files', t => {
  const { root, run, put } = setup(t);
  run('map');
  execFileSync('git', ['add', '.'], { cwd: root });
  execFileSync('git', ['-c', 'user.name=Reuse Test', '-c', 'user.email=reuse@example.invalid', 'commit', '-qm', 'baseline'], { cwd: root });
  const before = run('check', '--base', 'HEAD', '--json');
  assert.equal(before.status, 0, before.stderr);
  assert.equal(JSON.parse(before.stdout).changedFiles, 0);
  put('src/new.js', 'export const extra = () => 2;');
  const after = run('check', '--base', 'HEAD', '--json');
  assert.equal(after.status, 0, after.stderr);
  assert.equal(JSON.parse(after.stdout).changedFiles, 1);
});

test('default search remains lexical without downloading a model', t => {
  const { run } = setup(t);
  const result = run('search', 'load');
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).mode, 'lexical-not-semantic');
  assert.equal(run('search', 'load', '--semantic', '--lexical').status, 1);
});


test('historical pending decisions block checks without being changed or re-recorded', t => {
  const { root, run, put } = setup(t);
  run('map');
  const ledger = JSON.stringify({ version: 1, requests: [{ id: 'existing', status: 'pending', kind: 'index', summary: 'Unclear contract', options: ['Keep', 'Change'], evidence: [{ path: 'src/a.js', hash: '0'.repeat(64) }] }] });
  put('.reuse/decisions.json', ledger);
  for (const purpose of ['startup', 'audit']) {
    const result = run('check', '--purpose', purpose, '--json');
    assert.equal(result.status, 2, result.stderr);
    assert.equal(JSON.parse(result.stdout).pending[0].id, 'existing');
  }
  assert.equal(readFileSync(join(root, '.reuse/decisions.json'), 'utf8'), ledger);
  assert.equal(existsSync(join(root, '.cache/reuse/tasks')), false);
});

test('startup validates catalog without parsing unrelated source or producing reports', t => {
  const { root, run, put } = setup(t);
  run('map');
  put('llm_d_lens/unrelated.py', 'not valid python :::');
  const result = run('check', '--purpose', 'startup', '--json');
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).filesScanned, 0);
  assert.equal(existsSync(join(root, '.cache/reuse')), false);
  put('src/a.js', '// declaration removed');
  assert.equal(run('check', '--purpose', 'startup').status, 1);
});

test('removed bookkeeping commands and flags fail explicitly instead of claiming validation', t => {
  const { root, run } = setup(t);
  for (const command of ['begin', 'verify', 'verify-report', 'review', 'review-status', 'review-report', 'scope', 'cover', 'pause', 'resolve']) {
    assert.equal(run(command).status, 1, command);
  }
  for (const flag of ['--task', '--report', '--input']) assert.equal(run('check', flag, 'old').status, 1, flag);
  assert.equal(existsSync(join(root, '.cache/reuse/tasks')), false);
});

test('file recall bound is validated even in lexical mode and disclosed', t => {
  const { run } = setup(t);
  assert.equal(run('search', 'load', '--lexical', '--file-limit', '0').status, 1);
  const result = run('search', 'load', '--lexical', '--file-limit', '12');
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).fileLimit, 12);
});


test('changed historical approval evidence is reported without rewriting the record', t => {
  const { root, run, put } = setup(t);
  run('map');
  const ledger = JSON.stringify({ version: 1, requests: [{ id: 'old', status: 'resolved', kind: 'index', summary: 'Previous decision', options: ['Keep', 'Change'], evidence: [{ path: 'src/a.js', hash: '0'.repeat(64) }], decision: 'Keep', approvedBy: 'test-user', approvalRef: 'prior conversation' }] });
  put('.reuse/decisions.json', ledger);
  const result = run('check', '--purpose', 'startup', '--json');
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout).staleDecisionIds, ['old']);
  assert.equal(readFileSync(join(root, '.reuse/decisions.json'), 'utf8'), ledger);
});
