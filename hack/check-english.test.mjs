import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { checkEnglish } from './check-english.mjs';

// Construct Unicode test data at runtime so the test source stays English.
const sample = String.fromCodePoint(0x4e2d, 0x6587);

test('staged checks inspect the index rather than unstaged edits', t => {
  const root = mkdtempSync(join(tmpdir(), 'english-staged-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  execFileSync('git', ['init', '-q', root]);
  writeFileSync(join(root, 'a.txt'), sample);
  execFileSync('git', ['add', '.'], { cwd: root });
  writeFileSync(join(root, 'a.txt'), 'English');
  assert.equal(checkEnglish(root, [], true).length, 1);
  execFileSync('git', ['add', '.'], { cwd: root });
  writeFileSync(join(root, 'a.txt'), sample);
  assert.deepEqual(checkEnglish(root, [], true), []);
  assert.equal(checkEnglish(root).length, 1);
});

test('scans tracked and new text, dotfiles, filenames, supplementary Han and PR metadata', () => {
  const root = mkdtempSync(join(tmpdir(), 'english-check-'));
  try {
    execFileSync('git', ['init', '-q', root]);
    mkdirSync(join(root, '.instructions'));
    writeFileSync(join(root, 'tracked.md'), 'English documentation.\n');
    execFileSync('git', ['-C', root, 'add', 'tracked.md']);
    writeFileSync(join(root, 'tracked.md'), `English\n${sample}\n`);
    writeFileSync(join(root, '.instructions', 'new.md'), sample);
    writeFileSync(join(root, `${sample}.txt`), 'English');
    writeFileSync(join(root, 'extension.txt'), String.fromCodePoint(0x20000));
    writeFileSync(join(root, 'utf16.txt'), Buffer.concat([Buffer.from([0xff, 0xfe]), Buffer.from(sample, 'utf16le')]));
    writeFileSync(join(root, '.gitignore'), 'ignored.txt\nmetadata.txt\n');
    writeFileSync(join(root, 'ignored.txt'), sample);
    writeFileSync(join(root, 'metadata.txt'), sample);
    writeFileSync(join(root, 'asset.bin'), Buffer.from([0, 255, 128]));
    const findings = checkEnglish(root, ['metadata.txt']);
    assert.equal(findings.length, 6);
    assert.ok(findings.some(line => line.startsWith('tracked.md:2:')));
    assert.ok(findings.some(line => line.startsWith('.instructions/new.md:1:')));
    assert.ok(findings.some(line => line.includes('characters in filename')));
    assert.ok(findings.some(line => line.startsWith('extension.txt:1:')));
    assert.ok(findings.some(line => line.startsWith('utf16.txt:1:')));
    assert.ok(findings.some(line => line.startsWith('metadata.txt:1:')));
    assert.ok(!findings.some(line => line.startsWith('ignored.txt:')));
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('accepts English text and deleted tracked files; rejects undecodable text', () => {
  const root = mkdtempSync(join(tmpdir(), 'english-check-'));
  try {
    execFileSync('git', ['init', '-q', root]);
    writeFileSync(join(root, 'deleted.md'), 'English');
    execFileSync('git', ['-C', root, 'add', 'deleted.md']);
    rmSync(join(root, 'deleted.md'));
    writeFileSync(join(root, 'README.md'), 'English with Unicode punctuation — valid.');
    assert.deepEqual(checkEnglish(root), []);
    writeFileSync(join(root, 'invalid.txt'), Buffer.from([0xff]));
    assert.throws(() => checkEnglish(root), /non-UTF-8/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('exempts only Chinese locale content, not English docs or site source', () => {
  const root = mkdtempSync(join(tmpdir(), 'english-check-'));
  try {
    execFileSync('git', ['init', '-q', root]);
    const paths = [
      'docs/site/i18n/zh-CN/docusaurus-plugin-content-docs/current/overview.mdx',
      'docs/site/i18n/zh-CN/code.json',
      'docs/site/i18n/en/docusaurus-plugin-content-docs/current/overview.mdx',
      'docs/site/i18n/en/code.json',
      'docs/site/src/pages/index.mdx',
      'docs/design/cluster-design.md',
    ];
    for (const path of paths) {
      mkdirSync(join(root, path, '..'), { recursive: true });
      writeFileSync(join(root, path), sample);
    }
    const findings = checkEnglish(root);
    assert.equal(findings.length, 4);
    for (const path of paths.slice(2)) {
      assert.ok(findings.some(line => line.startsWith(`${path}:1:`)));
    }
    assert.equal(checkEnglish(root, [paths[0]]).length, 5);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('retains filename checks within translations and scopes the locale-label exception', () => {
  const root = mkdtempSync(join(tmpdir(), 'english-check-'));
  try {
    execFileSync('git', ['init', '-q', root]);
    mkdirSync(join(root, 'docs/site/i18n/zh-CN'), { recursive: true });
    writeFileSync(join(root, `docs/site/i18n/zh-CN/${sample}.md`), sample);
    const selfName = String.fromCodePoint(0x7b80, 0x4f53, 0x4e2d, 0x6587);
    const label = `  'zh-CN': { label: '${selfName}' },`;
    writeFileSync(join(root, 'docs/site/docusaurus.config.ts'), `${label}\n// ${sample}\n`);
    writeFileSync(join(root, 'other.config.ts'), label);
    const findings = checkEnglish(root);
    assert.equal(findings.length, 3);
    assert.ok(findings.some(line => line.includes('Chinese characters in filename')));
    assert.ok(findings.some(line => line.startsWith('docs/site/docusaurus.config.ts:2:')));
    assert.ok(findings.some(line => line.startsWith('other.config.ts:1:')));
  } finally { rmSync(root, { recursive: true, force: true }); }
});
