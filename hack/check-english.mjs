#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { lstatSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const han = /\p{Script=Han}/u;

// Chinese Docusaurus translations (docs/site/i18n/zh-CN/...) are
// the one explicit, user-approved exception to the English-only policy: its
// entire purpose is non-English, user-facing translated documentation and UI
// strings for the published docs site. Everything else in the repository —
// source, comments, commit/PR text, filenames, configuration — remains
// English-only and is still fully scanned below, including i18n/en/.
const i18nContentDir = /^docs\/site\/i18n\/zh-CN\//;

// The locale picker must label each locale with its own self-endonym (the
// universal convention used by every multilingual site: a language names
// itself in its own script, e.g. the Simplified Chinese and Japanese
// self-names, or "Espanol" - that is not "untranslated prose", it is the
// name of the language itself). This narrowly allows only that one known
// label line in the Docusaurus config, not non-English text anywhere else
// in the file. The expected value is expressed as Unicode escapes (U+7B80
// U+4F53 U+4E2D U+6587, i.e. "Simplified Chinese") so this file itself
// stays free of literal Han glyphs.
const zhCNSelfName = '\u7b80\u4f53\u4e2d\u6587';
const localeLabelException = {
  path: 'docs/site/docusaurus.config.ts',
  line: new RegExp(`^\\s*'zh-CN':\\s*\\{\\s*label:\\s*'${zhCNSelfName}'\\s*\\},?$`),
};

// Git owns the inventory, including dotfiles and untracked, non-ignored additions.
export function checkEnglish(root, textFiles = [], staged = false) {
  const paths = execFileSync('git', staged ? ['diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z'] : ['ls-files', '-z', '--cached', '--others', '--exclude-standard'], {
    cwd: root, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
  }).split('\0').filter(Boolean);
  const findings = [];
  for (const path of new Set(paths)) {
    if (han.test(path)) findings.push(`${path}: Chinese characters in filename`);
    if (i18nContentDir.test(path)) continue;
    if (staged) {
      const entry = execFileSync('git', ['ls-files', '--stage', '-z', '--', path], { cwd: root, encoding: 'utf8' });
      if (entry.startsWith('100')) scanBytes(execFileSync('git', ['show', `:${path}`], { cwd: root, maxBuffer: 64 * 1024 * 1024 }), path, findings);
      continue;
    }
    const full = resolve(root, path);
    let stat;
    try { stat = lstatSync(full); }
    catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    if (stat.isFile()) scan(full, path, findings);
  }
  for (const path of textFiles) scan(resolve(root, path), path, findings);
  return findings;
}

function scan(full, label, findings) {
  scanBytes(readFileSync(full), label, findings);
}

function scanBytes(bytes, label, findings) {
  let content;
  if (bytes[0] === 0xff && bytes[1] === 0xfe) content = new TextDecoder('utf-16le', { fatal: true }).decode(bytes);
  else if (bytes[0] === 0xfe && bytes[1] === 0xff) content = new TextDecoder('utf-16be', { fatal: true }).decode(bytes);
  else {
    if (bytes.includes(0)) return; // Binary assets are not repository text.
    try { content = new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
    catch { throw new Error(`${label}: non-UTF-8 text; convert to UTF-8 before checking`); }
  }
  const allowException = label === localeLabelException.path;
  content.split(/\r?\n/).forEach((line, index) => {
    if (allowException && localeLabelException.line.test(line)) return;
    if (han.test(line)) findings.push(`${label}:${index + 1}: ${line.trim()}`);
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const staged = process.argv.includes('--staged');
    const args = process.argv.slice(2).filter(arg => arg !== '--staged');
    const textFiles = [];
    for (let i = 0; i < args.length; i += 2) {
      if (args[i] !== '--text-file' || !args[i + 1]) throw new Error('Usage: npm run check:english -- [--staged] [--text-file PATH ...]');
      textFiles.push(args[i + 1]);
    }
    const root = execFileSync('git', ['rev-parse', '--show-toplevel'], { encoding: 'utf8' }).trim();
    const findings = checkEnglish(root, textFiles, staged);
    if (findings.length) {
      console.error(findings.join('\n'));
      console.error(`English check failed: ${findings.length} finding(s). Translate the text into English and rerun; do not delete content or encode it to bypass this check.`);
      process.exitCode = 1;
    } else console.log('English check passed: no Chinese characters in repository text or supplied PR text.');
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
