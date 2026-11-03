#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve, relative } from 'node:path';
import { pathToFileURL } from 'node:url';

// Fast local-file checks only. Docusaurus CI owns MDX, routes and anchors.
export function checkDocs(root, paths) {
  const errors = [];
  for (const path of new Set(paths)) {
    if (!/\.mdx?$/.test(path) || !existsSync(resolve(root, path))) continue;
    const text = readFileSync(resolve(root, path), 'utf8').replace(/^\s*(```|~~~)[\s\S]*?^\s*\1.*$/gm, '');
    const links = [...text.matchAll(/\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)/g)].map(match => match[1]);
    links.push(...[...text.matchAll(/^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)/gm)].map(match => match[1]));
    for (let link of links) {
      link = link.replace(/^<|>$/g, '');
      if (/^(?:[a-z][a-z\d+.-]*:|\/|#)/i.test(link) || link.includes('{')) continue;
      try { link = decodeURIComponent(link.split(/[?#]/)[0]); }
      catch { errors.push(`${path}: invalid link encoding: ${link}`); continue; }
      if (!link) continue;
      const target = resolve(root, dirname(path), link);
      // Extensionless Docusaurus routes need the full site build.
      if (!/\.[a-z\d]+$/i.test(link)) continue;
      if (relative(root, target).startsWith('../') || !existsSync(target)) errors.push(`${path}: missing local link: ${link}`);
    }
  }
  return errors;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const root = execFileSync('git', ['rev-parse', '--show-toplevel'], { encoding: 'utf8' }).trim();
    const args = process.argv.slice(2);
    let base = 'HEAD';
    if (args[0] === '--base') {
      if (!args[1]) throw new Error('--base requires a Git revision');
      base = args.splice(0, 2)[1];
    }
    if (args.some(arg => arg.startsWith('--'))) throw new Error('Usage: docs:check -- [--base REF] [FILE ...]');
    const git = argv => execFileSync('git', argv, { cwd: root, encoding: 'utf8' }).split('\0').filter(Boolean);
    const commit = execFileSync('git', ['rev-parse', '--verify', `${base}^{commit}`], { cwd: root, encoding: 'utf8' }).trim();
    const paths = args.length ? args : [...git(['diff', '--name-only', '-z', commit, '--']), ...git(['ls-files', '--others', '--exclude-standard', '-z'])];
    const errors = checkDocs(root, paths);
    if (errors.length) { console.error(errors.join('\n')); process.exitCode = 1; }
    else console.log('Changed documentation file links passed. MDX, routes and anchors require docs:build.');
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
