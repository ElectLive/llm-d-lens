import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, lstatSync, readFileSync, realpathSync } from 'node:fs';
import { resolve, relative, sep } from 'node:path';
import ts from 'typescript';

export const SOURCE_ROOTS = [
  'apps/frontend/src/',
  'apps/backend/src/llm_d_lens/',
  'packages/shared/src/',
  // Retained for repository-tool fixtures and historical task baselines.
  'src/',
  'server/',
  'llm_d_lens/',
  'llm_d_bench/',
  'hack/',
  'tools/',
  'scripts/',
];
export const digest = text => createHash('sha256').update(text).digest('hex');
export const git = (root, args) => execFileSync('git', args, { cwd: root, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });

export function isSource(path) {
  return SOURCE_ROOTS.some(prefix => path.startsWith(prefix))
    && /\.(?:[cm]?js|jsx|ts|tsx|py)$/.test(path)
    && !/(?:^|\/)(?:node_modules|__pycache__|dist|build|vendor|fixtures|tests|test)(?:\/|$)/.test(path)
    && !/(?:^|\/)test_[^/]+\.py$|(?:\.test|\.spec|\.smoke)\.[^/]+$|\.d\.ts$/.test(path)
    && !path.endsWith('/mcp/tools.ts');
}

export function safeFile(root, path) {
  if (typeof path !== 'string' || !path || path.includes('\\') || path.split('/').includes('..')) return false;
  const full = resolve(root, path);
  if (!full.startsWith(resolve(root) + sep) || !existsSync(full) || lstatSync(full).isSymbolicLink()) return false;
  return lstatSync(full).isFile() && !relative(realpathSync(root), realpathSync(full)).startsWith('..');
}

export function discover(root) {
  return [...new Set(git(root, ['ls-files', '-z', '--cached', '--others', '--exclude-standard']).split('\0'))]
    .filter(path => isSource(path) && safeFile(root, path)).sort();
}

// One binary-safe batch, addressed by object IDs rather than filename expressions.
export function revisionFiles(root, revision, include = () => true) {
  const commit = git(root, ['rev-parse', '--verify', `${revision}^{commit}`]).trim();
  const rows = git(root, ['ls-tree', '-r', '-z', commit]).split('\0').filter(Boolean).flatMap(record => {
    const tab = record.indexOf('\t');
    const [mode, type, oid] = record.slice(0, tab).split(' ');
    const path = record.slice(tab + 1);
    return include(path) ? [{ mode, type, oid, path }] : [];
  });
  const blobs = rows.filter(row => row.type === 'blob');
  if (!blobs.length) return rows;
  const output = execFileSync('git', ['cat-file', '--batch'], {
    cwd: root, input: blobs.map(row => row.oid).join('\n') + '\n', maxBuffer: 256 * 1024 * 1024,
  });
  let offset = 0;
  for (const row of blobs) {
    const end = output.indexOf(10, offset);
    const [oid, type, sizeText] = output.subarray(offset, end).toString().split(' ');
    const size = Number(sizeText);
    if (oid !== row.oid || type !== 'blob' || !Number.isSafeInteger(size) || size < 0 || end + size + 1 >= output.length) throw new Error('Invalid Git batch response');
    row.contents = output.subarray(end + 1, end + 1 + size);
    offset = end + size + 2;
  }
  return rows;
}

export function sources(root, revision, paths) {
  if (!revision) return discover(root).map(path => ({ path, text: readFileSync(resolve(root, path), 'utf8') }));
  return revisionFiles(root, revision, path => isSource(path) && (!paths || paths.has(path)))
    .filter(row => row.mode.startsWith('100'))
    .map(row => ({ path: row.path, text: row.contents.toString('utf8') }));
}

export function declarations(path, text) {
  if (path.endsWith('.py')) {
    return [...text.matchAll(/^(\s*)(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)\s*[:(]/gm)]
      .map(match => ({ symbol: match[2], line: text.slice(0, match.index + match[1].length).split('\n').length, callable: true }));
  }
  const source = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true);
  const found = [];
  const visit = node => {
    if ((ts.isFunctionDeclaration(node) || ts.isClassDeclaration(node) || ts.isVariableDeclaration(node)
        || ts.isInterfaceDeclaration(node) || ts.isTypeAliasDeclaration(node) || ts.isMethodDeclaration(node)) && node.name && ts.isIdentifier(node.name)) {
      found.push({ symbol: node.name.text, line: source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1,
        endLine: source.getLineAndCharacterOfPosition(node.end).line + 1,
        callable: ts.isFunctionDeclaration(node) || ts.isClassDeclaration(node) || ts.isMethodDeclaration(node)
          || (ts.isVariableDeclaration(node) && node.initializer && (ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer))),
      });
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return found;
}

// Bounded, overlapping chunks cover module-level code and long function tails.
export function chunks(path, text) {
  const lines = text.split('\n');
  const symbols = declarations(path, text).filter(item => item.callable);
  const starts = new Set([0, ...symbols.map(item => item.line - 1)]);
  for (let i = 0; i < lines.length; i += 30) starts.add(i);
  return [...starts].sort((a, b) => a - b).flatMap(start => {
    const body = lines.slice(start, start + 40).join('\n');
    if (!body.trim()) return [];
    const symbol = symbols.findLast(item => item.line <= start + 1 && (!item.endLine || item.endLine >= start + 1))?.symbol || '(module)';
    return [{ path, line: start + 1, endLine: Math.min(start + 40, lines.length), symbol, text: body,
      hash: digest(`${path}\n${symbol}\n${body}`) }];
  });
}

export function validateCatalog(root, catalog) {
  if (catalog?.version !== 1 || !Array.isArray(catalog.entries) || !catalog.entries.length) return ['Catalog must have version 1 and nonempty entries'];
  const errors = [];
  const ids = new Set();
  const sourcePaths = new Set(discover(root));
  for (const entry of catalog.entries) {
    if (!entry || typeof entry !== 'object') { errors.push('Invalid catalog entry'); continue; }
    const label = entry.id || '(missing id)';
    if (!/^[a-z][a-z0-9-]*$/.test(entry.id || '') || ids.has(entry.id)) errors.push(`${label}: missing/invalid/duplicate id`);
    ids.add(entry.id);
    for (const key of ['purpose', 'boundaries']) if (typeof entry[key] !== 'string' || !entry[key].trim()) errors.push(`${label}: missing ${key}`);
    if (!Array.isArray(entry.symbols) || !entry.symbols.length || entry.symbols.some(x => typeof x !== 'string')) errors.push(`${label}: symbols required`);
    if (!sourcePaths.has(entry.path)) errors.push(`${label}: invalid source path ${entry.path}`);
    else {
      const text = readFileSync(resolve(root, entry.path), 'utf8');
      // AST validation prevents examples inside Python docstrings counting as declarations.
      const symbols = entry.path.endsWith('.py')
        ? JSON.parse(execFileSync('python3', ['-c', 'import ast,json,sys; tree=ast.parse(sys.stdin.read()); print(json.dumps([n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]))'], { input: text, encoding: 'utf8' }))
        : declarations(entry.path, text).map(x => x.symbol);
      const declared = new Set(symbols);
      for (const symbol of entry.symbols || []) if (!declared.has(symbol)) errors.push(`${label}: declaration ${symbol} missing in ${entry.path}`);
    }
    for (const key of ['examples', 'tests']) {
      if (!Array.isArray(entry[key])) { errors.push(`${label}: ${key} must be an array`); continue; }
      if (key === 'examples' && !entry[key].length) errors.push(`${label}: a real example is required`);
      for (const path of entry[key]) if (!safeFile(root, path)) errors.push(`${label}: missing/unsafe ${key} reference ${path}`);
    }
    if (!entry.tests?.length && !entry.testNote?.trim()) errors.push(`${label}: document missing test coverage with testNote`);
  }
  return errors;
}

export function renderMap(catalog) {
  const lines = ['# Reusable Capability Index', '', '<!-- Generated by npm run reuse:map. Edit .reuse/catalog.json, not this file. -->', '',
    'This is a starting point for discovery, not a complete function inventory. Source code, callers, and tests are authoritative. See the [reuse development guide](../docs/design/reuse-workflow.md) for usage.', ''];
  const link = path => `[${path}](../${path})`;
  lines.push('Boundaries, examples and tests are maintained in [catalog.json](catalog.json); read the matching entry before reuse.', '');
  for (const entry of catalog.entries) lines.push(`## ${entry.id}`, '', entry.purpose,
    `Entry: ${link(entry.path)} — ${entry.symbols.map(x => `\`${x}\``).join(', ')}`, '');
  return lines.join('\n');
}

const keywords = new Set('function return if else for while const let var class new throw try catch finally async await import from export default def in with as except raise None True False null true false switch case break continue yield and or not is pass'.split(' '));

function tokens(text, python) {
  // A lexical heuristic, not a parser or proof of behavioral equivalence.
  const pattern = python
    ? /#[^\n]*|"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|[^\s]/g
    : /\/\/[^\n]*|\/\*[\s\S]*?\*\/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|[^\s]/g;
  const result = [];
  let line = 1, offset = 0;
  for (const match of text.matchAll(pattern)) {
    line += text.slice(offset, match.index).split('\n').length - 1;
    offset = match.index;
    const raw = match[0];
    if (python ? raw.startsWith('#') || raw.startsWith('"""') || raw.startsWith("'''") : raw.startsWith('//') || raw.startsWith('/*')) continue;
    const normalized = /^["'`\d]/.test(raw) ? 'LITERAL' : /^[A-Za-z_$]/.test(raw) && !keywords.has(raw) ? 'ID' : raw;
    result.push({ value: normalized, line });
  }
  return result;
}

export function duplicates(files, changed = new Set(files.map(x => x.path)), { minTokens = 70, cache = null } = {}) {
  if (!changed.size) return [];
  const previousFiles = cache?.version === 1 && cache.minTokens === minTokens ? cache.files : {};
  const nextFiles = {};
  const windows = new Map();
  const pairs = new Map();
  for (const file of [...files].sort((a, b) => a.path.localeCompare(b.path))) {
    const hash = digest(file.text);
    let indexed = previousFiles?.[file.path]?.hash === hash ? previousFiles[file.path].windows : null;
    if (!Array.isArray(indexed)) {
      const parsed = tokens(file.text, file.path.endsWith('.py'));
      const unique = new Map();
      for (let i = 0; i <= parsed.length - minTokens; i++) {
        const key = digest(`${file.path.endsWith('.py') ? 'py' : 'js'}:${parsed.slice(i, i + minTokens).map(x => x.value).join(' ')}`);
        if (!unique.has(key)) unique.set(key, [key, parsed[i].line, parsed[i + minTokens - 1].line]);
      }
      indexed = [...unique.values()];
    }
    nextFiles[file.path] = { hash, windows: indexed };
    for (const [key, line, endLine] of indexed) {
      const location = { path: file.path, line, endLine };
      const previous = windows.get(key) || new Map();
      for (const other of previous.values()) {
        if (other.path === file.path || (!changed.has(other.path) && !changed.has(file.path))) continue;
        const pair = `${other.path}\0${file.path}`;
        if (!pairs.has(pair)) pairs.set(pair, { id: digest(pair), left: other, right: location, minTokens,
          message: 'Normalized token overlap; inspect behavior and callers before deciding reuse.' });
      }
      if (!previous.has(file.path)) previous.set(file.path, location);
      windows.set(key, previous);
    }
  }
  if (cache) Object.assign(cache, { version: 1, minTokens, files: nextFiles });
  return [...pairs.values()];
}
