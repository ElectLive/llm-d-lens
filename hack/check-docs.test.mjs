import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { checkDocs } from './check-docs.mjs';

test('local links resolve relative to the changed document, with external and route links left to CI', t => {
  const root = mkdtempSync(join(tmpdir(), 'docs-check-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, 'docs'));
  writeFileSync(join(root, 'target file.md'), '# Target');
  writeFileSync(join(root, 'docs/a.mdx'), '[ok](../target%20file.md#target)\n[x](https://example.com/a.md)\n[route](/guide)\n[anchor](#here)\n[bad](gone.md)\n[ref]: missing.md\n```md\n[sample](sample.md)\n```\n');
  assert.deepEqual(checkDocs(root, ['docs/a.mdx', 'docs/deleted.md']), ['docs/a.mdx: missing local link: gone.md', 'docs/a.mdx: missing local link: missing.md']);
});
