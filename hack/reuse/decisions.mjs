// Read-only compatibility for existing decisions; never create or resolve records here.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { digest, safeFile } from './repository.mjs';

export function readDecisions(root) {
  const path = resolve(root, '.reuse/decisions.json');
  const ledger = existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : { version: 1, requests: [] };
  if (ledger.version !== 1 || !Array.isArray(ledger.requests)) throw new Error('Invalid decision ledger');
  const ids = new Set();
  for (const item of ledger.requests) {
    if (!item || typeof item.id !== 'string' || ids.has(item.id) || !['pending', 'resolved'].includes(item.status)
      || !['similarity', 'semantic', 'ci', 'index'].includes(item.kind) || !item.summary?.trim()
      || !Array.isArray(item.options) || item.options.length < 2 || item.options.some(x => typeof x !== 'string' || !x.trim())
      || !Array.isArray(item.evidence) || !item.evidence.length
      || item.evidence.some(x => typeof x.path !== 'string' || !/^[a-f0-9]{64}$/.test(x.hash))) throw new Error('Invalid decision ledger entry');
    ids.add(item.id);
    if (item.status === 'resolved' && (!item.decision?.trim() || !item.approvedBy?.trim() || !item.approvalRef?.trim())) throw new Error('Resolved decision is missing human approval evidence');
  }
  return ledger;
}

export function evidenceChanged(root, item) {
  return item.evidence.some(file => !safeFile(root, file.path) || digest(readFileSync(resolve(root, file.path))) !== file.hash);
}
