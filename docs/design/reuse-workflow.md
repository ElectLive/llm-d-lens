# Reuse tools and focused development

[AGENTS.md](../../AGENTS.md) owns daily development rules. Locate relevant source,
callers and tests with `rg`; consult the capability map when ownership is unclear.
The tools support discovery and deterministic checks, not proof that an Agent
understood or reviewed every file. Business decisions remain in the conversation. New features do not require catalog
registration or these discovery commands; use them when focused source search is insufficient.

## Commands

| Command | Purpose |
| --- | --- |
| `npm run reuse:map` | Validate the catalog and regenerate `.reuse/map.md` |
| `npm run reuse:search -- "terms"` | Lexical source search without model loading |
| `npm run reuse:search -- "description" --semantic` | Optional local semantic search |
| `npm run reuse:index` | Prepare the local model index; first use downloads assets |
| `npm run reuse:check -- --base HEAD` | Catalog/map checks and changed-source duplicate/registration candidates |
| `npm run reuse -- check --purpose startup` | Catalog/map and existing decision checks without source-wide analysis |
| `npm run reuse:test` | Tool and launcher regression tests |
| `npm run verify:full` | Manual local tests, linters and builds; see [contributing](../../CONTRIBUTING.md#periodic-full-local-validation) |

`check` returns 1 for invalid input/catalog/map and 2 for existing pending decisions.
Without `--base`, it examines all discovered source. A supplied base is resolved to
its commit and compared directly with the working tree, including untracked source.
`--json` prints details to stdout; no persistent audit report is generated.
Duplicate and registration candidates are advisory, not automatic refactoring or
proof of incompatibility. Inspect contracts and callers before changing behavior.

## Search and caches

Semantic search first ranks distinct file summaries, then chunks within selected
files. `--file-limit` defaults to 6 (range 1–100); `--limit` defaults to 8 (1–50).
`--offline` uses cached semantic model assets and fails if they are unavailable.
Lexical matching uses whitespace-separated substrings, not semantic equivalence.
Source inference stays local. Models load per semantic CLI invocation; unchanged
embeddings and duplicate-token windows are cached under `.cache/reuse/` and
invalidated by content/model changes. These are derived search caches, not test
result caches or approval records. Git history is read using binary-safe batches.

## Existing decisions and startup

`.reuse/decisions.json` is retained as read-only historical context. Malformed
records fail checks; existing pending records retain exit 2, and changed evidence
on resolved records is reported rather than reused as authorization. Do not delete
or silently resolve these records. New uncertainties are discussed directly with
the user; the CLI no longer creates or resolves decisions.

The launcher regenerates and validates the catalog before stopping services.
Invalid checks block startup. As before, existing pending decisions are printed
as warnings and local startup may continue; standalone checks and full validation
still fail on exit 2. No report file is needed in this path.

## Validation and reviews

Run actual tests directly. `begin`, `verify`, `verify-report`, `review`, `scope`,
`cover`, `review-status`, `review-report`, `pause`, `resolve` and their bookkeeping
flags have been removed; old invocations fail explicitly. Existing ignored task
files are left untouched but are no longer consulted. There is no automatic test
result reuse, fingerprint freshness gate or per-capability coverage ledger.

[The review skill](../../.agents/skills/repo-review/SKILL.md) guides explicitly
requested comprehensive reviews without these records. Report scope, findings,
changes, actual test results and remaining gaps. Manual full validation saves
per-command logs and fails if any command fails; it does not replace live cluster,
external-provider, container or release/security acceptance.

Implementation: [CLI](../../hack/reuse/cli.mjs),
[repository checks](../../hack/reuse/repository.mjs),
[registration discovery](../../hack/reuse/registration.mjs),
[search](../../hack/reuse/search.mjs) and
[historical decisions](../../hack/reuse/decisions.mjs).
