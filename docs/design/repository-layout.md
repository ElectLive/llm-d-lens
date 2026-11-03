# Repository layout

Status: proposed (October 2026). Owner: Lens maintainers.

## Intent

Lens is moving to the architecture in the October 2026 team-sync roadmap:
**the CLI is the backend**. One Python engine implements every lifecycle
operation (check, prepare, plan, standup, update, teardown, validate,
benchmark). The CLI runs those operations in-process; `serve` puts the
dashboard on top of the same engine; `orchestrate` runs the same CLI as a
Kubernetes Job or in CI. Kubernetes RBAC is the only permission check and
stack records live in the cluster, not on the host.

Today the repository has one Python package (`llm_d_lens`) that mixes
domain logic with FastAPI, auth, a database and features that are now out of
core scope. The root directory is also cluttered with tooling and lint
configuration.

This reorganization sets up the direction. It does **not** rewrite any
behavior. It:

1. creates a clear home for a new engine package, `lens`, at `apps/lens`,
   separate from the existing service;
2. keeps the existing `apps/` + `packages/` monorepo layout, so the working
   application stays where it is while the engine grows;
3. moves tooling, deployment and lint clutter off the root.

## Migration strategy (context for this and later work)

The existing backend is replaced gradually, not in one rewrite:

1. Implement an operation in `lens`.
2. Trial it from the `lens` CLI.
3. Point the existing FastAPI route at the engine operation.
4. Delete the old implementation from `llm_d_lens`.

Features outside core scope (Kubespray cluster bootstrap, Lens users, roles,
sessions and LDAP, GPU driver and monitoring stack installation, the Ubuntu
x86_64 installer) are not ported to `lens`; they are retired when the UI no
longer needs them.

**Dependency rule:** `llm_d_lens` (the legacy service) may import `lens`.
`lens` must never import `llm_d_lens`, FastAPI, the database layer or auth.
This keeps the engine installable alone for CI and in-cluster Jobs.

## Goals

- A new, independently installable Python package `lens` with its own
  `pyproject.toml` and a minimal dependency set.
- The existing service, frontend, tests, Docker image, CI and local dev
  scripts work exactly as before.
- A visibly smaller root: applications, docs, deployment and developer
  tooling each have one obvious directory.

## Non-goals

- No change to application behavior, APIs, database schema or UI.
- No move or rename of `apps/backend`, `apps/frontend`, `packages/shared`,
  the `llm_d_lens` Python package or the `@repo/*` npm workspaces.
- No lifecycle operations in `lens` yet; the CLI is created right after this
  reorganization.
- No removal of out-of-scope features yet.

## Anti-goals (failure even if everything "works")

- Any existing user-facing functionality or CI/CD workflow stops working.
  Everything is updated to the new structure, not dropped.
- The new engine depends on, or is mixed into, the legacy `llm_d_lens` code.
- Paths are fixed by symlinks or compatibility shims at the old locations
  instead of by updating references.
- Published docs updated in English but not Simplified Chinese.
- Lint, docs or English checks weakened to make the move pass.

## Target layout

```text
llm-d-lens/
├── apps/
│   ├── lens/                   NEW: Python distribution for the lens engine + CLI
│   │   ├── pyproject.toml
│   │   ├── lens/
│   │   │   ├── __init__.py
│   │   │   └── __main__.py     minimal entry point (`lens --version`)
│   │   └── tests/
│   │       └── test_boundary.py   lens must not import llm_d_lens
│   ├── backend/                unchanged (legacy FastAPI service)
│   └── frontend/               unchanged
├── packages/
│   └── shared/                 unchanged
├── docs/                       unchanged (design/ and site/)
├── deploy/
│   ├── Dockerfile              was ./Dockerfile
│   └── model-cache-downloader/ was docker/model-cache-downloader/
├── scripts/
│   └── LensInstaller-Ubuntu-x86_64-online.sh   stays: public install URL
├── hack/                       developer tooling: was scripts/ + tools/
│   ├── reuse/                  was tools/reuse/
│   ├── hooks/                  was scripts/hooks/
│   └── *.sh, *.mjs, *.py       remaining scripts/ and tools/ files
├── .github/  .agents/  .reuse/
├── .typos.toml                 was _typos.toml (renamed; typos discovers it)
└── README.md  LICENSE  CONTRIBUTING.md  CODE_OF_CONDUCT.md  AGENTS.md  OWNERS
    Makefile  package.json  package-lock.json  eslint.config.js
    tsconfig.base.json  .npmrc  .pre-commit-config.yaml  (other dotfiles)
```

Files that stay at the root do so because their tools discover them there:
npm workspaces (`package.json`, `package-lock.json`, `.npmrc`), ESLint flat
config, pre-commit, GitHub (`.github/`) and Prow (`OWNERS`).
`.hadolint.yaml`, `.yamllint.yml`, `.editorconfig` and similar dotfiles stay
where they are; they are already hidden.

## Decisions

### Made by the maintainer

- The engine is its own Python package named `lens`, at `apps/lens`.
- The `apps/` + `packages/` monorepo layout is kept. An earlier draft moved
  everything under a top-level `src/`; review pointed out that this produces
  `src/server/src/`, where the two `src/` directories mean different things,
  and that `apps/` + `packages/` is the established convention for separating
  applications from shared code.
- There is no root `pyproject.toml`. The repository holds two Python
  distributions (the engine and the legacy service) plus an npm workspace, so
  no single Python project owns the root. Each distribution keeps its own
  `pyproject.toml`, which keeps the engine free of the service's dependencies.
  The engine installs straight from Git with
  `pip install "git+https://github.com/llm-d-incubation/llm-d-lens#subdirectory=apps/lens"`.
- The existing functionality is preserved and keeps running during the
  migration; the UI is moved onto the engine operation by operation.
- Nothing may break going forward: the published install URL, fresh
  installs, CI/CD and the container build must all work on the new layout.
  Previously saved installer copies run against `main` are an existing user
  pattern the project cannot control; they are not supported.

### Made during design (check these)

- Directory names `apps/lens`, `deploy/` and `hack/`. The engine directory is
  named after its package rather than a role (`engine`, `core`).
- The CLI command is also `lens` (console script `lens = lens.__main__:main`).
- `apps/lens` uses a flat layout (`apps/lens/lens/`), avoiding a
  `apps/lens/src/` double nesting.
- `scripts/` and `tools/` merge into one `hack/` directory. File names do not
  collide. The one exception is the Ubuntu installer, which stays at
  `scripts/LensInstaller-Ubuntu-x86_64-online.sh` because that raw GitHub URL
  is published in the README and docs (`curl …/main/scripts/… | bash`) and
  must keep working. It is updated internally to the new layout
  (`hack/storage-env.sh`, `hack/run-backend.sh`, certificate script path).
  `installer-checks.yaml` keeps its paths.
- The `lens` skeleton uses only the standard library (`argparse`); choosing a
  CLI framework is left to the CLI work that follows.
- The PyPI name `lens` is already taken by an unrelated project (a pandas
  DataFrame explorer), so the two cannot be installed in the same environment.
  The import name and command stay `lens`. The distribution name in
  `apps/lens/pyproject.toml` is `llm-d-lens-engine` until publishing is
  decided; `llm-d-lens` is held by the legacy service.
- `llm_d_lens` does not declare a dependency on `lens` yet; it gets one when
  the first route is moved onto the engine.
- `apps/backend/Dockerfile` (backend-only, not referenced by CI or scripts)
  stays where it is; it is a candidate for retirement later.
- CI and `make` gate the engine from day one: `make test` and `make lint`
  include `apps/lens` (pytest, ruff), and `ci-pr-checks.yaml` installs
  `apps/lens` and runs its tests, so the dependency rule is enforced on
  every PR.

## Container image build

`deploy/Dockerfile` keeps the **repository root as its build context**. Every
build invocation passes the file explicitly and the context stays `.`:

```bash
docker buildx build -f deploy/Dockerfile .
podman build -f deploy/Dockerfile .
```

This applies to the `container-build` job in `ci-pr-checks.yaml` and to
`make image-build` for both the docker and podman branches.

Because `COPY` sources resolve against the context, not the Dockerfile's
location, the `apps/` and `packages/` sources are unchanged. Only the
startup scripts move:

| Before | After |
| --- | --- |
| `COPY scripts/run-backend.sh scripts/storage-env.sh ./scripts/` | `COPY hack/run-backend.sh hack/storage-env.sh ./hack/` |
| `CMD ["bash", "scripts/run-backend.sh"]` | `CMD ["bash", "hack/run-backend.sh"]` |

`run-backend.sh` sources `storage-env.sh` from `$ROOT_DIR/hack/`, so both
must land in `/app/hack/` inside the image.

`.dockerignore` stays at the root, where Docker still reads it for a root
context.

`deploy/model-cache-downloader/Dockerfile` has no `COPY` or `ADD` from the
repository; it builds from its own directory, so only the build command in
its header comment changes.

Verification: the image builds from `deploy/Dockerfile` with both the CI
command and `make image-build`, and a container started from it serves the
dashboard and API.

## What must be updated

The move is done with `git mv` so history follows the files. Every
reference to an old path must be updated. Known reference sites:

- **Build and CI:** root `package.json` (script paths), `Makefile`,
  `deploy/Dockerfile`, `.github/workflows/*.yaml` (path filters, `pip
  install`, `pytest` and script paths), `.github/instructions/`.
- **Lint config:** `eslint.config.js` (`tools/reuse` file glob becomes
  `hack/reuse`), `.typos.toml`.
- **Dev scripts:** `hack/lens-local.sh`, `hack/run-backend.sh`,
  `hack/storage-env.sh`, `hack/generate-mcp-tools.mjs`,
  `hack/export-openapi.py`, `hack/hooks/pre-commit`, the installer, and the
  `hack/reuse` modules and tests that hard-code `scripts/` or `tools/` paths.
- **Reuse catalog:** `.reuse/catalog.json`, `.reuse/decisions.json`; regenerate
  `.reuse/map.md` with `npm run reuse:map`.
- **Agent and contributor docs:** `AGENTS.md`, `.agents/skills/*/SKILL.md`,
  `CONTRIBUTING.md` (and its synced site page), `README.md`.
- **Design docs:** `docs/design/*.md` path references.
- **Published docs:** every English page under
  `docs/site/i18n/en/...` that cites a moved path, and the matching
  `docs/site/i18n/zh-CN/...` page in the same change.
- **`.gitignore`:** an allow-list entry for this document. Per `AGENTS.md`,
  the exact `.gitignore` diff and its impact must be shown to the maintainer
  and confirmed before it is applied.

## Approval scope

Approving this document is the `AGENTS.md` confirmation for:

- moving the files and directories listed in the target layout;
- creating `apps/lens/pyproject.toml`, `apps/lens/lens/__init__.py`,
  `apps/lens/lens/__main__.py` and `apps/lens/tests/test_boundary.py`.

It is not approval for the `.gitignore` change, which is confirmed separately,
or for deleting any file.

## Done when

- `pip install -e apps/lens` installs, and `lens --version` runs; CI runs
  the engine tests.
- The boundary test passes: importing `lens` pulls in no `llm_d_lens`
  module.
- `npm run build`, `npm run lint`, `npm run type-check`, `make test`,
  `npm run docs:check`, `npm run check:english` and
  `node hack/generate-mcp-tools.mjs --check` pass.
- The container image builds from `deploy/Dockerfile` (CI command and
  `make image-build`), and a container started from it serves the dashboard
  and API.
- `hack/lens-local.sh restart all` starts the app locally and the dashboard
  loads.

## Left to the builder

- Order of moves and how the work is split into commits.
- Exact `pyproject.toml` contents for `lens` beyond what is stated above
  (Python version should match the server, `>=3.11`).
- Whether a moved script needs small internal path fixes beyond the list
  above; fix whatever the checks surface.
