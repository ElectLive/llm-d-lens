## Governance Structure

`llm-d-lens` adopts the following hierarchical technical governance structure:

- A community of **contributors** who file issues and submit pull requests
- A body of **core maintainers** who own `llm-d-lens` overall and drive its development

All contributions are expected to follow `llm-d-lens` design principles and best practices, as enforced by core maintainers. While high-quality pull requests are appreciated and encouraged, all maintainers reserve the right to prioritize their own work over code reviews at-will, hence contributors should not expect their work to be reviewed promptly.

Contributors can maximize the chances of their work being accepted by maintainers by meeting a high quality bar before sending a PR to maintainers.

### Core maintainers

The core maintainers lead the development of `llm-d-lens` and define the benchmarking infrastructure and strategy for the broader `llm-d project`. Their responsibilities include:

- Proposing, implementing and reviewing load profiles, parameter configurations, run rules, data collections, and analysis of workloads to `llm-d`
- Enforcing code quality standards and adherence to core design principles

The core maintainers should publicly articulate their decision-making, and share the reasoning behind their decisions, vetoes, and dispute resolution.

List of core maintainers can be found in the [OWNERS](https://github.com/llm-d-incubation/llm-d-lens/blob/main/OWNERS) file.

### Decision Making

#### Uncontroversial Changes

We are committed to accepting functional bug fixes that meet our quality standards – and include minimized unit tests to avoid future regressions. Performance improvements generally fall under the same category, with the caveat that they may be rejected if the trade-off between usefulness and complexity is deemed unfavorable by core maintainers. Design changes that neither fix known functional nor performance issues are automatically considered controversial.

#### Controversial Changes

More controversial design changes (e.g., breaking changes to workload profiles, load generators, run rules or data collection and analysis tools) are evaluated on a case-by-case basis under the subjective judgment of core maintainers.

## Submitting a Pull Request

We welcome contributions to any aspect of `llm-d-lens`! If you have a bug fix, feature request, or improvement, please submit a pull request (PR) to the repository.

For every Pull Request submitted, ensure the following steps have been done:

1. [Sign your commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)
2. Make sure that the [pre-commit](https://pre-commit.com/) hooks have been run and pass **before you push** the contents of your PR. See [Local Development Checks](#development-and-validation) below.

## Publishing the online installer

Keep `LENS_SOURCE_VERSION="main"` in the main branch. For a release, create a
release branch from the intended commit and stamp the online installer before
creating the tag:

```bash
bash scripts/LensInstaller-Ubuntu-x86_64-online.sh release stamp vX.Y.Z
bash scripts/LensInstaller-Ubuntu-x86_64-online.sh release check vX.Y.Z
```

Replace `vX.Y.Z` with the actual tag. Commit the stamped script, tag that commit,
and wait for **Installer checks** on the tag to pass before announcing the raw
download URL. Keep release tags immutable. Do not merge the stamped version back
into main; main/PR checks require `main`. Tag checks require the exact tag name.
This workflow validates tags but does not create releases or enforce tag protection.
Older tags without the online entry point are not retroactively supported.

The only public installation entry is
`scripts/LensInstaller-Ubuntu-x86_64-online.sh`. The same script contains the
download stage, installation engine and Shell release-version commands.
After downloading the source archive, it invokes the archive's copy in internal
source-install mode instead of downloading recursively.
CI checks Shell syntax/lint, version consistency, standalone help and private
Python provisioning on Ubuntu 22.04.

## Repository language

Write all repository content and PR titles/descriptions in English. Run
`npm run check:english` before committing or opening/updating a PR. Agents must
translate findings automatically, review their meaning, and rerun until clean.
Use `-- --text-file /tmp/pr-title.txt --text-file /tmp/pr-body.md` to include
PR metadata in the check. The scanner reports Han characters in Git-visible text
and filenames; it is not a natural-language classifier for every other language.
Do not replace text with escapes or remove useful content to bypass the policy.
The one exception is `docs/site/i18n/zh-CN/` (the bilingual docs site's
Chinese translations), which the scanner skips on purpose. English documentation
under `docs/site/i18n/en/` remains fully scanned.

## Development and validation

Before changing behavior, follow [AGENTS.md](AGENTS.md) and the
[reuse-first workflow](docs/design/reuse-workflow.md). Find existing implementations,
compare contracts and callers, and record reuse/new-code rationale in the PR.
Pause dependent work for unresolved business, compatibility, CI exception or
catalog decisions; obtain and record the user's actual answer before proceeding.

Use targeted tests for bounded fixes. Run `npm run reuse:test` for reuse-tool
changes. Run existing test commands directly; no task JSON or automatic audit
reports are required. Report actual results and limitations; rerun affected tests
when their inputs or environment change.
Update `.reuse/catalog.json` and run `npm run reuse:map` when entries change.
Similarity candidates require analysis, not automatic refactoring.

`npm run docs:check` checks local file links in changed Markdown/MDX; pass
`--base REF` for a branch comparison or file paths explicitly. It does not compile
MDX or validate route/anchor links. `npm run docs:build` builds the bilingual site
for rendering, site configuration and dependency changes. Broken routes, Markdown
links and anchors are fatal, so the docs CI job fails and blocks merge when any
link no longer resolves. GitHub statistics refresh is explicit and optional.

### Periodic full local validation

Run `npm run verify:full` manually at a milestone or before review. It runs the
existing English/catalog/stack checks, linters, MCP generation, types, JS/Python
and tooling tests, frontend build and bilingual docs build without cached passes.
Install workspace and Python development/embedded-db dependencies first; set
`PRISM_PYTHON_BIN` if Python is not at `.venv/bin/python` (fallback: `python3`).
Each check prints PASS/FAIL and saves its full log in the printed unique directory
under `.cache/reuse/full-validation/`. Failures do not prevent later checks, and
any failure or missing tool makes the command exit nonzero. Test data is isolated.
This is a simple Makefile entry, not an evidence/report framework. Do not edit
source during the run; rerun after relevant source or environment changes.
Daily fixes use targeted tests; agents read failed logs only. Live cluster/provider
acceptance, containers, online stack checks, model smoke tests and the security
requirements below remain separate; test skips and build warnings remain in logs.

The optional repository hook (`git config core.hooksPath hack/hooks`) checks staged
whitespace and English text, without running regression suites. The pre-commit
framework configuration remains available for staged file hygiene and linters.
Full regression checks belong in CI; do not run both full and subset suites
before every commit.

Use Vite hot reload for frontend source changes. `hack/lens-local.sh restart backend`
or `restart server` restarts only the requested service. Use `restart all` for
port/environment changes and API changes that alter the server MCP catalog.
Actual service ports are recorded for targeted restarts. For a session started
with an older launcher, run `restart all` once to initialize that state.

Failed Deploy Kubernetes commands emit `deployment_kubernetes_command_failed`
in backend logs. The structured record identifies the transport, namespace, CLI
PID, selected API origin, kubeconfig/cache paths, sanitized proxy origins and
literal API-host entries in each `NO_PROXY` variant. CIDR/domain bypass rules
are not evaluated by this diagnostic. Credentials, full command arguments and
raw kubeconfig contents are not logged; command results and retry behavior are
unchanged.

Alembic logging configuration keeps existing application loggers enabled when
migration commands run in the same process.

Follow [README.md](README.md) for setup. Choose checks appropriate to the change:

```bash
npm run build
npm run type-check
make test-js
make test-python
make lint
```

Keep one canonical design per module or shared domain under `docs/design/`,
using lowercase, hyphen-separated English filenames without locale suffixes.
Merge related material into that design; remove superseded proposals rather than
keep historical appendices. Update file and section references, including docs
sync inputs and CI path filters, whenever designs move.
Maintain user guides and API references under `docs/site/i18n/en/docusaurus-plugin-content-docs/current/`. Keep API descriptions
and relevant documentation in sync with frontend and backend behavior. Real
cluster deployment and external provider calls require a configured environment;
report separately whether those flows were exercised.

## Security validation requirements

These requirements cover application code, automation, dependency manifests and
lockfiles, containers, installed dependencies shipped with the product, and
release artifacts. Treat PRs and dependency-update PRs as untrusted input: use
isolated runners and do not expose deployment credentials or long-lived secrets.
Runners, scanners and their rule/signature update channels are trusted
infrastructure; results apply only to scanned content, not the runtime environment.

**Repository configuration:** `.github/workflows/ci-pr-checks.yaml` targets PRs
to `main`. It checks English content and, for detected code changes, the pinned
llm-d stack, frontend build, MCP generation, TypeScript, JS tests, Ruff/format and
container build. It does not run `npm run lint`. No repository-owned CodeQL,
REUSE license or malware-scanning job is configured. `.github/dependabot.yml`
has weekly GitHub Actions and Docker updates, not npm/pip updates. Hosted
CodeQL setup, branch protection, required checks, Dependabot alerts and security
updates must be verified separately in GitHub settings.

The following is the **required validation policy**, not a claim that all gates
are wired into CI. Enabling missing gates or approving exceptions requires a
separate maintainer decision; moving this documentation does neither.

| ID | Acceptance requirement |
| --- | --- |
| CQ-1 | Existing build, type-check, tests, Ruff/format and container checks pass on code PRs; make `npm run lint` a required check. Do not merge failures. Exceptions need an owner, rationale and expiration. Build/test checks may be skipped for documentation-only PRs, but English and license checks still apply. |
| SAST-1 | Official CodeQL JavaScript/TypeScript and Python security suites complete on code PRs and publish SARIF. Analysis failure blocks merge. Triage findings; high/critical findings block absent an approved time-bounded exception. Keep queries/alerts current; local triage does not replace CI. |
| REUSE-1 | Run the official `reuse lint` on every PR, including documentation-only changes. Version-controlled files need valid SPDX license/copyright metadata or compliant license files/annotations; failures block merge. |
| MAL-1 | Scan repository content, shipped installed dependencies and actual deliverable artifacts with ClamAV or an equivalent scanner on every PR and before release. Update signatures; detections, scanner/update errors, stale signatures or unscanned deliverables block acceptance/release. Investigate false positives by file/hash, rationale and disposition, not global exclusions. |
| DEP-1 | Retain weekly Actions/Docker updates and add npm/pip coverage. Enable and verify Dependabot alerts/security updates. Dependency PRs pass CQ-1, REUSE-1 and MAL-1; prioritize high/critical alerts and record mitigations, owner and review date when an upgrade is unavailable. |

**REUSE-1 means REUSE.software license compliance**, not Lens's code-reuse
catalog or duplicate detection. `npm run reuse:*` does not replace `reuse lint`.

Maintainers must verify that required status checks enforce CQ-1, SAST-1,
REUSE-1 and MAL-1, and that releases scan the actual artifacts, not just source.
After scanner configuration changes, use an isolated standard detectable test
sample to confirm blocking behavior; never commit it or include it in releases.
Review alert backlogs, scanner/rule freshness, exception expiry and GitHub
required-check settings at least quarterly.

Keep current scan evidence and dispositions traceable to these five objective IDs.
If a repository threat model or SDL task list is adopted, map its specific IDs
to these cases. These requirements alone do not establish broader threat coverage;
old untriaged scan counts are not confirmed vulnerabilities or a current pass.

Preserve the upstream Apache 2.0 [license](LICENSE) and source copyright notices.
