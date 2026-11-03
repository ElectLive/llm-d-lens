---
name: deployment
description: Use when changing local startup scripts, Docker images or Compose, Kubernetes manifests, CI deployment configuration, or cluster deployment lifecycle behavior.
---

# Deployment

Follow [AGENTS.md](../../../AGENTS.md) for discovery,
verification and unresolved decisions. Paths below are repo-relative.

## Choose the relevant entry point

| Scope | Inspect before changing |
| --- | --- |
| Local service start/restart | `hack/lens-local.sh`, `hack/run-backend.sh`, `package.json` |
| Containers | The relevant Dockerfile under `deploy/` (built with the repository root as context) |
| Cluster provisioning/deployment | `apps/backend/src/llm_d_lens/cluster/`, `apps/backend/src/llm_d_lens/deploy/`, the manifest/template used by the caller |
| CI | The relevant workflow under `.github/workflows/` |

For backend deployment logic, also apply
[backend](../backend/SKILL.md). Reuse the existing
runtime composition, configuration and Kubernetes operations for the relevant
lifecycle; do not create a second deployment path without examining callers.

## Hardware-driven deployment

Runtime images, accelerator variants, device-plugin/DRA manifests, NFD/GFD/DCGM
and driver access modes come from the cluster's selected hardware profile
(`deployment.runtime_image`, `driver` modes/manifests, telemetry contributions)
via the hardware registry — never a hardcoded per-vendor image or manifest in a
deploy path. A new accelerator is a new profile/provider, not another
`if vendor …` branch in the deploy providers. Resolve through the same contract as
[backend](../backend/SKILL.md#hardware-profiles); when a value is unset, disable
the feature cleanly instead of substituting another vendor's default. See
[the AGENTS hardware rule](../../../AGENTS.md#hardware-is-profile-driven-never-hardcoded).

## Kubernetes transport and compatibility

Read the [Backend Kubernetes operating contract](../backend/SKILL.md#kubernetes-operations)
before changing cluster operations or backend-mode configuration, including its
coverage, credential/error boundaries and test-isolation rules.
With neither backend variable set, supported reads and writes use the official
SDK. `PRISM_KUBERNETES_BACKEND=cli` explicitly rolls back; the full backend setting
overrides the legacy read-only setting. Preserve this precedence.

Reuse the existing deployment runners and keep their scope/policy validation
before SDK or CLI execution. SDK failure is not permission to retry a write via
CLI. Client-side apply, Helm/Kustomize, exec, port-forward and other documented
exceptions retain CLI; keep their required tools installed. Do not substitute
server-side apply for client-side apply without addressing field ownership and
compatibility. The Backend skill owns the complete current boundary.

## Local development

The normal user entry is `./lens-local.sh restart` from `hack/`, equivalent to
`hack/lens-local.sh restart` from the repository root. Inspect `hack/lens-local.sh status`
and `$LENS_LOG_DIR/dev/` (default `~/.local/state/lens/logs/dev/`)
for service state and failures.

Use frontend hot reload for source/style edits. For running services that need a
restart, use `hack/lens-local.sh restart frontend|server|backend`; use `restart all`
for shared environment, port or API/MCP changes. Do not start the application
for tooling, tests or documentation-only work.

Startup validates the catalog/map and preserves pending decision warnings before
stopping services. It does not read historical source or run duplicate detection.
Invalid registrations still block restart. Run affected tests directly; startup
does not certify test results or produce an audit report.

For Docker-specific work, inspect actual service commands, mounts, ports and
runtime versions before choosing a launch command. The root image uses Node only
to build the frontend; its Python runtime serves the UI, API and MCP on one port.
`hack/lens-local.sh` requires Node 22 or newer for frontend development.
The MCP catalog is generated in Python from route metadata and packaged definitions;
`hack/generate-mcp-tools.mjs --check` validates it without a Node server artifact.
Do not assume a host Python environment or initialized database exists in a container.

## Storage paths in launchers and containers

Follow the [storage scheme](../../../docs/design/storage-layout.md). Shell launchers
reuse `hack/storage-env.sh`; Python and Node services use their shared path
helpers. Keep service environments and volume mounts consistent for shared files.
Resolve remote cache roots on the remote host rather than inserting the local
user's home path. Preserve explicit external source and model-volume inputs.
Changing configured paths neither migrates files nor switches running processes;
report those operations separately when they are part of the task.

## Validate proportionately

Use `bash -n` for changed shell scripts and build changed Dockerfiles
(`docker buildx build`), then relevant tests or image builds for the actual change.
Inspect persisted volumes, secrets, environment variables and rollback impact
before changing lifecycle behavior. A syntax check or image build does not prove
successful cluster deployment. State explicitly whether services were restarted
or a real cluster was exercised; execute those actions only within the user task.
