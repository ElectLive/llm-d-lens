# Lens Configuration API

The Configuration domain prepares immutable artifacts for deployment and
evaluation. The maintained [REST and Python API reference](../../../../../docs/site/i18n/en/docusaurus-plugin-content-docs/current/api-reference/modules/deployments.mdx#configuration-preparation-apis)
documents the concrete request and response models.

## Workflow and ownership

Users select a candidate source, supply or edit source configurations, and review
generated artifacts. Capability checking, normalization and validation are domain
operations rather than separate user-facing stages.

| Operation | Domain responsibility | Output |
| --- | --- | --- |
| `resolve_configurations` | Normalize each input, check model/hardware capability, then validate; retain a result per selected configuration | `CandidateConfig` plus errors and warnings |
| `render_configuration` | Validate candidate and supplied Guide resources, map role parameters and preserve source/bundle provenance | `DeployableConfiguration` |
| `save_configuration` | Check resource facts and checksums, write immutable artifact content and record its identity | Artifact ID and file metadata |
| Deployment | Consume the saved artifact, validate runtime requirements and manage Kubernetes resources | Execution, readiness, logs and cleanup |

The candidate source describes where a configuration came from, not its final
deployment type. Configuration owns input interpretation and publication; Deploy
owns Pod/Service lifecycle, teardown and rollback. Render and save alone do not
deploy a service. A file checksum protects bytes, not business semantics.

## Run locally

From the repository root, with the backend package and dependencies installed
from `apps/backend/pyproject.toml`:

```sh
python -m uvicorn llm_d_lens.api.main:app --host 127.0.0.1 --port 8090
```

The backend uses the following module boundaries:

- `llm_d_lens.api.main` owns the minimal FastAPI application and health endpoint.
- `configuration.router` owns HTTP request handling and error translation.
- `configuration.models` defines API contracts.
- `configuration.service` owns resolve, render, and persistence orchestration.
- `configuration.normalizers`, `validators`, and `capability` implement domain operations.

`llm_d_lens.configuration.app:app` remains available as a compatibility import.

Lens's unified Python entry serves configuration requests directly. Set the data
root with `LENS_DATA_DIR`; configuration payloads use `artifacts/configurations`
beneath it (default `~/.local/share/lens/artifacts/configurations`). The logical
file path returned to Deploy is `/configs/<artifact-id>/<file>`.

Resolve uses the in-process capability adapter in `llm_d_lens.aic` after
normalization and before validation. The adapter uses the pinned `aiconfigurator`
dependency and does not require a separate prediction service.

## Evaluation configuration integrity

The planner renders the selected Guide, applies explicit controls, and captures
the router inputs from the same resolved source. Render and save verify the
declared model, role topology, image, arguments, environment and storage against
the actual model-server resources. The deployment adapter repeats these checks;
recomputing a checksum cannot make contradictory configuration facts valid.
Unrecognized or compound shell invocations are rejected rather than rewritten.

`candidate_config.guide_settings` persists as `content.guideSettings`:

- `cacheCpuGiB`: CPU cache capacity per pod for Native CPU or LMCache CPU
  offload. Native capacity updates `cpu_bytes_to_use`; LMCache uses
  `LMCACHE_MAX_LOCAL_CPU_SIZE`. File-based LMCache settings require a compatible
  Guide source because an environment override cannot be assumed to win.
- `rdmaNicCount`: positive NIC count per pod for the P/D `vllm-rdma` variant.
  GPU TP, NIC counts and device selectors are independent.
- `routerValues`: YAML mapping merged over the Guide's Helm values. Precise
  routing binds tokenizer model identity and index block size to the model
  servers, and requires one EPP replica for its token-load accounting.

`render.deployment_bundle` persists as `officialGuide.deploymentBundle`. The
bundle records the resolved source commit, Helm chart/version, original and
effective router values, auxiliary resources, and calibration recipe where
required. Every embedded asset has a SHA-256 checksum. Deployment uses saved
inputs and records effective router values and installed manifest evidence;
calibration is recorded separately from the immutable requested configuration.

`GET /api/v1/configurations/artifacts/{artifact_id}/bundle` downloads these inputs
and the model-server YAML as a ZIP. The Evaluation YAML dialog provides the
download link. Older artifacts without a bundle must be regenerated to obtain
complete inputs. The standalone model-server YAML is not the entire Guide.

Official and remote sources must include the selected Guide's router recipes;
local sources must reside in a complete Guide checkout for those recipes to be
captured. Missing auxiliary sources fail generation instead of falling back to
an unrelated mutable local checkout. Filesystem offload remains unsupported.

LMCache option reference: [CPU RAM configuration](https://docs.lmcache.ai/kv_cache/storage_backends/cpu_ram.html).

## Maintaining AIConfigurator

The runtime SDK is pinned in `pyproject.toml` (currently `aiconfigurator==0.12.0`).
Its matching core package contains the support matrix and performance data.
Dependabot checks for AIC updates weekly and proposes a reviewable PR; it does
not automatically upgrade a running environment or merge the PR.

For an upgrade:

1. Review the [official releases](https://github.com/ai-dynamo/aiconfigurator/releases)
   and the target release's support matrix. Keep an exact released version in
   `pyproject.toml`; a passing matrix on `main` may not exist in a published wheel.
2. Install from the repository root with
   `.venv/bin/python -m pip install -e '.[embedded-db,ldap]'`.
3. Run `.venv/bin/python -m pip check` and
   `.venv/bin/python -m pytest llm_d_lens/aic llm_d_lens/configuration -q`.
4. Check both support flags and an actual candidate search for the target model,
   system, backend, GPU budget and workload. A support flag is not a deployment
   test and does not guarantee candidates for every budget or latency target.
5. Restart with `hack/lens-local.sh restart` and verify the running service. Record
   the installed SDK/core versions and search outcome in the upgrade PR.

Example smoke check (requires access to the model configuration, or a cached copy):

```python
from importlib.metadata import version
from llm_d_lens.aic.models import AICRequest
from llm_d_lens.aic.service import check_support_sync, search_sync

print({name: version(name) for name in ("aiconfigurator", "aiconfigurator-core")})
request = AICRequest(
    model_name="Qwen/Qwen3-0.6B",
    aic_system_name="b60",
    aic_backend_name="vllm",
    gpu_count=4,
    mean_input_tokens=1024,
    mean_output_tokens=256,
)
assert check_support_sync(request).disagg_supported
assert any(item["mode"] == "disagg" for item in search_sync(request).configs)
```

If validation fails, restore the prior dependency pin, reinstall with the same
command and restart. Do not edit the installed support CSV to force acceptance.
