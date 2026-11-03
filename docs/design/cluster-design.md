# Cluster Design

Cluster owns onboarding, bootstrap and provisioning orchestration. The wizard and bootstrap contracts share one design while retaining their distinct lifecycle stages. Database connection policy belongs to the database domain; shared Gateway publication belongs to model service.

This is the single canonical design for this module. Section numbers retain their
original part-local context; use the links below for unambiguous navigation.

| Part | Scope and status |
| --- | --- |
| [Creation wizard](#cluster-creation-wizard) | Onboarding, application database gate, versions and provisioning. |
| [Kubernetes bootstrap](#kubernetes-bootstrap) | Official-tool research, node inventory, Kubespray and SSH lifecycle. |

<!-- design-part: wizard -->

## Cluster creation wizard

- **Change directory:** `cluster-design.md`
- **Reference architecture:**
  - Frontend: `apps/frontend/src/pages/clusters/ui/CreateClusterWizard.jsx`
    and its `ClusterBootstrap/` components.
  - Backend: `llm_d_lens/cluster/` (registry/service/router),
    `llm_d_lens/storage/` (Storage volumes), `llm_d_lens/model_cache/` (Model Cache
    download job orchestration), `llm_d_lens/monitoring/cluster_stack/repo.py`
    (llm-d repository cache), `llm_d_lens/evaluation/router.py`
    (llm-d-benchmark repository cache, `_managed_benchmark_root`),
    `llm_d_lens/db/` (see
    `database-design.md`: the relational data access layer,
    with both external and embedded PostgreSQL connection modes).

---

### 1. Background and goals

The cluster wizard coordinates kubeconfig onboarding, proxy configuration,
version selection and optional model-cache preparation. Database setup is an
application-level prerequisite, not a database deployed into each target cluster.
The registry persists cluster metadata through the database domain.

The wizard composes existing storage, model-cache, repository-cache and
model-service capabilities. Detailed current API/UI contracts live in the
[Clusters API manual](../site/i18n/en/docusaurus-plugin-content-docs/current/api-reference/modules/clusters.mdx)
and [user guide](../site/i18n/en/docusaurus-plugin-content-docs/current/user-guide/clusters.mdx).

#### Goals

- The create-cluster wizard includes the following steps, in order:
  0. **Database setup** (shown only when the application has not yet been
     configured with a database, and only needs to succeed once globally):
     connect to an already deployed database, or choose to use embedded
     PostgreSQL.
  1. **Basic information** (same as today): name, description, kubeconfig file.
  2. **Network proxy**: `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`, with options to
     "use the default proxy" or "customize for this cluster".
  3. **Software versions**: llm-d and llm-d-benchmark version number or commit
     SHA; the backend prewarms the corresponding repository / install caches at
     creation time. Also pin the model-service data-plane version set (Gateway
     provider, GIE, Router, IPP) and install the shared components into the target
     cluster.
  4. **Model Cache** (storage required, prewarm optional): the user provides
     `HF_TOKEN`, Lens creates a Secret in the target cluster and always creates a
     Storage volume; optionally choose which models to prewarm (HuggingFace repo
     id + revision) and automatically trigger the download after creation.
  5. **Confirmation**: read-only summary, then submit.
- Afterward, modules such as Evaluate and Monitoring should, for the same cluster,
  use the llm-d / llm-d-benchmark versions pinned on that cluster **by default**,
  without requiring the user to override them via environment variables or request
  bodies each time.
- In the wizard, steps 2-3 are optional / skippable: skipping is equivalent to the
  current behavior (no proxy override, use global default versions), ensuring the
  existing minimal flow is not broken. Step 4 always creates a Storage volume;
  only the model prewarming remains optional.

#### Non-goals

- Do not implement "automatic discovery of the cluster's own proxy configuration"
  (for example probing a proxy ConfigMap in kube-system or node
  `/etc/environment`). The precise meaning of "use the cluster's own proxy" in
  this proposal is described in §4.2 (intentionally simplified and marked as an
  assumption pending confirmation).
- Do not introduce a new Model Catalog data source; the Model Cache step supports
  only HuggingFace (same scope as the current `model_cache` module;
  `ModelSourceKind.MODEL_CATALOG` remains a placeholder).
- Do not redesign the independent operation capabilities of the existing Storage /
  Model Cache pages—the wizard only orchestrates their "creation" into the
  create-cluster flow. Those two pages remain independently usable, and
  wizard-created Storage volumes and Model Cache entries are indistinguishable from
  manually created ones, sharing the same store/API.
- Step 0 (database setup) is **not one configuration per cluster**, and is not
  attached to any `Cluster` record: it is a global setting at the Lens backend
  process level. Once configured, it applies to all clusters and all future
  create-cluster wizards, and remains hidden thereafter (see §4.0). This proposal
  does not implement later operational capabilities like "migrate an existing
  database to another database", which belongs to
  `database-design.md`.
- Do not expose advanced database parameters in the wizard (pool size, read
  replicas, SSL certificate upload, etc.); Step 0 covers only the two minimal
  cases of "use embedded" or "provide a connection string". Advanced parameters
  continue to use environment variables (see
  `database-design.md` §4.1 such as
  `LLM_D_BENCH_DB_POOL_SIZE`).

---

### 2. Overall design

#### 2.1 Frontend: multi-step wizard

In `OptimizationClusterOverview.jsx`, replace the current single-form
`Modal(showCreate)` with a wizard modal with step indicators (visually based on
the `wizardStep` / step-dot implementation in `SubmitValidationPage.jsx`),
`size="lg"`:

```
When the application has not yet configured a database, opening the wizard first
shows a standalone prerequisite page (not counted in the step indicator):

┌───────────────────────────────────────────────────────────┐
│                    ⓪ Database setup (required the first time) │
├───────────────────────────────────────────────────────────┤
│   ○ Connect to an existing database (fill in connection info) │
│   ○ Use embedded PostgreSQL (no additional input required)    │
├───────────────────────────────────────────────────────────┤
│                                        [Test connection and continue] │
└───────────────────────────────────────────────────────────┘

After success (or when the application has already been configured, so Step 0 is
skipped directly), the normal five-step wizard is shown:

┌───────────────────────────────────────────────────────────┐
│  ① Basic info   ② Network proxy   ③ Software versions   ④ Model Cache   ⑤ Confirm  │
├───────────────────────────────────────────────────────────┤
│                     <current step form content>                       │
├───────────────────────────────────────────────────────────┤
│                              [Previous]  [Next / Create cluster]     │
└───────────────────────────────────────────────────────────┘
```

- Step 0 does not appear in the ①-⑤ step indicator and has no "skip" switch—it
  is a one-time **prerequisite gate**, not a configuration item of the cluster
  itself. When the application has already completed database setup, the wizard
  begins rendering directly from Step 1 and the user never sees Step 0 at all
  (see §4.0).
- The local form state of each step is merged into a single `wizardForm` object
  (instead of the current three fields `{ name, description, file }`, expanded to
  the full payload in §3).
- Steps 2-3 each have a "skip this step" switch at the top, off by default. When a
  step is skipped, its form fields are not rendered, not validated, and the
  submitted payload does not include the corresponding segment. Step 4 always
  creates a Storage volume; only its model prewarm switch is optional and off by
  default.
- After clicking "Create cluster", the wizard **does not close immediately**:
  instead, it enters a "creating" progress view (see the state machine in §5) and
  polls the cluster's provisioning status until all selected steps are completed
  or failed, and only then allows closing or retrying individual failed substeps.
  This differs from the current behavior (synchronous creation and immediate modal
  close), as detailed in §5.

#### 2.2 Backend: from "one-shot synchronous creation" to "creation + background orchestration"

Currently `POST /api/cluster/clusters` is synchronous: write metadata + kubeconfig
file, then immediately return 201. That step (§3 Step 1) remains synchronously
unchanged—it is inherently fast and is a prerequisite dependency for later steps
(without `cluster_id`, proxy configuration, Storage volume, and Model Cache
cannot proceed).

Steps 2-4 **cannot** be stuffed into the same synchronous request:

- Step 3 (software version prewarming) may trigger git clone + `pip install`
  (`_install_benchmark` already needs a 30-minute timeout today), so it must be
  asynchronous.
- Step 4 (Model Cache) is already asynchronous
  (`ModelCacheService.provision()` is a "fire-and-forget background task"), and
  depends on the Storage volume created in Step 4 becoming `READY` first
  (`storage/service.py::create_volume_objects` also does asynchronous
  `kubectl apply` + verification).

Therefore the create-cluster backend flow changes as follows:

1. `POST /api/cluster/clusters` (existing required fields remain: `name`,
   `description`, `file`; new optional fields are listed in §3) completes
   synchronously:
   - Write Cluster metadata + kubeconfig (unchanged from today).
   - **Synchronously** write fields submitted from Steps 2/3 (proxy config,
     llm-d/llm-d-benchmark versions) into the expanded Cluster metadata (these are
     only persisted configuration and do not involve network I/O).
   - If a model prewarm was requested, **synchronously** validate input
     (repo id format, non-empty token, etc.), but do not block waiting for the
     Storage volume to become ready or for the download to finish.
   - Return `202` (instead of today's `201`—semantically more accurate to express
     "background work still remains"), with a new `provisioning` section in the
     response body (see §5), which the frontend uses to decide whether to show the
     progress view.
   - When no optional steps are selected (equivalent to current behavior),
     `provisioning` is empty / `null`, and the frontend behaves exactly as it does
     today (close immediately after creation).
2. A new background task
   `llm_d_lens/cluster/provisioning.py::provision_cluster(cluster_id)`
   (`asyncio.create_task`, using the same fire-and-forget task pattern as the
   existing `deployment_run_manager` and `model_cache`) executes the selected
   substeps in order:
   a. Always: call `storage.service` to create a `local-disk` Storage volume (see
      §4.4 for the discussion of automatic configuration), and wait for it to
      become `READY`.
   b. If a model prewarm was requested: call the new
      `model_cache.jobs.create_pasted_token_secret()` to create a Secret in the
      target cluster using the user-provided `HF_TOKEN` (see §4.4), and then call
      existing `ModelCacheService.request_download()` + `provision()`.
   c. If software version pinning is enabled: call the new utility (see §4.3)
      `llm_d_lens/utils/repo_cache.py::ensure_cached(name, url, ref)` to prewarm
      caches for llm-d and llm-d-benchmark respectively.
   - Each substep updates the Cluster's provisioning status record (see §5), which
     the frontend polls.
3. The frontend polls the new endpoint
   `GET /api/cluster/clusters/{id}/provisioning` until all substeps are
   `ready`/`failed`.

---

### 3. Data model changes

#### 3.0 (new) `llm_d_lens/db/bootstrap_config.py`: application-level database configuration

This configuration describes **how the Lens backend process connects to the
database**, so it **cannot be stored in the database itself** (chicken-and-egg).
It is persisted in a small standalone file that exists before the database,
written atomically the same way as other stores (temporary file + `os.replace`),
at `~/.llm-d-bench/db-bootstrap.json` (overridable via
`LLM_D_BENCH_DB_BOOTSTRAP_FILE` to support containerized deployments mounting it
to a persistent volume):

```python
class DatabaseBootstrapConfig(StrictModel):
    mode: Literal["external", "embedded"]
    # Required when mode="external"; no redaction before or after writing to disk
    # (this file itself is the only source of "how to connect to the database"),
    # but it must never be echoed back verbatim through any read-only API; see §4.0.
    database_url: str | None = Field(default=None, alias="databaseUrl")
    configured_at: datetime = Field(default_factory=utcnow, alias="configuredAt")


class DatabaseStatus(StrictModel):
    """Response of `GET /api/v1/system/database`; only exposes whether it is
    configured and which mode, and never echoes `database_url`
    (which may contain a password)."""

    configured: bool
    mode: Literal["external", "embedded"] | None = None
    # In external mode, redacted display info (host/port/dbname), for read-only
    # hints such as "Connected to xxx:5432/llm_d_lens"; password/username do not
    # appear here.
    display_target: str | None = Field(default=None, alias="displayTarget")


class DatabaseSetupRequest(StrictModel):
    mode: Literal["external", "embedded"]
    database_url: str | None = Field(default=None, alias="databaseUrl")

    @model_validator(mode="after")
    def _require_url_for_external(self):
        if self.mode == "external" and not (self.database_url or "").strip():
            raise ValueError("external mode requires databaseUrl")
        if self.mode == "embedded" and self.database_url:
            raise ValueError("embedded mode must not include databaseUrl")
        return self
```

This section is independent from the rest of §3.1-3.2: `DatabaseBootstrapConfig`
is not attached to `Cluster` / `ClusterRecord`, and does not appear in the
`POST /api/cluster/clusters` payload (see Non-goal #4 in §1).

#### 3.1 `llm_d_lens/cluster/registry.py::Cluster`

Currently `Cluster` is a frozen dataclass with only four fields. Expand it to:

```python
@dataclass(frozen=True)
class Cluster:
    id: str
    name: str
    description: str
    created_at: str
    proxy: ProxyConfig | None = None  # new; see below
    llm_d_ref: str | None = None  # new: llm-d version or commit SHA
    llm_d_lensmark_ref: str | None = None  # new: llm-d-benchmark version or commit SHA
    data_plane: DataPlaneVersions | None = (
        None  # new: model-service data-plane version set
    )
```

```python
@dataclass(frozen=True)
class DataPlaneVersions:
    """Pinned version set for the model-service data plane; components are
    compatibility-constrained, so they are selected as a group."""

    gateway_provider: str | None = None  # istio | gke | agentgateway | envoy-ai-gateway
    gateway_version: str | None = None
    gie_version: str | None = None  # GAIE / Gateway API CRD release
    router_version: str | None = None  # llm-d Router (EPP / Proxy) chart or image
    ipp_version: str | None = None  # llm-d-inference-payload-processor chart or image
```

```python
@dataclass(frozen=True)
class ProxyConfig:
    mode: Literal["auto", "custom"] = "auto"
    # When mode="custom", the following three are the effective values (user-entered).
    http_proxy: str | None = None
    https_proxy: str | None = None
    no_proxy: str | None = None
    # When mode="auto", the following are a detected snapshot (read-only, may be
    # re-detected and overwritten at any time); on detection failure all three are
    # None and detection_error is non-empty.
    detected_http_proxy: str | None = None
    detected_https_proxy: str | None = None
    detected_no_proxy: str | None = None
    detected_at: str | None = (
        None  # ISO timestamp of the most recent successful/failed detection
    )
    detection_error: str | None = None  # human-readable reason when detection fails
```

- `mode="auto"`: the detection flow in §4.2 writes `detected_*` /
  `detected_at` / `detection_error`; the runtime effective values use
  `detected_http_proxy` and friends (if detection fails or they are empty, this is
  treated as "no proxy", and no longer falls back to the Lens backend process's
  own environment variables—that was the old design. "auto" is now explicitly
  "detect this target cluster", not "inherit the backend process").
- `mode="custom"`: ignore `detected_*`; runtime effective values use the
  manually entered `http_proxy` etc.
- Add corresponding keys to the metadata JSON and expand
  `Cluster.from_dict` / `to_dict` accordingly. When old files lack these keys,
  treat them all as `None` / `mode="auto"` and "not yet detected", which is
  backward-compatible without a migration script. When first reading an old
  cluster, the `ready` / overview APIs could opportunistically trigger one
  backfill detection, or we could simply require the user to click "re-detect
  proxy" once in the cluster detail page, avoiding silently stuffing a potentially
  slow I/O operation into an endpoint that was originally just a query—the latter
  is preferable, and better matches the current code style where "detection-like
  operations are explicit endpoints", such as `cluster_is_ready()` being
  explicitly awaited inside `list_clusters` / `overview`, but that is only a
  single `--raw=/readyz` probe on a different scale.

#### 3.2 `llm_d_lens/cluster/models.py`

```python
class ProxyConfigDTO(StrictModel):
    mode: Literal["auto", "custom"] = "auto"
    http_proxy: str | None = Field(default=None, alias="httpProxy")
    https_proxy: str | None = Field(default=None, alias="httpsProxy")
    no_proxy: str | None = Field(default=None, alias="noProxy")
    detected_http_proxy: str | None = Field(default=None, alias="detectedHttpProxy")
    detected_https_proxy: str | None = Field(default=None, alias="detectedHttpsProxy")
    detected_no_proxy: str | None = Field(default=None, alias="detectedNoProxy")
    detected_at: str | None = Field(default=None, alias="detectedAt")
    detection_error: str | None = Field(default=None, alias="detectionError")


class HfTokenSecretCreateRequest(StrictModel):
    """Standalone "create HF Token Secret" request, not attached to a specific
    Model Cache entry."""

    namespace: str = Field(min_length=1, max_length=253)
    # Default name includes a short UUID suffix to avoid name collisions; users
    # can also provide their own, in which case it must be unique within the
    # namespace.
    name: str = Field(
        default_factory=lambda: f"hf-token-{uuid4().hex[:8]}", max_length=253
    )
    token: str = Field(min_length=1)


class HfTokenSecretRef(StrictModel):
    namespace: str
    name: str


class ModelCacheSeedRequest(StrictModel):
    """Step 4 payload; submitted together with the create request only when the
    user did not skip this step.

    Storage volumes no longer have an "auto-create" default capacity/path—
    `storage` is either an existing volume id, or a fully completed
    `StorageVolumeCreateRequest` entered manually in the wizard
    (exactly the same fields as creating one manually on the Storage page; see 4.4).
    """

    hf_token_secret: HfTokenSecretCreateRequest = Field(alias="hfTokenSecret")
    repo_id: str = Field(
        alias="repoId", pattern=_REPO_ID_PATTERN
    )  # reuse model_cache regex
    revision: str = Field(default="main")
    storage: (
        str | StorageVolumeCreateRequest
    )  # existing volume id, or complete create-volume form


class ClusterRecord(StrictModel):
    id: str
    name: str
    description: str
    created_at: str = Field(alias="createdAt")
    ready: bool = False
    proxy: ProxyConfigDTO = Field(default_factory=ProxyConfigDTO)  # new
    llm_d_ref: str | None = Field(default=None, alias="llmDRef")  # new
    llm_d_lensmark_ref: str | None = Field(
        default=None, alias="llmDBenchmarkRef"
    )  # new
    data_plane: DataPlaneVersionsDTO | None = Field(
        default=None, alias="dataPlane"
    )  # new


class DataPlaneVersionsDTO(StrictModel):
    """Pinned version set for the model-service data plane; selected as a group, see 4.3."""

    gateway_provider: str | None = Field(default=None, alias="gatewayProvider")
    gateway_version: str | None = Field(default=None, alias="gatewayVersion")
    gie_version: str | None = Field(default=None, alias="gieVersion")
    router_version: str | None = Field(default=None, alias="routerVersion")
    ipp_version: str | None = Field(default=None, alias="ippVersion")


class ClusterProvisioningStep(StrictModel):
    """Status of a provisioning substep, used by the frontend to render the progress view."""

    id: Literal[
        "proxy-detection",
        "storage",
        "model-cache",
        "llm-d-repo",
        "llm-d-benchmark-repo",
        "data-plane",
    ]
    status: Literal["skipped", "pending", "running", "ready", "failed"] = "skipped"
    detail: str | None = (
        None  # human-readable current action, e.g. "cloning llm-d@<sha>"
    )
    failure_detail: str | None = Field(default=None, alias="failureDetail")


class ClusterProvisioningStatus(StrictModel):
    cluster_id: str = Field(alias="clusterId")
    steps: list[ClusterProvisioningStep]

    @property
    def done(self) -> bool:
        return all(step.status in ("skipped", "ready", "failed") for step in self.steps)
```

`POST /api/cluster/clusters` adds the following form fields (still
`multipart/form-data`, because kubeconfig is a file upload):

| Field | Type | Description |
|---|---|---|
| `proxy` | JSON string (`{"mode": "auto"}` or `{"mode": "custom", "httpProxy": ..., ...}`) | Optional; default `{"mode": "auto"}`; when `mode="auto"`, the background task triggers one detection after creation (§4.2) |
| `llmDRef` | string | Optional |
| `llmDBenchmarkRef` | string | Optional |
| `dataPlane` | JSON string (`DataPlaneVersionsDTO`) | Optional; pinned version set for the model-service data plane (provider / GIE / Router / IPP), see §4.3 |
| `modelCacheSeed` | JSON string (`ModelCacheSeedRequest`) | Optional; present means Step 4 was not skipped |

The response body expands to:

```json
{
  "cluster": { "...ClusterRecord..." },
  "sessionId": "…",
  "provisioning": { "clusterId": "…", "steps": [ /* ClusterProvisioningStep[] */ ] }
}
```

When `provisioning` is `null`, it means there are no background substeps
(equivalent to today's behavior).

New endpoints for Step 0 (fully independent from `POST /api/cluster/clusters`, with
no `clusterId`; see §4.0):

| Endpoint | Description |
|---|---|
| `GET /api/v1/system/database` | Returns `DatabaseStatus`; the wizard calls this first when it opens to decide whether to render Step 0 |
| `POST /api/v1/system/database` | Request body `DatabaseSetupRequest`; on success, persist `DatabaseBootstrapConfig`, rebuild the database engine, and run one Alembic migration, then return `DatabaseStatus`; on failure (for example external mode cannot connect), return 409/422 + human-readable reason, and Step 0 remains visible |

---

### 4. Detailed per-step design

#### 4.0 Step 0: database setup (first-time global gate, unrelated to any specific cluster)

> **Update (subsequent change):** This section originally designed Step 0 as a
> prerequisite page inside the "Create cluster" wizard. It has now been changed to
> be completed once in the installer script
> `scripts/LensInstaller-Ubuntu-x86_64-online.sh` (new
> `configure_database` / `db_interactive_configure` steps; silent install
> `--silent` defaults directly to embedded PostgreSQL, while interactive install
> prompts for the same fields described here), and no longer appears in the
> wizard. Reason: database connection is an operational decision about **how the
> entire backend process connects to its own database**, and asking once during
> installation better matches the user's mental model than checking "is it
> configured?" every time the wizard opens. The `POST /api/v1/system/database`
> API itself (together with the probing/migration/persistence behavior described
> below) **remains unchanged**; only the caller changes from the React frontend to
> the new `llm_d_lens/db/bootstrap_cli.py` (both share the same
> `apply_database_setup()` implementation). `GET /api/v1/system/database` also
> remains, for operational status queries. `CreateClusterWizard.jsx` now only
> performs a read-only "is it configured?" check when opened, and if not
> configured, prompts the user to rerun the installer script instead of rendering
> the form described below. The body below is retained as a design record of that
> earlier form; the fields/validation logic were moved unchanged into the
> installer script.

**Trigger condition:** when the wizard modal opens, first call
`GET /api/v1/system/database`; only when `configured: false` should Step 0 be
rendered. If `true`, jump directly to Step 1 (the user notices nothing). This
check runs every time the wizard opens (rather than only once and caching it in
frontend local state), because `DATABASE_URL` / `LLM_D_BENCH_DB_MODE` may also
already be configured through backend startup environment variables (see
`database-design.md` §4.1); that should likewise count as
"already configured" and skip Step 0.

**Form:** two mutually exclusive radio options; the corresponding input area
expands only after selection:
1. **Connect to an existing deployed database**: first choose the database engine
   (PostgreSQL / MySQL / Oracle / SQL Server dropdown, corresponding to
   `settings.py::SUPPORTED_EXTERNAL_ENGINES`), then fill in five separate fields:
   Host, Port (default filled based on engine, e.g. PostgreSQL 5432 / MySQL 3306 /
   Oracle 1521 / SQL Server 1433, editable), Database name, Username, Password—
   the user is **not** required to directly enter / assemble a full connection
   string. The password field is masked with `type="password"`. Before
   submission, the frontend assembles these five fields into the single DSN format
   already defined by the backend (`******host:port/dbname`, with user/password
   URL-encoded before assembly), then submits that together with the engine to
   `POST /api/v1/system/database`—the backend API itself remains unchanged and
   still only accepts / validates a complete `databaseUrl` (consistent with the
   `DATABASE_URL` format in `database-design.md` §4.1); the
   assembly logic happens only in the frontend. The selected engine must match the
   DSN scheme (`engine://user:password@host:port/dbname`) in the assembled URL (validated by backend `DatabaseSetupRequest`,
   422 on mismatch). **Only PostgreSQL is end-to-end verified** (migration + DAO
   layer fully covered); MySQL / Oracle / SQL Server currently have only
   "UI + driver"-level experimental support—they may establish a connection, but
   some migrations use PostgreSQL-specific types (such as `sa.ARRAY`), so
   `alembic upgrade head` may fail on those engines. When a non-PostgreSQL engine
   is selected, the wizard shows a prominent experimental-support warning.
2. **Use embedded PostgreSQL**: no input is required; selecting it is sufficient
   to click submit (corresponding to the embedded mode in
   `database-design.md` §4.3, where `pgserver` automatically
   runs `initdb` under `~/.llm-d-bench/embedded-pg/data`). Embedded mode is
   **always PostgreSQL** (`pgserver` only provides that engine), and does not
   offer engine selection.
   `database-design.md` §4.3 embedded mode, where `pgserver`
   automatically `initdb` to `~/.llm-d-bench/embedded-pg/data`). Embedded mode is
   **always PostgreSQL** (`pgserver` only provides that one engine) and offers no
   engine selector.

**Submission behavior** (`POST /api/v1/system/database`, see the API table in
§3.2):

- External mode: the backend first performs a connectivity probe (`SELECT 1`).
  After success, persist `DatabaseBootstrapConfig` and run
  `alembic upgrade head`; if the probe fails (network unreachable / authentication
  failure / invalid DSN / missing driver package for the engine), return 422/409 +
  the specific reason (when the driver is missing, include a hint such as
  `pip install llm-d-lens[mysql|oracle|mssql]`). Step 0 remains visible
  and does not allow proceeding; the user can correct the connection string and
  retry, or switch to embedded mode.
- Embedded mode: call `pgserver.get_server(data_dir)` (initializing with `initdb`
  on first use), obtain the DSN, then likewise run one migration. This typically
  completes within a few seconds, so the frontend only needs a brief loading
  state and does not need the asynchronous Provisioning state machine in §5
  (the database must be synchronously ready before the wizard can continue; there
  is no meaningful intermediate state such as "allow failure and retry later").
- After success in either mode, the application-level database is considered
  configured. Thereafter `GET /api/v1/system/database` returns
  `configured: true`, and Step 0 is **permanently skipped** (unless operations
  manually delete `db-bootstrap.json` or change environment variables—which is an
  operational action outside the scope of this wizard).

**Relationship to the cluster itself:** only after Step 0 succeeds does Step 1
`registry.create_cluster()` actually write to the database for the first time
(rather than today's local JSON files). But this change is completely transparent
to the rest of Steps 1-5: data models such as `Cluster` / `ClusterRecord`
(§3.1-3.2) do not need to know whether the underlying storage is files or a
database; the DAO layer absorbs that difference (see
`database-design.md` §6.3).

**Concurrency / multi-replica considerations:** if multiple backend replicas start
for the first time simultaneously and receive Step 0 submissions at the same time,
persist `db-bootstrap.json` using the same "temporary file + `os.replace`"
atomic-write strategy as other stores; `alembic upgrade head` itself is idempotent
(running it twice does not create tables twice), so even if two replicas both
submit once, data corruption will not occur. At worst, one overwrites the other's
`configured_at` timestamp, which is acceptable because under normal operation both
submissions should use the same mode/connection string.

#### 4.1 Step 1: basic information (unchanged from current behavior)

Name, description, kubeconfig file; after submission, immediately persist with the
existing `registry.create_cluster()` to obtain `cluster_id` for later steps.
**Inside the wizard**, the create API is actually called when the user clicks
"Next" to leave Step 1 (rather than waiting until the final "Confirm" step),
because Steps 2-4 all require an existing `cluster_id` (especially because
kubeconfig must be persisted first before Step 4's Storage / Model Cache can issue
kubectl commands to the target cluster). If the user cancels midway through Steps
2-4, the wizard must call `DELETE /api/cluster/clusters/{id}` to roll back
(reusing the current "delete cluster" logic; since there is not yet any Storage
volume at this point, the new "cannot delete a cluster with Storage underneath it"
check added in §8.4 will not block this rollback).

> This is an intentional design tradeoff: splitting "create" into "immediately
> create in Step 1 + Step 2-4 are follow-up PATCH operations", rather than
> "collect the full wizard form and submit once at the end". Reason: kubeconfig
> validation itself may fail (file format error, cannot connect to cluster), and
> the user should get that feedback in Step 1 rather than after filling out all
> four steps only to discover the very first and most fundamental one was invalid.

#### 4.2 Step 2: network proxy

**Semantics clarification (confirmed with the requester):** selecting "use the
cluster's own proxy" means **automatically detecting the proxy configured on the
target Kubernetes cluster itself**, not "inherit the environment variables of the
Lens backend process" (that assumption in the old draft was wrong and has been
discarded).

Form fields:

```
( ) Automatically detect this cluster's proxy (recommended)
    [Detected] HTTP_PROXY=http://10.x.x.x:3128  HTTPS_PROXY=...  NO_PROXY=...
             (detected at 2024-xx-xx 12:00, [Detect again])
    or
    [Detection failed] Could not detect proxy configuration from the cluster: <reason>.
             Will run as "no proxy";
             if a proxy is actually required, switch to "Custom" and fill it in manually.
( ) Custom proxy for this cluster
    HTTP_PROXY  [__________]
    HTTPS_PROXY [__________]
    NO_PROXY    [__________]
```

**Detection mechanism:** Kubernetes itself has no first-class concept of a
"cluster-level proxy" (proxy is usually runtime configuration of each node's
containerd/docker/kubelet, or system-level environment variables in
`/etc/environment`), so detection can only "guess using a method that can read
node/system configuration". Proposal:

1. Submit a one-off **diagnostic Job** to the target cluster (`kind: Job`,
   `restartPolicy: Never`, `activeDeadlineSeconds` with a short timeout such as
   60s; reusing the existing "temporary Job + poll + cleanup" pattern in
   `deploy/runtime`). The Job's single container does two things and writes the
   result to container stdout (the backend reads the result from `kubectl logs`,
   so no extra API is required):
   - Directly read the `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` environment
     variables inherited by the container, plus lowercase variants (in many
     clusters kubelet propagates node-level proxy settings into containers, which
     is the most common and highest-success-rate path).
   - If those are empty, attempt a read-only hostPath mount of `/etc/environment`
     (Debian/Ubuntu) and `/etc/profile.d/proxy.sh` (if present) and use regex
     extraction as a fallback.
2. **Known limitations (documented explicitly, not hidden):**
   - If the cluster enforces stricter PodSecurity Admission (`restricted`
     profile), hostPath mounts may be rejected outright. In that case the
     diagnostic Job only has the "container inherited environment variables" path;
     if that produces nothing, then it simply produces nothing. That should be
     treated as "proxy not detected" (not an error, but one reasonable result
     indicating the cluster may not have a proxy).
   - The detection result is only a point-in-time snapshot and is not
     continuously synchronized; if the cluster-side proxy configuration changes,
     the user must manually click "Detect again" to refresh it (no periodic
     polling; the cost/benefit is not worth it).
   - When detection fails or finds nothing, **do not fall back to any other
     source** (do not read the Lens backend's own environment variables). In the
     semantics of "automatic detection", only the detection result is trusted; if
     nothing is found, it means "this cluster has no proxy", avoiding confusing
     results like "detection found A, but fallback mixed in backend process B".
   - If the user believes detection is wrong or the environment is unusual, they
     can switch directly to "Custom" and fill values manually. This is the final
     fallback that guarantees the feature remains usable even if detection success
     rate is low.

**How it takes effect at runtime:** after saving `ProxyConfig` via
`PATCH /api/cluster/clusters/{id}` (when `mode="auto"`, the background
Provisioning task triggers one detection and writes back `detected_*`; when
`mode="custom"`, the manual values take effect directly),
`llm_d_lens/utils/kubernetes.py::kubeconfig_environment()` and
`model_cache/jobs.py::_proxy_environment()` are changed to preferentially read the
"effective values" of that cluster's `ProxyConfig` (`custom` uses the manually
entered values; `auto` uses the `detected_*` values; when all are empty, no
proxy-related environment variables are set), no longer reading `os.environ`, so
that:

- Model Cache download Jobs (`jobs.py`)
- Pods launched by the Deploy side
  (`deploy/runtime/composition.py::_model_environment`, which currently also reads
  proxy values from `os.environ`, and is likewise migrated to "read the effective
  value from this cluster's `ProxyConfig`")
- Any future in-cluster workload that needs to access the public internet

all uniformly follow the rule "cluster-level proxy configuration; if absent, fall
back to the process-level default", instead of, as today, each module scattering
its own direct reads of `os.environ`.

#### 4.3 Step 3: software versions (llm-d / llm-d-benchmark / data plane)

Form fields:

```
llm-d version/commit         [main ▾]  (dropdown of common tags + free-form commit SHA)
llm-d-benchmark version/commit [87d03da…] (default = current _BENCHMARK_REVISION)

── Model-service data plane (pinned version set per cluster; optional) ──────────
Gateway provider             [istio ▾]  (istio | gke | agentgateway | envoy-ai-gateway)
Gateway / provider version   [1.29.4]
GIE / Gateway API CRD        [v1.6.2]
LeaderWorkerSet              [v1.0.0]  (DisaggregatedSet API for P/D)
llm-d Router version         [v1.0.0]  (EPP / Proxy chart or image)
IPP version                  [v1.0.0]  (payload-processor chart or image)
```

**Backend changes (reuse + generalize the two existing repository caches rather
than inventing a new one):**

1. Add `llm_d_lens/utils/repo_cache.py`, extracting the reusable parts of
   `_managed_benchmark_root` / `_install_benchmark` from `evaluate/router.py`
   (clone → checkout → optional build step → file lock against concurrent
   duplicate installs):

   ```python
   async def ensure_cached(
       *,
       name: str,
       repository: str,
       ref: str,
       build: Callable[[Path], None] | None = None,
   ) -> Path:
       """Return <cache_root>/<name>/<key(repository, ref)>. If missing, do
       clone+checkout (+ optional build); use file locks to prevent duplicate
       concurrent installation. `name` separates cache buckets for llm-d,
       llm-d-benchmark, etc."""
   ```

   `cache_root` defaults to `~/.cache/llm-d-bench/repos` (consistent with the
   existing `~/.cache/llm-d-bench/*` directory family). The existing Evaluate path
   `~/.cache/llm-d-bench/backends/<hash>` can be converged into
   `repos/llm-d-benchmark/<ref>` in the next major version. In the first version
   of this proposal, keep Evaluate unchanged and only let new code use the new
   utility function, to avoid an overly broad change to stable existing paths.
2. Change `monitoring/cluster_stack/repo.py` to accept a `ref` parameter (it
   currently hardcodes `--branch main`); the default remains `main` (preserving
   current behavior), and when `ref` is provided use
   `ensure_cached(name="llm-d", repository=_LLM_D_REPO_URL, ref=ref)`.
3. Keep `evaluate/router.py::_managed_benchmark_root` unchanged in signature and
   behavior (avoiding impact to currently running evaluate tasks and their
   existing `~/.cache/llm-d-bench/backends` path), but add a resolution priority:
   if the request does not explicitly pass repository/revision, and the current
   cluster identified by `cluster_session_id` has `llm_d_lensmark_ref`, use that
   as the default. Today's priority is "request body > environment variable
   default"; change it to "request body > cluster default > environment variable
   default".
4. When Step 3 is not skipped, the Cluster creation wizard's background task calls
   `ensure_cached(...)` once for llm-d and once for llm-d-benchmark (prewarm only,
   not immediate use), moving the "invisible backend wait" earlier to cluster
   creation rather than making the user wait for a multi-minute git clone + pip
   install the first time they click "run evaluate".

**Data-plane components (different from the two repositories above: installed
into the target cluster, not cached as local source):**

The model-service data plane (Gateway/Proxy + IPP + InferencePool + EPP) needs a
pinned version per cluster, but these are **Helm charts / images installed into the
target cluster**, not source trees Lens uses locally, so the `ensure_cached`
clone-source approach does not apply:

- **Version set**: `data_plane` (`DataPlaneVersionsDTO`) records the provider, GIE,
  Router, and IPP versions. These components are compatibility-constrained
  (IPP ↔ Router ↔ GIE ↔ provider ↔ Kubernetes); validate them **as a group** and
  reject any mismatch instead of allowing independent free choice.
- **Shared components installed at cluster creation**: Gateway API / GAIE CRDs,
  Gateway provider, Gateway, and IPP are installed during cluster creation (new
  provisioning substep `data-plane`), so publishing a model later only adds
  InferencePool / HTTPRoute / IPP config.
- **Model-level components installed at publish time**: InferencePool, EPP,
  HTTPRoute, and the IPP model-mapping ConfigMap are rendered and installed by the
  model-service Reconciler at the cluster's pinned versions.
- **Compatibility check**: at cluster creation, run a preflight over provider vs.
  Kubernetes version, GIE CRD version, and Router/IPP versions; on failure return
  the missing items and a compatible version set rather than silently continuing.
- **Idempotency**: like the existing provisioning substeps, shared components that
  are already installed at the same version are not reinstalled; version changes
  upgrade as needed.

#### 4.4 Step 4: Model Cache (storage required, prewarm optional)

Form fields (**corrected after requirement clarification: there is no
"automatic/default" behavior at all; the Storage portion is a fully manual form,
exactly the same as creating a volume on the Storage page**):

```
[x] Prewarm a model cache for this cluster now

── HuggingFace Token ──────────────────────────────────
Namespace    [dropdown listing all namespaces in this cluster]
Secret name  [hf-token-a1b2c3d4]  (defaults to a short-UUID suffix to avoid name
                                   collisions; editable;
                                   validation: no Secret with the same name may
                                   already exist in the selected namespace)
Token value  [__________________]  (password input)

── Model ────────────────────────────────────────────────
repo id      [meta-llama/Llama-3.1-8B-Instruct]
revision     [main]

── Storage location (required) ────────────────────────────
A new Storage volume is always created — the following is exactly the same as
the Storage page's "New volume" form, with no default values; the user must fill
out everything manually:
    kind          ( ) local-disk  ( ) nfs  ( ) ... (same as StorageVolumeCreateRequest)
    capacity      [__________]  (required, no default)
    hostPath      [__________]  (required when kind=local-disk, no default)
    nodeSelector  [__________]  (expand corresponding fields by kind; copied one
                                 by one from the existing form)
    readOnly      [ ]
    purposes      [ ] model-cache  [ ] ...
```

**Dependencies and implementation approach:**

- **HF Token Secret is no longer a new `TokenSourceMode` inside Model Cache**.
  Instead, it is split into an independent, generic action: "create a Secret
  containing `HF_TOKEN` in the user-specified namespace". Add a new endpoint that
  is not attached to any specific Model Cache entry:

  ```
  POST /api/cluster/clusters/{cluster_id}/hf-token-secrets
  Body: HfTokenSecretCreateRequest { namespace, name?, token }
  Resp: HfTokenSecretRef { namespace, name }
  ```

  The implementation directly reuses the manifest structure of
  `deploy/runtime/composition.py::create_model_secret_from_token()`
  (`stringData: {HF_TOKEN: token}`), but changes it to explicitly
  **"create if absent, 409 if already exists"** (either by checking once with
  `kubectl get secret <name> -n <ns>` first, or directly using
  `kubectl create secret`—which already naturally means "error if exists", and
  avoids silent overwrite unlike the current Deploy logic with `kubectl apply`).
  The duplicate-name validation can reuse the same "list all secrets across
  namespaces" call as `cluster/service.py::list_model_secrets()`, or it can issue
  a direct precise `kubectl get secret <name> -n <namespace>` (more efficient;
  implementation can choose). The wizard's default name is
  `hf-token-<uuid4().hex[:8]>`; if the user edits it, validate again.
- After the Secret is created, the Model Cache entry continues to use the existing
  `TokenSourceMode.EXISTING_SECRET` (`namespace` / `name` pointing to the newly
  created Secret), fully reusing the mature existing path that already supports
  cross-namespace `copy_model_secret()`. **No new `TokenSourceMode` enum value is
  needed.** The wizard
  simply "creates a Secret from the user's input first, then passes it into the
  Model Cache request as an existing Secret", so the semantics, security model,
  and persisted format all fully reuse the current implementation, with no new
  special branch to maintain.
- Storage volumes: there is **no automatic creation / no default values**. The
  user either selects an existing READY volume, or fills out the entire manual
  volume-creation form from the Storage page in the wizard
  (`StorageVolumeCreateRequest` copied verbatim, with fields, validation rules,
  and defaults—including the fact that there are no defaults—kept completely in
  sync with the Storage page, rather than inventing a second variant). The wizard
  background task calls `storage.service.StorageVolumeService.create()` under the
  `cluster_id` created in Step 1, obtains the resulting `volume_id`, and only
  continues after it becomes `READY`.
- Regardless of whether the user "chooses an existing volume" or "creates a new
  one", after the Storage volume is `READY`, the background task reuses the
  existing `ModelCacheService.request_download()` + `provision()`, exactly the
  same code path used when the user manually starts a download from the Model
  Cache page today. The only difference is that the initiator is the wizard's
  background task rather than a button click by the user.

#### 4.5 Step 5: confirmation

Read-only summary of the choices in Steps 1-4 (skipped steps show as "not
enabled"), with a "Create cluster" button. If the cluster was already created
mid-wizard in Step 1 (see §4.1), clicking this step actually sends
`PATCH /api/cluster/clusters/{id}` (persist Step 2/3 config and start Step 4
background work), rather than another `POST`.

---

### 5. Provisioning state machine

```
                 ┌─────────┐
   (step not enabled) │ skipped │
                 └─────────┘

   (enabled)    ┌─────────┐    ┌─────────┐    ┌───────┐
                │ pending │ →  │ running │ →  │ ready │
                └─────────┘    └─────────┘    └───────┘
                                     │
                                     ▼
                                ┌────────┐
                                │ failed │ ──(user may retry individual substep)──┐
                                └────────┘                                      │
                                     ▲────────────────────────────────────────────┘
```

- The six substeps (`proxy-detection`, `storage`, `model-cache`, `llm-d-repo`,
  `llm-d-benchmark-repo`, `data-plane`) advance their states independently.
  `model-cache` depends on `storage` reaching `ready` first (if `storage` becomes
  `failed`, `model-cache` is marked `failed` directly, and `detail` explains the
  reason without duplicating the error). `proxy-detection` is non-`skipped` only
  when `mode="auto"`, and it is a best-effort probe: even if detection fails, the
  result is still marked `ready` +
  `detail="proxy not detected; proceeding without proxy"`; this **does not count
  as `failed`** (not being able to detect a proxy is a valid final state and
  should not delay/block completion of the wizard). `data-plane` is non-`skipped`
  only when a data-plane version set is selected in Step 3; on failure it keeps the
  already-installed parts and reports the missing items.
- `GET /api/cluster/clusters/{id}/provisioning` is used by the frontend for
  polling (reusing the polling interval conventions from the Model Cache page).
  Once all substeps enter terminal states (`skipped` / `ready` / `failed`), the
  frontend stops polling.
- An individual `failed` substep can be retried independently
  (`POST /api/cluster/clusters/{id}/provisioning/{step_id}/retry`) without
  rerunning the entire wizard—for example, if Model Cache download fails due to a
  transient network issue, the user retries only that part; "detect again" for
  `proxy-detection` reuses the same retry endpoint.
- The wizard modal can be minimized/closed before all substeps reach terminal
  states (closing the modal does not cancel background tasks), and the user can
  reopen "view creation progress" from the cluster list page at any time. This
  matches the user expectation already established by the Model Cache page, where
  closing the page does not stop the background download Job.

---

### 6. Persistence ownership

| Data | Owner and storage contract |
| --- | --- |
| Cluster metadata and kubeconfig | `ClusterDao` and the `clusters` table; local kubeconfig files are materialized caches, not the source of truth |
| Storage volumes and model-cache entries | Their domain DAOs; wizard actions reuse those services |
| HF token | Secret in the target cluster; cluster metadata stores its namespace/name reference |
| Repository checkouts | Shared repository-cache helpers and the [storage layout](storage-layout.md), with repository/revision isolation |
| Data-plane configuration | Cluster metadata records the selected versions; shared resources live in Kubernetes |
| Provisioning progress | Owning operation lifecycle; do not add a second persistence mechanism in the wizard |
| Application database setup | Database bootstrap configuration, separate from managed-cluster records |

See the [database design](database-design.md) for relational boundaries.

### 7. Confirmed design decisions

1. **Exact semantics of "use the cluster's own proxy":** **automatically detect it
   from the target cluster**, not "inherit the Lens backend process's environment
   variables". Implemented as a diagnostic Job detection flow; see §4.2.
2. **Where Provisioning status is stored:** standalone file
   `<clusters_dir>/<id>.provisioning.json` (last row in §6), not merged into
   Cluster metadata.
3. **Default capacity/hostPath for automatically created Storage volumes:** **no
   defaults at all**. The Storage portion of Step 4 is simply the manual
   create-volume form from the Storage page copied into the wizard, and the user
   must fill in all fields themselves (capacity, hostPath, etc.); the wizard does
   not provide a "one-click default" option.
4. **Secret namespace and naming for `TokenSourceMode.PASTE`:** namespace is
   chosen by the user in the wizard (dropdown listing all namespaces in that
   cluster); the default name is `hf-token-<uuid4().hex[:8]>` (short UUID to
   avoid collisions), and the user may change it, after which validate that "the
   same name must not already exist in that namespace" (reject with 409).
   **Architectural simplification:** do not add a `TokenSourceMode.PASTE` enum
   value; instead create the Secret first through a standalone endpoint, then let
   the Model Cache entry reference it using the existing `EXISTING_SECRET` mode
   (see §4.4).
5. **Whether Storage in Step 4 is "automatically created":** no. After the user
   manually completes the full form, the wizard calls the existing Storage create
   API—exactly the same code path as manually creating a volume on the Storage
   page. The wizard simply embeds that form into the flow to reduce user
   navigation.
6. **Whether database setup (Step 0) is a global one-time gate or configured per
   cluster:** **global, one-time**, not attached to `Cluster` records and not
   appearing in the step indicator / skip switch. Once the application has a
   configured database, it is permanently skipped; see §4.0.

---



<!-- /design-part: wizard -->

<!-- design-part: bootstrap -->

## Kubernetes bootstrap

- **Location:** `docs/design/cluster-design.md`.
- **Reference architecture:**
  - Frontend: `apps/frontend/src/pages/clusters/ui/CreateClusterWizard.jsx` (the multi-step “Create Cluster” wizard,
    see `cluster-design.md`).
  - Backend: `llm_d_lens/cluster/` (registry/service/router, which manages metadata for registered
    clusters), `llm_d_lens/deploy/runtime/composition.py` (the “optional external executable”
    resolution pattern in `RuntimeEnvironment`, via `kubectl_path`/`helm_path`/`docker_path`
    — this proposal reuses the same pattern for `ansible-playbook`), and
    `llm_d_lens/utils/ssh.py` (existing paramiko-based remote execution utilities, currently unused;
    in this proposal they are downgraded to lightweight preflight connectivity probing, while actual
    deployment work is delegated to the official tools researched below).

---

### 0. Research conclusion (answering "is there an official tool" first)

The Kubernetes official documentation, [Installing Kubernetes with deployment tools](https://kubernetes.io/docs/setup/production-environment/tools/),
explicitly lists four official/SIG-recognized cluster deployment tools. The following evaluates them
one by one for this proposal's scenario (**the user provides several networked bare-metal or virtual
machines that can be reached over SSH, with no cloud-provider API available**):

| Tool | Ownership | Suitable scenario | Suitable for this proposal? |
|---|---|---|---|
| **kubeadm** | Kubernetes core subproject | Single-node init/join primitives; it does not provide multi-node orchestration, retries or CNI selection by itself | Use it as a **dependency** of Kubespray, not a separate Lens orchestration engine |
| **[Cluster API](https://cluster-api.sigs.k8s.io/) (CAPI)** | Kubernetes SIG-cluster-lifecycle subproject | Declarative, designed for multi-cluster lifecycle management, but requires a **management cluster** first (typically bootstrapped with `kind`/`k3s`) plus an Infrastructure Provider for the target platform (for bare metal, Metal3/CAPM3, which depends on BMC/Redfish/PXE bootstrapping and out-of-band management) — far heavier than this proposal's assumption of “only SSH credentials” | Not suitable for MVP; reserve for Phase 2+ (see §8), and reconsider only if future node inventory includes BMC |
| **[kOps](https://kops.sigs.k8s.io/)** | Kubernetes SIG-cluster-lifecycle subproject | Historically focused on cloud-provider automation (primarily AWS, secondarily GCE/Azure) to create full cloud resources such as VPCs, LBs, and ASGs. Although its docs include [Bare Metal](https://kops.sigs.k8s.io/getting_started/bare_metal/) support, its maturity and community usage are clearly weaker than in cloud scenarios | Not suitable (this proposal has no cloud-provider API available, and bare metal is a secondary scenario for it) |
| **[Kubespray](https://kubespray.io/)** | Kubernetes SIG-cluster-lifecycle subproject | A set of **Ansible playbooks**. Its official README explicitly states support for **Bare Metal**; it only requires SSH + sudo; the input is an Ansible inventory (which maps almost exactly to the “node list” this proposal needs to collect in §3); it supports HA multi-control-plane, selectable CNI (default **Calico**, matching existing precedent in the current environment), and major Linux distributions including **Ubuntu/Debian/RHEL family (CentOS Stream, RHEL, Rocky, AlmaLinux, Oracle Linux) / Fedora / openSUSE** (details in §3.2). It has extensive built-in preflight checks, and `cluster.yml`/`scale.yml`/`remove-node.yml` correspond directly to “create cluster”/“add node”/“remove node”. It is **idempotent and rerunnable** (if one node fails, fix it and rerun the same playbook; there is no need to wipe everything and start over) | ✅ **Chosen for this proposal** |

**Conclusion: use Kubespray rather than implementing kubeadm orchestration in Lens.**
Lens only needs to: ① render the node list collected by the wizard into the inventory format
expected by Kubespray, ② run `ansible-playbook cluster.yml` as a subprocess, and ③ read the
`admin.conf` produced by Kubespray itself.
This delegates the most error-prone and maintenance-heavy part of the proposal (retry semantics for
multi-node joins, CNI installation, per-distribution dependency differences) wholesale to an
official tool already validated at large production scale.

---

### 1. Background and goals

#### 1.1 Current state

Step 1 (“Basic information”) of `CreateClusterWizard.jsx` assumes the user **already** has a usable
Kubernetes cluster and already has a `kubeconfig` file in hand. The user uploads it, and only then
can Lens register the cluster into `llm_d_lens/cluster/registry.py` (writing `<id>.json` +
`<id>.kubeconfig`).

For a benchmark engineer who has just obtained several bare-metal machines (or several VMs that do
not yet have Kubernetes installed), there is still an entire unsupported gap between “bare metal”
and “a cluster ready to run llm-d”. They have to manually follow kubeadm tutorials and debug each
machine's dependencies and networking issues on their own, completely disconnected from Lens. When
something goes wrong, Lens has no visibility at all.

#### 1.2 Goal

Insert a new segment in front of the existing “Create Cluster” wizard entry so that the user can
choose:

- **Path A (current behavior):** I already have a cluster; upload kubeconfig directly — completely
  unchanged. The wizard starts from Step 1 as today; see
  `cluster-design.md`.
- **Path B (new):** I only have several machines and no Kubernetes yet — Lens uses **Kubespray** to
  install a production-usable cluster on those machines, then automatically feeds the kubeconfig
  produced by Kubespray into Step 1 so the existing wizard continues seamlessly (proxy/version/Model
  Cache and all later steps remain unaffected).

The core constraint of the design is: **Path B only inserts one new step in front of Path A that
produces a kubeconfig string; it does not modify any Step 1–6 code/API in the existing wizard.**
This minimizes implementation and review cost for the proposal and also ensures the existing minimal
“upload kubeconfig only” flow and existing automation scripts remain unaffected.

#### 1.3 Non-goals

- No integration with cloud-provider managed Kubernetes provisioning APIs (GKE/EKS/AKS, etc.) —
  those are entirely different authentication/billing/resource models; although kOps officially
  supports them, they are outside the scope of this proposal.
- No Cluster API/Metal3 — that requires BMC/PXE out-of-band management and is far heavier than the
  “only SSH credentials” assumption here; leave it for Phase 2+ (see §8).
- No operating-system-level node initialization (partitioning, OS installation, network setup) —
  assume the user-provided machines are already reachable by SSH using an account with sudo, and can
  reach the Internet or an internal image mirror. Kubespray itself handles retrieval of containerd,
  Kubernetes component packages, and CNI images; for offline image schemes see
  [Kubespray's Air-Gap documentation](https://github.com/kubernetes-sigs/kubespray/blob/master/docs/operations/offline-environment.md).
  MVP assumes nodes can directly reach the Internet or the company's proxy and does not implement
  offline image distribution.
- No CNI selector beyond Kubespray — MVP fixes Kubespray's default Calico; however, **the OS support
  matrix is aligned directly with the Kubespray official support list** (see §3.2), rather than
  artificially narrowing it to a single distribution matching the local development environment.

---

### 2. User flow (the split point between the two paths)

At the top of Step 1 in `CreateClusterWizard.jsx`, add a two-option entry point (it is not part of
`STEPS`; it is a pre-branch inside Step 1):

```
┌───────────────────────────────────────────────────┐
│  This cluster is…                                 │
│  ○ I already have kubeconfig (current behavior)   │
│  ○ I only have a few machines; help me install    │
│    Kubernetes (new, based on Kubespray)           │
├───────────────────────────────────────────────────┤
│  [Expanded node list + Bootstrap progress view    │
│   after selecting the second option]              │
└───────────────────────────────────────────────────┘
```

After selecting the second option, Step 1 temporarily switches into a “sub-wizard” (reusing the UI
in §4). Once it finishes and obtains the kubeconfig string, it is **automatically backfilled** into
where Step 1 originally expected an uploaded kubeconfig file (displayed as a read-only “upload
succeeded” state, with an option to “choose again” and return to manual upload). The user then
clicks “Next” as normal and proceeds into the existing Step 2 (proxy) through Step 6 (confirmation).
To the rest of the wizard, this auto-generated kubeconfig is no different from a manually uploaded
file.

---

### 3. Node inventory and preflight validation

#### 3.1 Inputs

The first screen of the sub-wizard collects a node table (not just a pile of form fields; model it
on the node selector interaction in `StorageManagement`). Its fields correspond directly to the
minimum data Kubespray needs for a single host in the inventory:

| Column | Description | Corresponding Kubespray inventory field |
|---|---|---|
| Host | One row may contain **a single IP/hostname**, **multiple IPs/hostnames separated by commas**, or **an IP range** (short form `10.0.0.10-20` means the last octet runs from 10 to 20; it also supports full ranges across subnets such as `10.0.0.250-10.0.1.5`). The range is expanded in the backend into multiple independent nodes, all inheriting the same checked Role(s) and authentication method from that row. A single expansion is capped at 256 nodes; exceeding that, or an invalid range itself (e.g. start/end reversed or field overflow), causes a submission-time error. Strings containing `-` that do not form a valid range (such as hostname `worker-1.internal`) are treated as literal hostnames and do not error | After expansion, each host's own `ansible_host` |
| SSH Port | Default 22 | `ansible_port` |
| Role(s) | **Checkboxes**; `control-plane` and `worker` may both be selected. At least one node must select `control-plane`; multiple control-plane nodes are supported by Kubespray. A node may hold both roles for single-node validation or small clusters. | Selecting `control-plane` puts it into `kube_control_plane` and `etcd`; selecting `worker` puts it into `kube_node`; selecting both enters both groups |
| Authentication method | Paste private-key text, or username + password | `ansible_user` + `ansible_ssh_private_key_file` (for password mode, use `ansible_ssh_pass`, which requires `sshpass` on the target side of execution) |

At least one node must be marked `control-plane`. It may or may not also be marked `worker`
(single-node validation/development scenario, corresponding to Kubespray's all-in-one topology —
with only one node, check both `control-plane` and `worker`, and it will appear in all three groups
`kube_control_plane`/`kube_node`/`etcd`).

The pasted private key is written into a temporary file that exists only for the lifecycle of the
current Bootstrap Job (`0600`), and is immediately `unlink`ed on success/failure. It is never
written into the persistent Cluster directory. It is only used for
`ansible-playbook --private-key=<temporary-file>`.

**Network proxy on target nodes:** below the node table there is also an independent “Network proxy
for target nodes” switch (`auto`/`custom`), rendered into Kubespray's
`group_vars/k8s_cluster/k8s-cluster.yml` (`http_proxy`/`https_proxy`/`no_proxy`, which are
officially recognized Kubespray variables; see its
`inventory/sample/group_vars/all/all.yml`). This is distinct from Step 2 “Network Proxy” (which only
affects the Lens backend itself when pulling the llm-d/llm-d-benchmark repositories). Here it
controls **the target nodes themselves** when they run OS package manager installs, and when roles
such as `container-engine/runc` download binaries directly from GitHub. If the target nodes lack a
direct route to the public Internet (common in enterprise proxy environments), failing to set these
variables causes errors like `ansible-playbook exited with status 2`, often with messages similar
to `container-engine/runc : Download_file | Download item` repeatedly retrying and then failing.
`auto` (default) probes each target node via SSH for its already-configured proxy settings (reading
`/etc/environment` and exported `*_proxy`/`*_PROXY` variables after preflight succeeds but before
Kubespray is actually invoked), takes the first non-empty `http_proxy`/`https_proxy` found across
the nodes, and uses the union of all nodes' `no_proxy`. If none of the target nodes themselves have
proxy settings, it is treated as “no proxy needed” (it will not fall back to the Lens backend
process's own environment variables; the two may be in completely different networks). `custom`
allows specifying proxy settings for this batch of target nodes directly, and they take effect
immediately on submission (no probing required).

**`no_proxy` auto-completion (important):** Kubespray itself **does not** automatically append any
cluster-internal addresses to `no_proxy` — its default is to pass through exactly what the caller
provides (see `roles/kubespray_defaults/defaults/main/main.yml`:
`no_proxy: "{{ no_proxy | default('') }}"`). This means that once a node has `http_proxy`/
`https_proxy`, **all** its HTTPS traffic — including `kubectl`/kubelet/etcd access to
`127.0.0.1:6443` or peer node IPs — will be sent to the enterprise proxy. Proxies usually reject
this kind of internal direct-connect traffic, which manifests as a cluster that finished
installation but then `kubectl get pods -A` suddenly reports
`Unable to connect to the server: Forbidden` (HTTP 403 from the proxy, not an actual inability to
reach the API server). Therefore, when a proxy is configured (whether `auto` or `custom`),
`write_inventory`/`inventory.py::_augment_no_proxy` automatically merges the following into
`no_proxy` (deduplicated while preserving user-supplied values): `localhost,127.0.0.1`, the
Kubespray defaults when not overridden for `kube_service_addresses`/`kube_pods_subnet`
(`10.233.0.0/18` / `10.233.64.0/18`),
`kubernetes,kubernetes.default,kubernetes.default.svc,kubernetes.default.svc.cluster.local,.svc,.svc.cluster.local,.cluster.local`,
and each target node's own host/IP from this submission.

#### 3.2 Preflight checks (per-node, concurrent, the fast-fail layer before Kubespray)

Kubespray's own `bootstrap-os`/`preinstall` roles already include many preflight checks, but those
only show up when the playbook actually starts running (possibly after several minutes). To let the
user know within seconds of finishing the node list that “this machine is unreachable / lacks sudo”,
clicking “Next” (not “Finish”) first runs a read-only quick check using
`llm_d_lens/utils/ssh.py::remote_exec`. Only after all checks pass may the user proceed to the next
screen and trigger Kubespray for real:

- SSH reachable + authentication succeeds (`remote_exec(target, "true")`).
- Passwordless sudo via `sudo -n true` works (Kubespray requires root throughout, corresponding to
  `ansible-playbook -b`).
- The operating system is in Kubespray's supported list (`ID`/`VERSION_ID` from `/etc/os-release`,
  aligned with [Kubespray v2.31.0 README “Supported Linux Distributions”](https://github.com/kubernetes-sigs/kubespray/blob/v2.31.0/README.md#supported-linux-distributions),
  rather than the narrower set represented by the local development environment. Currently included:
  Ubuntu 22.04/24.04, Debian 11/12/13, CentOS Stream/RHEL 9/10, Fedora 39-42, openSUSE Leap
  15.x/Tumbleweed, Oracle Linux 9/10, AlmaLinux 9/10, Rocky Linux 9/10. Distributions Kubespray
  marks as “experimental” — Flatcar, Fedora CoreOS, Kylin, UOS, openEuler, Amazon Linux 2 — are
  excluded from MVP because they require additional distro-specific bootstrap steps.)
- **The node must not already have Kubernetes deployed** — read-only checks on
  `/etc/kubernetes/admin.conf`, `/etc/kubernetes/manifests/kube-apiserver.yaml`, whether `kubelet`
  is running, and whether the `kubeadm` command exists. If any condition matches, the node is
  judged to “already have a Kubernetes installation”, and the request is rejected immediately with a
  prompt to run `kubeadm reset` first (or switch to a clean node). Running `cluster.yml` again on an
  already-existing cluster is not allowed, to avoid kubelet/etcd/certificate conflicts causing
  unpredictable state.

If any node fails this preflight layer, only **that node** is blocked. The user may fix its
configuration in place and retry, or remove it from the list and continue, but **at least one
`control-plane`** must pass to proceed. After this fast-fail layer, finer-grained dependency and
kernel-parameter checks are left to Kubespray's own `preinstall` role (no need to reinvent them).

---

### 4. Bootstrap execution engine (based on Kubespray)

#### 4.1 Integrating Kubespray as a managed external dependency

`llm_d_lens/cluster/bootstrap/kubespray.py` owns a pinned Kubespray checkout
and dedicated virtualenv built from that checkout's `requirements.txt`.
The cache root comes from `storage_path("cache", "kubespray")`; do not share
Lens's Python environment or substitute an arbitrary system Ansible version.

`require_ready()` performs a readiness check without installation.
If the checkout is unavailable, the background bootstrap job calls
`ensure_ready()` through `asyncio.to_thread`, then invokes the checkout's
`cluster.yml` with its own `ansible-playbook`. Dependency download/build errors
fail the job explicitly. Existing ready checkouts are reused.

#### 4.2 Phases

Kubespray handles node joining, CNI installation and control-plane orchestration;
Lens observes the job and subprocess lifecycle.

```
queued → preflight (§3.2, done by Lens itself) → [provisioning (only on first run; see below)]
       → running (`ansible-playbook cluster.yml` subprocess) → kubeconfig_ready → succeeded
                                                                  ↘ failed (non-zero subprocess exit, carrying stdout/stderr tail)
```

- **`provisioning` (automatic Kubespray provisioning):** `kubespray.require_ready()` only checks;
  when the local cache (`~/.cache/llm-d-bench/kubespray/<tag>`) is missing, **operations no longer
  need to manually run `ensure_ready()` in advance**. Instead, the Job itself enters the
  `provisioning` phase and calls `kubespray.ensure_ready()` (clone the specified tag + create a
  dedicated venv using its bundled `requirements.txt`). On success it continues into `running`; on
  failure (no network, clone/pip error, etc.) it goes directly to `failed`, returning the original
  error to the user. Only the **first Bootstrap Job on the same backend process** passes through
  this phase; later jobs reuse the cached checkout and skip `provisioning`. Because this involves a
  real `git clone` / `pip install` (which may take minutes), it should be offloaded into a thread
  pool via `asyncio.to_thread` so the event loop is not blocked.

- **Render inventory:** convert the node list collected in §3.1 into the YAML inventory required by
  Kubespray (`all.hosts.<name>.ansible_host/ansible_port/ansible_user/...` + the three child groups
  `kube_control_plane` / `kube_node` / `etcd`) and write it to
  `<bootstrap_job temporary directory>/inventory.yml`.
- **group_vars overrides:** write a minimal `group_vars/k8s_cluster/k8s-cluster.yml` override that
  locks three things:
  `kube_network_plugin: calico` (although this is Kubespray's default, write it explicitly so a
  future Kubespray default change does not silently change this proposal's behavior),
  `kubeconfig_localhost: true` + `kubectl_localhost: true` (critical — these make Kubespray output
  kubeconfig directly into `inventory/<job>/artifacts/admin.conf` on the **machine running
  Ansible** — i.e. the Lens backend itself — eliminating the entire extra step from the previous
  draft of “SSH back into the control plane and read `/etc/kubernetes/admin.conf`”), and
  `override_system_hostname: false` (Kubespray defaults this to `true`, which changes the real
  `/etc/hostname` on each target machine to the node name in the inventory. For machines that
  already have hostnames assigned by ops/IT, this is unnecessary and surprising, so we explicitly
  disable it. Kubespray still uses the inventory names for internal grouping/addressing, and this
  does not affect whether the cluster itself starts correctly).
- **running:** start
  `ansible-playbook -i inventory.yml -b --private-key=<temporary private key> cluster.yml`
  asynchronously, read stdout line by line
  and incrementally feed it into the Bootstrap Job's log buffer (reusing the `_MAX_OUTPUT` tail
  truncation pattern), and let the frontend poll and display “last N lines of Ansible output” as a
  progress indicator. Kubespray's Ansible output is already grouped by task, so Lens does not need
  to parse finer-grained phases such as “what exact stage is it on now” — another “free” capability
  obtained by using a mature tool.
- **kubeconfig_ready:** after the subprocess exits with code 0, read the contents of
  `inventory/<job>/artifacts/admin.conf` as the kubeconfig text.
- **failed:** when the subprocess exits non-zero, the Bootstrap Job records the stdout/stderr tail
  (Ansible failures usually identify the specific host and task that failed, allowing the user to
  tell whether this was a network issue or an environment problem on a particular machine).

#### 4.3 Adding/removing nodes (mapping to Kubespray's `scale.yml` / `remove-node.yml`)

Not in MVP scope (see §8), but worth documenting: because Kubespray was chosen, the future need of
“add/remove worker nodes for an already registered Lens cluster” does not need a fresh design.
As long as Cluster metadata retains an optional marker such as “was this cluster created by Lens
using Kubespray, and which inventory corresponds to it” (for example
`bootstrapProvider: "kubespray"` + inventory directory path), the Storage/Monitoring pages could
later add a “scale out” button that simply rerenders the inventory (adding the new nodes) and runs
`scale.yml`. This is an immediately visible compounding benefit of choosing an official tool rather
than hand-writing kubeadm orchestration.

#### 4.4 Data model (in-memory only, not persisted to Cluster registry)

A Bootstrap Job is **not** a variant of `llm_d_lens/cluster/registry.py::Cluster`; its lifecycle
ends once the kubeconfig string is obtained:

```python
@dataclass
class BootstrapNode:
    host: str
    # Non-empty list; one node may hold both control-plane and worker roles
    # simultaneously (see §3.1).
    roles: list[Literal["control-plane", "worker"]]
    preflight: Literal["pending", "passed", "failed"]
    preflight_error: str | None = None


@dataclass
class BootstrapJob:
    id: str  # uuid4
    phase: Literal[
        "queued",
        "preflight",
        "provisioning",
        "running",
        "kubeconfig_ready",
        "succeeded",
        "failed",
    ]
    nodes: list[BootstrapNode]
    inventory_dir: (
        Path  # temporary dir, containing inventory.yml + group_vars + artifacts/
    )
    log_tail: str  # truncated tail of ansible-playbook output
    kubeconfig_text: str | None = (
        None  # only present on succeeded, one-time consumption
    )
    created_at: str
```
Store it in a pure in-memory `dict[str, BootstrapJob]`, following the pattern of
`llm_d_lens/cluster/sessions.py::ClusterSession` (lost on process restart — acceptable, because
Bootstrap is inherently a one-shot operation; if interrupted, the user can just initiate it again).
`inventory_dir` is cleaned up once the Job reaches a terminal state (`succeeded`/`failed`) and the
frontend has read the kubeconfig/logs.

---

### 5. API surface

Add a set of endpoints independent from the existing cluster CRUD routes in
`llm_d_lens/cluster/router.py`:

| Method & path | Description |
|---|---|
| `POST /api/cluster/bootstrap/preflight` | Takes the node list, synchronously runs the quick checks in §3.2, and returns per-node pass/fail details. |
| `POST /api/cluster/bootstrap` | Called after all preflight checks pass: renders the inventory, starts a background task running `ansible-playbook cluster.yml`, and returns `{ bootstrapId }`. |
| `GET /api/cluster/bootstrap/{id}` | Polls status + log tail; when `phase == "succeeded"`, the response body also includes `kubeconfig` (returned only once; after it is read, the field is cleared from the in-memory Job and `inventory_dir` is cleaned up). |
| `POST /api/cluster/bootstrap/{id}/cancel` | If the user closes the sub-wizard mid-run: terminate the running `ansible-playbook` subprocess (`asyncio.subprocess` `.terminate()`, following the existing Helm subprocess management pattern in `precise_prefix_cache_routing.py`). |

Existing Cluster DTOs such as `ClusterSettingsUpdateRequest` remain **unchanged**.

---

### 6. Security considerations

- **SSH credentials only flow through memory/temporary files:** private keys/passwords are supplied
  once in the request body for one-time use. The private-key file is written with strict `0600`
  permissions and deleted immediately when the Job ends, whether successfully or not.
  For password-authenticated nodes, the password **is not** written into the shared `inventory.yml`
  (that file itself is not permission-hardened and is easy to leak through `cat` or logging).
  Instead, `render_host_vars()` generates a separate `host_vars/<host>.yml` for each password-based
  node (`{ansible_ssh_pass: ...}`), written under `work_dir/host_vars/` with `0600` permissions as
  well. Ansible automatically loads `host_vars/` from the inventory directory, so there is no need
  for extra `--extra-vars` or command-line arguments (which are visible to other users on the same
  host via `ps`). The Job's `finally` block deletes the entire `host_vars/` directory together with
  the private-key temp directory via `rmtree`, preventing plaintext passwords from remaining on
  disk.
  In addition, password authentication depends on Ansible's `ssh` connection plugin invoking the
  `sshpass` binary on the Lens **backend** host, not on the target nodes. Therefore, before starting
  a Job, the backend checks whether `sshpass` exists (`shutil.which`); if it does not, fail fast and
  tell the user to install `sshpass` on the Lens backend host instead of letting Ansible fail
  halfway through. Logs/`log_tail` must be redacted before being stored into `BootstrapJob`
  (verbose Ansible output may echo variables such as `ansible_ssh_pass`; filter out lines
  containing those field names while reading subprocess output, or rely on `no_log: true` where
  Kubespray already does so in its built-in tasks).
- **One-time kubeconfig delivery:** once `GET /api/cluster/bootstrap/{id}` returns kubeconfig, the
  field is immediately cleared from the in-memory Job and `artifacts/admin.conf` is deleted from
  disk, preventing the same sensitive file from lingering on the Lens backend host.
- **sudo boundary:** `ansible-playbook -b` (`become`) delegates privilege elevation
  to Kubespray's task definitions.
- **Host-key verification:** Before the first authenticated connection, the UI calls
  `POST /api/cluster/bootstrap/host-keys` to retrieve each expanded host's SSH public-key SHA-256
  fingerprint without credentials. It displays the `host:port`, algorithm and fingerprint, and
  requires the operator to confirm an independent check against a trusted source. Both preflight
  and job creation require `hostKeys` pins covering every expanded node; the backend rejects
  missing, extra or duplicate pins. Paramiko accepts only an exactly matching server key before
  authentication. Before Ansible runs, the backend checks the key again and writes a per-job
  `known_hosts` file with `0600` permissions. `ANSIBLE_SSH_COMMON_ARGS` sets
  `StrictHostKeyChecking=yes`, `UserKnownHostsFile`, and an isolated global known_hosts file.
  If the key changes, the job fails and requires renewed verification. Temporary files are removed
  when the job ends. **Displaying a fingerprint and checking a box does not establish trust:**
  the operator must verify it through an independent trusted channel, such as a cloud console,
  asset inventory or on-site records. Otherwise initial discovery remains exposed to a
  man-in-the-middle attack; connection-time pinning prevents replacement after confirmation.

---

### 7. Failure handling and retries

- Kubespray's `cluster.yml` is **idempotent and rerunnable**. Retry with the same
  inventory after correcting a transient failure; Ansible identifies already
  satisfied steps. Lens does not duplicate Kubespray's per-node task state machine.
- The only retry semantic Lens itself needs to handle: if the user **changes the node list** before
  retrying (for example, removing a persistently unreachable worker), Lens must rerender the
  inventory and rerun, rather than simply retrying the old inventory.

---

### 8. Non-goals

- Cluster API + Metal3 (bare metal requires BMC/PXE out-of-band management) — see the research
  conclusion in §0; reconsider only when future node inventory truly includes out-of-band control.
- Cloud-provider managed Kubernetes (GKE/EKS/AKS) or kOps cloud automation — completely different
  API/billing/IAM models.
- CNI choice (Cilium/Flannel/kube-ovn, etc.; Kubespray itself supports them, but MVP does not
  expose that choice in the UI).
- Operating-system-level node initialization (OS install, partitioning, NIC configuration) — assume
  the machines are already SSH-accessible.
- Offline/Air-Gap image distribution (Kubespray has documentation for it, but it introduces
  substantial UI and storage complexity, so postpone it until a real offline deployment need
  appears).
- The “scale an already registered cluster” idea mentioned in §4.3 — finish the “create a cluster
  from zero to one” path first, then treat scaling as a high-reuse follow-on proposal.
- Preserving Bootstrap Job state across process restarts: jobs are process-local;
  after a restart the user must initiate the action again.

---

### 9. Acceptance boundaries

- Inventory tests cover node roles, proxy exclusion, pinned SSH host keys and
  credential-file permissions without requiring remote machines.
- Service tests cover missing-checkout provisioning, failure/cancellation,
  redacted logs and one-time kubeconfig consumption.
- Real-cluster acceptance separately verifies package/image access, Kubespray
  completion and successful onboarding of the returned kubeconfig.
- Creating an existing-cluster connection must not invoke Kubespray.

<!-- /design-part: bootstrap -->
