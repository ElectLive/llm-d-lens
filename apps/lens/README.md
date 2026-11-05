# `llmd`

`llmd` plans, stands up, rolls out, and removes [llm-d](https://github.com/llm-d/llm-d) guides inside one
existing Kubernetes or OpenShift namespace. It does not need cluster-admin rights, and it does not create
namespaces, CRDs, cluster roles, PersistentVolumes, SCCs or network attachments: it reports those for an
administrator.

A *stack* is what a team deploys into one namespace:

- an llm-d guide, such as `guides/optimized-baseline`. `llmd` reads the guide's `guide.yaml`, the file the
  guide's README commands are generated from, and follows its `deploy` steps without running a shell. It
  renders the router chart with `helm template`, using the same chart, version and values as the guide's
  `helm install`, and renders each model server overlay with `kustomize build`;
- one kustomize directory, such as one model server overlay of a guide;
- manifest files, read as they are.

An *instance* is a stack running in a namespace. It is recorded in a ConfigMap `lens-instance-<name>` in that
namespace, so every command works from any machine. The record's fields and phases, and how `llmd` maps onto the
Lens roadmap, are described in the [Lens CLI design](../../docs/design/lens-cli-design.md).

`llmd` is part of the Lens engine package (`apps/lens`). It is pure Python and depends only on PyYAML. It
does not need the Lens service, its database or Node.

## Install

From the repository root, in any Python 3.11+ virtual environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -e ./apps/lens
.venv/bin/llmd --version
```

`llmd` runs `kubectl` (or `oc`, with `--kube-cli oc`), `helm`, `git`, and `kustomize` (or `kubectl kustomize`
when kustomize is not installed). Install them on the `PATH`.

## Sources

A guide comes from an llm-d checkout, or from upstream directly:

```sh
llmd sources add llm-d https://github.com/llm-d/llm-d.git#main
llmd stacks                           # the guides of the source
llmd stacks optimized-baseline        # one guide's variables, support matrix and steps
```

`--source` takes a registered name, a local path, or `URL#REF`. REF can be a branch, a tag, a commit, or any
ref the remote serves, so a pull request can be planned before it merges:
`--source https://github.com/llm-d/llm-d.git#refs/pull/<N>/head`. With exactly one registered source,
`--source` may be left out. A remote checkout is cached; `--refresh` fetches it again.

The namespace comes from `-n`, else `$LLMD_NAMESPACE`, else the kube context.

## Guide variables

A guide's README sets variables with `export` lines (`ACCELERATOR_TYPE`, `MODEL_SERVER`, `INFRA_PROVIDER`, and
the guide's own). `--var NAME=VALUE` sets them for `llmd`. The guide's `when:` filters then pick the same steps
the README would run:

```sh
llmd plan optimized-baseline -n <namespace> --var INFRA_PROVIDER=ocp --offline
```

`--set KEY=VALUE` passes a value to the Helm charts the guide installs (the router), as `helm install --set`
would. `NAMESPACE` comes from `-n`, and `REPO_ROOT` is the source checkout. The plan lists each guide step and
what `llmd` did with it:

- `followed`: the step's `helm install` or `kubectl apply -k|-f` is rendered.
- `yours`: creating the namespace or a Secret is left to you. The plan checks that the Secret exists.
- `administrator`: a cluster-wide prerequisite (CRDs from the network, an operator).
- `other variant`: the step applies a directory this checkout does not have for these variables.
- `not followed`: `llmd` cannot reproduce the step (a pipe through `sed` or `envsubst`, a script). The plan is
  blocked, unless the guide's own CI skips that step too (`skip_in: [ci]`).

Gateway Mode needs a cluster Gateway, so `llmd` follows Standalone Mode.

## An empty namespace

```sh
llmd plan optimized-baseline -n <namespace>       # read-only: findings, prerequisites, a server dry run
llmd standup optimized-baseline -n <namespace> --expected-digest <digest from the plan> --wait
llmd status optimized-baseline -n <namespace>
llmd teardown optimized-baseline -n <namespace>
```

`plan` renders the guide, moves every object into the namespace, and reports what blocks a standup:

- objects only an administrator may apply. `--admin-yaml FILE` writes them; pass `--skip-admin-parts` once
  they are applied;
- privileged containers and hostPath volumes. Pass `--allow-privileged` once an administrator has granted them;
- missing Secrets and claims, missing permissions, and objects another tool already deployed.

The llm-d router chart creates a ClusterRole and ClusterRoleBinding so that the endpoint picker can check who
scrapes its metrics. Without an administrator, `--set router.monitoring.prometheus.auth.enabled=false` turns
that check off. The metrics are then served without authentication inside the cluster.

`standup` records the instance before it applies anything, then applies with server-side apply. It also
records the `--var` and `--set` values with the instance. `teardown` deletes exactly the recorded objects whose
UID still matches. Claims are kept unless you pass `--delete-volumes`.

## A guide already deployed with its README's commands

```sh
llmd import optimized-baseline -n <namespace> --name <instance> --dry-run   # do the running objects match?
llmd import optimized-baseline -n <namespace> --name <instance>
```

`import` records the running objects by UID and changes nothing else: no labels, no apply, and no field
ownership, so the team's own `kubectl apply -k` and `helm upgrade` keep working. `--dry-run` compares the
running objects with the guide's render first. `llmd teardown <instance> --forget` stops managing the instance
and leaves every object running.

## Rollouts

A rollout moves an instance to what another ref of the same stack renders, in place. In a guide whose model
server is one DisaggregatedSet (`guides/pd-disaggregation`, `guides/wide-ep`), a change to the pod templates
makes its controller (lws) replace every group. With `maxSurge: 0` and `maxUnavailable: N` per role, each step
removes up to N old groups of every role and creates as many new ones.

```sh
NEW=https://github.com/llm-d/llm-d.git#refs/pull/<N>/head
llmd rollouts plan pd-disaggregation -n <namespace> --name <instance> --source "$NEW"
llmd rollouts start pd-disaggregation -n <namespace> --name <instance> --source "$NEW" --no-drain --watch
```

`rollouts plan` is read-only. It shows the objects that change, whether the DisaggregatedSet rolls or only
scales, the steps, and the groups each role keeps serving during a step. It refuses a rollout in these cases:

- a roll is in progress, or a group is not serving;
- several DisaggregatedSets would roll at once;
- roles are added or removed;
- a step would leave a role below its `--floor`;
- a step would leave a role with no group serving. `--allow-outage` accepts that outage.

The llm-d guides' vLLM overlays drain each pod as it stops: a `preStop` sleep, then vLLM's
`--shutdown-timeout`. `--no-drain` relies on that. A stack that holds the controller's scale-down until a drain
tool takes a group out of the pool passes that tool with `--drain`. `llmd` runs it once per role, filling in
`{role}`, `{source}`, `{namespace}`, `{set}`, `{old_revision}` and `{max_unavailable}`, and `--pool-label`
names the label it sets. Each drainer's output is under `$LENS_DATA_DIR/llmd/rollouts/<namespace>/<instance>/`.

```sh
llmd rollouts status <instance> -n <namespace> [--watch]   # per role and revision; the drainers
llmd rollouts pause <instance> -n <namespace>               # stop the drainers
llmd rollouts resume <instance> -n <namespace> [--watch]    # start them again, on the same machine
llmd rollouts abort <instance> -n <namespace> [--watch]     # re-apply what ran before; it rolls back the same way
llmd rollouts finish <instance> -n <namespace>              # end it once the old revision is gone
```

## Development

```sh
cd apps/lens
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/ruff check lens tests
.venv/bin/ruff format --check lens tests
```

The tests use an in-memory cluster (`tests/fakes.py`). It answers the kubectl calls `llmd` makes and acts as
the Deployment and lws controllers. Git, kustomize and Helm run for real, against a small checkout in llm-d's
layout (`tests/data/llmd`).
