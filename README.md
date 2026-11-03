<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <img alt="Lens" src="apps/frontend/public/lens_logo.png" width="150">
</p>

<h3 align="center">
The ultimate control plane for AI inference: Single-click deployment, automated benchmarking, and full-stack cluster monitoring.
</h3>

<p align="center">
| <a href="https://llm-d-incubation.github.io/llm-d-lens/"><b>Documentation</b></a> | <a href="https://llm-d-incubation.github.io/llm-d-lens/quickstart"><b>Quickstart</b></a> | <a href="https://llm-d-incubation.github.io/llm-d-lens/user-guide/model-market"><b>User Guide</b></a> | <a href="https://llm-d-incubation.github.io/llm-d-lens/architecture"><b>Architecture</b></a> |
</p>

---

## About

Lens is an open-source control plane for deploying and operating large language models on [llm-d](https://llm-d.ai/), the Kubernetes-native distributed inference stack, across your Kubernetes clusters. It plans and executes inference deployments, runs guided evaluations and traffic simulations, and keeps clusters, storage and external providers under one roof, so a team can go from a registered model to a benchmarked, production-serving deployment without stitching together separate tools.

Lens is built on llm-d with:

- Prefill/decode disaggregation, KV-cache-aware routing, KV cache offloading to CPU/SSD tiers, and latency-predicted routing that raise throughput and cut tail latency on the same accelerators
- Multi-cluster planning, approval and execution with agentic, workload-guided placement
- A multi-protocol model service gateway (OpenAI chat completions and embeddings, Anthropic messages) behind HTTPRoute → InferencePool → EPP, with per-model selection policies
- Role-based access control, per-cluster scoping and audit logging across every page
- A bootstrap-and-maintain monitoring stack (Prometheus/Grafana) for accelerator utilization and deployment health

Lens manages these areas from one console:

- **Model market** — browse models and start a one-click deployment
- **Deployments** — plan, approve and run inference deployments across clusters
- **Evaluation** — design benchmark experiments, run them against live deployments, and analyze or export the results
- **Simulation** — replay registered request traces against an endpoint to validate behavior and capacity
- **Model services** — publish cluster deployments as gateway-served model endpoints (OpenAI chat completions and embeddings, Anthropic messages)
- **Clusters** — register, bootstrap and monitor Kubernetes clusters
- **Storage** — manage volumes and file storage
- **Model cache** — share model weights across deployments
- **External providers** — connect external inference providers
- **Observability** — run the cluster monitoring stack
- **Lens Assistant** — an in-product assistant over the Lens API
- **Administration** — manage users, groups, roles, identity providers and API keys

## Getting Started

On Ubuntu 22.04+ (x86_64), use the online installer without cloning the repository.
It provisions Node, compatible Python, an embedded PostgreSQL database, TLS and the application:

Set `VERSION` to `main` for the development version, or an actual release tag
(for example, replace `main` with `vX.Y.Z`):

With **curl**:

```bash
VERSION=main && curl -fsSL "https://raw.githubusercontent.com/llm-d-incubation/llm-d-lens/${VERSION}/scripts/LensInstaller-Ubuntu-x86_64-online.sh" | bash
```

Or with **wget**:

```bash
VERSION=main && wget -qO- "https://raw.githubusercontent.com/llm-d-incubation/llm-d-lens/${VERSION}/scripts/LensInstaller-Ubuntu-x86_64-online.sh" | bash
```

Press Enter to accept the defaults (install directory `$HOME/llm-d-lens`, unified app/API port `3000`). When it finishes, open the printed URL (`https://localhost:3000` by default) and sign in with the auto-generated admin credentials — the installer prints the path to the mode-600 file that holds them (`cat` that file to read the username and password; it's removed once you change the password).

The installer reads prompts from your terminal and, after validating the source
archive, saves a uniquely named installer copy in the current directory.
The downloaded script selects its version automatically.
Only releases containing the online installer are supported. Existing nonempty
install directories are rejected. See [Installation](https://llm-d-incubation.github.io/llm-d-lens/installation)
for prerequisites, proxy settings and options.

To install dependencies and run directly from the source checkout:

```bash
hack/lens-local.sh start
hack/lens-local.sh status
```

The source frontend runs on port `5173` and the unified Python API on `3000`. `hack/lens-local.sh` prepares dependencies, manages both processes and validates the Python MCP catalog. To constrain deployments to one node, use `hack/lens-local.sh restart --target-node node-a --kubeconfig "$HOME/.kube/config"`, or set `TARGET_NODE` and `KUBECONFIG`; no target-specific configuration file is needed. `npm run dev:web` also starts both development processes. For production, `npm run build` followed by `npm start` serves the built frontend, API and MCP from Python on one port. See the [CLI Reference](https://llm-d-incubation.github.io/llm-d-lens/cli-reference) for every subcommand.

Build and test:

```bash
npm run build
npm run type-check
make test-js
make test-python
```

Visit the [documentation](https://llm-d-incubation.github.io/llm-d-lens/) to learn more:

- [Installation](https://llm-d-incubation.github.io/llm-d-lens/installation)
- [Quickstart](https://llm-d-incubation.github.io/llm-d-lens/quickstart)
- [User Guide](https://llm-d-incubation.github.io/llm-d-lens/user-guide/model-market)

## Contributing

We welcome and value any contributions. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) before opening an issue or pull request.

Before changing behavior, follow [AGENTS.md](AGENTS.md) and the [reuse-first workflow](docs/design/reuse-workflow.md): find existing implementations, compare contracts and callers, and record reuse or new-code rationale in the PR. All repository text and PR titles and descriptions must be English; run `npm run check:english` before committing.

## License

Lens is released under the [Apache 2.0 license](LICENSE). The project retains the upstream Apache 2.0 license and copyright notices.

## Contact Us

- For bugs and feature requests, use GitHub [Issues](https://github.com/llm-d-incubation/llm-d-lens/issues).
- For security reports, use GitHub's [Security Advisories](https://github.com/llm-d-incubation/llm-d-lens/security/advisories).
