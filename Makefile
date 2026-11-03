# Project configuration
IMAGE ?= lens
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
PLATFORMS ?= linux/amd64,linux/arm64

# Container engine: docker buildx if present, else podman. Override with
# CONTAINER_TOOL=podman to force one.
CONTAINER_TOOL ?= $(shell docker buildx version >/dev/null 2>&1 && echo docker || echo podman)

.DEFAULT_GOAL := help

##@ General

.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: build
build: ## Build the frontend
	npm run build

.PHONY: test test-js test-app-js test-reuse test-python
test: test-js test-python ## Run Node and Python tests

test-js: test-app-js test-reuse ## Run frontend, shared and reuse unit tests

test-app-js: ## Run frontend and shared unit tests
	TSX_TSCONFIG_PATH=tsconfig.base.json node --import tsx --test $$(find apps/frontend/src packages/shared/src -type f \( -name '*.test.js' -o -name '*.test.jsx' -o -name '*.test.ts' \))

test-reuse: ## Run reuse tooling tests
	npm run reuse:test

PYTHON ?= .venv/bin/python
test-python: ## Run Python engine and backend tests
	$(PYTHON) -m pytest apps/lens/tests
	$(PYTHON) -m pytest apps/backend/tests

.PHONY: lint lint-js lint-python
lint: lint-js lint-python ## Run JavaScript and Python linters

lint-js:
	npm run lint

lint-python:
	ruff check apps/lens apps/backend/src apps/backend/tests
	ruff format --check apps/lens apps/backend/src apps/backend/tests

.PHONY: fmt
fmt: ## Format Python code
	ruff format apps/lens apps/backend/src apps/backend/tests

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Manual local checkpoint; keep detailed test output out of the agent context.
.PHONY: verify-full
verify-full: ## Run all local checks with compact output and per-check logs
	@bash -eu -c '\
	mkdir -p .cache/reuse/full-validation; \
	logs=$$(mktemp -d "$$PWD/.cache/reuse/full-validation/run-XXXXXX"); \
	export PRISM_PYTHON_BIN="$${PRISM_PYTHON_BIN:-$$(if test -x .venv/bin/python; then echo "$$PWD/.venv/bin/python"; else echo python3; fi)}"; \
	export LENS_DATA_DIR="$$logs/data" LENS_CACHE_DIR="$$logs/cache" LENS_LOG_DIR="$$logs/logs" LENS_SCRATCH_DIR="$$logs/scratch" LLM_D_BENCH_DB_ALLOW_SQLITE=1; \
	failed=0; \
	run() { name=$$1; shift; echo "RUN $$name"; if "$$@" >"$$logs/$$name.log" 2>&1; then echo "PASS $$name"; else echo "FAIL $$name: $$logs/$$name.log"; failed=1; fi; }; \
	echo "Full local validation; logs: $$logs"; \
	run english npm run check:english; \
	run catalog npm run reuse:check; \
	run stack npm run check:stack -- --offline; \
	run lint npm run lint; \
	run python-lint "$$PRISM_PYTHON_BIN" -m ruff check .; \
	run python-format "$$PRISM_PYTHON_BIN" -m ruff format --check .; \
	run mcp node hack/generate-mcp-tools.mjs; \
	run types npm run type-check; \
	run javascript-tests $(MAKE) test-app-js; \
	run reuse-tests $(MAKE) test-reuse; \
	run tool-tests node --test hack/check-docs.test.mjs hack/check-english.test.mjs hack/check-llm-d-stack.test.mjs; \
	run engine-tests "$$PRISM_PYTHON_BIN" -m pytest apps/lens/tests -q; \
	run python-tests "$$PRISM_PYTHON_BIN" -m pytest apps/backend/tests apps/backend/src/llm_d_lens -q; \
	run frontend-build npm run build; \
	run docs-build npm run docs:build; \
	if test $$failed = 0; then echo "PASS: all local checks"; else echo "FAIL: see logs above"; fi; \
	exit $$failed'

##@ Container

.PHONY: image-build
image-build: ## Build multi-arch container image (local only)
ifeq ($(CONTAINER_TOOL),docker)
	docker buildx build \
		--platform $(PLATFORMS) \
		--tag $(IMAGE):$(VERSION) \
		--tag $(IMAGE):latest \
		-f deploy/Dockerfile \
		.
else
	podman manifest rm $(IMAGE):$(VERSION) 2>/dev/null || true
	podman build \
		--platform $(PLATFORMS) \
		--manifest $(IMAGE):$(VERSION) \
		-f deploy/Dockerfile \
		.
endif
