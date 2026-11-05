#!/usr/bin/env bash
# Shared variables, as llm-d's guides/env.sh defines them.
export REPO_ROOT=${REPO_ROOT:-$(realpath "$(git rev-parse --show-toplevel 2>/dev/null)" 2>/dev/null)}
export GAIE_VERSION=${GAIE_VERSION:-latest}
if [[ $GAIE_VERSION == "latest" ]]; then
  export GAIE_URL=releases/latest/download
else
  export GAIE_URL=releases/download/${GAIE_VERSION}
fi
export ROUTER_CHART_VERSION=${ROUTER_CHART_VERSION:-0.1.0}
export ROUTER_STANDALONE_CHART=${ROUTER_STANDALONE_CHART:-${REPO_ROOT}/charts/router}
