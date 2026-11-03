// Deployment API client shared by configuration, monitoring and task pages.

import { requestJson, postJson as post } from '../../../../shared/api/httpClient';

async function rebindLocalDeploymentRun(runId) {
    const clusterSessionId = sessionStorage.getItem('prism_cluster_session_id');
    if (!clusterSessionId) return;
    await post(`/api/v1/deployments/runs/${encodeURIComponent(runId)}/cluster-session`, {
        cluster_session_id: clusterSessionId,
    });
}

export async function listDeploymentNodeLabels(clusterId, signal) {
    const payload = await requestJson(`/api/v1/deployments/node-labels?cluster_id=${encodeURIComponent(clusterId)}`, { signal });
    return payload.items || [];
}

export async function startLocalDeployment(configurations, provenance, placement) {
    return post('/api/v1/deployments/runs', {
        configurations,
        provenance,
        ...(placement ? { runtime_binding: { placement } } : {}),
    });
}

export async function getLocalDeploymentRun(runId) {
    const payload = await requestJson(`/api/v1/deployments/runs/${encodeURIComponent(runId)}`);
    return payload;
}

export async function listDeploymentExecutions({ query, statuses, status, clusterId, limit = 200, signal } = {}) {
    const params = new URLSearchParams();
    if (status) params.set('status', status);
    for (const value of statuses || []) {
        if (value) params.append('statuses', value);
    }
    if (clusterId) params.set('cluster_id', clusterId);
    if (query) params.set('query', query);
    if (limit) params.set('limit', String(limit));
    const suffix = params.size ? `?${params.toString()}` : '';
    const payload = await requestJson(`/api/v1/deployments/executions${suffix}`, { signal });
    if (!Array.isArray(payload.items)) throw new Error('Deployment execution list response is invalid');
    return payload.items;
}

export async function updateDeploymentMetadata(executionId, { displayName, description } = {}) {
    const body = {};
    if (displayName !== undefined) body.display_name = displayName;
    if (description !== undefined) body.description = description;
    const payload = await requestJson(`/api/v1/deployments/executions/${encodeURIComponent(executionId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    return payload;
}

export async function scaleDeploymentExecution(executionId, { prefillReplicas, decodeReplicas } = {}) {
    const body = {};
    if (prefillReplicas !== undefined && prefillReplicas !== null) body.prefill_replicas = prefillReplicas;
    if (decodeReplicas !== undefined && decodeReplicas !== null) body.decode_replicas = decodeReplicas;
    return requestJson(`/api/v1/deployments/executions/${encodeURIComponent(executionId)}/scale`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
}

export async function deleteDeploymentExecution(executionId, { deleteNamespace, runId } = {}) {
    // Deleting can clean up cluster resources (see delete_namespace), which needs
    // a live cluster session. Rebind proactively, matching stop/restart/clean below,
    // instead of failing on a stale session bound at deployment-create time.
    if (runId) {
        await rebindLocalDeploymentRun(runId);
    }
    const params = new URLSearchParams();
    if (deleteNamespace !== undefined) params.set('delete_namespace', deleteNamespace ? 'true' : 'false');
    const query = params.toString();
    const url = `/api/v1/deployments/executions/${encodeURIComponent(executionId)}${query ? `?${query}` : ''}`;
    // Use the shared transport so the request carries credentials and the
    // double-submit CSRF header that Python requires for unsafe methods, while
    // still enforcing the endpoint's strict 204 contract.
    await requestJson(url, { method: 'DELETE' }, { expectedStatus: 204 });
}

export async function getDeploymentExecutions({ status, clusterId } = {}) {
    const items = await listDeploymentExecutions({ status, clusterId });
    // Ready deployments need a reachable host-side URL for existing-endpoint
    // consumers. Establish/reuse the tunnel opportunistically; benchmark
    // execution can still fall back to the in-cluster endpoint on failure.
    if (status === "ready") {
        const resolved = await Promise.all(items.map(async (item) => {
            if (!item?.execution_id) return null;
            let next = item;
            if (!item.forwarded_endpoint) {
                try {
                    const payload = await requestJson(`/api/v1/deployments/executions/${encodeURIComponent(item.execution_id)}/endpoint`, { method: 'POST' });
                    next = { ...item, ...payload, forwarded_endpoint: payload.forwarded_endpoint || item.forwarded_endpoint };
                } catch { /* keep the ready deployment */ }
            }
            // Do not probe forwarded localhost URLs from the browser: 127.0.0.1
            // refers to the user's machine and HTTPS pages may block the HTTP request.
            return { ...next, endpoint_health: 'unknown' };
        }));
        return resolved.filter(Boolean);
    }
    return items;
}

export async function getDeploymentExecution(executionId) {
    const payload = await requestJson(`/api/v1/deployments/executions/${encodeURIComponent(executionId)}`);
    return payload;
}

export async function getDeploymentExecutionPods(executionId) {
    const payload = await requestJson(`/api/v1/deployments/executions/${encodeURIComponent(executionId)}/pods`);
    return payload;
}

export async function getDeploymentExecutionPodLogs(executionId, podName, { container, tail = 200, previous = false } = {}) {
    const params = new URLSearchParams();
    if (container) params.set('container', container);
    if (tail) params.set('tail', String(tail));
    if (previous) params.set('previous', 'true');
    const suffix = params.size ? `?${params.toString()}` : '';
    const payload = await requestJson(`/api/v1/deployments/executions/${encodeURIComponent(executionId)}/pods/${encodeURIComponent(podName)}/logs${suffix}`);
    return payload;
}

export async function connectDeploymentExecution(executionId) {
    return post(`/api/v1/deployments/executions/${encodeURIComponent(executionId)}/endpoint`, {});
}

export async function getLocalDeploymentCase(runId, caseId) {
    const payload = await requestJson(`/api/v1/deployments/runs/${encodeURIComponent(runId)}/cases/${encodeURIComponent(caseId)}`);
    return payload;
}

export async function refreshLocalDeploymentCase(runId, caseId) {
    return post(`/api/v1/deployments/runs/${encodeURIComponent(runId)}/cases/${encodeURIComponent(caseId)}/refresh`, {});
}

export async function stopLocalDeploymentCase(runId, caseId) {
    await rebindLocalDeploymentRun(runId);
    return post(`/api/v1/deployments/runs/${encodeURIComponent(runId)}/cases/${encodeURIComponent(caseId)}/stop`, {});
}

export async function restartLocalDeploymentCase(runId, caseId) {
    await rebindLocalDeploymentRun(runId);
    return post(`/api/v1/deployments/runs/${encodeURIComponent(runId)}/cases/${encodeURIComponent(caseId)}/restart`, {});
}

export async function cleanLocalDeploymentCase(runId, caseId, preserveRenderedOverlay = false) {
    await rebindLocalDeploymentRun(runId);
    return post(`/api/v1/deployments/runs/${encodeURIComponent(runId)}/cases/${encodeURIComponent(caseId)}/clean`, {
        preserve_rendered_overlay: preserveRenderedOverlay,
    });
}

export async function searchLocalDeploymentRuns(query, { signal } = {}) {
    const payload = await requestJson(`/api/v1/deployments/runs?query=${encodeURIComponent(query)}`, { signal }, { fallback: [] });
    if (!Array.isArray(payload)) throw new Error('Deployment run list response is invalid');
    return payload;
}

export function pendingDeploymentExecutions(runs, executionIds = new Set()) {
    return runs.flatMap((run) => (run.cases || [])
        .filter((deploymentCase) => !deploymentCase.execution_id && !executionIds.has(deploymentCase.execution_id))
        .map((deploymentCase) => {
            const configuration = run.source_configurations?.[deploymentCase.source_configuration_ordinal] || {};
            const content = configuration.content || {};
            const decode = content.decode || content.serving || {};
            const runtime = content.runtime || {};
            const customParameters = content.customParameters || [];
            const paramsMap = {};
            for (const p of customParameters) {
                if (p?.name) paramsMap[p.name] = p.value;
            }
            const provenance = deploymentCase.create_request?.provenance || run.provenance || {};
            return {
                execution_id: `pending:${run.id}:${deploymentCase.id}`,
                pending: true,
                status: deploymentCase.status || run.status || 'queued',
                cluster_id: provenance.cluster_server_id || run.provenance?.cluster_server_id || null,
                cluster_session_id: provenance.cluster_session_id || run.provenance?.cluster_session_id || null,
                name: provenance.deployment_name || provenance.model_market?.deployment_name || content.model?.name || 'Deployment',
                display_name: provenance.deployment_name || provenance.model_market?.deployment_name || '',
                description: deploymentCase.failure?.detail || 'Preparing deployment',
                model: content.model?.name || null,
                backend: 'vLLM',
                guide: deploymentCase.provider_ref || configuration.provider_ref || null,
                replicas: decode.replicaCount ?? null,
                tensor_parallel_size: decode.tensorParallelSize ?? null,
                max_model_len: decode.maxModelLen ?? null,
                gpu_memory_utilization: paramsMap['gpu-memory-utilization'] ?? null,
                enable_prefix_caching: paramsMap['enable-prefix-caching'] ?? null,
                max_num_seqs: paramsMap['max-num-seqs'] ?? null,
                max_num_batched_tokens: paramsMap['max-num-batched-tokens'] ?? null,
                storage_type: runtime.storageType || (runtime.storageVolumeId ? 'model-cache' : runtime.pvcName ? 'pvc' : runtime.mountPath ? 'local-cache' : null),
                storage_volume_id: runtime.storageVolumeId || null,
                mount_path: runtime.mountPath || null,
                pvc_name: runtime.pvcName || null,
                image: runtime.image || null,
                custom_parameters: customParameters,
                failure: deploymentCase.failure || null,
                preserve_deployment: false,
                created_at: run.created_at,
                updated_at: run.started_at || run.created_at,
            };
        }));
}
