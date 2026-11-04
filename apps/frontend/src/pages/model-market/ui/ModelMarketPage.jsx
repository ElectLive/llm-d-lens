import { requestJson } from '../../../shared/api/httpClient';
import { openClusterSession } from '../../optimization';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { usePolling } from '../../../shared/lib/usePolling';
import { ArrowLeft, Box, CheckCircle2, ChevronDown, Cpu, Database, Rocket, Search, Sparkles } from 'lucide-react';
import { Button, Input, ModuleHeader, ModulePage, Badge, EmptyState } from '../../../shared/ui';
import { CATEGORIES, MODELS } from '@repo/shared/catalog';
import { listStorageVolumes } from '../../storage';
import { listModelCacheEntries } from '../../model-cache';
import { StorageVolumeSelect } from '../../../shared/ui/StorageVolumeSelect';
import { AgenticDeploymentWorkspace } from '../../deployment';
import { storeEvaluationIntent } from '../../evaluation';
import { consumePendingModelMarketCluster } from '../model/transfer';
import { standardRecommendation } from '../model/standardRecommendation';
import { useAuth } from '../../auth';
import { recommendationBudget } from '../../evaluation';

function formatDefaultName(model) {
    const suffix = (globalThis.crypto?.randomUUID?.()
        ?? `${Math.random().toString(16).slice(2)}${Math.random().toString(16).slice(2)}`).slice(0, 8);
    const base = String(model.id || 'model').slice(0, Math.max(1, 63 - suffix.length - 1));
    return `${base}-${suffix}`;
}
const VLLM_ARGUMENT_NAME = /^[a-z][a-z0-9-]*$/;

function parseVllmArguments(input) {
    const argumentsByName = new Map();
    const lines = input.split('\n').map((line) => line.trim()).filter(Boolean);
    for (const line of lines) {
        const match = line.match(/^--?([a-z][a-z0-9-]*)(?:[=\s]+(.+))?$/);
        if (!match || !VLLM_ARGUMENT_NAME.test(match[1])) {
            return { arguments: [], error: `Invalid vLLM parameter: ${line}` };
        }
        argumentsByName.set(match[1], match[2]?.trim() || 'true');
    }
    return { arguments: [...argumentsByName].map(([name, value]) => ({ name, value })), error: '' };
}

export function ModelCard({ model, onSelect }) {
    return (
        <button
            type="button"
            onClick={() => onSelect(model)}
            className="group flex min-h-52 flex-col rounded-lg border border-[#b7d4e6] bg-white p-5 text-left transition hover:border-[#037ffc99] hover:bg-gradient-to-br hover:from-white hover:via-[#037ffc05] hover:to-[#037ffc0d] focus:outline-none focus:ring-2 focus:ring-[#037ffc] dark:border-slate-800 dark:bg-slate-900 dark:bg-gradient-to-br dark:from-slate-900 dark:via-slate-900/70 dark:to-slate-950 dark:shadow-none dark:hover:border-cyan-400/60 dark:hover:from-slate-900 dark:hover:via-slate-900/70 dark:hover:to-slate-900 dark:focus:ring-cyan-400"
        >
            <div className="flex items-start justify-between gap-4">
                <div>
                    <p className="text-xs font-medium uppercase text-[#037ffc] dark:text-cyan-300">{model.family}</p>
                    <h2 className="mt-2 text-lg font-semibold text-slate-900 dark:text-white">{model.name}</h2>
                </div>
                <Box size={19} className="text-[#037ffc] transition group-hover:text-[#037ffc] dark:text-slate-500 dark:group-hover:text-cyan-300" aria-hidden="true" />
            </div>
            <dl className="mt-5 grid grid-cols-2 gap-x-3 gap-y-4 text-xs">
                <div><dt className="text-slate-500">Released</dt><dd className="mt-1 text-slate-700 dark:text-slate-200">{model.releaseDate}</dd></div>
                <div><dt className="text-slate-500">Precision</dt><dd className="mt-1 text-slate-700 dark:text-slate-200">{model.dtype}</dd></div>
                <div><dt className="text-slate-500">Weights</dt><dd className="mt-1 text-slate-700 dark:text-slate-200">{model.weightGiB ? `${model.weightGiB} GiB` : 'Not listed'}</dd></div>
                <div><dt className="text-slate-500">Context</dt><dd className="mt-1 text-slate-700 dark:text-slate-200">{model.contextLength ? model.contextLength.toLocaleString() : 'Not listed'}</dd></div>
                <div className="col-span-2"><dt className="text-slate-500">Model type</dt><dd className="mt-1 text-slate-700 dark:text-slate-200">{model.modelType}</dd></div>
            </dl>
        </button>
    );
}

export default function ModelMarketPage({ onNavigate, isActive = true }) {
    const { can } = useAuth();
    const canDeploy = can('deployment:run:create');
    const [selected, setSelected] = useState(() => {
        try {
            const draft = JSON.parse(sessionStorage.getItem('prism_model_market_return_draft') || 'null');
            sessionStorage.removeItem('prism_model_market_return_draft');
            return MODELS.find((model) => model.repository === draft?.model) || null;
        } catch {
            sessionStorage.removeItem('prism_model_market_return_draft');
            return null;
        }
    });
    const [category, setCategory] = useState('All');
    const [search, setSearch] = useState('');
    const [deployName, setDeployName] = useState('');
    const [deploymentDescription, setDeploymentDescription] = useState('');
    const [cluster, setCluster] = useState('');
    const [replicas, setReplicas] = useState(1);
    const [tensorParallelism, setTensorParallelism] = useState(1);
    const [mode, setMode] = useState('advanced');
    const [resourceBasis] = useState('available');
    const [modelCacheVolumes, setModelCacheVolumes] = useState([]);
    const [storageVolumeId, setStorageVolumeId] = useState('');
    const [storageLoading, setStorageLoading] = useState(false);
    const [cacheCheckLoading, setCacheCheckLoading] = useState(false);
    const [cacheReady, setCacheReady] = useState(false);
    const [cacheCheckError, setCacheCheckError] = useState('');
    const [vllm, setVllm] = useState({
        gpuMemoryUtilization: 0.9,
        maxModelLen: null,
        maxNumSeqs: 256,
        maxNumBatchedTokens: 8192,
        enablePrefixCaching: true,
        enforceEager: false,
        enableAutoToolChoice: false,
        toolCallParser: 'hermes',
    });
    const [placementReplicas] = useState('prefer-same-node');
    const [placementNodeName, setPlacementNodeName] = useState('');
    const [customVllmArguments, setCustomVllmArguments] = useState('');
    const [useClusterProxy] = useState(true);
    const [proxy] = useState({ httpProxy: '', httpsProxy: '', noProxy: '' });
    const [clusters, setClusters] = useState([]);
    const [hardware, setHardware] = useState(null);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState('');
    const [agenticDraft, setAgenticDraft] = useState(null);
    const [pendingCluster] = useState(() => consumePendingModelMarketCluster());

    useEffect(() => {
        if (!selected) return;
        const draft = JSON.parse(sessionStorage.getItem('prism_model_market_return_draft_values') || 'null');
        sessionStorage.removeItem('prism_model_market_return_draft_values');
        if (!draft) return;
        setDeployName(draft.deployName || formatDefaultName(selected));
        setDeploymentDescription(draft.description || '');
        setCluster(draft.cluster || '');
        setReplicas(Number(draft.replicas) || 1);
        setTensorParallelism(Number(draft.tp) || 1);
        setStorageVolumeId(draft.storageVolumeId || '');
        setMode('advanced');
    }, [selected]);

    const loadClusters = useCallback(async () => {
        try {
            const payload = await requestJson('/api/cluster/clusters');
            const items = payload.items || [];
            setClusters(items);
            const preferredClusterId = pendingCluster && items.some((item) => item.id === pendingCluster)
                ? pendingCluster
                : items.find((item) => item.ready)?.id;
            setCluster((current) => current || preferredClusterId || '');
        } catch (failure) {
            setError(failure.message);
        }
    }, [pendingCluster]);

    useEffect(() => {
        loadClusters();
    }, [loadClusters]);
    usePolling(loadClusters, { enabled: isActive });

    const plan = useMemo(() => selected && standardRecommendation(selected, hardware, resourceBasis), [selected, hardware, resourceBasis]);
    const visibleModels = useMemo(() => {
        const terms = search.trim().toLowerCase().split(/\s+/).filter(Boolean);
        return MODELS.filter((model) => {
            if (category !== 'All' && model.category !== category) return false;
            const text = [model.name, model.family, model.repository, model.category, model.modelType, model.dtype].join(' ').toLowerCase();
            return terms.every((term) => text.includes(term));
        });
    }, [category, search]);
    const availableGpuCount = recommendationBudget(hardware, resourceBasis);
    const hasGpuCapacity = Number.isInteger(availableGpuCount);
    const requiresModelCache = mode === 'standard' || mode === 'agentic';
    const requestedGpuCount = replicas * tensorParallelism;
    const parsedVllmArguments = parseVllmArguments(customVllmArguments);
    const vllmArguments = [...new Map([
        ...(selected?.vllmArguments || []).map((argument) => [argument.name, argument]),
        ...(vllm.enableAutoToolChoice ? [
            { name: 'enable-auto-tool-choice', value: 'true' },
            { name: 'tool-call-parser', value: vllm.toolCallParser },
        ] : []).map((argument) => [argument.name, argument]),
        ...parsedVllmArguments.arguments.map((argument) => [argument.name, argument]),
    ]).values()];
    const vllmArgumentControls = <>
        <fieldset className="border-t border-slate-800 pt-3">
            <legend className="text-xs font-medium text-slate-300">Tool calling</legend>
            <label className="mt-3 flex items-center justify-between gap-3 text-xs text-slate-300"><span>Enable automatic tool choice</span><input className="h-4 w-4 accent-cyan-400" type="checkbox" checked={vllm.enableAutoToolChoice} onChange={(event) => setVllm((current) => ({ ...current, enableAutoToolChoice: event.target.checked }))} /></label>
            {vllm.enableAutoToolChoice && <label className="mt-3 block text-xs text-slate-400">Tool call parser<input className="mt-1 h-9 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 text-sm text-slate-200" value={vllm.toolCallParser} onChange={(event) => setVllm((current) => ({ ...current, toolCallParser: event.target.value }))} required /></label>}
        </fieldset>
        <label className="border-t border-slate-800 pt-3 text-xs text-slate-400">Custom vLLM parameters<textarea className="mt-1 min-h-24 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-2 font-mono text-xs text-slate-200" value={customVllmArguments} onChange={(event) => setCustomVllmArguments(event.target.value)} placeholder={'--max-log-len 4096\n--enable-chunked-prefill'} /></label>
        {parsedVllmArguments.error && <p role="alert" className="text-xs text-rose-300">{parsedVllmArguments.error}</p>}
    </>;
    const topologyError = hasGpuCapacity && plan && (tensorParallelism > availableGpuCount || requestedGpuCount > availableGpuCount)
        ? `This topology needs ${requestedGpuCount} GPUs (${replicas} replica${replicas === 1 ? '' : 's'} x TP ${tensorParallelism}), but the cluster has ${availableGpuCount} available.`
        : '';

    useEffect(() => {
        if (!cluster) {
            setHardware(null);
            return undefined;
        }
        let cancelled = false;
        requestJson(`/api/cluster/overview?clusterId=${encodeURIComponent(cluster)}`)
            .then((payload) => {
                if (!cancelled) {
                    setHardware(payload.kubernetes?.hardware || null);
                }
            })
            .catch(() => !cancelled && setHardware(null));
        return () => { cancelled = true; };
    }, [cluster]);

    useEffect(() => {
        setPlacementNodeName('');
        setStorageVolumeId('');
    }, [cluster]);

    useEffect(() => {
        if (!isActive) return undefined;
        if (!cluster) {
            setModelCacheVolumes([]);
            setStorageVolumeId('');
            return undefined;
        }
        let cancelled = false;
        setStorageLoading(true);
        // Fetch every status (not just "ready") so the picker can show
        // non-ready volumes as visible-but-unselectable instead of hiding
        // them entirely.
        listStorageVolumes({ clusterId: cluster, purpose: 'model-cache' })
            .then((volumes) => {
                if (cancelled) return;
                setModelCacheVolumes(volumes);
                // Auto-select the first ready volume so the operator doesn't
                // have to manually pick storage every time a cluster is chosen.
                const firstReady = volumes.find((volume) => volume.status === 'ready');
                setStorageVolumeId((current) => volumes.some((volume) => volume.id === current && volume.status === 'ready')
                    ? current
                    : firstReady?.id || '');
            })
            .catch((failure) => {
                if (!cancelled) {
                    setModelCacheVolumes([]);
                    setStorageVolumeId('');
                    setError(failure.message || 'Unable to load model cache storage');
                }
            })
            .finally(() => {
                if (!cancelled) setStorageLoading(false);
            });
        return () => { cancelled = true; };
    }, [cluster, isActive]);

    useEffect(() => {
        if (!isActive) return undefined;
        setCacheReady(false);
        setCacheCheckError('');
        if (mode === 'agentic' || !selected?.repository || !cluster || !storageVolumeId) {
            setCacheCheckLoading(false);
            return undefined;
        }
        let cancelled = false;
        setCacheCheckLoading(true);
        listModelCacheEntries({ clusterId: cluster, storageVolumeId })
            .then((entries) => {
                if (cancelled) return;
                const expected = selected.repository.trim().toLowerCase();
                setCacheReady(entries.some((entry) => (
                    entry.status === 'ready'
                    && entry.source?.kind === 'huggingface'
                    && String(entry.source?.huggingface?.repoId || '').trim().toLowerCase() === expected
                )));
            })
            .catch((failure) => {
                if (!cancelled) {
                    setCacheReady(false);
                    setCacheCheckError(failure.message || 'Unable to check model cache');
                }
            })
            .finally(() => {
                if (!cancelled) setCacheCheckLoading(false);
            });
        return () => { cancelled = true; };
    }, [selected?.repository, cluster, storageVolumeId, isActive, mode]);

    useEffect(() => {
        if (selected && hardware) {
            setReplicas(plan.replicas);
            setTensorParallelism(plan.tensorParallelism);
        }
    }, [selected, hardware, plan]);

    const chooseModel = (model) => {
        setSelected(model);
        setDeployName(formatDefaultName(model));
        setDeploymentDescription('');
        setReplicas(standardRecommendation(model, hardware, resourceBasis).replicas);
        setTensorParallelism(standardRecommendation(model, hardware, resourceBasis).tensorParallelism);
        const isPoolingModel = ['Encoder', 'Rerank'].includes(model.category);
        setVllm((current) => ({
            ...current,
            maxModelLen: standardRecommendation(model, hardware, resourceBasis).maxModelLen,
            maxNumSeqs: isPoolingModel ? 512 : 256,
            maxNumBatchedTokens: isPoolingModel ? 16384 : 8192,
            enablePrefixCaching: !isPoolingModel,
        }));
        // A previously captured Agentic draft snapshots `model` at the moment
        // the Agentic tab was clicked (see the Mode fieldset below). Picking a
        // different model here must invalidate that stale snapshot -- and
        // fall back to Manual -- so the workspace can't silently redeploy
        // whatever model was selected before.
        setMode('advanced');
        setAgenticDraft(null);
    };

    const openAdvanced = () => {
        storeEvaluationIntent({
            entrySource: 'model-market',
            operation: 'create-evaluation',
            initial_step: 'configurations',
            return_target: 'optimization-evaluate-new',
            name: deployName || formatDefaultName(selected),
            deployment_name: deployName || formatDefaultName(selected),
            deployment_description: deploymentDescription,
            cluster_id: cluster,
            workloads: [{ model: selected.repository, replicas, tensor_parallel_size: tensorParallelism }],
            runtime: {
                model_server: 'vllm',
                model_source: storageVolumeId ? 'auto-cache' : 'huggingface',
                storage_volume_id: storageVolumeId,
                max_model_len: vllm.maxModelLen ?? plan.maxModelLen,
            },
        });
        onNavigate('optimization-evaluate-new');
    };

    const submit = async (requestedResourceBasis = resourceBasis) => {
        setError('');
        if (parsedVllmArguments.error) {
            setError(parsedVllmArguments.error);
            return;
        }
        const requestedGpuBudget = recommendationBudget(hardware, requestedResourceBasis);
        if (mode === 'standard' && (tensorParallelism > requestedGpuBudget || requestedGpuCount > requestedGpuBudget)) {
            setError(`This topology needs ${requestedGpuCount} GPUs (${replicas} replica${replicas === 1 ? '' : 's'} x TP ${tensorParallelism}), but the selected budget has ${requestedGpuBudget} available. Change the replica count or choose a sufficient budget.`);
            return;
        }
        if (requiresModelCache && !storageVolumeId) {
            setError('Select a model cache storage for Standard or Agentic deployment, or choose Advanced to configure another model source.');
            return;
        }
        if (mode === 'standard' && !cacheReady) {
            setError('This model is not ready in the selected model cache storage. Ask an admin to download it before deploying.');
            return;
        }
        if (mode === 'advanced') {
            openAdvanced();
            return;
        }
        if (mode === 'agentic') {
            setAgenticDraft({
                model: selected.repository,
                deploymentName: deployName,
                description: deploymentDescription,
                clusterId: cluster,
                modelWeightGiB: selected.sizeGiB,
                contextLength: vllm.maxModelLen || plan.maxModelLen,
                storageVolumeId,
                maxNumSeqs: vllm.maxNumSeqs,
                maxNumBatchedTokens: vllm.maxNumBatchedTokens,
                gpuMemoryUtilization: vllm.gpuMemoryUtilization,
                vllmArguments,
            });
            return;
        }
        setSubmitting(true);
        try {
            const clusterSessionId = await openClusterSession(cluster);
            await requestJson('/api/v1/deployments/standard-vllm-runs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: selected.repository,
                    deployment_name: deployName,
                    description: deploymentDescription,
                    cluster_session_id: clusterSessionId,
                    replicas,
                    tensor_parallel_size: tensorParallelism,
                    placement_replicas: placementReplicas,
                    placement_node_label: placementReplicas === 'same-node' && placementNodeName ? { key: 'kubernetes.io/hostname', value: placementNodeName } : null,
                    storage_type: 'model-cache',
                    storage_volume_id: storageVolumeId,
                    max_model_len: vllm.maxModelLen ?? plan.maxModelLen,
                    gpu_memory_utilization: vllm.gpuMemoryUtilization,
                    max_num_seqs: vllm.maxNumSeqs,
                    max_num_batched_tokens: vllm.maxNumBatchedTokens,
                    enable_prefix_caching: vllm.enablePrefixCaching,
                    enforce_eager: vllm.enforceEager,
                    vllm_arguments: vllmArguments,
                    use_cluster_proxy: useClusterProxy,
                    http_proxy: proxy.httpProxy,
                    https_proxy: proxy.httpsProxy,
                    no_proxy: proxy.noProxy,
                }),
            });
            onNavigate('optimization-deployments');
        } catch (failure) {
            setError(failure.message || 'Unable to create deployment');
        } finally {
            setSubmitting(false);
        }
    };

    if (!selected) {
        return (
            <ModulePage contentClassName="flex flex-col gap-4 py-4 lg:py-5">
                <ModuleHeader icon={Database} title="Model Catalog" />
                <div className="flex flex-wrap items-center gap-3">
                    <div className="flex min-w-0 w-full flex-wrap gap-2 lg:w-auto lg:flex-1" role="tablist" aria-label="Model categories">{CATEGORIES.map((item) => <button key={item} type="button" role="tab" aria-selected={category === item} onClick={() => setCategory(item)} className={`rounded-md border px-3 py-1.5 text-xs ${category === item ? 'border-[#037ffc] bg-[#037ffc] text-white dark:border-cyan-400 dark:bg-cyan-400/10 dark:text-cyan-200' : 'border-slate-700 text-slate-400 hover:border-slate-500'}`}>{item}</button>)}</div>
                    <div className="ml-auto flex max-w-full flex-wrap items-center gap-2">
                        <span role="status" className="text-xs text-theme-muted">{visibleModels.length} models</span>
                        <div className="w-52 sm:w-64">
                        <Input type="search" icon={Search} value={search} onChange={(event) => setSearch(event.target.value)}
                            aria-label="Search models" placeholder="Search models, publishers, repositories, types, or precision..." />
                        </div>
                        {(search || category !== 'All') && <Button type="button" variant="secondary" size="sm" onClick={() => { setSearch(''); setCategory('All'); }}>Clear filters</Button>}
                    </div>
                </div>
                <section className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                    {visibleModels.map((model) => <ModelCard key={model.id} model={model} onSelect={chooseModel} />)}
                    {!visibleModels.length && <EmptyState className="col-span-full" icon={<Search size={22} aria-hidden="true" />}
                        title="No matching models" message="Try another search or change the model category." />}
                </section>
            </ModulePage>
        );
    }

    const catalogBackButton = <Button variant="secondary" onClick={() => setSelected(null)} className="self-start"><ArrowLeft size={15} aria-hidden="true" />Model Catalog</Button>;

    if (!canDeploy) {
        return (
            <ModulePage contentClassName="flex flex-col gap-5">
                <ModuleHeader icon={Rocket} title="One-click deployment" />
                {catalogBackButton}
                <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-6">
                    <p className="text-sm font-medium text-slate-200">Your role cannot create deployments.</p>
                    <p className="mt-2 text-sm leading-6 text-slate-400">You can use deployments that already exist or were shared with you. Open the Deployments page to view them, or ask an administrator to grant you the deployment-create permission.</p>
                    <div className="mt-5"><Button variant="sky" onClick={() => onNavigate?.('optimization-deployments')}>Open Deployments</Button></div>
                </div>
            </ModulePage>
        );
    }

    return (
        <ModulePage contentClassName="flex flex-col gap-5">
            <ModuleHeader icon={Rocket} title="One-click deployment" />
            {catalogBackButton}
            <div className="grid grid-cols-1 gap-5">
                <form id="model-market-deployment-form" className="rounded-lg border border-slate-800 bg-slate-900/70 p-6" onSubmit={(event) => { event.preventDefault(); submit(); }}>
                    <div className="flex items-start justify-between gap-4 border-b border-slate-800 pb-5"><div><p className="text-xs uppercase text-cyan-300">{selected.family}</p><h1 className="mt-1 text-xl font-semibold text-white">{selected.name}</h1></div><span className="text-sm text-slate-400">{selected.sizeGiB} GiB</span></div>
                    <div className="mt-6 grid gap-5 md:grid-cols-2">
                        <label className="text-sm text-slate-300">Deployment name<Input className="mt-2" value={deployName} onChange={(event) => setDeployName(event.target.value)} required /></label>
                        <label className="text-sm text-slate-300 md:col-span-2">Description<textarea className="mt-2 min-h-24 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200 outline-none transition placeholder:text-slate-600 focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/10" value={deploymentDescription} onChange={(event) => setDeploymentDescription(event.target.value)} maxLength={1000} placeholder="Optional deployment description" /></label>
                        <label className="text-sm text-slate-300">Cluster<select className="mt-2 h-10 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 text-sm text-slate-200" value={cluster} onChange={(event) => setCluster(event.target.value)} required><option value="">Select cluster</option>{clusters.map((item) => <option key={item.id} value={item.id} disabled={!item.ready}>{item.name}{item.ready ? '' : ' (not ready)'}</option>)}</select></label>
                        {mode !== 'agentic' && <><label className="order-3 text-sm text-slate-300">Replicas<input className="mt-2 h-10 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 text-sm text-slate-200" type="number" min="1" max="32" value={replicas} onChange={(event) => setReplicas(Number(event.target.value))} /></label><label className="order-3 text-sm text-slate-300">Tensor parallelism<input className="mt-2 h-10 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 text-sm text-slate-200" type="number" min="1" max="32" value={tensorParallelism} onChange={(event) => setTensorParallelism(Number(event.target.value))} /></label></>}
                        <label className="order-2 md:col-span-2 text-sm text-slate-300">
                            Model cache storage <span className="text-xs text-slate-500">{requiresModelCache ? 'Required for Agentic' : 'Optional for Manual'}</span>
                            <p className="mt-1 text-xs leading-5 text-slate-500">Select a registered hostPath or NFS volume for model weights. This is not the Tiered Prefix Cache KV filesystem.</p>
                            <StorageVolumeSelect clearable className="mt-2" value={storageVolumeId} onChange={setStorageVolumeId} volumes={modelCacheVolumes} loading={storageLoading} disabled={!cluster} placeholder={requiresModelCache ? 'Select registered model storage' : 'No model storage selected'} emptyLabel={storageLoading ? 'Loading model cache storage...' : 'No ready model-cache storage'} optionDetail={(volume) => volume.capacity || volume.kind} />
                            <div className="mt-2 flex gap-3">
                                {storageVolumeId && <button type="button" onClick={() => setStorageVolumeId('')} className="text-xs text-slate-400 underline hover:text-cyan-300">Clear selection</button>}
                                <button type="button" onClick={() => onNavigate('model-cache', { downloadModel: { clusterId: cluster, repoId: selected.repository, storageVolumeId } })} className="text-xs text-cyan-300 underline">Manage model cache</button>
                            </div>
                            {!storageLoading && cluster && modelCacheVolumes.length === 0 && <p role="alert" className="mt-2 text-xs text-amber-300">No registered model-cache volume exists. Register a Model Cache volume in Storage before downloading this model.</p>}
                            {mode === 'agentic'
                                ? <p className="mt-2 text-xs text-theme-muted">Agentic recommendations skip model download/readiness checks. The selected cache must contain the ready model before deployment approval.</p>
                                : <>
                                    {storageVolumeId && cacheCheckLoading && <p className="mt-2 text-xs text-slate-500">Checking whether this model is cached...</p>}
                                    {storageVolumeId && cacheCheckError && <p role="alert" className="mt-2 text-xs text-rose-300">{cacheCheckError}</p>}
                                    {storageVolumeId && !cacheCheckLoading && !cacheCheckError && !cacheReady && <p role="alert" className="mt-2 text-xs text-amber-300">This model is not downloaded in the selected storage. Clear it or download the model there first.</p>}
                                    {storageVolumeId && !cacheCheckLoading && cacheReady && <p className="mt-2 text-xs text-emerald-300">This model is ready in the selected cache.</p>}
                                </>}
                        </label>
                    </div>
                    {!selected.vllmSupported && <p role="alert" className="mt-5 text-sm text-amber-300">{selected.type} requires the {selected.runtime} runtime. Raw vLLM Standard deployment is unavailable until this provider is added.</p>}
                    {mode === 'standard' && topologyError && <p role="alert" className="mt-5 text-sm text-amber-300">{topologyError} The recommended TP remains TP {plan.tensorParallelism}; modify replicas or TP before creating this deployment.</p>}
                    <fieldset className="mt-6"><legend className="text-sm text-slate-300">Mode</legend><div className="mt-2 grid gap-2 md:grid-cols-2">{[['advanced', 'Manual', 'Template-based deployment', false], ['agentic', 'Agentic', 'Recommended guide and topology', true]].map(([value, label, description, experimental]) => <button key={value} type="button" onClick={() => {
                        if (value === 'agentic') {
                            setMode(value);
                            setAgenticDraft({ model: selected.repository, deploymentName: deployName, description: deploymentDescription, clusterId: cluster, modelWeightGiB: selected.sizeGiB, contextLength: vllm.maxModelLen || plan.maxModelLen, storageVolumeId, maxNumSeqs: vllm.maxNumSeqs, maxNumBatchedTokens: vllm.maxNumBatchedTokens, gpuMemoryUtilization: vllm.gpuMemoryUtilization, vllmArguments: selected.vllmArguments || [] });
                            return;
                        }
                        setMode(value);
                    }} className={`mode-option rounded-lg border p-3 text-left ${mode === value ? 'mode-option-selected border-cyan-400 bg-cyan-400/10' : 'border-slate-700 bg-slate-950 hover:border-slate-600'}`}><span className="flex items-center gap-2 text-sm font-medium text-slate-900 dark:text-white">{label}{experimental && <Badge tone="warning" size="xs" className="experimental-chip">Experimental</Badge>}</span><span className="mt-1 block text-xs text-slate-500">{description}</span></button>)}</div></fieldset>
                    {error && <p role="alert" className="mt-5 text-sm text-rose-300">{error}{error.startsWith('This model is not ready') && <> <button type="button" className="underline" onClick={() => onNavigate?.('model-cache')}>Open Model Cache</button></>}</p>}
                    {mode === 'advanced' && <div className="mt-7 flex justify-end"><Button type="submit" disabled={!cluster || !deployName || submitting}><Rocket size={16} aria-hidden="true" />{submitting ? 'Creating deployment...' : 'Configure deployment'}</Button></div>}
                </form>
                {mode === 'agentic' && <aside><AgenticDeploymentWorkspace embedded draft={agenticDraft && { ...agenticDraft, storageVolumeId, vllmArguments }} vllmParameterControls={vllmArgumentControls} onBack={() => { setAgenticDraft(null); setMode('advanced'); }} onDeploymentStarted={() => onNavigate('optimization-deployments')} /></aside>}
            </div>
        </ModulePage>
    );
}
