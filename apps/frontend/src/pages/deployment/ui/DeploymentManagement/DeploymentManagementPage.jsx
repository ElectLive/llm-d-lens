import { useDebouncedValue } from '../../../../shared/lib/useDebouncedValue';
import { useNotice } from '../../../../shared/lib/useNotice';
import { paginate } from '../../../../shared/lib/pagination';
import { AsyncState } from '../../../../shared/ui/AsyncState';
import { PaginationControls } from '../../../../shared/ui/PaginationControls';
import { useClusterNames } from '../../../optimization';
import { useResourceList } from '../../../../shared/lib/useResourceList';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, Plus, Rocket, Search, ServerCog } from 'lucide-react';
import {
    deleteDeploymentExecution,
    connectDeploymentExecution,
    listDeploymentExecutions,
    pendingDeploymentExecutions,
    scaleDeploymentExecution,
    searchLocalDeploymentRuns,
    updateDeploymentMetadata,
} from '../../../optimization';
import { manageDeploymentMonitoring } from '../../../clusters';
import { MultiSelectDropdown } from '../../../../shared/ui/MultiSelectDropdown';
import { EmptyState } from '../../../../shared/ui/EmptyState';
import { Button, RefreshButton } from '../../../../shared/ui/Button';
import { StatCardRow } from '../../../../shared/ui/StatCard';
import { Input, Label, Select } from '../../../../shared/ui/FormControls';
import { Modal } from '../../../../shared/ui/Modal';
import { FormError } from '../../../../shared/ui/FormError';
import { useSubmission } from '../../../../shared/lib/useSubmission';
import { SectionLabel } from '../../../../shared/ui/SectionLabel';
import { ModuleHeader } from '../../../../shared/ui/ModuleHeader';
import { ModulePage } from '../../../../shared/ui/ModulePage';
import { DeploymentTable } from './DeploymentTable';
import { DeploymentDetailsModal } from './DeploymentDetailsModal';
import { EditDeploymentModal } from './EditDeploymentModal';
import { DeleteDeploymentModal } from './DeleteDeploymentModal';
import { DeploymentPodsLogsModal } from './DeploymentPodsLogsModal';
import { Toast } from '../../../../shared/ui/Toast';
import { useAuth } from '../../../auth';
import { usePolling } from '../../../../shared/lib/usePolling';
import {
    STATUS_FILTER_OPTIONS,
    clusterOptions,
    deploymentTitle,
    errorMessage,
    statusLabel,
    summarizeStatuses,
} from './deploymentPresentation';

const SEARCH_DEBOUNCE_MS = 300;
const DEPLOYMENT_REFRESH_INTERVAL_MS = 5000;
const PAGE_SIZE_OPTIONS = [10, 20, 50, 100];
const DEFAULT_PAGE_SIZE = 20;

function StatCard({ label, value, tone = 'text-white' }) {
    return (
        <div className="flex h-full flex-col gap-2 rounded-xl border border-slate-700/60 bg-slate-800/60 p-4 text-left shadow-lg backdrop-blur-xl">
            <div className="truncate text-xs font-medium text-slate-400">{label}</div>
            <div className={`truncate text-3xl font-bold leading-none ${tone}`}>{value}</div>
        </div>
    );
}

function ScaleDeploymentModal({ deployment, onCancel, onScale }) {
    const prefillSupported = deployment?.scale?.prefillReplicas != null;
    const currentDecode = deployment?.scale?.decodeReplicas ?? deployment?.replicas ?? '';
    const currentPrefill = deployment?.scale?.prefillReplicas ?? '';
    const [decodeReplicas, setDecodeReplicas] = useState(String(currentDecode));
    const [prefillReplicas, setPrefillReplicas] = useState(String(currentPrefill));
    const { pending: scaling, error, run } = useSubmission('Failed to scale deployment', { keepPendingOnSuccess: true });
    const decodeRef = useRef(null);

    useEffect(() => {
        decodeRef.current?.focus();
    }, []);

    if (!deployment) return null;

    const parseReplicas = (value) => {
        const parsed = Number(value);
        return Number.isInteger(parsed) && parsed >= 1 && parsed <= 100 ? parsed : null;
    };
    const decode = parseReplicas(decodeReplicas);
    const prefill = prefillSupported ? parseReplicas(prefillReplicas) : null;
    const unchanged = decode === currentDecode && (!prefillSupported || prefill === currentPrefill);
    const invalid = decode === null || (prefillSupported && prefill === null) || unchanged;

    const submit = async (event) => {
        event.preventDefault();
        if (invalid || scaling) return;
        await run(() => onScale({
            ...(decode !== currentDecode ? { decodeReplicas: decode } : {}),
            ...(prefillSupported && prefill !== currentPrefill ? { prefillReplicas: prefill } : {}),
        }));
    };

    return (
        <Modal
            isOpen
            onClose={scaling ? undefined : onCancel}
            title="Scale deployment"
            subtitle={deploymentTitle(deployment)}
            closeOnBackdrop={!scaling}
            closeOnEscape={!scaling}
            footer={(
                <>
                    <Button variant="secondary" onClick={onCancel} disabled={scaling}>Cancel</Button>
                    <Button variant="sky" type="submit" form="scale-deployment-form" isLoading={scaling} disabled={invalid}>
                        {scaling ? 'Scaling' : 'Scale deployment'}
                    </Button>
                </>
            )}
        >
            <form id="scale-deployment-form" onSubmit={submit} className="flex flex-col gap-4">
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    {prefillSupported && (
                        <div>
                            <Label htmlFor="scale-prefill-replicas">Prefill replicas</Label>
                            <Input
                                id="scale-prefill-replicas"
                                type="number"
                                min="1"
                                max="100"
                                step="1"
                                value={prefillReplicas}
                                onChange={(event) => setPrefillReplicas(event.target.value)}
                                disabled={scaling}
                                error={Boolean(prefillReplicas) && parseReplicas(prefillReplicas) === null}
                            />
                            <p className="mt-1 text-[10px] text-slate-500">Current: {currentPrefill || '—'}</p>
                        </div>
                    )}
                    <div>
                        <Label htmlFor="scale-decode-replicas">{prefillSupported ? 'Decode replicas' : 'Replicas'}</Label>
                        <Input
                            id="scale-decode-replicas"
                            ref={decodeRef}
                            type="number"
                            min="1"
                            max="100"
                            step="1"
                            value={decodeReplicas}
                            onChange={(event) => setDecodeReplicas(event.target.value)}
                            disabled={scaling}
                            error={Boolean(decodeReplicas) && decode === null}
                        />
                        <p className="mt-1 text-[10px] text-slate-500">Current: {currentDecode || '—'}</p>
                    </div>
                </div>
                <p className="rounded-lg border border-slate-800/60 bg-slate-900/40 px-3 py-2 text-[11px] text-slate-500">
                    This changes live Kubernetes replicas only. Restarting the deployment restores its configured replica count.
                </p>
                {(Number(deployment.scale?.slices) || 1) > 1 && (
                    <p className="text-[11px] text-theme-muted">
                        Values are per slice; this deployment has {deployment.scale.slices} slices
                        ({(Number(deployment.scale.slices) || 1) * (
                            (Number(decode) || 0) + (Number(prefill) || 0)
                        )} model-server Pods after scaling).
                    </p>
                )}
                {prefillSupported && (
                    <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-200">
                        Scaling down Prefill can interrupt in-flight KV transfers. When a{' '}
                        <span className="font-mono">--kv-transfer-config</span> is present, Lens sets{' '}
                        <span className="font-mono">kv_load_failure_policy=recompute</span> so Decode can recompute.
                    </p>
                )}
                <FormError message={error} />
            </form>
        </Modal>
    );
}

export function DeploymentManagementPage({ onNavigate, initialSearch = '', initialClusterId = '' } = {}) {
    const [searchInput, setSearchInput] = useState(initialSearch);
    const query = useDebouncedValue(searchInput, SEARCH_DEBOUNCE_MS).trim();
    const [statuses, setStatuses] = useState(new Set());
    const [clusterId, setClusterId] = useState(initialClusterId);
    const [dialog, setDialog] = useState(null);
    const [selected, setSelected] = useState(null);
    const { can } = useAuth();
    const canScale = can('deployment:execution:scale');
    const [mutatingId, setMutatingId] = useState(null);
    const [connectingEndpointId, setConnectingEndpointId] = useState('');
    const [notice, setNotice] = useNotice();
    const [page, setPage] = useState(0);
    const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);

    // Cluster choices come from the unfiltered list so narrowing by cluster
    // never removes the option the user just picked.
    const [knownClusters, setKnownClusters] = useState([]);
    const { clusterNameById } = useClusterNames();


    const fetchItems = useCallback(async ({ signal }) => {
        const executions = await listDeploymentExecutions({ query, statuses: [...statuses], clusterId, signal });
        signal.throwIfAborted();
        const runs = await searchLocalDeploymentRuns(query, { signal });
        const pending = pendingDeploymentExecutions(runs, new Set(executions.map(item => item.execution_id)))
            .filter(item => !clusterId || item.cluster_id === clusterId)
            .filter(item => statuses.size === 0 || statuses.has(item.status));
        return [...pending, ...executions];
    }, [query, statuses, clusterId]);
    const rememberClusters = useCallback(results => {
        if (!query && statuses.size === 0 && !clusterId) setKnownClusters(clusterOptions(results));
    }, [query, statuses, clusterId]);
    const { items, setItems, loading, refreshing, error, setError, load } = useResourceList(fetchItems, {
        onLoaded: rememberClusters, errorFallback: 'Failed to load deployments',
    });
    usePolling(() => load({ quiet: true }), { intervalMs: DEPLOYMENT_REFRESH_INTERVAL_MS });




    const summary = useMemo(() => summarizeStatuses(items), [items]);
    const clusters = useMemo(() => {
        const merged = new Set([...knownClusters, ...clusterOptions(items)]);
        if (clusterId) merged.add(clusterId);
        return [...merged].sort();
    }, [knownClusters, items, clusterId]);

    const hasFilters = Boolean(query || statuses.size || clusterId);

    useEffect(() => {
        setPage(0);
    }, [query, statuses, clusterId, items.length]);

    const { totalPages, currentPage, pageStart, pageEnd, pagedItems } = useMemo(
        () => paginate(items, page, pageSize),
        [items, page, pageSize]
    );

    const closeDialog = () => {
        setDialog(null);
        setSelected(null);
    };

    const openDialog = (kind) => (item) => {
        setSelected(item);
        setDialog(kind);
    };

    const handleSave = async ({ displayName, description }) => {
        if (selected.pending) return;
        const executionId = selected.execution_id;
        setMutatingId(executionId);
        try {
            const updated = await updateDeploymentMetadata(executionId, { displayName, description });
            setItems((current) => current.map((item) => (item.execution_id === executionId ? updated : item)));
            setNotice('Deployment updated');
            closeDialog();
        } finally {
            setMutatingId(null);
        }
    };

    const handleScale = async ({ prefillReplicas, decodeReplicas }) => {
        if (!selected?.execution_id) return;
        const executionId = selected.execution_id;
        setMutatingId(executionId);
        try {
            const updated = await scaleDeploymentExecution(executionId, { prefillReplicas, decodeReplicas });
            setItems((current) => current.map((item) => (
                item.execution_id === executionId ? { ...item, ...updated } : item
            )));
            setNotice('Deployment scaled');
            closeDialog();
        } finally {
            setMutatingId(null);
        }
    };

    const handleDelete = async ({ deleteNamespace = false } = {}) => {
        if (selected.pending) return;
        const executionId = selected.execution_id;
        setMutatingId(executionId);
        try {
            try {
                await manageDeploymentMonitoring(executionId, 'uninstall', { clusterId: selected.cluster_id });
            } catch (monitoringError) {
                // Monitoring may not be installed for this deployment; don't block deletion on it.
                console.warn('Failed to uninstall deployment monitoring before delete', monitoringError);
            }
            await deleteDeploymentExecution(executionId, { deleteNamespace, runId: selected.run_id });
            setItems((current) => current.filter((item) => item.execution_id !== executionId));
            setNotice('Deployment deleted');
            closeDialog();
        } finally {
            setMutatingId(null);
        }
    };

    const handleConnectLocalEndpoint = async (item) => {
        if (!item?.execution_id) return;
        setConnectingEndpointId(item.execution_id);
        setError('');
        try {
            const result = await connectDeploymentExecution(item.execution_id);
            const forwarded = result?.forwarded_endpoint || result?.endpoint || '';
            if (forwarded) {
                setItems((current) => current.map((deployment) => (
                    deployment.execution_id === item.execution_id
                        ? { ...deployment, forwarded_endpoint: forwarded }
                        : deployment
                )));
                setNotice('Local endpoint connected');
            }
        } catch (failure) {
            setError(errorMessage(failure, 'Unable to connect local endpoint'));
        } finally {
            setConnectingEndpointId('');
        }
    };

    const toggleStatus = (value) => {
        // MultiSelectDropdown emits '' for the "All" entry.
        if (!value) {
            setStatuses(new Set());
            return;
        }
        setStatuses((current) => {
            const next = new Set(current);
            if (next.has(value)) next.delete(value);
            else next.add(value);
            return next;
        });
    };

    return (
        <ModulePage className="deployment-page" contentClassName="flex flex-col gap-4">
            <ModuleHeader
                icon={Rocket}
                title="Deployments"
                description="Track active model serving deployments, inspect endpoints, and update or remove existing services."
            />
            <StatCardRow>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                    <StatCard label="Total" value={summary.total} />
                    <StatCard label="Ready" value={summary.Ready} />
                    <StatCard label="In progress" value={summary['In progress']} />
                    <StatCard label="Failed" value={summary.Failed} tone={summary.Failed > 0 ? 'text-rose-500 font-extrabold' : 'text-white'} />
                    <StatCard label="Cleaned" value={summary.Cleaned} />
                </div>
            </StatCardRow>

            <section
                aria-busy={loading || refreshing}
                className="relative rounded-2xl border border-slate-800/80 bg-gradient-to-br from-slate-900/90 via-slate-900/50 to-slate-950/90 shadow-2xl backdrop-blur-xl"
            >
              <div className="relative overflow-visible p-6">
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <SectionLabel className="mb-0">{`Deployments (${items.length})`}</SectionLabel>
                    <div className="ml-auto flex items-center gap-2">
                        <RefreshButton onClick={load} disabled={refreshing} refreshing={refreshing} label="Refresh deployments" />
                        {can('deployment:run:create') && (
                            <Button variant="sky" size="sm" onClick={() => onNavigate?.('model-market')}>
                                <Plus size={14} /> Create
                            </Button>
                        )}
                    </div>
                </div>
                <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-[1fr_220px_220px]">
                    <div className="relative">
                        <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" aria-hidden="true" />
                        <Input
                            type="search"
                            value={searchInput}
                            onChange={(event) => setSearchInput(event.target.value)}
                            placeholder="Search by name, description, model, namespace, guide or ID"
                            aria-label="Search deployments"
                            className="h-10 py-2 pl-9 text-xs"
                        />
                    </div>
                    <MultiSelectDropdown
                        label="Status"
                        options={STATUS_FILTER_OPTIONS}
                        selected={statuses}
                        onChange={toggleStatus}
                        formatLabel={statusLabel}
                        className="h-10"
                    />
                    <div className="relative">
                        <select
                            value={clusterId}
                            onChange={(event) => setClusterId(event.target.value)}
                            aria-label="Filter by cluster"
                            className="h-10 w-full appearance-none rounded-xl border border-[#cbd5e1] bg-white px-3 pr-8 text-xs text-[#33475b] outline-none transition hover:border-[#94a3b8] hover:bg-[#e6f6fe] focus:border-[#037ffc] focus:ring-2 focus:ring-[#037ffc]/20 dark:border-slate-800/40 dark:bg-[#0b0f17] dark:text-slate-300 dark:hover:border-slate-700/45 dark:hover:bg-[#101622] dark:focus:border-cyan-400 dark:focus:ring-cyan-500/10"
                        >
                            <option value="">All clusters</option>
                            {clusters.map((cluster) => (
                                <option key={cluster} value={cluster}>{clusterNameById[cluster] || cluster}</option>
                            ))}
                        </select>
                        <ChevronDown size={12} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 dark:text-slate-500" aria-hidden="true" />
                    </div>
                </div>
                <div className="mt-3">
                <AsyncState loading={loading} error={error} empty={items.length === 0}
                    loadingContent={(
                    <p className="px-4 py-12 text-center text-xs text-slate-500">Loading deployments…</p>
                    )}
                    errorContent={(
                    <EmptyState
                        icon={<ServerCog size={28} />}
                        title="Could not load deployments"
                        message={error}
                        action={<Button variant="secondary" onClick={load}>Try again</Button>}
                    />
                    )}
                    emptyContent={(
                    <EmptyState
                        icon={<Rocket size={28} />}
                        iconClassName="deployments-empty-icon"
                        title={hasFilters ? 'No deployments match your filters' : 'No deployments yet'}
                        message={
                            hasFilters
                                ? 'Adjust the search text, status or cluster filter to widen the results.'
                                : 'Deployments created from the Evaluation workspace will appear here.'
                        }
                        action={
                            hasFilters ? (
                                <Button
                                    variant="secondary"
                                    onClick={() => {
                                        setSearchInput('');
                                        setStatuses(new Set());
                                        setClusterId('');
                                    }}
                                >
                                    Clear filters
                                </Button>
                            ) : undefined
                        }
                    />
                    )}
                >

                    <>
                    <DeploymentTable
                        items={pagedItems}
                        mutatingId={mutatingId}
                        connectingEndpointId={connectingEndpointId}
                        clusterNameById={clusterNameById}
                        canScale={canScale}
                        onInspect={openDialog('details')}
                        onViewPodsLogs={openDialog('pods-logs')}
                        onEdit={openDialog('edit')}
                        onDelete={openDialog('delete')}
                        onNavigate={onNavigate}
                        onConnectLocalEndpoint={handleConnectLocalEndpoint}
                        onScale={openDialog('scale')}
                    />
                    <PaginationControls className="mt-2" page={currentPage} totalPages={totalPages} onPageChange={setPage}
                        total={items.length} pageStart={pageStart} pageEnd={pageEnd} itemLabel="deployments"
                        pageSize={pageSize} pageSizeOptions={PAGE_SIZE_OPTIONS} onPageSizeChange={setPageSize} pageSizeId="deployments-page-size" />
                    </>
                </AsyncState>
                </div>
              </div>
            </section>

            {dialog === 'details' && <DeploymentDetailsModal deployment={selected} clusterNameById={clusterNameById} onClose={closeDialog} onViewPodsLogs={openDialog('pods-logs')} />}
            {dialog === 'edit' && <EditDeploymentModal deployment={selected} onCancel={closeDialog} onSave={handleSave} />}
            {dialog === 'scale' && <ScaleDeploymentModal deployment={selected} onCancel={closeDialog} onScale={handleScale} />}
            {dialog === 'delete' && <DeleteDeploymentModal deployment={selected} clusterNameById={clusterNameById} onCancel={closeDialog} onDelete={handleDelete} />}
            {dialog === 'pods-logs' && <DeploymentPodsLogsModal deployment={selected} onClose={closeDialog} />}
            <Toast
                message={notice}
                title="Deployments"
                tone="info"
                onDismiss={() => setNotice('')}
            />
        </ModulePage>
    );
}

export default DeploymentManagementPage;
