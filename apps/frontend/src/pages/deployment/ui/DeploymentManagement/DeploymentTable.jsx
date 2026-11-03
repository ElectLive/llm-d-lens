import { useClipboard } from '../../../../shared/lib/useClipboard';
import { Fragment, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Check, ChevronDown, ChevronRight, Copy, FileText, Info, Link2, Loader2, Menu, Pencil, Scaling, Trash2 } from 'lucide-react';
import { Button } from '../../../../shared/ui';
import { cn } from '../../../../shared/lib/cn';
import {
    deploymentKind,
    deploymentLocalEndpoint,
    deploymentTitle,
    effectiveStatusGroup,
    effectiveStatusLabel,
    statusDotClassForGroup,
    statusGroup,
} from './deploymentPresentation';
import { getDeploymentExecutionPods } from '../../../optimization';
import { DeploymentMonitoringButton } from './DeploymentMonitoringButton';
import { PodStatusBadge } from './PodStatusBadge';

const PODS_POLL_INTERVAL_MS = 5000;

function StatusDot({ item, pods }) {
    if (item) {
        const group = effectiveStatusGroup(item, pods);
        const label = effectiveStatusLabel(item, pods);
        return (
            <span
                className={`h-1.5 w-1.5 shrink-0 rounded-full motion-safe:animate-pulse [animation-duration:2.4s] ${statusDotClassForGroup(group)}`}
                title={label}
                aria-label={label}
            />
        );
    }
    return null;
}

function CopyButton({ value, copied, onCopy, label = 'Copy' }) {
    if (!value) return null;
    return (
        <button
            type="button"
            onClick={() => onCopy(value)}
            className="inline-flex shrink-0 items-center gap-1 rounded-md border border-slate-700 px-2 py-1 text-[10px] font-medium text-slate-300 transition-colors hover:border-emerald-500/40 hover:text-slate-100"
            title={label}
            aria-label={label}
        >
            {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
        </button>
    );
}

function RowActionItem({ icon, label, onClick, disabled = false, danger = false, trailing = null, isLoading = false }) {
    return (
        <Button
            type="button"
            variant="ghost"
            size="sm"
            role="menuitem"
            onClick={onClick}
            disabled={disabled}
            isLoading={isLoading}
            className={cn(
                'w-full justify-start gap-2 rounded-md px-2.5 py-2 text-left text-xs font-normal',
                danger
                    ? 'text-rose-300 hover:bg-rose-500/10 hover:text-rose-200'
                    : 'text-slate-200 hover:bg-slate-800 hover:text-white',
                'disabled:pointer-events-auto disabled:cursor-not-allowed disabled:hover:bg-transparent'
            )}
        >
            {icon && <span className={cn('shrink-0', danger ? 'text-rose-400' : 'text-slate-400')} aria-hidden="true">{icon}</span>}
            <span className="min-w-0 flex-1 truncate">{label}</span>
            {trailing}
        </Button>
    );
}

function RowActions({ item, busy, connecting, canScale, onInspect, onViewPodsLogs, onEdit, onDelete, onNavigate, onConnectLocalEndpoint, onScale }) {
    const name = deploymentTitle(item);
    const isInProgress = statusGroup(item.status) === 'In progress';
    const localEndpoint = deploymentLocalEndpoint(item);
    const canConnect = !localEndpoint && !item.pending && item.status === 'ready' && item.endpoint;
    const scaleStatus = statusGroup(item.status);
    const canScaleNow = canScale && !item.pending
        && item.scale?.decodeReplicas != null
        && scaleStatus !== 'Cleaned' && scaleStatus !== 'In progress';
    const [open, setOpen] = useState(false);
    const [menuPosition, setMenuPosition] = useState(null);
    const triggerRef = useRef(null);

    useEffect(() => {
        if (!open) return undefined;
        const closeOnEscape = (event) => {
            if (event.key === 'Escape') setOpen(false);
        };
        window.addEventListener('keydown', closeOnEscape);
        return () => window.removeEventListener('keydown', closeOnEscape);
    }, [open]);

    const toggleMenu = () => {
        if (!open) {
            const rect = triggerRef.current?.getBoundingClientRect();
            if (rect) {
                const menuWidth = 224;
                const right = Math.max(8, Math.min(window.innerWidth - rect.right, window.innerWidth - menuWidth - 8));
                const spaceBelow = window.innerHeight - rect.bottom;
                setMenuPosition(spaceBelow >= 240
                    ? { top: rect.bottom + 6, right, maxHeight: Math.max(120, spaceBelow - 10) }
                    : { bottom: window.innerHeight - rect.top + 6, right, maxHeight: Math.max(120, rect.top - 14) });
            }
        }
        setOpen((current) => !current);
    };

    const select = (action) => {
        setOpen(false);
        action?.();
    };

    return (
        <div className="flex items-center justify-end gap-1">
            {busy && <Loader2 size={14} className="mr-1 animate-spin text-[#037ffc] dark:text-cyan-300" aria-hidden="true" />}
            <Button
                ref={triggerRef}
                type="button"
                variant="ghost"
                size="icon"
                onClick={toggleMenu}
                disabled={busy}
                aria-label={`Actions for ${name}`}
                aria-haspopup="menu"
                aria-expanded={open}
                title="Deployment actions"
                className={cn('border border-slate-700/70 bg-slate-900/40', open && 'border-slate-600 bg-slate-800 text-slate-100')}
            >
                <Menu size={16} />
            </Button>
            {open && typeof document !== 'undefined' && createPortal(
                <>
                    <button
                        type="button"
                        aria-label="Close actions menu"
                        className="fixed inset-0 z-[200] cursor-default"
                        onClick={() => setOpen(false)}
                    />
                    <div
                        role="menu"
                        aria-label={`Actions for ${name}`}
                        style={menuPosition || undefined}
                        className="fixed z-[210] w-56 max-w-[calc(100vw-1rem)] overflow-y-auto rounded-lg border border-slate-700/60 bg-slate-900 p-1 shadow-2xl"
                    >
                        {canConnect && (
                            <RowActionItem
                                icon={connecting ? null : <Link2 size={14} />}
                                label={connecting ? 'Connecting…' : 'Create local port-forward'}
                                onClick={() => select(() => onConnectLocalEndpoint?.(item))}
                                disabled={busy || connecting}
                                isLoading={connecting}
                            />
                        )}
                        <RowActionItem
                            icon={<Info size={14} />}
                            label="View details"
                            onClick={() => select(() => onInspect(item))}
                            disabled={busy}
                        />
                        {!item.pending && (
                            <RowActionItem
                                icon={<Pencil size={14} />}
                                label="Edit name and description"
                                onClick={() => select(() => onEdit(item))}
                                disabled={busy}
                            />
                        )}
                        {canScaleNow && (
                            <RowActionItem
                                icon={<Scaling size={14} />}
                                label="Scale deployment"
                                onClick={() => select(() => onScale?.(item))}
                                disabled={busy}
                            />
                        )}
                        <RowActionItem
                            icon={<FileText size={14} />}
                            label="View pods & logs"
                            onClick={() => select(() => onViewPodsLogs?.(item))}
                            disabled={busy}
                            trailing={isInProgress ? <span className="ml-auto h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400 motion-safe:animate-pulse" aria-hidden="true" /> : null}
                        />
                        {!item.pending && <DeploymentMonitoringButton deployment={item} onNavigate={onNavigate} />}
                        {!item.pending && <div className="my-1 h-px bg-slate-700/60" />}
                        {!item.pending && (
                            <RowActionItem
                                icon={<Trash2 size={14} />}
                                label="Delete deployment"
                                onClick={() => select(() => onDelete(item))}
                                disabled={busy}
                                danger
                            />
                        )}
                    </div>
                </>,
                document.body
            )}
        </div>
    );
}

function PodsExpandPanel({ pods, loading, error }) {
    if (loading && pods === undefined) {
        return (
            <div className="flex items-center gap-2 px-4 py-3 text-[11px] text-slate-500">
                <Loader2 size={12} className="animate-spin" aria-hidden="true" /> Loading pods…
            </div>
        );
    }
    const list = pods || [];
    return (
        <div className="px-4 py-3">
            {error && <p className="mb-2 text-[11px] text-rose-300">{error}</p>}
            {list.length === 0 ? (
                <p className="text-[11px] text-slate-500">No pods found in namespace.</p>
            ) : (
                <table className="w-full min-w-[40rem] border-collapse text-left text-[11px]">
                    <thead className="text-slate-500">
                        <tr>
                            <th scope="col" className="py-1 pl-10 pr-3 font-medium">Pod</th>
                            <th scope="col" className="py-1 pr-3 font-medium">Status</th>
                            <th scope="col" className="py-1 pr-3 font-medium">Restarts</th>
                            <th scope="col" className="py-1 pr-3 font-medium">Node</th>
                            <th scope="col" className="py-1 pr-3 font-medium">Pod IP</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-slate-800/40">
                        {list.map((pod) => (
                            <tr key={pod.name}>
                                <td className="py-1 pl-10 pr-3 font-mono text-slate-300">{pod.name}</td>
                                <td className="py-1 pr-3">
                                    <div className="flex items-center gap-1.5 text-slate-300">
                                        <PodStatusBadge phase={pod.phase} reason={pod.status_reason} ready={pod.ready} />
                                        <span>{pod.status_reason || pod.phase || 'Unknown'}</span>
                                    </div>
                                </td>
                                <td className="py-1 pr-3 text-slate-400">{pod.restarts ?? 0}</td>
                                <td className="py-1 pr-3 text-slate-400">{pod.node_name || '—'}</td>
                                <td className="py-1 pr-3 font-mono text-slate-500">{pod.pod_ip || '—'}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
}

function ExpandToggle({ expanded, onToggle, name }) {
    return (
        <button
            type="button"
            onClick={onToggle}
            aria-expanded={expanded}
            aria-label={expanded ? `Hide pods for ${name}` : `Show pods for ${name}`}
            title={expanded ? 'Hide pods' : 'Show pods'}
            className="shrink-0 rounded p-0.5 text-slate-500 transition hover:bg-slate-800/70 hover:text-[#037ffc] dark:hover:text-slate-200"
        >
            {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        </button>
    );
}

export function DeploymentTable({ items, mutatingId, connectingEndpointId = '', clusterNameById = {}, canScale = false, onInspect, onViewPodsLogs, onEdit, onDelete, onNavigate, onConnectLocalEndpoint, onScale }) {
    const { copiedValue: copiedEndpoint, copy: copyEndpoint } = useClipboard();
    const [expandedId, setExpandedId] = useState('');
    const [podsById, setPodsById] = useState({});
    const [podsLoadingId, setPodsLoadingId] = useState('');
    const [podsErrorById, setPodsErrorById] = useState({});
    const pollTimerRef = useRef(null);

    const fetchPodsFor = async (executionId, silent = false) => {
        if (!executionId) return;
        if (!silent) setPodsLoadingId(executionId);
        try {
            const data = await getDeploymentExecutionPods(executionId);
            const list = Array.isArray(data?.pods) ? data.pods : [];
            setPodsById((current) => ({ ...current, [executionId]: list }));
            setPodsErrorById((current) => ({ ...current, [executionId]: data?.error || '' }));
        } catch (err) {
            setPodsErrorById((current) => ({ ...current, [executionId]: err.message || 'Failed to fetch pods' }));
        } finally {
            if (!silent) setPodsLoadingId((current) => (current === executionId ? '' : current));
        }
    };

    const toggleExpand = (executionId) => {
        setExpandedId((current) => {
            const next = current === executionId ? '' : executionId;
            if (next) fetchPodsFor(next);
            return next;
        });
    };

    // Poll pods for the visible page so the status dot reflects live service
    // availability and reduced capacity, not only the slower backend record.
    const executionIdsKey = items.filter((item) => !item.pending && item.execution_id)
        .map((item) => item.execution_id).join('|');
    useEffect(() => {
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        const executionIds = executionIdsKey ? executionIdsKey.split('|') : [];
        if (!executionIds.length) return undefined;
        const poll = () => Promise.all(executionIds.map((id) => fetchPodsFor(id, true)));
        poll();
        pollTimerRef.current = setInterval(poll, PODS_POLL_INTERVAL_MS);
        return () => clearInterval(pollTimerRef.current);
        // Polling is keyed by the visible execution ids; fetchPodsFor closes
        // over the current API client and is intentionally not a dependency.
    }, [executionIdsKey]);

    return (
        <>
            {/* Desktop table */}
            <div className="hidden overflow-x-auto rounded-xl border border-slate-800/60 lg:block">
                <table className="w-full min-w-[68rem] border-collapse text-left text-xs">
                    <thead className="bg-slate-950/60 text-slate-400">
                        <tr>
                            <th scope="col" className="px-4 py-3 text-[10px] font-semibold uppercase tracking-wider">Deployment</th>
                            <th scope="col" className="px-4 py-3 text-[10px] font-semibold uppercase tracking-wider">Model</th>
                            <th scope="col" className="px-4 py-3 text-[10px] font-semibold uppercase tracking-wider">Backend</th>
                            <th scope="col" className="px-4 py-3 text-[10px] font-semibold uppercase tracking-wider">Type</th>
                            <th scope="col" className="px-4 py-3 text-[10px] font-semibold uppercase tracking-wider">Cluster</th>
                            <th scope="col" className="px-4 py-3 text-[10px] font-semibold uppercase tracking-wider">Namespace</th>
                            <th scope="col" className="px-4 py-3 text-right text-[10px] font-semibold uppercase tracking-wider">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-slate-800/70">
        {items.map((item) => {
                            const busy = mutatingId === item.execution_id;
                            const connecting = connectingEndpointId === item.execution_id;
                            const expanded = expandedId === item.execution_id;
                            const pods = podsById[item.execution_id];
                            const clusterLabel = clusterNameById[item.cluster_id] || item.cluster_id || '';
                            return (
                                <Fragment key={item.execution_id}>
                                <tr
                                    aria-busy={busy}
                                    className="align-top transition-colors hover:bg-[#037ffc0d] dark:hover:bg-slate-800/40"
                                >
                                    <td className="max-w-[260px] px-4 py-3 text-left">
                                        <div className="flex items-start gap-2">
                                            <div className="mt-0.5 flex w-8 shrink-0 items-center gap-1">
                                                {!item.pending && (
                                                    <ExpandToggle expanded={expanded} onToggle={() => toggleExpand(item.execution_id)} name={deploymentTitle(item)} />
                                                )}
                                                <StatusDot item={item} pods={pods} />
                                            </div>
                                            <div className="min-w-0 text-left">
                                                <span className="block truncate text-sm font-semibold text-slate-100" title={deploymentTitle(item)}>
                                                    {deploymentTitle(item)}
                                                </span>
                                                <div className="mt-0.5 line-clamp-2 text-left text-[11px] text-slate-500" title={item.description || ''}>
                                                    {item.description || 'No description'}
                                                </div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="max-w-[180px] truncate px-4 py-3 text-slate-300" title={item.model || ''}>{item.model || '—'}</td>
                                    <td className="px-4 py-3 text-slate-300">{item.backend || 'vLLM'}</td>
                                    <td className="px-4 py-3 text-slate-400">{deploymentKind(item)}</td>
                                    <td className="max-w-[180px] px-4 py-3">
                                        {clusterLabel ? (
                                            <div className="flex items-center gap-1.5">
                                                <span className="min-w-0 truncate text-slate-300" title={clusterLabel}>{clusterLabel}</span>
                                                <CopyButton value={clusterLabel} copied={copiedEndpoint === clusterLabel} onCopy={copyEndpoint} label="Copy cluster" />
                                            </div>
                                        ) : (
                                            <span className="text-slate-600">—</span>
                                        )}
                                    </td>
                                    <td className="max-w-[180px] px-4 py-3">
                                        {item.namespace ? (
                                            <div className="flex items-center gap-1.5">
                                                <span className="min-w-0 truncate font-mono text-[11px] text-slate-400" title={item.namespace}>{item.namespace}</span>
                                                <CopyButton value={item.namespace} copied={copiedEndpoint === item.namespace} onCopy={copyEndpoint} label="Copy namespace" />
                                            </div>
                                        ) : (
                                            <span className="text-slate-600">—</span>
                                        )}
                                    </td>
                                    <td className="px-4 py-3">
                                        <RowActions item={item} busy={busy} connecting={connecting} canScale={canScale} onInspect={onInspect} onViewPodsLogs={onViewPodsLogs} onEdit={onEdit} onDelete={onDelete} onNavigate={onNavigate} onConnectLocalEndpoint={onConnectLocalEndpoint} onScale={onScale} />
                                    </td>
                                </tr>
                                {expanded && (
                                    <tr className="bg-slate-950/50">
                                        <td colSpan={7} className="border-t border-slate-200 p-0 dark:border-slate-800/60">
                                            <PodsExpandPanel
                                                pods={pods}
                                                loading={podsLoadingId === item.execution_id}
                                                error={podsErrorById[item.execution_id]}
                                            />
                                        </td>
                                    </tr>
                                )}
                                </Fragment>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {/* Narrow-screen cards */}
            <div className="flex flex-col gap-3 p-3 lg:hidden">
                {items.map((item) => {
                    const busy = mutatingId === item.execution_id;
                    const connecting = connectingEndpointId === item.execution_id;
                    const expanded = expandedId === item.execution_id;
                    const pods = podsById[item.execution_id];
                    const clusterLabel = clusterNameById[item.cluster_id] || item.cluster_id || '';
                    return (
                        <div
                            key={item.execution_id}
                            aria-busy={busy}
                            className="rounded-xl border border-slate-800/60 bg-slate-950/40 p-3"
                        >
                            <div className="flex items-start justify-between gap-2">
                                <div className="flex min-w-0 flex-1 items-start gap-2">
                                    <div className="mt-0.5 flex w-8 shrink-0 items-center gap-1">
                                        {!item.pending && (
                                            <ExpandToggle expanded={expanded} onToggle={() => toggleExpand(item.execution_id)} name={deploymentTitle(item)} />
                                        )}
                                        <StatusDot item={item} pods={pods} />
                                    </div>
                                    <div className="min-w-0 text-left">
                                        <span className="block truncate text-sm font-semibold text-slate-100">{deploymentTitle(item)}</span>
                                        <div className="mt-0.5 line-clamp-2 text-left text-[11px] text-slate-500">{item.description || 'No description'}</div>
                                    </div>
                                </div>
                            </div>
                            <dl className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
                                <div><dt className="text-slate-600">Model</dt><dd className="truncate text-slate-300">{item.model || '—'}</dd></div>
                                <div><dt className="text-slate-600">Backend</dt><dd className="truncate text-slate-300">{item.backend || 'vLLM'}</dd></div>
                                <div><dt className="text-slate-600">Type</dt><dd className="truncate text-slate-300">{deploymentKind(item)}</dd></div>
                                <div>
                                    <dt className="text-slate-600">Cluster</dt>
                                    <dd className="flex items-center gap-1.5 truncate text-slate-300">
                                        <span className="truncate">{clusterLabel || '—'}</span>
                                        {clusterLabel && <CopyButton value={clusterLabel} copied={copiedEndpoint === clusterLabel} onCopy={copyEndpoint} label="Copy cluster" />}
                                    </dd>
                                </div>
                                <div>
                                    <dt className="text-slate-600">Namespace</dt>
                                    <dd className="flex items-center gap-1.5 truncate text-slate-300">
                                        <span className="truncate font-mono">{item.namespace || '—'}</span>
                                        {item.namespace && <CopyButton value={item.namespace} copied={copiedEndpoint === item.namespace} onCopy={copyEndpoint} label="Copy namespace" />}
                                    </dd>
                                </div>
                            </dl>
                            <div className="mt-3 border-t border-slate-200 pt-2 dark:border-slate-800/60">
                                <RowActions item={item} busy={busy} connecting={connecting} canScale={canScale} onInspect={onInspect} onViewPodsLogs={onViewPodsLogs} onEdit={onEdit} onDelete={onDelete} onNavigate={onNavigate} onConnectLocalEndpoint={onConnectLocalEndpoint} onScale={onScale} />
                            </div>
                            {expanded && (
                                <div className="mt-3 rounded-lg border border-slate-800/60 bg-slate-950/60">
                                    <PodsExpandPanel
                                        pods={pods}
                                        loading={podsLoadingId === item.execution_id}
                                        error={podsErrorById[item.execution_id]}
                                    />
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </>
    );
}

export default DeploymentTable;
