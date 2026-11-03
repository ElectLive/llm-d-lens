// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

import { useMonitoringOperation } from './useMonitoringOperation';
import { openMonitoringLink } from './monitoringLinks';
import { requestMonitoringJson } from '../../../../shared/api/monitoringClient';
import React, { useEffect, useState } from 'react';
import {
    AlertTriangle,
    AlertOctagon,
    Boxes,
    CheckCircle2,
    CloudCog,
    Cpu,
    RefreshCw,
    Rocket,
    ShieldCheck,
    Webcam,
    X,
} from 'lucide-react';
import { Badge, Button, RefreshButton, EmptyState, Input, Label, LoadingState, SectionLabel, Select, ModuleHeader, ModulePage, Panel } from '../../../../shared/ui';
import { cn } from '../../../../shared/lib/cn';
import { AcceleratorObservabilityPanel, GrafanaLogo, PrometheusLogo } from './AcceleratorObservabilityPanel';
import { DeploymentObservabilityPanel } from './DeploymentObservabilityPanel';
import { DeploymentProfilingPanel } from './DeploymentProfilingPanel';
import { ClusterStackComponentTable } from './ClusterStackComponentTable';
import { ClusterStackOperationProgress } from './ClusterStackOperationProgress';
import { ClusterStackSummary } from './ClusterStackSummary';
import { InstallClusterStackModal } from './InstallClusterStackModal';
import { getClusterStackLinks, getClusterStackOperation } from './clusterMonitoringStackBackend';
import { useClusterMonitoringStackStatus } from './useClusterMonitoringStackStatus';

const DEFAULT_NAMESPACE = 'llm-d-monitoring';

const SERVICE_TABS = [
    { value: 'cluster', label: 'Cluster', icon: CloudCog },
    { value: 'accelerator', label: 'Accelerators', icon: Cpu },
    { value: 'deployment', label: 'Deployments', icon: Rocket },
];

export default function ClusterMonitoringStackDashboard({ onToggleMobileNav, initialClusterId, initialProfileDeployment }) {
    const [namespaceInput, setNamespaceInput] = useState(DEFAULT_NAMESPACE);
    const [namespace, setNamespace] = useState(DEFAULT_NAMESPACE);
    const [clusters, setClusters] = useState([]);
    const [clusterId, setClusterId] = useState('');
    const [clustersError, setClustersError] = useState(null);
    const [installOpen, setInstallOpen] = useState(false);
    const [reinstallMode, setReinstallMode] = useState(false);
    const [activeServiceTab, setActiveServiceTab] = useState('cluster');
    const [profilingDeployment, setProfilingDeployment] = useState(null);
    const [operation, setOperation] = useState(null);
    const [operationError, setOperationError] = useState(null);
    const [installSuccess, setInstallSuccess] = useState(null);
    const [completedOperationId, setCompletedOperationId] = useState(null);
    const [openingLink, setOpeningLink] = useState(null);
    const [linksError, setLinksError] = useState(null);
    const status = useClusterMonitoringStackStatus(namespace, clusterId);
    const refreshStatus = status.refresh;
    const snapshot = status.data;
    const snapshotOperationId = snapshot?.active_operation_id;
    const operationId = operation?.operation_id
        || (snapshotOperationId !== completedOperationId ? snapshotOperationId : null);

    useEffect(() => {
        if (initialProfileDeployment) {
            setActiveServiceTab('deployment');
            setProfilingDeployment(initialProfileDeployment);
        }
    }, [initialProfileDeployment]);

    useMonitoringOperation({
        operationId, operationStatus: operation?.status, completedOperationId, namespace,
        loadOperation: getClusterStackOperation, refreshStatus, setOperation, setOperationError,
        setCompletedOperationId, setInstallSuccess,
    });

    useEffect(() => {
        if (!installSuccess) return undefined;
        const timer = window.setTimeout(() => setInstallSuccess(null), 10000);
        return () => window.clearTimeout(timer);
    }, [installSuccess]);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const payload = await requestMonitoringJson('/api/cluster/clusters');
                if (cancelled) return;
                const ready = (payload?.items || []).filter((cluster) => cluster.ready);
                setClusters(ready);
                setClustersError(null);
                if (ready.length === 0) {
                    setClusterId('');
                    return;
                }
                const preferred = initialClusterId && ready.some((cluster) => cluster.id === initialClusterId)
                    ? initialClusterId
                    : ready[0].id;
                setClusterId(preferred);
            } catch (error) {
                if (cancelled) return;
                setClusters([]);
                setClusterId('');
                setClustersError(error);
            }
        })();
        return () => { cancelled = true; };
    }, []);

    const applyNamespace = (event) => {
        event.preventDefault();
        const nextNamespace = namespaceInput.trim();
        if (nextNamespace && nextNamespace !== namespace) {
            setOperation(null);
            setOperationError(null);
            setInstallSuccess(null);
            setCompletedOperationId(null);
            setNamespace(nextNamespace);
        } else if (nextNamespace) {
            refreshStatus();
        }
    };

    const openClusterLink = async (kind) => {
        setOpeningLink(kind);
        setLinksError(null);
        try {
            const payload = await getClusterStackLinks(namespace, { clusterId });
            openMonitoringLink(payload, kind);
        } catch (error) {
            setLinksError(error.message || `Unable to open ${kind} dashboard`);
        } finally {
            setOpeningLink(null);
        }
    };

    const startOperation = (nextOperation) => {
        if (nextOperation.namespace && nextOperation.namespace !== namespace) {
            setNamespaceInput(nextOperation.namespace);
            setNamespace(nextOperation.namespace);
        }
        setOperation(nextOperation);
        setOperationError(null);
        setInstallSuccess(null);
        setCompletedOperationId(null);
        refreshStatus();
    };

    const openInstall = () => {
        setReinstallMode(false);
        setInstallOpen(true);
    };
    const openReinstall = () => {
        setReinstallMode(true);
        setInstallOpen(true);
    };

    const context = snapshot?.cluster?.context || 'Current kube context';
    const unhealthyStack = snapshot && ['absent', 'degraded', 'unreachable', 'unsupported'].includes(snapshot.status);
    const unhealthyDetails = snapshot?.components
        ?.filter((component) => ['degraded', 'missing'].includes(component.status))
        .map((component) => String(component.name || '').replace(/_/g, ' '))
        .filter(Boolean) || [];

    return (
        <ModulePage className="observability-page" contentClassName="flex flex-col gap-3 lg:py-6">
                <ModuleHeader
                    icon={Webcam}
                    title="Observability"
                    description="Install and monitor the cluster observability stack powering metrics, dashboards, and accelerator profiling."
                    onToggleMobileNav={onToggleMobileNav}
                />
                <Panel padding="none" className="shadow-none">
                    <div className="flex items-center justify-between gap-3 border-b border-theme-border px-4 py-3">
                        <div
                            className="flex min-w-0 flex-wrap items-center gap-1"
                            role="tablist"
                            aria-label="Observability service"
                        >
                            {SERVICE_TABS.map((tab) => {
                                const active = tab.value === activeServiceTab;
                                const Icon = tab.icon;
                                return (
                                    <button
                                        key={tab.value}
                                        type="button"
                                        role="tab"
                                        aria-selected={active}
                                        onClick={() => setActiveServiceTab(tab.value)}
                                        className={cn(
                                            'inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/50',
                                            active
                                                ? 'observability-service-tab-active bg-sky-500/20 text-white shadow-[0_0_18px_rgba(56,189,248,0.18)] ring-1 ring-sky-400/30'
                                                : 'text-theme-muted hover:bg-theme-bg hover:text-theme-text'
                                        )}
                                    >
                                        <Icon className="h-4 w-4 shrink-0" />
                                        {tab.label}
                                    </button>
                                );
                            })}
                        </div>

                        <RefreshButton onClick={refreshStatus} disabled={status.loading} refreshing={status.loading} label="Refresh observability status" />
                    </div>
                    <div className={cn(
                        'grid items-start gap-4 p-4',
                        activeServiceTab === 'cluster' && clusterId
                            ? 'md:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)_auto]'
                            : 'max-w-sm'
                    )}>
                            <div className="min-w-0">
                                <Label htmlFor="observability-cluster" className="mb-2 block">Target cluster</Label>
                                <Select
                                    id="observability-cluster"
                                    aria-label="Target cluster"
                                    value={clusterId}
                                    onChange={(event) => setClusterId(event.target.value)}
                                    disabled={clusters.length === 0}
                                    className="h-9 w-full min-w-0 py-1 text-xs"
                                >
                                    {clusters.length === 0 && <option value="">No ready clusters</option>}
                                    {clusters.map((cluster) => (
                                        <option key={cluster.id} value={cluster.id}>
                                            {cluster.name || cluster.id}
                                        </option>
                                    ))}
                                </Select>
                                <span className="mt-2 flex min-w-0 items-center gap-1.5 font-mono text-[10px] text-theme-muted" title={context}>
                                    <ShieldCheck className="h-3.5 w-3.5 shrink-0" />
                                    <span className="truncate">{context}</span>
                                </span>
                                {clustersError && (
                                    <p className="mt-2 flex items-center gap-1.5 text-[11px] text-red-600 dark:text-red-300/90">
                                        <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                                        {clustersError.message || 'Unable to load clusters'}
                                    </p>
                                )}
                            </div>
                {activeServiceTab === 'cluster' && clusterId && (
                <>
                        <form onSubmit={applyNamespace} className="min-w-0">
                            <Label htmlFor="monitoring-namespace" className="mb-2 block">Namespace</Label>
                            <div className="flex min-w-0 items-center gap-2">
                                <Input
                                    id="monitoring-namespace"
                                    value={namespaceInput}
                                    onChange={(event) => setNamespaceInput(event.target.value)}
                                    placeholder={DEFAULT_NAMESPACE}
                                    className="h-9 min-w-0 flex-1 py-1 text-xs"
                                />
                                <Button type="submit" variant="secondary" size="sm" className="h-9 shrink-0">Inspect</Button>
                            </div>
                        </form>
                        <div className="min-w-0">
                            <Label className="mb-2 block">GUIs</Label>
                            <div className="flex flex-wrap items-center gap-2">
                                <Button
                                    variant="outline"
                                    size="sm"
                                    className="h-9"
                                    onClick={() => openClusterLink('prometheus')}
                                    isLoading={openingLink === 'prometheus'}
                                    disabled={openingLink !== null}
                                >
                                    <PrometheusLogo className="h-3.5 w-3.5 text-[#E6522C]" /> Prometheus
                                </Button>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    className="h-9"
                                    onClick={() => openClusterLink('grafana')}
                                    isLoading={openingLink === 'grafana'}
                                    disabled={openingLink !== null}
                                >
                                    <GrafanaLogo className="h-3.5 w-3.5 text-[#F46800]" /> Grafana
                                </Button>
                            </div>
                                {linksError && (
                                    <span className="mt-2 flex items-center gap-1.5 text-[11px] text-red-600 dark:text-red-300/90">
                                        <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                                        {linksError}
                                    </span>
                                )}
                        </div>
                </>
                )}
                    </div>
                </Panel>

                {activeServiceTab === 'cluster' && (
                    !clusterId ? (
                        <section className="relative overflow-hidden rounded-2xl border border-slate-800/80 bg-gradient-to-br from-slate-900/80 via-slate-900/40 to-slate-950/90 p-5 shadow-xl backdrop-blur-xl">
                            <div className="relative">
                                <div className="mb-4">
                                    <SectionLabel tone="sky" className="text-[#037ffc]">Cluster infrastructure</SectionLabel>
                                </div>
                                <EmptyState
                                    icon={<Boxes className="h-10 w-10" />}
                                    title="No ready cluster selected"
                                    message="Cluster infrastructure observability requires a ready Kubernetes cluster. No status is shown until a ready cluster is available."
                                />
                            </div>
                        </section>
                    ) : (
                    <>
                {status.error && (
                    <div
                        role="alert"
                        aria-live="assertive"
                        className="relative overflow-hidden rounded-xl border border-red-500/40 bg-gradient-to-r from-red-950/70 via-red-950/40 to-slate-950 p-4 shadow-[0_0_30px_rgba(239,68,68,0.08)]"
                    >
                        <div className="flex items-start justify-between gap-4">
                            <div className="flex items-start gap-3">
                                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-red-500/30 bg-red-500/15">
                                    <AlertOctagon className="h-5 w-5 text-red-300" />
                                </span>
                                <div>
                                    <div className="text-sm font-bold text-red-200">Observability service is not responding normally</div>
                                    <p className="mt-1 text-xs leading-relaxed text-red-200/75">
                                        {status.error.message}
                                        {snapshot && ' The last successful snapshot is shown below and may be stale.'}
                                    </p>
                                </div>
                            </div>
                            <Button variant="secondary" size="sm" onClick={refreshStatus} isLoading={status.loading}>
                                <RefreshCw className="h-3.5 w-3.5" /> Retry
                            </Button>
                        </div>
                    </div>
                )}

                {installSuccess && (
                    <div
                        role="status"
                        aria-live="polite"
                        className="relative overflow-hidden rounded-xl border border-emerald-500/35 bg-gradient-to-r from-emerald-950/70 via-emerald-950/30 to-slate-950 p-4 shadow-[0_0_30px_rgba(16,185,129,0.1)]"
                    >
                        <div className="flex items-start justify-between gap-4">
                            <div className="flex items-start gap-3">
                                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-emerald-500/30 bg-emerald-500/15">
                                    <CheckCircle2 className="h-5 w-5 text-emerald-300" />
                                </span>
                                <div>
                                    <div className="text-sm font-bold text-emerald-200">
                                        Cluster observability stack installed successfully
                                    </div>
                                    <p className="mt-1 text-xs text-emerald-200/70">
                                        Prometheus and Grafana in <span className="font-mono">{installSuccess.namespace}</span> passed readiness verification.
                                    </p>
                                </div>
                            </div>
                            <button
                                type="button"
                                onClick={() => setInstallSuccess(null)}
                                className="rounded-lg p-1.5 text-emerald-300/60 transition-colors hover:bg-emerald-500/10 hover:text-emerald-200"
                                aria-label="Dismiss installation success message"
                            >
                                <X className="h-4 w-4" />
                            </button>
                        </div>
                    </div>
                )}

                {unhealthyStack && !status.error && (
                    <div
                        role="alert"
                        aria-live="polite"
                        className={`relative overflow-hidden rounded-xl border p-4 shadow-xl ${
                            snapshot.status === 'absent' || snapshot.status === 'unsupported'
                                ? 'border-[#d1d5db] bg-white dark:border-amber-500/35 dark:bg-slate-950 dark:bg-gradient-to-r dark:from-amber-950/60 dark:via-amber-950/25 dark:to-slate-950'
                                : 'border-[#d1d5db] bg-white dark:border-red-500/40 dark:bg-slate-950 dark:bg-gradient-to-r dark:from-red-950/70 dark:via-red-950/35 dark:to-slate-950'
                        }`}
                    >
                        <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                            <div className="flex items-start gap-3">
                                <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border ${
                                    snapshot.status === 'absent' || snapshot.status === 'unsupported'
                                        ? 'border-amber-200 bg-amber-50 dark:border-amber-500/30 dark:bg-amber-500/15'
                                        : 'border-red-200 bg-red-50 dark:border-red-500/30 dark:bg-red-500/15'
                                }`}>
                                    <AlertTriangle className={`h-5 w-5 ${
                                        snapshot.status === 'absent' || snapshot.status === 'unsupported'
                                            ? 'text-amber-600 dark:text-amber-300'
                                            : 'text-red-600 dark:text-red-300'
                                    }`} />
                                </span>
                                <div>
                                    <div className="flex flex-wrap items-center gap-2">
                                        <span className="text-sm font-bold text-theme-text dark:text-white">Observability stack is not fully operational</span>
                                        <Badge
                                            tone={snapshot.status === 'absent' || snapshot.status === 'unsupported' ? 'warning' : 'danger'}
                                            size="xs"
                                        >
                                            {snapshot.status}
                                        </Badge>
                                    </div>
                                    <p className="mt-1 text-xs leading-relaxed text-theme-muted dark:text-slate-300">{snapshot.message}</p>
                                    {unhealthyDetails.length > 0 && (
                                        <p className="mt-1 text-[10px] text-slate-600 dark:text-slate-500">
                                            Affected components: {unhealthyDetails.join(', ')}
                                        </p>
                                    )}
                                </div>
                            </div>
                            {snapshot.status === 'absent' && !operationId ? (
                                <Button variant="sky" size="sm" onClick={openInstall}>
                                    Install stack
                                </Button>
                            ) : snapshot.status === 'degraded' && !operationId ? (
                                <Button variant="sky" size="sm" onClick={openReinstall}>
                                    Reinstall
                                </Button>
                            ) : (
                                <Button variant="secondary" size="sm" onClick={refreshStatus} isLoading={status.loading}>
                                    <RefreshCw className="h-3.5 w-3.5" /> Recheck
                                </Button>
                            )}
                        </div>
                    </div>
                )}

                {status.loading && !snapshot ? (
                    <LoadingState label="Discovering observability infrastructure…" />
                ) : snapshot ? (
                    <>
                        <ClusterStackSummary snapshot={snapshot} />
                        <ClusterStackOperationProgress
                            operation={operation}
                            error={operationError}
                            onRetryPreflight={snapshot.status === 'absent' ? openInstall : null}
                        />
                        <ClusterStackComponentTable
                            components={snapshot.components}
                            actions={snapshot.status === 'ready' && !operationId ? (
                                <Button variant="sky" size="sm" onClick={openReinstall}>
                                    <RefreshCw className="h-3.5 w-3.5" /> Reinstall
                                </Button>
                            ) : null}
                        />

                        {snapshot.status === 'absent' && !operationId && (
                            <div className="relative overflow-hidden rounded-2xl border border-[#d1d5db] bg-white p-2 shadow-xl dark:border-cyan-500/20 dark:bg-slate-950 dark:bg-gradient-to-br dark:from-cyan-950/40 dark:via-slate-900/80 dark:to-slate-950">
                                <EmptyState
                                    icon={<CloudCog className="h-10 w-10" />}
                                    title="Observability stack is not installed"
                                    message="Run mandatory preflight checks, review the installation plan, then install Prometheus and Grafana infrastructure."
                                    action={<Button variant="sky" onClick={openInstall}>Install observability stack</Button>}
                                />
                            </div>
                        )}

                        {snapshot.status === 'degraded' && (
                            <div className="relative overflow-hidden rounded-2xl border border-[#d1d5db] bg-white p-2 shadow-xl dark:border-amber-500/20 dark:bg-slate-950 dark:bg-gradient-to-br dark:from-amber-950/30 dark:via-slate-900/80 dark:to-slate-950">
                                <EmptyState
                                    icon={<AlertTriangle className="h-10 w-10" />}
                                    iconClassName="border-amber-200 bg-amber-50 text-amber-600 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300"
                                    title="Observability stack is degraded"
                                    message="Review component diagnostics, or reinstall the stack. Reinstalling removes the existing release and deletes stored metrics."
                                    action={<Button variant="sky" onClick={openReinstall}>Reinstall</Button>}
                                />
                            </div>
                        )}
                    </>
                ) : (
                    <div className="rounded-2xl border border-slate-800 bg-slate-900/70">
                        <EmptyState
                            title="Cluster status unavailable"
                            message="Check access to the current Kubernetes context and try again."
                            action={<Button variant="outline" onClick={refreshStatus}>Retry</Button>}
                        />
                    </div>
                )}
                    </>
                    )
                )}

                {activeServiceTab === 'accelerator' && (
                    <AcceleratorObservabilityPanel clusterId={clusterId} />
                )}

                {activeServiceTab === 'deployment' && (
                    profilingDeployment ? (
                        <DeploymentProfilingPanel
                            clusterId={clusterId}
                            deployment={profilingDeployment}
                            onBack={() => setProfilingDeployment(null)}
                        />
                    ) : (
                        <DeploymentObservabilityPanel
                            clusterId={clusterId}
                            onOpenProfiling={setProfilingDeployment}
                        />
                    )
                )}

            <InstallClusterStackModal
                isOpen={installOpen}
                namespace={namespace}
                clusterId={clusterId}
                reinstall={reinstallMode}
                onClose={() => setInstallOpen(false)}
                onOperationStarted={startOperation}
            />
        </ModulePage>
    );
}
