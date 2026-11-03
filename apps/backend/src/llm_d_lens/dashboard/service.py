"""Welcome issue composition, authorization and prioritization."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from llm_d_lens.ai_providers.service import default_service as default_ai_provider_service
from llm_d_lens.auth.access import filter_by_cluster, resource_readable, visible_cluster_ids
from llm_d_lens.auth.contracts import Principal
from llm_d_lens.auth.policy import authorize
from llm_d_lens.cluster import registry
from llm_d_lens.core.exceptions import ForbiddenError, NotFoundError
from llm_d_lens.db.dao.deployment_batch import DeploymentBatchDao
from llm_d_lens.db.dao.group import GroupDao, GroupRoleBindingDao, UserGroupDao
from llm_d_lens.db.dao.role import UserRoleBindingDao
from llm_d_lens.db.dao.user import UserDao
from llm_d_lens.db.dao.welcome_issue_snapshot import WelcomeIssueSnapshotDao
from llm_d_lens.deploy.executions import list_execution_contexts
from llm_d_lens.infrastructure.kubernetes import run_kubectl
from llm_d_lens.model_service.gateway_contracts import GatewayStatus, GatewayStatusCluster
from llm_d_lens.model_service.service import default_service as default_model_service
from llm_d_lens.monitoring.profiling.pressure import collect_pressure

from .collector import DashboardCollector, default_collector
from .contracts import (
    ClusterRow,
    ComponentRow,
    Coverage,
    DashboardIssue,
    DashboardScope,
    DeploymentActivityItem,
    DeploymentPanel,
    GatewayState,
    PressureSummary,
    ProviderCounts,
    ProviderHealth,
    SectionEnvelope,
    SectionReason,
    ServiceRow,
    WelcomeIssueSnapshot,
    WelcomeIssuesResponse,
    WelcomeOverview,
)
from .rules import (
    cache_issue,
    cluster_issues,
    deployment_issues,
    deployment_monitoring_issue,
    deployment_pod_issue,
    empty_model_cache_issue,
    empty_model_cache_storage_issue,
    gateway_cluster_issues,
    missing_model_cache_storage_issue,
    no_ai_provider_issue,
    no_cluster_issue,
    no_role_groups_issue,
    no_role_user_issue,
    service_issues,
    sort_issues,
    storage_issue,
    unpublished_deployment_issues,
    worst_cluster_status,
)

logger = logging.getLogger(__name__)

_pressure_tasks: set[asyncio.Task[Any]] = set()
_snapshot_tasks: set[asyncio.Task[Any]] = set()
DEPLOYMENT_ACTIVE_STATUSES = {"draft", "validated", "rendered", "deploying", "rolling_back"}
CAPACITY_UNITS = {
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
}
_CAPACITY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(Mi|Gi|Ti|M|G|T)\s*$")


@dataclass
class Composition:
    cluster_rows: list[ClusterRow]
    service_rows: list[ServiceRow]
    deployment_panel: DeploymentPanel | None
    issues: list[DashboardIssue]
    sections: dict[str, SectionEnvelope[Any]] = field(default_factory=dict)
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class DashboardService:
    def __init__(
        self,
        *,
        collector: DashboardCollector | None = None,
        model_service=None,
        snapshot_dao: WelcomeIssueSnapshotDao | None = None,
    ) -> None:
        self._collector = collector or default_collector()
        self._model_service = model_service or default_model_service()
        self._snapshot_dao = snapshot_dao or WelcomeIssueSnapshotDao()
        self._snapshot_lock = asyncio.Lock()

    async def welcome_overview(
        self,
        principal: Principal | None,
        *,
        cluster_id: str | None = None,
        refresh: bool = False,
    ) -> WelcomeOverview:
        now = datetime.now(UTC)
        clusters = self._scope_clusters(principal, cluster_id)
        scope_clusters = self._scope_clusters(principal, None) if cluster_id else clusters
        snapshot = self._snapshot_dao.get_current()
        if snapshot is not None:
            if refresh:
                self.request_snapshot_check()
            issues = self._visible_issues(principal, snapshot.issues, cluster_id=cluster_id)
            return WelcomeOverview(
                generated_at=now,
                scope=self._scope(principal, cluster_id, scope_clusters),
                issues=SectionEnvelope(
                    state=snapshot.state,
                    generated_at=snapshot.generated_at,
                    observed_at=snapshot.observed_at,
                    stale=snapshot.stale,
                    coverage=snapshot.coverage,
                    data=issues,
                ),
            )
        composition = await self._compose(principal, clusters, cluster_id=cluster_id, refresh=refresh)
        return WelcomeOverview(
            generated_at=now,
            scope=self._scope(principal, cluster_id, scope_clusters),
            issues=composition.sections["issues"],
        )

    async def alerts(
        self,
        principal: Principal | None,
        *,
        cluster_id: str | None = None,
        severity: str | None = None,
        category: str | None = None,
        search: str = "",
        cursor: str | None = None,
        limit: int = 25,
    ) -> WelcomeIssuesResponse:
        self._scope_clusters(principal, cluster_id)
        snapshot = self._snapshot_dao.get_current()
        if snapshot is not None:
            snapshot_id = snapshot.snapshot_id
            issues = self._visible_issues(principal, snapshot.issues, cluster_id=cluster_id)
        else:
            clusters = self._scope_clusters(principal, cluster_id)
            composition = await self._compose(principal, clusters, cluster_id=cluster_id, refresh=False)
            snapshot_id, issues = composition.snapshot_id, composition.issues

        offset = 0
        restarted = False
        if cursor:
            decoded = _decode_cursor(cursor)
            if decoded is None or decoded.get("snapshot") != snapshot_id:
                restarted = True
            else:
                offset = max(0, int(decoded.get("offset") or 0))
        filtered = _filter_issues(issues, severity=severity, category=category, search=search)
        page = filtered[offset : offset + limit]
        next_offset = offset + len(page)
        next_cursor = None
        if next_offset < len(filtered):
            next_cursor = _encode_cursor(snapshot_id, next_offset)
        return WelcomeIssuesResponse(
            generated_at=datetime.now(UTC),
            snapshot_id=snapshot_id,
            total=len(filtered),
            items=page,
            next_cursor=next_cursor,
            restarted=restarted,
        )

    async def run_snapshot_check(self) -> WelcomeIssueSnapshot | None:
        """Compose every cluster as the system principal and persist the result."""
        if self._snapshot_lock.locked():
            return None
        async with self._snapshot_lock:
            previous = self._snapshot_dao.get_current()
            first_seen = dict(previous.first_seen) if previous is not None else {}
            if previous is not None:
                for issue in previous.issues:
                    first_seen.setdefault(issue.id, issue.first_seen_at or issue.observed_at or previous.generated_at)
            clusters = self._collector.cluster_list()
            composition = await self._compose(None, clusters, cluster_id=None, refresh=True)
            envelope = composition.sections["issues"]
            issues = []
            current_ids = set()
            for issue in composition.issues:
                current_ids.add(issue.id)
                seen_at = first_seen.get(issue.id) or issue.first_seen_at or issue.observed_at or envelope.generated_at
                first_seen[issue.id] = seen_at
                issues.append(issue.model_copy(update={"first_seen_at": seen_at}))
            if envelope.state == "complete":
                first_seen = {issue_id: value for issue_id, value in first_seen.items() if issue_id in current_ids}
            snapshot = WelcomeIssueSnapshot(
                snapshot_id=composition.snapshot_id,
                generated_at=envelope.generated_at,
                observed_at=envelope.observed_at,
                state=envelope.state,
                stale=envelope.stale,
                coverage=envelope.coverage,
                issues=issues,
                first_seen=first_seen,
            )
            self._snapshot_dao.put(snapshot)
            return snapshot

    def request_snapshot_check(self) -> None:
        """Schedule a refresh without blocking the current request."""
        if self._snapshot_lock.locked():
            return
        task = asyncio.create_task(self.run_snapshot_check())
        _snapshot_tasks.add(task)
        task.add_done_callback(_finish_snapshot_task)

    # --- composition -----------------------------------------------------

    async def _compose(
        self,
        principal: Principal | None,
        clusters: list[registry.Cluster],
        *,
        cluster_id: str | None,
        refresh: bool,
    ) -> Composition:
        now = datetime.now(UTC)
        allowed = self._allowed_ids(principal)
        scope_ids = {cluster.id for cluster in clusters}
        services_permitted = self._allows_any(principal, "model-service:group:read", scope_ids)
        gateway_permitted = self._allows_any(principal, "model-service:gateway:read", scope_ids)
        members = self._visible_members(principal, scope_ids) if services_permitted else []
        groups = self._visible_groups(principal, scope_ids) if services_permitted else []
        member_by_group = _group_members(groups, members)
        execution_ids = {
            member.execution_id
            for group_members in member_by_group.values()
            for member in group_members
            if member.execution_id and member.status == "active" and member.weight > 0
        }
        if services_permitted and gateway_permitted:
            gateway_status, gateway_observed_at, gateway_stale, gateway_reason = await self._collector.gateway_status(
                force=refresh
            )
        else:
            gateway_status, gateway_observed_at, gateway_stale, gateway_reason = None, None, False, None
        contexts = self._visible_execution_contexts(principal, scope_ids)
        context_by_id = {context.execution_id: context for context in contexts}
        pressure = await self._collect_pressure(principal, execution_ids, context_by_id)
        deployment_monitoring_issues = await self._deployment_monitoring_issues(
            principal, contexts, context_by_id, scope_ids, refresh=refresh
        )
        service_rows, service_state, service_reason = self._service_rows(
            principal,
            clusters,
            groups,
            member_by_group,
            gateway_status,
            context_by_id,
            pressure,
            allowed,
            gateway_permitted,
        )
        deployment_panel, deployment_state = self._deployment_panel(principal, contexts, clusters)
        service_cluster_ids = {row.cluster_id for row in service_rows}
        cluster_rows, cluster_state, cluster_coverage = await self._cluster_rows(principal, clusters, refresh=refresh)

        issues: list[DashboardIssue] = []
        if not clusters:
            issues.append(no_cluster_issue())
        missing_ai_provider = self._missing_ai_provider_issue(principal)
        if missing_ai_provider is not None:
            issues.append(missing_ai_provider)
        issues.extend(self._access_issues(principal))
        for row in cluster_rows:
            issues.extend(cluster_issues(row, has_services=row.id in service_cluster_ids))
        issues.extend(issue for row in service_rows for issue in service_issues(row))
        if gateway_permitted and gateway_status is not None:
            service_names_by_cluster: dict[str, list[str]] = {}
            for group in groups:
                if group.cluster_id:
                    service_names_by_cluster.setdefault(group.cluster_id, []).append(group.name)
            cluster_names = {cluster.id: cluster.name for cluster in clusters}
            for gateway_cluster in gateway_status.clusters:
                if gateway_cluster.cluster_id not in scope_ids:
                    continue
                issues.extend(
                    gateway_cluster_issues(
                        gateway_cluster,
                        cluster_name=cluster_names.get(
                            gateway_cluster.cluster_id or "", gateway_cluster.cluster_name or ""
                        ),
                        service_names=service_names_by_cluster.get(gateway_cluster.cluster_id or "", []),
                    )
                )
        if deployment_panel is not None:
            issues.extend(deployment_issues(deployment_panel))
        issues.extend(await self._deployment_pod_issues(contexts, {cluster.id: cluster.name for cluster in clusters}))
        issues.extend(deployment_monitoring_issues)
        if services_permitted:
            published_execution_ids = {member.execution_id for member in members if member.execution_id}
            issues.extend(
                unpublished_deployment_issues(
                    contexts,
                    published_execution_ids,
                    {cluster.id: cluster.name for cluster in clusters},
                )
            )
        issues.extend(await self._supporting_issues(principal, clusters, scope_ids))
        issues = sort_issues(_dedupe_issues(issues))
        issue_by_cluster = Counter(issue.cluster_id for issue in issues if issue.cluster_id)
        for row in cluster_rows:
            row.issue_count = issue_by_cluster.get(row.id, 0)
            if row.available:
                row.status = worst_cluster_status(  # type: ignore[assignment]
                    [issue for issue in issues if issue.cluster_id == row.id]
                )
            else:
                row.status = "unknown"

        observed_at = max((row.observed_at for row in cluster_rows if row.observed_at), default=None)
        coverage = Coverage(
            observed=sum(1 for row in cluster_rows if row.observed_at is not None),
            expected=len(cluster_rows),
            unit="clusters",
        )
        sections: dict[str, SectionEnvelope[Any]] = {
            "clusters": SectionEnvelope(
                state=cluster_state,
                generated_at=now,
                observed_at=observed_at,
                stale=cluster_state != "complete",
                coverage=coverage,
                reason=SectionReason(code="cluster_observation", message="Some clusters could not be observed")
                if cluster_state != "complete"
                else None,
                data=cluster_rows,
            ),
            "issues": SectionEnvelope(
                state=cluster_state,
                generated_at=now,
                observed_at=observed_at,
                stale=cluster_state != "complete",
                coverage=coverage,
                data=issues,
            ),
            "services": SectionEnvelope(
                state=service_state,
                generated_at=now,
                observed_at=gateway_observed_at,
                stale=bool(gateway_stale),
                coverage=Coverage(observed=len(service_rows), expected=len(service_rows), unit="services"),
                reason=SectionReason(code="gateway_status", message=gateway_reason or "Gateway status is collecting")
                if service_state != "complete"
                else None,
                data=service_rows,
            )
            if self._allows_any(principal, "model-service:group:read", allowed)
            else None,
            "deployments": SectionEnvelope(
                state=deployment_state,
                generated_at=now,
                observed_at=now,
                coverage=Coverage(
                    observed=sum(deployment_panel.totals.values()) if deployment_panel else 0,
                    expected=sum(deployment_panel.totals.values()) if deployment_panel else 0,
                    unit="deployments",
                ),
                data=deployment_panel,
            )
            if deployment_panel is not None
            else None,
        }
        sections = {key: value for key, value in sections.items() if value is not None}
        return Composition(
            cluster_rows=cluster_rows,
            service_rows=service_rows,
            deployment_panel=deployment_panel,
            issues=issues,
            sections=sections,
        )

    async def _cluster_rows(
        self,
        principal: Principal | None,
        clusters: list[registry.Cluster],
        *,
        refresh: bool,
    ) -> tuple[list[ClusterRow], str, Coverage]:
        rows: list[ClusterRow] = []
        observations: list[tuple[registry.Cluster, dict[str, Any] | None, datetime | None, bool, str | None]] = []
        for cluster in clusters:
            overview, observed_at, stale, reason = await self._collector.cluster_overview(cluster, force=refresh)
            observations.append((cluster, overview, observed_at, stale, reason))
        secret_statuses = await asyncio.gather(
            *(self._collector.hf_token_secret_status(cluster, force=refresh) for cluster in clusters),
            return_exceptions=True,
        )
        for index, (cluster, overview, observed_at, stale, reason) in enumerate(observations):
            secret_status = secret_statuses[index]
            if not isinstance(secret_status, str):
                secret_status = "unreachable"  # noqa: S105
            secret_ref = None
            secret_namespace = getattr(cluster, "hf_token_secret_namespace", None)
            secret_name = getattr(cluster, "hf_token_secret_name", None)
            if secret_namespace and secret_name:
                secret_ref = f"{secret_namespace}/{secret_name}"
            rows.append(
                _cluster_row(
                    cluster.id,
                    cluster.name,
                    overview,
                    observed_at,
                    stale,
                    reason,
                    hf_token_secret_status=secret_status,
                    hf_token_secret_ref=secret_ref,
                )
            )
        observed = sum(1 for _, _, observed_at, _, _ in observations if observed_at is not None)
        coverage = Coverage(observed=observed, expected=len(observations), unit="clusters")
        if not observations:
            state = "complete"
        elif observed == 0:
            state = "unavailable"
        elif observed < len(observations):
            state = "partial"
        else:
            state = "complete"
        return rows, state, coverage

    def _service_rows(
        self,
        principal: Principal | None,
        clusters: list[registry.Cluster],
        groups: list,
        member_by_group: dict[str, list],
        gateway_status: GatewayStatus | None,
        context_by_id: dict[str, Any],
        pressure: dict[str, Any],
        allowed: set[str] | None,
        gateway_permitted: bool,
    ) -> tuple[list[ServiceRow], str, str | None]:
        cluster_names = {cluster.id: cluster.name for cluster in clusters}
        gateway_by_cluster = {
            cluster.cluster_id: cluster for cluster in (gateway_status.clusters if gateway_status else [])
        }
        all_execution_ids = Counter(
            member.execution_id
            for group_members in member_by_group.values()
            for member in group_members
            if member.execution_id
        )
        rows: list[ServiceRow] = []
        for group in groups:
            group_members = member_by_group.get(group.id, [])
            gateway = gateway_by_cluster.get(group.cluster_id or "")
            providers = _provider_counts(group_members, context_by_id)
            provider_details = _provider_details(group_members, context_by_id)
            execution_ids = [
                member.execution_id
                for member in group_members
                if member.execution_id and member.status == "active" and member.weight > 0
            ]
            availability = _service_availability(
                group.status,
                providers,
                gateway,
                context_by_id,
                execution_ids,
                has_deployment_access=self._allows_any(
                    principal, "deployment:run:read", {group.cluster_id} if group.cluster_id else allowed
                ),
                has_gateway_access=self._allows_any(
                    principal,
                    "model-service:gateway:read",
                    {group.cluster_id} if group.cluster_id else allowed,
                ),
            )
            pressure_summary = None
            if execution_ids and self._allows_any(
                principal, "monitoring:profiling:read", {group.cluster_id} if group.cluster_id else allowed
            ):
                evidence = next(
                    (pressure[execution_id] for execution_id in execution_ids if execution_id in pressure),
                    None,
                )
                if evidence is not None:
                    pressure_summary = _pressure_summary(
                        evidence,
                        shared=any(all_execution_ids.get(execution_id, 0) > 1 for execution_id in execution_ids),
                    )
            deployment_observed = self._allows_any(
                principal, "deployment:run:read", {group.cluster_id} if group.cluster_id else allowed
            )
            rows.append(
                ServiceRow(
                    group_id=group.id,
                    name=group.name,
                    cluster_id=group.cluster_id or "",
                    cluster_name=cluster_names.get(group.cluster_id or "", ""),
                    status=group.status,
                    availability=availability,
                    providers=providers,
                    provider_details=provider_details,
                    deployment_observed=deployment_observed,
                    gateway=_gateway_state(gateway),
                    pressure=pressure_summary,
                    execution_ids=execution_ids,
                )
            )
        rows.sort(key=_service_sort_key)
        if not rows:
            state = "complete"
            reason = None
        elif gateway_permitted and gateway_status is None:
            state = "partial"
            reason = "Gateway status is still collecting"
        else:
            state = "complete"
            reason = None
        return rows, state, reason

    def _deployment_panel(
        self,
        principal: Principal | None,
        contexts: list[Any],
        clusters: list[registry.Cluster],
    ) -> tuple[DeploymentPanel | None, str]:
        if not self._allows_any(principal, "deployment:run:read", self._allowed_ids(principal)):
            return None, "complete"
        cluster_names = {cluster.id: cluster.name for cluster in clusters}
        totals = Counter(context.status for context in contexts)
        now = datetime.now(UTC)
        in_progress: list[DeploymentActivityItem] = []
        recent_failures: list[DeploymentActivityItem] = []
        for context in contexts:
            item = _deployment_item(context, cluster_names)
            if context.status in DEPLOYMENT_ACTIVE_STATUSES:
                in_progress.append(item)
            if context.status == "failed":
                timestamp = _as_datetime(context.updated_at) or _as_datetime(context.created_at)
                if timestamp is None or timestamp >= now - timedelta(hours=24):
                    recent_failures.append(item)
        missing = datetime.min.replace(tzinfo=UTC)
        in_progress.sort(key=lambda item: item.updated_at or item.created_at or missing, reverse=True)
        recent_failures.sort(key=lambda item: item.updated_at or item.created_at or missing, reverse=True)
        return (
            DeploymentPanel(
                totals=dict(totals),
                ready=totals.get("ready", 0),
                in_progress=in_progress[:20],
                recent_failures=recent_failures[:20],
            ),
            "complete",
        )

    async def _deployment_pod_issues(self, contexts: list[Any], cluster_names: dict[str, str]) -> list[DashboardIssue]:
        """Flag ready deployments whose live model-server Pods disappeared or shrank."""
        targets = [
            context
            for context in contexts
            if context.status == "ready"
            and not getattr(context, "evaluate_owned", False)
            and getattr(context, "namespace", None)
        ]
        semaphore = asyncio.Semaphore(4)

        async def inspect(context: Any) -> DashboardIssue | None:
            async with semaphore:
                try:
                    result = await run_kubectl(
                        ["get", "pods", "--namespace", context.namespace, "-o", "json"],
                        cluster_id=context.cluster_id,
                        timeout=10,
                    )
                except Exception:  # noqa: BLE001 - a failed observation is not an outage
                    return None
            if result.returncode != 0:
                return None
            try:
                items = json.loads(result.stdout or "{}").get("items") or []
            except (TypeError, ValueError):
                return None
            live = [item for item in items if (item.get("status") or {}).get("phase") != "Succeeded"]
            name = context.display_name or context.name or context.execution_id
            if not live:
                return deployment_pod_issue(
                    execution_id=context.execution_id,
                    execution_name=name,
                    cluster_id=context.cluster_id,
                    cluster_name=cluster_names.get(context.cluster_id or "", ""),
                    ready=0,
                    expected=context.expected_model_pods,
                    observed_at=_as_datetime(context.updated_at) or _as_datetime(context.created_at),
                )

            def labels(item: dict[str, Any]) -> dict[str, Any]:
                return (item.get("metadata") or {}).get("labels") or {}

            def is_gateway(item: dict[str, Any]) -> bool:
                pod_labels = labels(item)
                return (
                    bool(pod_labels.get("llm-d-router-gateway"))
                    or "-epp" in str((item.get("metadata") or {}).get("name") or "").lower()
                )

            model = [
                item
                for item in live
                if labels(item).get("llm-d.ai/inference-serving") == "true"
                or labels(item).get("llm-d.ai/role") in {"decode", "prefill"}
            ]
            if not model:
                if all(is_gateway(item) for item in live):
                    return deployment_pod_issue(
                        execution_id=context.execution_id,
                        execution_name=name,
                        cluster_id=context.cluster_id,
                        cluster_name=cluster_names.get(context.cluster_id or "", ""),
                        ready=0,
                        expected=context.expected_model_pods,
                        observed_at=_as_datetime(context.updated_at) or _as_datetime(context.created_at),
                    )
                return None
            ready = sum(
                1
                for item in model
                if any(
                    condition.get("type") == "Ready" and condition.get("status") == "True"
                    for condition in (item.get("status") or {}).get("conditions") or []
                )
                and not (item.get("metadata") or {}).get("deletionTimestamp")
            )
            expected = context.expected_model_pods
            if ready == 0 or ready < expected:
                return deployment_pod_issue(
                    execution_id=context.execution_id,
                    execution_name=name,
                    cluster_id=context.cluster_id,
                    cluster_name=cluster_names.get(context.cluster_id or "", ""),
                    ready=ready,
                    expected=expected,
                    observed_at=_as_datetime(context.updated_at) or _as_datetime(context.created_at),
                )
            return None

        return [issue for issue in await asyncio.gather(*(inspect(context) for context in targets)) if issue]

    async def _supporting_issues(
        self,
        principal: Principal | None,
        clusters: list[registry.Cluster],
        scope_ids: set[str],
    ) -> list[DashboardIssue]:
        cluster_names = {cluster.id: cluster.name for cluster in clusters}
        issues: list[DashboardIssue] = []
        volumes: list = []
        cache_entries: list = []
        if self._allows_any(principal, "storage:volume:read", scope_ids):
            volumes = filter_by_cluster(self._collector.storage_volumes(), lambda volume: volume.cluster_id, principal)
            volumes = [volume for volume in volumes if volume.cluster_id in scope_ids]
            for volume in volumes:
                allowed_here = {volume.cluster_id} if volume.cluster_id else scope_ids
                if not self._allows_any(principal, "storage:volume:read", allowed_here):
                    continue
                issues.extend(
                    storage_issue(
                        cluster_id=volume.cluster_id,
                        cluster_name=cluster_names.get(volume.cluster_id, ""),
                        volume_id=volume.id,
                        volume_name=volume.name,
                        status=volume.status.value,
                        used_ratio=_used_ratio(volume),
                        observed_at=_as_datetime(volume.updated_at),
                    )
                )
        if self._allows_any(principal, "model-cache:entry:read", scope_ids):
            entries = await self._collector.model_cache_entries()
            entries = filter_by_cluster(entries, lambda entry: entry.cluster_id, principal)
            entries = [entry for entry in entries if entry.cluster_id in scope_ids]
            cache_entries = entries
            for entry in entries:
                allowed_here = {entry.cluster_id} if entry.cluster_id else scope_ids
                if not self._allows_any(principal, "model-cache:entry:read", allowed_here):
                    continue
                failed = sum(1 for node in entry.node_progress if node.status == "failed")
                pending = sum(1 for node in entry.node_progress if node.status in {"pending", "downloading"})
                issues.extend(
                    cache_issue(
                        cluster_id=entry.cluster_id,
                        cluster_name=cluster_names.get(entry.cluster_id, ""),
                        entry_id=entry.id,
                        entry_name=entry.source.display_name(),
                        status=entry.status.value,
                        failed_nodes=failed,
                        pending_nodes=pending,
                        observed_at=_as_datetime(entry.updated_at),
                    )
                )
        if self._allows_any(principal, "storage:volume:read", scope_ids):
            for cluster in clusters:
                if not self._allows_any(principal, "storage:volume:read", {cluster.id}):
                    continue
                cluster_volumes = [volume for volume in volumes if volume.cluster_id == cluster.id]
                cache_volume_ids = {
                    volume.id
                    for volume in cluster_volumes
                    if any(getattr(purpose, "value", purpose) == "model-cache" for purpose in volume.purposes)
                }
                cluster_entries = [
                    entry
                    for entry in cache_entries
                    if entry.cluster_id == cluster.id and entry.storage_volume_id in cache_volume_ids
                ]
                if not cache_volume_ids and not cluster_entries:
                    issues.append(
                        missing_model_cache_storage_issue(
                            cluster_id=cluster.id,
                            cluster_name=cluster_names.get(cluster.id, cluster.id),
                        )
                    )
                    continue
                if (
                    cache_volume_ids
                    and self._allows_any(principal, "model-cache:entry:read", {cluster.id})
                    and not any(entry.status.value == "ready" for entry in cluster_entries)
                ):
                    issues.append(
                        empty_model_cache_issue(
                            cluster_id=cluster.id,
                            cluster_name=cluster_names.get(cluster.id, cluster.id),
                            volume_count=len(cache_volume_ids),
                        )
                    )
                if self._allows_any(principal, "model-cache:entry:read", {cluster.id}):
                    for volume in cluster_volumes:
                        if getattr(volume.status, "value", volume.status) != "ready":
                            continue
                        if not any(getattr(purpose, "value", purpose) == "model-cache" for purpose in volume.purposes):
                            continue
                        if any(entry.storage_volume_id == volume.id for entry in cluster_entries):
                            continue
                        issues.append(
                            empty_model_cache_storage_issue(
                                cluster_id=cluster.id,
                                cluster_name=cluster_names.get(cluster.id, cluster.id),
                                volume_id=volume.id,
                                volume_name=volume.name,
                            )
                        )
        return issues

    def _access_issues(self, principal: Principal | None) -> list[DashboardIssue]:
        if principal is not None and not (
            self._allows_any(principal, "user:user:read", None) or self._allows_any(principal, "group:group:read", None)
        ):
            return []
        try:
            users = [user for user in UserDao().list(status="active") if user.id != "system"]
            groups = GroupDao().list()
            user_bindings = UserRoleBindingDao().list_all()
            group_bindings = GroupRoleBindingDao().list_all()
            memberships = UserGroupDao().list_all()
        except Exception:  # noqa: BLE001 - access hygiene is supporting context
            logger.warning("welcome access-role check failed", exc_info=True)
            return []
        role_less_users = _role_less_users(users, user_bindings, group_bindings, memberships)
        role_less_groups = _role_less_groups(groups, group_bindings)
        issues: list[DashboardIssue] = []
        for user in role_less_users:
            issues.append(no_role_user_issue(user_id=user.id, username=user.username))
        if role_less_groups:
            issues.append(no_role_groups_issue([group.name for group in role_less_groups]))
        return issues

    async def _deployment_monitoring_issues(
        self,
        principal: Principal | None,
        contexts: list[Any],
        context_by_id: dict[str, Any],
        scope_ids: set[str],
        *,
        refresh: bool,
    ) -> list[DashboardIssue]:
        if not self._allows_any(principal, "monitoring:deployment:read", scope_ids):
            return []
        cluster_ids = sorted({context.cluster_id for context in contexts if context.cluster_id})
        if not cluster_ids:
            return []
        results = await asyncio.gather(
            *(self._collector.deployment_monitoring_status(cluster_id, force=refresh) for cluster_id in cluster_ids),
            return_exceptions=True,
        )
        cluster_names = {cluster.id: cluster.name for cluster in self._collector.cluster_list()}
        issues: list[DashboardIssue] = []
        for cluster_id, result in zip(cluster_ids, results, strict=False):
            if not isinstance(result, dict):
                continue
            for item in result.get("items") or []:
                execution_id = item.get("execution_id")
                context = context_by_id.get(execution_id)
                if context is None:
                    continue
                issue = deployment_monitoring_issue(
                    execution_id=execution_id,
                    execution_name=context.display_name or context.name or execution_id,
                    cluster_id=cluster_id,
                    cluster_name=cluster_names.get(cluster_id, cluster_id),
                    status=str(item.get("status") or "unknown"),
                    stack_ready=item.get("stack_ready"),
                    message=item.get("message"),
                )
                if issue is not None:
                    issues.append(issue)
        return issues

    async def _collect_pressure(
        self,
        principal: Principal | None,
        execution_ids: set[str],
        context_by_id: dict[str, Any],
    ) -> dict[str, Any]:
        if not execution_ids or not any(
            self._allows_any(principal, "monitoring:profiling:read", {context_by_id[execution_id].cluster_id})
            for execution_id in execution_ids
            if execution_id in context_by_id and context_by_id[execution_id].cluster_id
        ):
            return {}
        executions = [
            (execution_id, context_by_id[execution_id].cluster_id)
            for execution_id in sorted(execution_ids)
            if execution_id in context_by_id
            and self._allows_any(
                principal,
                "monitoring:profiling:read",
                {context_by_id[execution_id].cluster_id},
            )
        ]
        if not executions:
            return {}
        try:
            return await asyncio.wait_for(collect_pressure(executions, limit=20), timeout=5.0)
        except TimeoutError:
            logger.info("dashboard pressure collection exceeded its request budget")
            return {}

    # --- authorization and scope -----------------------------------------

    def _visible_groups(self, principal: Principal | None, allowed: set[str] | None) -> list:
        groups = []
        for group in self._model_service.list_groups():
            if allowed is not None and group.cluster_id not in allowed:
                continue
            if not resource_readable(
                principal,
                permission="model-service:group:read",
                resource_type="cluster",
                resource_id=group.cluster_id or "",
                cluster_id=group.cluster_id,
            ):
                continue
            groups.append(group)
        groups.sort(key=lambda group: (group.cluster_id or "", group.name))
        return groups

    def _visible_members(self, principal: Principal | None, allowed: set[str] | None) -> list:
        members = []
        for member in self._model_service.list_members():
            if allowed is not None and member.cluster_id not in allowed:
                continue
            if not resource_readable(
                principal,
                permission="model-service:group:read",
                resource_type="cluster",
                resource_id=member.cluster_id,
                cluster_id=member.cluster_id,
            ):
                continue
            members.append(member)
        return members

    def _visible_execution_contexts(self, principal: Principal | None, allowed: set[str] | None) -> list:
        if not self._allows_any(principal, "deployment:run:read", allowed):
            return []
        contexts = list_execution_contexts(allowed_cluster_ids=allowed, limit=500)
        if principal is None:
            return contexts
        owners = DeploymentBatchDao().owner_map([context.run_id for context in contexts])
        return [context for context in contexts if _execution_readable(principal, context, owners.get(context.run_id))]

    def _scope_clusters(self, principal: Principal | None, cluster_id: str | None) -> list[registry.Cluster]:
        if cluster_id:
            cluster = registry.get_cluster(cluster_id)
            if cluster is None:
                raise NotFoundError("cluster not found")
            if not self._allows_any(principal, "cluster:cluster:read", {cluster_id}):
                raise ForbiddenError("cluster is not accessible")
            return [cluster]
        clusters = self._collector.cluster_list()
        return filter_by_cluster(clusters, lambda cluster: cluster.id, principal)

    def _scope(
        self,
        principal: Principal | None,
        cluster_id: str | None,
        clusters: list[registry.Cluster],
    ) -> DashboardScope:
        selected = next((cluster for cluster in clusters if cluster.id == cluster_id), None)
        return DashboardScope(
            mode="cluster" if cluster_id else "all",
            cluster_id=cluster_id,
            cluster_name=selected.name if selected else None,
            clusters=[{"id": cluster.id, "name": cluster.name} for cluster in clusters],
        )

    def _allowed_ids(self, principal: Principal | None) -> set[str] | None:
        return visible_cluster_ids(principal)

    def _allows_any(
        self,
        principal: Principal | None,
        permission: str,
        cluster_ids: set[str] | None,
    ) -> bool:
        if principal is None:
            return True
        if cluster_ids:
            return any(
                authorize(principal, permission, target_cluster_id=cluster_id).allowed for cluster_id in cluster_ids
            )
        return authorize(principal, permission).allowed

    def _missing_ai_provider_issue(self, principal: Principal | None) -> DashboardIssue | None:
        if principal is not None and not authorize(principal, "playground:chat:use").allowed:
            return None
        try:
            providers = default_ai_provider_service().list()
        except Exception:  # noqa: BLE001 - provider configuration is supporting context
            return None
        if any(getattr(provider, "provider_type", "") == "openai" for provider in providers):
            return None
        return no_ai_provider_issue()

    def _visible_issues(
        self,
        principal: Principal | None,
        issues: list[DashboardIssue],
        *,
        cluster_id: str | None = None,
    ) -> list[DashboardIssue]:
        visible = issues
        if cluster_id:
            visible = [issue for issue in visible if issue.cluster_id in (None, cluster_id)]
        if principal is None:
            return list(visible)
        return [issue for issue in visible if self._issue_visible(principal, issue)]

    def _issue_visible(self, principal: Principal, issue: DashboardIssue) -> bool:
        if issue.cluster_id and not self._allows_any(principal, "cluster:cluster:read", {issue.cluster_id}):
            return False
        permission = _issue_permission(issue)
        if permission is None:
            return False
        if issue.cluster_id:
            return self._allows_any(principal, permission, {issue.cluster_id})
        return self._allows_any(principal, permission, None)


# --- helpers ----------------------------------------------------------------


def _role_less_users(users: list, user_bindings: list, group_bindings: list, memberships: list) -> list:
    direct = {binding.user_id for binding in user_bindings}
    groups_with_role = {binding.group_id for binding in group_bindings}
    inherited = {membership.user_id for membership in memberships if membership.group_id in groups_with_role}
    return [user for user in users if user.id not in direct and user.id not in inherited]


def _role_less_groups(groups: list, group_bindings: list) -> list:
    bound = {binding.group_id for binding in group_bindings}
    return [group for group in groups if group.id not in bound]


def _group_members(groups: list, members: list) -> dict[str, list]:
    group_ids = {group.id for group in groups}
    grouped: dict[str, list] = {group_id: [] for group_id in group_ids}
    for member in members:
        if member.group_id in grouped:
            grouped[member.group_id].append(member)
    return grouped


def _provider_details(members: list, context_by_id: dict[str, Any]) -> list[ProviderHealth]:
    details: list[ProviderHealth] = []
    for member in members:
        context = context_by_id.get(member.execution_id)
        health = member.health_json or {}
        reason = health.get("error")
        if not reason:
            components = health.get("components") or {}
            degraded = [
                f"{name}={state}"
                for name, state in components.items()
                if state not in {"ready", "standby", "n/a", "unknown"}
            ]
            reason = ", ".join(degraded) or None
        details.append(
            ProviderHealth(
                execution_id=member.execution_id,
                name=(context.display_name or context.name) if context is not None else member.execution_id,
                status=member.status,
                weight=member.weight,
                ready=bool(context is not None and context.status == "ready"),
                reason=reason,
            )
        )
    return details


def _provider_counts(members: list, context_by_id: dict[str, Any]) -> ProviderCounts:
    counts = ProviderCounts(total=len(members))
    for member in members:
        if member.status == "disabled":
            counts.disabled += 1
            continue
        if member.status == "unhealthy":
            counts.unhealthy += 1
        if member.status == "active":
            if member.weight <= 0:
                counts.standby += 1
            else:
                counts.active += 1
                context = context_by_id.get(member.execution_id)
                if context is not None and context.status == "ready":
                    counts.ready += 1
    return counts


def _service_availability(
    status: str,
    providers: ProviderCounts,
    gateway: GatewayStatusCluster | None,
    context_by_id: dict[str, Any],
    execution_ids: list[str],
    *,
    has_deployment_access: bool,
    has_gateway_access: bool,
) -> str:
    if status == "disabled":
        return "disabled"
    if providers.active == 0:
        return "unavailable"
    if has_gateway_access and gateway is not None:
        if gateway.gateway_state == "missing" or gateway.gateway_ready is False or gateway.gateway_running is False:
            return "unavailable"
        if gateway.ipp_state in {"missing", "unreachable"} and gateway.ipp_ready is False:
            return "unavailable"
    if not has_deployment_access:
        return "unknown"
    known = [context_by_id.get(execution_id) for execution_id in execution_ids]
    if not any(known):
        return "unknown"
    ready = sum(1 for context in known if context is not None and context.status == "ready")
    if ready == 0:
        return "unavailable"
    if ready < len(execution_ids):
        return "degraded"
    return "available"


def _gateway_state(gateway: GatewayStatusCluster | None) -> GatewayState | None:
    if gateway is None:
        return None
    return GatewayState(
        state=gateway.gateway_state,
        ready=gateway.gateway_ready,
        running=gateway.gateway_running,
        ipp_state=gateway.ipp_state,
        ipp_ready=gateway.ipp_ready,
    )


def _pressure_summary(pressure: Any, *, shared: bool) -> PressureSummary:
    return PressureSummary(
        level=pressure.level,  # type: ignore[arg-type]
        execution_id=pressure.execution_id,
        affected_instances=pressure.affected_instances,
        observed_instances=pressure.observed_instances,
        expected_instances=pressure.expected_instances,
        waiting_current=pressure.waiting_current,
        waiting_peak=pressure.waiting_peak,
        waiting_fraction=pressure.waiting_fraction,
        kv_cache_usage_percent=pressure.kv_cache_usage_percent,
        kv_cache_fraction=pressure.kv_cache_fraction,
        error_rate=pressure.error_rate,
        outcomes=pressure.outcomes,
        window_seconds=pressure.window_seconds,
        shared_execution=shared,
        stale=pressure.stale,
    )


def _service_sort_key(row: ServiceRow) -> tuple[int, int, str, str]:
    priority = {"unavailable": 0, "degraded": 1, "unknown": 2, "available": 3, "disabled": 4}
    pressure = row.pressure is not None and row.pressure.level == "elevated"
    return (0 if pressure else 1, priority.get(row.availability, 9), row.cluster_name, row.name)


def _deployment_item(context: Any, cluster_names: dict[str, str]) -> DeploymentActivityItem:
    return DeploymentActivityItem(
        execution_id=context.execution_id,
        name=context.name or "",
        display_name=context.display_name or "",
        model=context.model,
        cluster_id=context.cluster_id,
        cluster_name=cluster_names.get(context.cluster_id or "", ""),
        status=context.status,
        created_at=_as_datetime(context.created_at),
        updated_at=_as_datetime(context.updated_at),
        failure=context.failure,
    )


def _execution_readable(principal: Principal, context: Any, owner: tuple[str | None, str | None] | None) -> bool:
    owner_user_id, owner_group_id = owner if owner else (None, None)
    if context.run_id and resource_readable(
        principal,
        permission="deployment:run:read",
        resource_type="deployment_run",
        resource_id=context.run_id,
        cluster_id=context.cluster_id,
        owner_user_id=owner_user_id,
        owner_group_id=owner_group_id,
    ):
        return True
    return resource_readable(
        principal,
        permission="deployment:run:read",
        resource_type="deployment_execution",
        resource_id=context.execution_id,
        cluster_id=context.cluster_id,
        owner_user_id=owner_user_id,
        owner_group_id=owner_group_id,
    )


def _cluster_row(
    cluster_id: str,
    name: str,
    overview: dict[str, Any] | None,
    observed_at: datetime | None,
    stale: bool,
    reason: str | None,
    *,
    hf_token_secret_status: str = "unknown",  # noqa: S107
    hf_token_secret_ref: str | None = None,
) -> ClusterRow:
    if overview is None:
        return ClusterRow(
            id=cluster_id,
            name=name,
            status="unknown",
            available=False,
            reason=reason,
            observed_at=observed_at,
            stale=stale,
            hf_token_secret_status=hf_token_secret_status,
            hf_token_secret_ref=hf_token_secret_ref,
        )
    # `build_overview` returns the Kubernetes summary directly; older callers
    # may wrap it in a ``kubernetes`` field. Accept both shapes.
    kubernetes = overview.get("kubernetes") if isinstance(overview.get("kubernetes"), dict) else overview
    hardware = kubernetes.get("hardware") or {}
    nodes = kubernetes.get("nodes") or []
    components = kubernetes.get("components") or {}
    node_disks = []
    for node in nodes:
        percent = _float_or_none(node.get("diskUsagePercent"))
        if percent is not None:
            node_disks.append((str(node.get("name") or ""), percent))
    disk_pressure_node, max_disk_usage_percent = (
        max(node_disks, key=lambda item: item[1]) if node_disks else (None, None)
    )
    component_rows = []
    monitoring = components.get("monitoring")
    if isinstance(monitoring, dict):
        component_rows.append(
            ComponentRow(
                name="monitoring",
                status=str(monitoring.get("status") or ("absent" if not monitoring.get("installed") else "unknown")),
            )
        )
    # Only report device plugins for accelerator profiles actually present in
    # the cluster. The component summary deliberately includes every registered
    # vendor profile so the hardware page can show "not installed"; treating
    # those as cluster issues would flag an NVIDIA plugin on an XPU-only
    # cluster.
    present_profiles = {
        str(item.get("id"))
        for item in hardware.get("accelerators") or []
        if str(item.get("id")) and int(item.get("gpuCount") or 0) > 0
    }
    accelerator_plugins = components.get("acceleratorDevicePlugins") or {}
    if accelerator_plugins:
        for key, value in accelerator_plugins.items():
            if key not in present_profiles or not isinstance(value, dict):
                continue
            component_rows.append(
                ComponentRow(name=f"{key} accelerator-driver", status=str(value.get("status") or "unknown"))
            )
    else:
        intel = components.get("intelDevicePlugin")
        if isinstance(intel, dict) and intel.get("installed") and "intel-xpu" in present_profiles:
            component_rows.append(
                ComponentRow(name="intel-xpu accelerator-driver", status=str(intel.get("status") or "unknown"))
            )
    return ClusterRow(
        id=cluster_id,
        name=name,
        status="healthy",
        available=True,
        reason=None,
        observed_at=observed_at,
        stale=stale,
        nodes_ready=sum(1 for node in nodes if node.get("ready")),
        nodes_total=len(nodes),
        cpu_cores=float(hardware.get("cpuCores") or 0),
        cpu_usage_percent=_float_or_none(hardware.get("cpuUsagePercent")),
        memory_bytes=int(hardware.get("memoryBytes") or 0),
        memory_usage_percent=_float_or_none(hardware.get("memoryUsagePercent")),
        gpu_total=int(hardware.get("totalGpuCount") or hardware.get("gpuCount") or 0),
        gpu_allocated=int(hardware.get("allocatedGpuCount") or 0),
        gpu_available=int(hardware.get("availableGpuCount") or 0),
        usable_gpu_count=int(hardware.get("usableGpuCount") or 0),
        configured_gpu_limit=hardware.get("configuredGpuLimit"),
        gpu_usage_percent=_float_or_none(hardware.get("gpuUsagePercent")),
        vram_usage_percent=_float_or_none(hardware.get("vramUsagePercent")),
        disk_usage_percent=_float_or_none(hardware.get("diskUsagePercent")),
        max_disk_usage_percent=max_disk_usage_percent,
        disk_pressure_node=disk_pressure_node or None,
        hf_token_secret_status=hf_token_secret_status,
        hf_token_secret_ref=hf_token_secret_ref,
        accelerators=[
            {
                "id": str(item.get("id") or ""),
                "gpuCount": int(item.get("gpuCount") or 0),
                "models": list(item.get("models") or []),
            }
            for item in hardware.get("accelerators") or []
        ],
        device_metrics=list(hardware.get("deviceMetrics") or []),
        components=component_rows,
        monitoring_dashboard_only=bool(monitoring.get("dashboardOnly")) if isinstance(monitoring, dict) else False,
    )


def _used_ratio(volume: Any) -> float | None:
    used = volume.used_bytes
    if used is None:
        return None
    capacity = _capacity_bytes(volume.capacity)
    if not capacity:
        return None
    return min(1.0, max(0.0, float(used) / capacity))


def _capacity_bytes(value: str | None) -> int | None:
    if not value:
        return None
    match = _CAPACITY_RE.match(value)
    if match is None:
        return None
    return int(float(match.group(1)) * CAPACITY_UNITS[match.group(2)])


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _filter_issues(
    issues: list[DashboardIssue],
    *,
    severity: str | None,
    category: str | None,
    search: str,
) -> list[DashboardIssue]:
    needle = search.strip().lower()
    filtered = []
    for issue in issues:
        if severity and issue.severity != severity:
            continue
        if category and issue.category != category:
            continue
        if needle:
            haystack = " ".join(
                str(value)
                for value in (
                    issue.title,
                    issue.message,
                    issue.code,
                    issue.cluster_name,
                    issue.resource.get("name"),
                    issue.resource.get("id"),
                )
                if value
            ).lower()
            if needle not in haystack:
                continue
        filtered.append(issue)
    return filtered


def _dedupe_issues(issues: list[DashboardIssue]) -> list[DashboardIssue]:
    seen: set[str] = set()
    result: list[DashboardIssue] = []
    for issue in issues:
        if issue.id in seen:
            continue
        seen.add(issue.id)
        result.append(issue)
    return result


def _encode_cursor(snapshot_id: str, offset: int) -> str:
    raw = json.dumps({"snapshot": snapshot_id, "offset": offset}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _decode_cursor(cursor: str) -> dict[str, Any] | None:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode())
        value = json.loads(raw)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _finish_pressure_task(task: asyncio.Task[Any]) -> None:
    _pressure_tasks.discard(task)
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()


def _finish_snapshot_task(task: asyncio.Task[Any]) -> None:
    _snapshot_tasks.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001 - background checks must not break the app
        logger.exception("welcome snapshot check failed")


def _issue_permission(issue: DashboardIssue) -> str | None:
    """Permission a principal needs to see this issue's resource details."""
    if issue.code == "ai-provider.none_configured":
        return "playground:chat:use"
    if issue.code == "cluster.observation_unavailable":
        return "cluster:cluster:read"
    if issue.code.startswith("monitoring.cluster_stack"):
        return "monitoring:cluster-stack:read"
    if issue.code == "monitoring.deployment_unavailable":
        return "monitoring:deployment:read"
    if issue.code == "access.user_without_role":
        return "user:user:read"
    if issue.code == "access.groups_without_role":
        return "group:group:read"
    return {
        "cluster": "cluster:cluster:read",
        "model-service": "model-service:group:read",
        "deployment": "deployment:run:read",
        "storage": "storage:volume:read",
        "model-cache": "model-cache:entry:read",
        "monitoring": "monitoring:cluster-stack:read",
        "ai-provider": "playground:chat:use",
        "access": "user:user:read",
    }.get(issue.category)


_service: DashboardService | None = None


def default_service() -> DashboardService:
    global _service
    if _service is None:
        _service = DashboardService()
    return _service
