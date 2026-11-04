"""Derive validated candidate-generation seeds from benchmark history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import log2

from .benchmark_evidence import (
    BenchmarkEvidenceRetriever,
    BenchmarkQuery,
    BenchmarkRecord,
    CandidateConfiguration,
    historical_query_context,
)
from .benchmark_store import JsonBenchmarkRecordStore
from .generator import CandidateProposal
from .planner import PlanningFacts


@dataclass(frozen=True)
class HistoricalCandidateSeed:
    proposal: CandidateProposal
    evidence_refs: tuple[str, ...]


class HistoricalCandidateSeedRetriever:
    """Convert sufficiently supported historical configurations into proposals."""

    def __init__(self, store: JsonBenchmarkRecordStore | None = None) -> None:
        self._store = store or JsonBenchmarkRecordStore()
        self._retriever = BenchmarkEvidenceRetriever()

    def retrieve(
        self,
        model_name: str,
        facts: PlanningFacts,
        *,
        records: list[BenchmarkRecord] | None = None,
    ) -> tuple[HistoricalCandidateSeed, ...]:
        """Use caller-authorized records when supplied, including an empty authorized result."""
        if facts.workload_profile is None:
            return ()
        query = self._query(
            model_name,
            facts,
            CandidateConfiguration(
                topology="optimized-baseline",
                decode_tp=1,
                decode_replicas=1,
            ),
        )
        if records is None:
            records = self._store.find_seed_candidates(query, observed_after=self._minimum_timestamp())
        configurations = {
            self._configuration_key(record.configuration): record.configuration
            for record in records
            if record.configuration.topology
            in {
                "optimized-baseline",
                "pd-disaggregation",
                "tiered-prefix-cache",
                "precise-prefix-cache-routing",
            }
        }
        seeds: list[HistoricalCandidateSeed] = []
        for _configuration_key, configuration in sorted(configurations.items()):
            matching_records = [record for record in records if record.configuration == configuration]
            evidence = self._retriever.retrieve(
                self._candidate_id(configuration),
                self._query(model_name, facts, configuration),
                matching_records,
            )
            if not self._is_qualified(
                evidence.status, evidence.confidence, evidence.effective_sample_size, evidence.negative_evidence
            ):
                continue
            seeds.append(
                HistoricalCandidateSeed(
                    proposal=CandidateProposal(
                        provider_ref=configuration.topology,
                        replicas=configuration.decode_replicas,
                        tensor_parallel_size=configuration.decode_tp,
                        prefill_replicas=configuration.prefill_replicas,
                        prefill_tensor_parallel_size=configuration.prefill_tp,
                        rationale="Qualified historical benchmark configuration.",
                    ),
                    evidence_refs=tuple(f"benchmark:{reference}" for reference in evidence.benchmark_refs),
                )
            )
        return tuple(seeds)

    def _query(
        self,
        model_name: str,
        facts: PlanningFacts,
        candidate: CandidateConfiguration,
    ) -> BenchmarkQuery:
        return BenchmarkQuery(
            **historical_query_context(model_name, facts),
            candidate=candidate,
        )

    def _minimum_timestamp(self) -> datetime:
        maximum_age_days = self._retriever.half_life_days * -log2(self._retriever.min_weight)
        return datetime.now(UTC) - timedelta(days=maximum_age_days)

    def _is_qualified(
        self,
        status: str,
        confidence: float,
        effective_sample_size: float,
        negative_evidence: list[str],
    ) -> bool:
        minimum_samples = self._retriever.min_effective_samples - 1e-9
        return (
            not negative_evidence
            and confidence >= 0.5
            and (status == "measured" or (status == "interpolated" and effective_sample_size >= minimum_samples))
        )

    @staticmethod
    def _configuration_key(configuration: CandidateConfiguration) -> tuple[object, ...]:
        return (
            configuration.topology,
            configuration.decode_tp,
            configuration.decode_replicas,
            configuration.prefill_tp or 0,
            configuration.prefill_replicas or 0,
            configuration.optimizations,
        )

    @staticmethod
    def _candidate_id(configuration: CandidateConfiguration) -> str:
        if configuration.topology == "pd-disaggregation":
            return (
                f"pd-disaggregation-p{configuration.prefill_replicas}-tp{configuration.prefill_tp}"
                f"-d{configuration.decode_replicas}-tp{configuration.decode_tp}"
            )
        return f"{configuration.topology}-tp{configuration.decode_tp}-r{configuration.decode_replicas}"
