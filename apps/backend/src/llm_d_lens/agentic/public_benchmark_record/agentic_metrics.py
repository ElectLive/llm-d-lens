"""Only measured fields of the existing normalized Agentic record contract."""

from dataclasses import dataclass
from decimal import Decimal

from llm_d_lens.agentic.benchmark_evidence import BenchmarkRecord
from llm_d_lens.agentic.public_benchmark_record.contracts import effective_statistic, metric_aliases
from llm_d_lens.agentic.public_benchmark_record.query_contracts import QueryFilters


@dataclass(frozen=True)
class AgenticMetric:
    field: str
    metric: str
    statistic: str | None
    unit: str
    normalized: bool


AGENTIC_METRICS = (
    AgenticMetric("ttft_p95_ms", "ttft", "p95", "ms", True),
    AgenticMetric("tpot_p95_ms", "tpot", "p95", "ms/token", False),
    AgenticMetric("e2e_p95_ms", "e2e", "p95", "ms", True),
    AgenticMetric("throughput_tokens_per_s", "throughput", None, "tokens/s", True),
    AgenticMetric("error_rate", "error_rate", None, "fraction", False),
)


def metric_matches(spec: AgenticMetric, filters: QueryFilters) -> bool:
    return (
        (filters.metric is None or spec.metric in metric_aliases(filters.metric))
        and (filters.statistic is None or effective_statistic(filters.statistic) == effective_statistic(spec.statistic))
        and (filters.unit is None or filters.unit == spec.unit)
        and (not filters.normalized or spec.normalized)
    )


def measured_values(record: BenchmarkRecord, filters: QueryFilters) -> tuple[tuple[AgenticMetric, Decimal], ...]:
    values = []
    for spec in AGENTIC_METRICS:
        value = getattr(record.metrics, spec.field)
        if value is None or not metric_matches(spec, filters):
            continue
        number = Decimal(str(value))
        if not number.is_finite():
            continue
        if filters.value_min is not None and number < filters.value_min:
            continue
        if filters.value_max is not None and number > filters.value_max:
            continue
        values.append((spec, number))
    return tuple(values)
