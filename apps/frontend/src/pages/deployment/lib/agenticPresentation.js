import { formatTimestamp } from '../../../shared/lib/formatTimestamp.js';

export function retrievalBranchSummaries(run) {
    const evidence = run?.planning_evidence || [];
    const publicSql = evidence.find((item) => item.id === 'public-rag:query');
    const cases = evidence.find((item) => item.source === 'configuration-cases')?.details;
    const branches = cases?.branch_status;
    const internal = branches?.internal_sql;
    const retainedCases = cases?.cases || [];
    const sourceCases = (internal) => retainedCases.filter((item) =>
        internal ? item.origins?.includes('agentic:evaluate') : item.origins?.some((origin) => origin !== 'agentic:evaluate'));
    const rows = [];
    const diagnostic = (status, reason) => `${status.replaceAll('_', ' ')}${reason ? `: ${reason.replaceAll('_', ' ')}` : ''}.`;
    if (publicSql || branches?.public_sql) {
        const { retrieval = branches?.public_sql, response, code } = publicSql?.details || {};
        rows.push({
            id: 'public_sql',
            label: 'Public SQL measured observations',
            status: publicSql?.status || branches.public_sql.status,
            evidence: publicSql,
            cases: sourceCases(false),
            summary: ['available', 'no_matches'].includes(publicSql?.status || branches.public_sql.status) && retrieval
                ? `Retrieved ${retrieval.returned_observations ?? 'unknown'} of ${retrieval.matched_observations ?? 'unknown'} matching observations; ${response?.evidence?.length ?? sourceCases(false).reduce((count, item) => count + (item.measurements?.length ?? 0), 0)} retained for planning.`
                : diagnostic(publicSql?.status || branches.public_sql.status, code),
        });
    }
    if (internal) {
        rows.push({
            id: 'internal_sql',
            label: 'Internal SQL experiments',
            status: internal.status,
            cases: sourceCases(true),
            summary: ['available', 'empty'].includes(internal.status)
                ? `Retrieved ${internal.returned_count ?? 'unknown'} of ${internal.candidate_total ?? 'unknown'} matching experiments; ${sourceCases(true).length} retained for planning.${internal.reason ? ` ${diagnostic(internal.reason)}` : ''}`
                : diagnostic(internal.status, internal.reason),
        });
    }
    for (const row of rows) {
        const fallbackCount = row.cases?.filter((item) => item.ranking?.queue === 'fallback').length ?? 0;
        if (fallbackCount) {
            row.warning = `${fallbackCount} of ${row.cases.length} retained cases have neither comparable workload nor valid latency; these are fallback references, not low-latency evidence.`;
        }
    }
    return rows;
}

export function planningEvidenceLabel(item, hasAicPrediction) {
    if (item.id === 'public-rag:query' || item.source === 'configuration-cases') return null;
    if (item.id === 'aic:support' && hasAicPrediction) return null;
    if (item.id === 'aic:search') return 'AIC performance prediction';
    if (item.id.startsWith('cluster-overview:')) return 'Live cluster capacity';
    if (item.id === 'deploy-history') return 'Deployment history';
    if (item.id === 'simulation-history') return 'Simulation history';
    return item.source;
}

export function publicRagMatchSummary(match) {
    if (!match) return [];
    const lines = Object.entries(match.workload?.fields || {}).map(([name, field]) =>
        `${name}: requested ${field.requested}, reported ${field.reported}, difference ${field.difference}; ${field.basis}.`);
    if (match.workload?.unknown_fields?.length) {
        lines.push(`Unknown workload fields: ${match.workload.unknown_fields.join(', ')}.`);
    }
    if (match.workload?.similarity !== null && match.workload?.similarity !== undefined) {
        lines.push(`Workload similarity: ${Math.round(match.workload.similarity * 100)}%; coverage: ${Math.round(match.workload.coverage * 100)}%. Not performance confidence.`);
    }
    if (match.workload?.nominal_prompt_shape) {
        lines.push(`Reported nominal prompt shape: ${JSON.stringify(match.workload.nominal_prompt_shape)}. Not a measured mean or P95.`);
    }
    if (match.ranking?.queue === 'fallback') {
        lines.push('Fallback queue: neither comparable workload nor valid latency; configuration reference only, not low-latency evidence.');
    }
    if (match.latency) {
        lines.push(`Published latency target status: ${match.latency.status.replaceAll('_', ' ')}. Not current deployment SLO compliance.`);
        for (const [metric, field] of Object.entries(match.latency.fields || {})) {
            for (const target of field.targets ?? [field]) {
                const statistic = target.statistic ? ` ${target.statistic}` : '';
                const operator = target.operator ? `${target.operator} ` : '';
                lines.push(`${metric.toUpperCase()}${statistic} target ${operator}${target.target_ms} ms: ${target.observations.length
                    ? target.observations.map((item) => `${item.value_ms} ms (${item.statistic || 'unspecified statistic'}; deviation ${(item.relative_deviation * 100).toFixed(1)}%)`).join('; ')
                    : 'unknown'}.`);
            }
        }
    }
    if (match.configuration_basis) {
        lines.push(`Configuration basis: ${JSON.stringify(match.configuration_basis)}.`);
    }
    return lines;
}

export function historicalEvidenceSummary(samples) {
    const history = Array.isArray(samples) ? samples : [];
    if (!history.length) return null;
    const failed = history.filter((sample) => sample.outcome === 'failed').length;
    return {
        sampleCount: history.length,
        successfulCount: history.length - failed,
        failedCount: failed,
    };
}

export function generatorLabel(generator) {
    return generator === 'ai-mcp' ? 'AI MCP generator' : 'Deterministic generator';
}

export function relaxedAicCandidates(run) {
    const trace = run?.generator_tool_trace?.find((entry) => entry.tool === 'search_aic_with_relaxation' && entry.status === 'success');
    return Array.isArray(trace?.result?.candidates) ? trace.result.candidates : [];
}

export function historicalBenchmarkLabel(sample) {
    const observedAt = formatTimestamp(sample.observed_at, { invalid: () => 'unknown date' });
    const relevance = Number.isFinite(sample.match_confidence)
        ? `${Math.round(sample.match_confidence * 100)}% relevance`
        : 'unknown relevance';
    const metrics = [
        sample.ttft_p95_ms !== null && sample.ttft_p95_ms !== undefined ? `TTFT ${sample.ttft_p95_ms} ms` : null,
        sample.tpot_p95_ms !== null && sample.tpot_p95_ms !== undefined ? `TPOT ${sample.tpot_p95_ms} ms` : null,
        sample.throughput_tokens_per_s !== null && sample.throughput_tokens_per_s !== undefined
            ? `${sample.throughput_tokens_per_s} tokens/s` : null,
        sample.failure_reason ? `failure: ${sample.failure_reason}` : null,
    ].filter(Boolean).join(', ');
    return `${sample.benchmark_id} | ${observedAt} | ${sample.outcome} | ${relevance}${metrics ? ` | ${metrics}` : ''}`;
}

export function candidateAnalysis(candidate, index, aicConfigs, relaxedConfigs, clusterDetails) {
    const aicApplicable = ['baseline-vllm', 'optimized-baseline', 'pd-disaggregation'].includes(candidate.provider_ref);
    const matchingPrediction = (config) => {
        const mode = config.topologyMode ?? config.mode;
        const decodeTp = config.decodeTp ?? config.decode_tp ?? config.tp;
        const decodeReplicas = config.decodeReplicas ?? config.decode_replicas ?? config.replicas;
        const prefillTp = config.prefillTp ?? config.prefill_tp;
        const prefillReplicas = config.prefillReplicas ?? config.prefill_replicas;
        return (mode === 'agg' && ['baseline-vllm', 'optimized-baseline'].includes(candidate.provider_ref)
            && Number(decodeTp) === candidate.tensor_parallel_size && Number(decodeReplicas) === candidate.replicas)
            || (mode === 'disagg' && candidate.provider_ref === 'pd-disaggregation'
                && Number(decodeTp) === candidate.tensor_parallel_size && Number(decodeReplicas) === candidate.replicas
                && Number(prefillTp) === candidate.prefill_tensor_parallel_size && Number(prefillReplicas) === candidate.prefill_replicas);
    };
    const prediction = aicConfigs.find(matchingPrediction);
    const relaxedPrediction = prediction ? null : relaxedConfigs.find(matchingPrediction);
    const metrics = prediction?.predicted || prediction;
    const relaxedMetrics = relaxedPrediction?.predicted;
    const estimate = candidate.performance_estimate;
    const availability = clusterDetails?.available_gpu_count;
    const performance = candidate.performance_estimate_source === 'aic_estimate' && estimate
        ? `AIC estimate (exact topology${candidate.slo_status === 'not_satisfied' ? '; original SLO not met' : ''}): TTFT ${estimate.ttft_ms?.toFixed(1) ?? 'n/a'} ms, TPOT ${estimate.tpot_ms?.toFixed(1) ?? 'n/a'} ms, throughput ${estimate.throughput_tokens_per_sec?.toFixed(1) ?? 'n/a'} tokens/s.`
        : prediction
        ? `AIC estimate: TTFT ${metrics.ttftMs?.toFixed(1) || metrics.ttft_ms?.toFixed(1) || 'n/a'} ms, TPOT ${metrics.tpotMs?.toFixed(1) || metrics.tpot_ms?.toFixed(1) || 'n/a'} ms, throughput ${metrics.throughputTps?.toFixed(1) || metrics.throughput_tokens_per_sec?.toFixed(1) || 'n/a'} tokens/s.`
        : relaxedPrediction
            ? `Relaxed AIC estimate (does not meet the original SLO): TTFT ${relaxedMetrics?.ttftMs?.toFixed(1) ?? 'n/a'} ms, TPOT ${relaxedMetrics?.tpotMs?.toFixed(1) ?? 'n/a'} ms, throughput ${relaxedMetrics?.throughputTps?.toFixed(1) ?? 'n/a'} tokens/s.`
        : aicApplicable
            ? 'No exact AIC topology prediction is available for this baseline or PD shape.'
            : 'AIC performance predictions do not apply to this guide; it is ranked using operator preference, guide suitability, and resource cost.';
    const resources = Number.isFinite(availability)
        ? `Hardware: requires ${candidate.required_gpus} GPU; ${availability} GPU are currently free. Capacity is rechecked immediately before deployment.`
        : `Hardware: requires ${candidate.required_gpus} GPU. Capacity is rechecked immediately before deployment.`;
    const ranking = candidate.performance_estimate_source === 'aic_estimate' && estimate
        ? `Rank ${index + 1}. This fixed-topology estimate informs the SLO assessment; ranking also considers operator preference, guide suitability, and GPU cost.`
        : prediction
        ? `Rank ${index + 1}. Exact AIC predictions are ordered by TTFT, then TPOT, then throughput before guide heuristics.`
        : aicApplicable
            ? `Rank ${index + 1}. No AIC prediction meets the original SLO, so this candidate is ordered by operator preference, guide suitability, and GPU cost.`
            : `Rank ${index + 1}. This guide has no comparable AIC prediction, so it is ordered by operator preference, guide suitability, and GPU cost.`;
    return { performance, resources, ranking };
}
