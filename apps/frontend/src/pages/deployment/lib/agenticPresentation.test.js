import assert from 'node:assert/strict';
import test from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { RetrievalEvidenceSources } from '../ui/AgenticDeploymentWorkspace.jsx';
import { candidateAnalysis, generatorLabel, historicalBenchmarkLabel, historicalEvidenceSummary, planningEvidenceLabel, publicRagMatchSummary, relaxedAicCandidates, retrievalBranchSummaries } from './agenticPresentation.js';

test('retrieval summaries distinguish public observations from internal experiments', () => {
    const summaries = retrievalBranchSummaries({ planning_evidence: [
        { id: 'public-rag:query', status: 'no_matches', details: {
            retrieval: { returned_observations: 0, matched_observations: 0 }, response: { evidence: [] },
        } },
        { source: 'configuration-cases', details: { branch_status: {
            internal_sql: { status: 'empty', returned_count: 0, candidate_total: 0 },
        } } },
    ] });
    assert.deepEqual(summaries.map(({ label, status }) => ({ label, status })), [
        { label: 'Public SQL measured observations', status: 'no_matches' },
        { label: 'Internal SQL experiments', status: 'empty' },
    ]);
    assert.match(summaries[0].summary, /Retrieved 0 of 0 matching observations; 0 retained/);
    assert.match(summaries[1].summary, /Retrieved 0 of 0 matching experiments/);
});

test('unavailable, forbidden and absent branches are not presented as zero matches', () => {
    assert.deepEqual(retrievalBranchSummaries({}), []);
    const summaries = retrievalBranchSummaries({ planning_evidence: [
        { id: 'public-rag:query', status: 'unavailable', details: { code: 'TimeoutError' } },
        { source: 'configuration-cases', details: { branch_status: {
            internal_sql: { status: 'forbidden', returned_count: 0, candidate_total: 0, reason: 'query_forbidden' },
        } } },
    ] });
    for (const row of summaries) assert.doesNotMatch(row.summary, /Retrieved 0/);
    assert.match(summaries[0].summary, /TimeoutError/);
    assert.match(summaries[1].summary, /query forbidden/);
});

test('retrieval summaries keep retrieved and budget-retained evidence counts separate', () => {
    const summaries = retrievalBranchSummaries({ planning_evidence: [
        { id: 'public-rag:query', status: 'available', details: {
            retrieval: { returned_observations: 50, matched_observations: 504 }, response: { evidence: [{}] },
        } },
    ] });
    assert.match(summaries[0].summary, /Retrieved 50 of 504 matching observations; 1 retained/);
});

test('public RAG match presentation preserves differences, unknowns, statistics and configuration basis', () => {
    assert.deepEqual(publicRagMatchSummary(null), []);
    const lines = publicRagMatchSummary({
        workload: {
            fields: { mean_input_tokens: { requested: 4000, reported: 7200, difference: 3200, basis: 'nominal_length_proxy' } },
            unknown_fields: ['p95_input_tokens'], similarity: 0.75, coverage: 2 / 3,
            nominal_prompt_shape: { shared_prefix_tokens: 6000, question_tokens: 1200, output_tokens: 1000 },
        },
        latency: { status: 'unknown', fields: {
            ttft: { target_ms: 2000, observations: [{ value_ms: 900, statistic: 'p90', relative_deviation: -0.55 }] },
            tpot: { target_ms: 30, observations: [] },
        } },
        configuration_basis: { configuration: { tp: 2, replicas: 8 } },
    }).join('\n');
    assert.match(lines, /requested 4000, reported 7200, difference 3200/);
    assert.match(lines, /Unknown workload fields: p95_input_tokens/);
    assert.match(lines, /coverage: 67%/);
    assert.match(lines, /Not a measured mean or P95/);
    assert.match(lines, /900 ms \(p90; deviation -55.0%\)/);
    assert.match(lines, /TPOT target 30 ms: unknown/);
    assert.match(lines, /Not current deployment SLO compliance/);
    assert.match(lines, /"tp":2,"replicas":8/);
});

test('historical evidence presentation distinguishes empty, successful, and failed records', () => {
    assert.equal(historicalEvidenceSummary([]), null);
    assert.deepEqual(historicalEvidenceSummary([
        { outcome: 'succeeded' }, { outcome: 'failed' }, { outcome: 'succeeded' },
    ]), { sampleCount: 3, successfulCount: 2, failedCount: 1 });
});

test('latency presentation explains every target including the failing statistic', () => {
    const targets = [
        { target_ms: 100, statistic: 'mean', operator: '<=', observations: [
            { value_ms: 200, statistic: 'mean', relative_deviation: 1, target_satisfied: false },
        ] },
        { target_ms: 300, statistic: 'p95', operator: '<', observations: [
            { value_ms: 250, statistic: 'p95', relative_deviation: -1 / 6, target_satisfied: true },
        ] },
        { target_ms: 50, statistic: 'p99', operator: '>=', observations: [] },
    ];
    const lines = publicRagMatchSummary({
        latency: { status: 'not_satisfied', fields: { ttft: { ...targets[1], targets } } },
    });
    assert.match(lines[0], /not satisfied/);
    assert.equal(lines.length, 4);
    assert.match(lines[1], /TTFT mean target <= 100 ms: 200 ms \(mean; deviation 100.0%\)/);
    assert.match(lines[2], /TTFT p95 target < 300 ms: 250 ms \(p95; deviation -16.7%\)/);
    assert.match(lines[3], /TTFT p99 target >= 50 ms: unknown/);
});

test('single latency targets retain their statistic and operator', () => {
    const lines = publicRagMatchSummary({
        latency: { status: 'satisfied', fields: {
            tpot: { target_ms: 20, statistic: 'mean', operator: '=', observations: [
                { value_ms: 20, statistic: 'mean', relative_deviation: 0 },
            ] },
        } },
    });
    assert.match(lines[1], /TPOT mean target = 20 ms: 20 ms/);
});

test('fallback references are explicitly labeled without a low-latency claim', () => {
    const ranking = { queue: 'fallback', fallback_reason: 'no_comparable_workload_or_valid_latency' };
    const lines = publicRagMatchSummary({ ranking }).join('\n');
    assert.match(lines, /Fallback queue/);
    assert.match(lines, /neither comparable workload nor valid latency/);
    assert.match(lines, /not low-latency evidence/);
    const summaries = retrievalBranchSummaries({ planning_evidence: [{
        source: 'configuration-cases',
        details: {
            cases: [{ origins: ['prism'], ranking: { queue: 'ranked' } }, { origins: ['prism'], ranking }],
            branch_status: { public_sql: { status: 'available', returned_observations: 2, matched_observations: 2 } },
        },
    }] });
    assert.equal(summaries.length, 1);
    assert.equal(summaries[0].label, 'Public SQL measured observations');
    assert.match(summaries[0].warning, /1 of 2 retained cases/);
    assert.match(summaries[0].warning, /not low-latency evidence/);
});

test('source evidence is grouped once, collapsed by default, with counts and measured citations', () => {
    const publicEvidence = {
        id: 'public-rag:query', source: 'public-benchmark-rag', status: 'available',
        details: {
            retrieval: { returned_observations: 2, matched_observations: 9, mode: 'ranked' },
            response: {
                evidence: ['one', 'two'].map((item_id) => ({
                    item_id, metric: 'ttft', statistic: 'p95', value: 40, unit: 'ms',
                    citation: { source_path: `${item_id}.json`, source_revision: 'rev-public', sha256: 'public-hash' },
                })),
                truncated: true,
            },
        },
    };
    const caseEvidence = {
        id: 'configuration-cases:query', source: 'configuration-cases',
        details: {
            branch_status: {
                public_sql: { status: 'available' },
                internal_sql: { status: 'available', returned_count: 4, candidate_total: 12 },
            },
            cases: [{
                case_id: 'case-internal', origins: ['agentic:evaluate'], experiment_id: 'internal-run',
                model: 'test-model', hardware: 'test-device', applicability: 'adaptable', latency_status: 'unknown',
                configuration: { tp: 2 }, workload: { concurrency: 4 }, ranking: { queue: 'fallback' },
                known_differences: ['runtime'], unknown_fields: ['model.revision'],
                measurements: [{ evidence_id: 'benchmark:bench-one', metric: 'tpot', statistic: 'mean', value: 20, unit: 'ms', quality: 'informative' }],
                provenance: [{ benchmark_id: 'bench-one' }], validation_required: ['runtime compatibility'],
            }],
        },
    };
    const run = {
        planning_evidence: [publicEvidence, caseEvidence],
        decision_evidence_ids: ['public-rag:query', 'public-rag:one', 'benchmark:bench-one'],
    };
    const html = renderToStaticMarkup(React.createElement(RetrievalEvidenceSources, { run }));
    assert.equal((html.match(/<details\b/g) || []).length, 2);
    assert.equal((html.match(/<summary\b/g) || []).length, 2);
    assert.doesNotMatch(html, /<details[^>]*\bopen(?:=|\s|>)/);
    for (const label of ['Public SQL measured observations', 'Internal SQL experiments']) {
        assert.equal(html.split(label).length - 1, 1);
    }
    assert.match(html, /Retrieved 2 of 9 matching observations; 2 retained/);
    assert.match(html, /Retrieved 4 of 12 matching experiments; 1 retained/);
    assert.match(html, /one.json.*rev-public/);
    assert.match(html, /public-hash/);
    assert.match(html, /More observations matched/);
    assert.match(html, /Known differences: runtime/);
    assert.match(html, /Unknown fields: model.revision/);
    assert.match(html, /Validation required: runtime compatibility/);
    assert.match(html, /1 of 1 retained cases.*not low-latency evidence/);
    assert.match(html, /Compare model, hardware, runtime and workload/);
    assert.equal((html.match(/used for selection/g) || []).length, 4);
    assert.equal((html.match(/context only/g) || []).length, 1);
    assert.doesNotMatch(html, /configuration-cases|priority over AIC|Document RAG/);
    for (const item of run.planning_evidence) assert.equal(planningEvidenceLabel(item, false), null);
    assert.equal(planningEvidenceLabel({ id: 'aic:search', source: 'aic' }, true), 'AIC performance prediction');
    assert.equal(planningEvidenceLabel({ id: 'cluster-overview:one', source: 'cluster' }, true), 'Live cluster capacity');
});

test('source grouping preserves unavailable and forbidden diagnostics without fabricated zero counts', () => {
    const html = renderToStaticMarkup(React.createElement(RetrievalEvidenceSources, { run: {
        planning_evidence: [
            { id: 'public-rag:query', status: 'unavailable', details: { code: 'TimeoutError' } },
            { source: 'configuration-cases', details: { branch_status: {
                internal_sql: { status: 'forbidden', reason: 'query_forbidden' },
            } } },
        ],
    } }));
    assert.equal((html.match(/<details\b/g) || []).length, 2);
    assert.match(html, /TimeoutError/);
    assert.match(html, /query forbidden/);
    assert.doesNotMatch(html, /Retrieved 0|undefined|configuration-cases/);
});

test('historical benchmark label preserves measured metrics, recency, relevance, and failure reasons', () => {
    const label = historicalBenchmarkLabel({
        benchmark_id: 'bench-7', observed_at: '2026-09-20T10:00:00+00:00', outcome: 'failed',
        match_confidence: 0.875, ttft_p95_ms: 410, tpot_p95_ms: 31,
        throughput_tokens_per_s: 720, failure_reason: 'OOM',
    });

    assert.match(label, /^bench-7 \| .*2026.* \| failed \| 88% relevance \| TTFT 410 ms, TPOT 31 ms, 720 tokens\/s, failure: OOM$/);
});

test('generator presentation distinguishes AI MCP and deterministic candidate generation', () => {
    assert.equal(generatorLabel('ai-mcp'), 'AI MCP generator');
    assert.equal(generatorLabel('deterministic'), 'Deterministic generator');
    assert.equal(generatorLabel('unknown'), 'Deterministic generator');
});

test('relaxed AIC candidates remain visible even when no prediction survives planning evidence filtering', () => {
    const candidates = [{topologyMode: 'agg', decodeTp: 1, decodeReplicas: 1, predicted: {ttftMs: 44.872, tpotMs: 6.454}}];
    assert.deepEqual(relaxedAicCandidates({planning_evidence: [], generator_tool_trace: [
        {tool: 'search_candidates', status: 'success', result: {candidates: []}},
        {tool: 'search_aic_with_relaxation', status: 'success', result: {candidates}},
    ]}), candidates);
    assert.deepEqual(relaxedAicCandidates({generator_tool_trace: [
        {tool: 'search_aic_with_relaxation', status: 'error', result: {candidates}},
    ]}), []);
});

test('candidate analysis displays a fixed-topology AIC estimate when search returned no candidates', () => {
    const candidate = {
        provider_ref: 'baseline-vllm', replicas: 1, tensor_parallel_size: 1, required_gpus: 1,
        slo_status: 'not_satisfied', performance_estimate_source: 'aic_estimate',
        performance_estimate: { ttft_ms: 222.535, tpot_ms: 47.497, throughput_tokens_per_sec: 20.674 },
    };
    const analysis = candidateAnalysis(candidate, 0, [], [], null);

    assert.match(analysis.performance, /AIC estimate \(exact topology; original SLO not met\): TTFT 222\.5 ms, TPOT 47\.5 ms, throughput 20\.7 tokens\/s/);
    assert.match(analysis.ranking, /fixed-topology estimate informs the SLO assessment/);
    assert.match(candidateAnalysis({ ...candidate, performance_estimate: null }, 0, [], [], null).performance,
        /No exact AIC topology prediction is available/);
});
