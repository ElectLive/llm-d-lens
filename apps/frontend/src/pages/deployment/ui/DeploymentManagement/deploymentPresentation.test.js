import assert from 'node:assert/strict';
import test from 'node:test';

import { derivePodStatusGroup, effectiveStatusGroup, effectiveStatusLabel, expectedModelPodCount } from './deploymentPresentation.js';

const pod = (overrides = {}) => ({
    name: 'model',
    phase: 'Running',
    ready: true,
    status_reason: 'Running',
    restarts: 0,
    ...overrides,
});

test('a completed (calibration) pod does not mark a ready deployment not-ready', () => {
    const pods = [
        pod(),
        { name: 'calibrate-peak-throughput-9dgr6', phase: 'Succeeded', ready: false, status_reason: 'Completed', restarts: 0 },
        pod({ name: 'epp' }),
    ];

    assert.equal(derivePodStatusGroup(pods), 'Ready');
    assert.equal(effectiveStatusGroup({ status: 'ready' }, pods), 'Ready');
    assert.equal(effectiveStatusLabel({ status: 'ready' }, pods), 'READY');
});

test('only terminal pods yield no conclusive pod status', () => {
    assert.equal(derivePodStatusGroup([{ phase: 'Succeeded', ready: false }]), null);
});

test('a running not-ready pod still downgrades a ready deployment', () => {
    const pods = [pod(), pod({ name: 'epp', ready: false })];

    assert.equal(derivePodStatusGroup(pods), 'In progress');
    assert.equal(effectiveStatusLabel({ status: 'ready' }, pods), 'READY · CAPACITY REDUCED');
});

test('a completed pod does not hide a genuine pod failure', () => {
    const pods = [
        { name: 'calibrate-peak-throughput-9dgr6', phase: 'Succeeded', ready: false },
        pod({ name: 'model', phase: 'Failed', ready: false, status_reason: 'Failed' }),
    ];

    assert.equal(derivePodStatusGroup(pods), 'Failed');
    assert.equal(effectiveStatusLabel({ status: 'ready' }, pods), 'READY · POD ERROR');
});

test('no pods make a ready deployment unavailable', () => {
    assert.equal(effectiveStatusGroup({ status: 'ready' }, []), 'Failed');
    assert.equal(effectiveStatusLabel({ status: 'ready' }, []), 'READY · POD UNAVAILABLE');
});

test('an EPP-only namespace makes a ready deployment unavailable', () => {
    const pods = [pod({ name: 'optimized-baseline-epp-abc', labels: { 'llm-d-router-gateway': 'optimized-baseline-epp' } })];

    assert.equal(effectiveStatusGroup({ status: 'ready' }, pods), 'Failed');
});

test('fewer ready model-server pods than requested replicas is degraded', () => {
    const pods = [
        pod({ name: 'decode-a', labels: { 'llm-d.ai/role': 'decode', 'llm-d.ai/inference-serving': 'true' } }),
        pod({ name: 'decode-b', phase: 'Pending', ready: false, labels: { 'llm-d.ai/role': 'decode' } }),
    ];

    assert.equal(effectiveStatusGroup({ status: 'ready', replicas: 2 }, pods), 'Degraded');
    assert.equal(effectiveStatusLabel({ status: 'ready', replicas: 2 }, pods), 'READY · CAPACITY REDUCED');
});

test('expected model pods include prefill and decode replicas', () => {
    assert.equal(expectedModelPodCount({ replicas: 2 }), 2);
    assert.equal(expectedModelPodCount({ replicas: 2, scale: { prefillReplicas: 1, decodeReplicas: 2 } }), 3);
    assert.equal(expectedModelPodCount({ replicas: 2, scale: { prefillReplicas: 1, decodeReplicas: 2, slices: 2 } }), 6);
    assert.equal(expectedModelPodCount({ replicas: 1, scale: { prefillReplicas: null, decodeReplicas: 1 } }), 1);
});

test('a P/D deployment is degraded when one of three expected model pods is missing', () => {
    const pods = [
        pod({ name: 'prefill-a', labels: { 'llm-d.ai/role': 'prefill', 'llm-d.ai/inference-serving': 'true' } }),
        pod({ name: 'decode-a', labels: { 'llm-d.ai/role': 'decode', 'llm-d.ai/inference-serving': 'true' } }),
        pod({ name: 'decode-b', phase: 'Pending', ready: false, labels: { 'llm-d.ai/role': 'decode' } }),
    ];

    assert.equal(expectedModelPodCount({ replicas: 2, scale: { prefillReplicas: 1, decodeReplicas: 2 } }), 3);
    assert.equal(expectedModelPodCount({ replicas: 2, scale: { prefillReplicas: 1, decodeReplicas: 2, slices: 2 } }), 6);
    assert.equal(effectiveStatusGroup({ status: 'ready', replicas: 2, scale: { prefillReplicas: 1, decodeReplicas: 2 } }, pods), 'Degraded');
});
