import assert from 'node:assert/strict';
import test from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MODELS } from '@repo/shared/catalog';
import { AuthContext } from '../../auth/AuthContext.js';
import AgenticDeploymentWorkspace from '../../deployment/ui/AgenticDeploymentWorkspace.jsx';
import ModelMarketPage from './ModelMarketPage.jsx';
import { standardRecommendation } from '../model/standardRecommendation.js';

test('deployment context uses the exact model card metadata for every listed context', () => {
    for (const model of MODELS.filter((model) => model.contextLength != null)) {
        assert.equal(standardRecommendation(model, null).maxModelLen, model.contextLength, model.repository);
    }
    const qwen = MODELS.find((model) => model.repository === 'Qwen/Qwen3-0.6B');
    assert.equal(standardRecommendation(qwen, null).maxModelLen, 40960);
});

test('deployment context keeps the default only when model card context is missing', () => {
    for (const contextLength of [null, undefined]) {
        assert.equal(standardRecommendation({ contextLength, sizeGiB: 1 }, null).maxModelLen, 4096);
    }
});

test('Agentic deployment shows reference placeholders without setting latency or workload defaults', () => {
    const html = renderToStaticMarkup(createElement(AgenticDeploymentWorkspace, {
        draft: {}, embedded: true,
    }));
    for (const [label, example] of [
        ['TTFT SLO \\(ms\\)', '2000'],
        ['TPOT SLO \\(ms\\)', '30'],
        ['Mean input tokens per request', '4000'],
        ['Mean output tokens per request', '1000'],
    ]) {
        assert.match(html, new RegExp(`${label}<input[^>]*placeholder:text-theme-muted[^>]*placeholder="e.g. ${example}"[^>]*value=""`));
    }
    assert.match(html, /Concurrent requests<input[^>]*value=""/);
    assert.match(html, /Request rate \(QPS\)<input[^>]*value=""/);
    assert.doesNotMatch(html, /End-to-end latency/);
    assert.doesNotMatch(html, /Planning hardware identity|Simulated target hardware model|Cluster default \(may include a test override\)/);
});

test('Agentic deployment expands SLO and workload sections and keeps vLLM parameters collapsed', () => {
    const html = renderToStaticMarkup(createElement(AgenticDeploymentWorkspace, {
        draft: {}, embedded: true,
        vllmParameterControls: createElement('span', null, 'Parameter controls'),
    }));
    const sections = [...html.matchAll(/<details\b([^>]*)><summary[^>]*>(.*?)<\/summary>/g)];
    const slo = sections.findIndex(([, , summary]) => summary.includes('SLO'));
    const preference = sections.findIndex(([, , summary]) => summary.includes('Preference'));
    const workload = sections.findIndex(([, , summary]) => summary.includes('Workload estimate'));
    const vllm = sections.findIndex(([, , summary]) => summary.includes('vLLM parameters'));
    assert.notEqual(slo, -1);
    assert.equal(preference, slo + 1);
    assert.equal(workload, preference + 1);
    assert.equal(vllm, workload + 1);
    assert.match(sections[slo][1], /\bopen=""/);
    assert.match(sections[slo][2], /Optional target/);
    assert.match(sections[workload][1], /\bopen=""/);
    assert.doesNotMatch(sections[vllm][1], /\bopen\b/);
    assert.match(html, /Parameter controls/);
    const sloSection = html.match(/<details\b[^>]*><summary[^>]*>.*?SLO .*?<\/summary>(.*?)<\/details>/)?.[1];
    assert.ok(sloSection);
    assert.match(sloSection, /^<div class="[^"]*\bgrid-cols-2\b[^"]*"><label[^>]*>TTFT SLO/);
    assert.match(sloSection, /<\/label><label[^>]*>TPOT SLO/);
});

test('Agentic deployment submits empty SLO and workload fields as null, not placeholder examples', async (t) => {
    let tree;
    function Capture() {
        tree = AgenticDeploymentWorkspace({
            draft: { clusterId: 'cluster-1', storageVolumeId: 'cache-1', model: 'Qwen/test' },
            embedded: true,
        });
        return tree;
    }
    renderToStaticMarkup(createElement(Capture));
    function findGenerate(node) {
        if (!node || typeof node !== 'object') return undefined;
        if (node.props?.onClick?.name === 'createPlan') return node;
        return [node.props?.children].flat(Infinity).map(findGenerate).find(Boolean);
    }
    const payloads = [];
    t.mock.method(globalThis, 'fetch', async (url, options) => {
        if (url === '/api/cluster/session?clusterId=cluster-1') {
            return Response.json({ sessionId: 'session-1' });
        }
        assert.equal(url, '/api/agentic-deployments/stream');
        payloads.push(JSON.parse(options.body));
        return new Response('event: complete\ndata: {"id":"run-1"}\n\n', {
            headers: { 'Content-Type': 'text/event-stream' },
        });
    });
    await findGenerate(tree).props.onClick();
    assert.equal(payloads.length, 1);
    const facts = payloads[0].planning_facts;
    assert.equal(facts.ttft_slo_ms, null);
    assert.equal(facts.tpot_slo_ms, null);
    assert.deepEqual(facts.workload_profile, {
        mean_input_tokens: null,
        mean_output_tokens: null,
        concurrency: null,
        request_rate: null,
    });
});

test('model catalog keeps search with category filters and preserves model configuration actions', (testContext) => {
    let hasReturnDraft = false;
    const originalStorage = Object.getOwnPropertyDescriptor(globalThis, 'sessionStorage');
    Object.defineProperty(globalThis, 'sessionStorage', {
        configurable: true,
        value: {
            getItem: (key) => hasReturnDraft && key === 'prism_model_market_return_draft'
                ? JSON.stringify({ model: MODELS[0].repository }) : null,
            removeItem() {},
        },
    });
    testContext.after(() => {
        if (originalStorage) Object.defineProperty(globalThis, 'sessionStorage', originalStorage);
        else delete globalThis.sessionStorage;
    });
    const renderPage = (canDeploy = true) => renderToStaticMarkup(createElement(AuthContext.Provider, { value: { can: () => canDeploy } },
        createElement(ModelMarketPage, { onNavigate() {} })));
    const catalog = renderPage();
    const header = catalog.match(/<header\b[\s\S]*?<\/header>/)?.[0] || '';
    assert.match(header, />Model Catalog<\/h1>/);
    assert.doesNotMatch(header, /aria-label="Search models"/);
    assert.match(catalog, /role="tablist" aria-label="Model categories"[\s\S]*aria-label="Search models"[\s\S]*<section/);
    assert.doesNotMatch(catalog, /Select a model|Popular open models across/);
    hasReturnDraft = true;
    const html = renderPage();
    assert.match(html, />Model Catalog<\/button>/);
    for (const page of [html, renderPage(false)]) {
        assert.doesNotMatch(page.match(/<header\b[\s\S]*?<\/header>/)?.[0] || '', />Model Catalog<\/button>/);
        assert.match(page, /<\/header>[\s\S]*>Model Catalog<\/button>[\s\S]*<(?:form|div)/);
    }
    assert.match(html, />Manual</);
    assert.match(html, /Template-based deployment/);
    assert.match(html, /Configure deployment/);
    assert.doesNotMatch(html, /Template-based benchmark configuration|>Advanced</);
});
