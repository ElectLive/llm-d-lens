import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

test('compatibility launcher validates the Python catalog without generating a Node server artifact', () => {
    const result = spawnSync(process.execPath, ['hack/generate-mcp-tools.mjs', '--check', '--json'], {
        cwd: new URL('../', import.meta.url),
        encoding: 'utf8',
        maxBuffer: 16 * 1024 * 1024,
    });
    assert.equal(result.status, 0, result.error?.message || result.stderr);
    const tools = JSON.parse(result.stdout);
    assert.equal(new Set(tools.map(tool => tool.name)).size, tools.length);
    const query = tools.find(tool => tool.name === 'get_cluster_overview');
    assert.ok(query);
    assert.equal(query.riskTier, 'read');
    assert.equal(query.inputSchema.type, 'object');
    assert.ok(query.inputSchema.required.includes('clusterId'));
    assert.equal(query.route, '/api/cluster/overview');
});

test('compatibility launcher rejects unsupported arguments', () => {
    const result = spawnSync(process.execPath, ['hack/generate-mcp-tools.mjs', '--unsupported'], {
        cwd: new URL('../', import.meta.url),
        encoding: 'utf8',
    });
    assert.equal(result.status, 2);
    assert.match(result.stderr, /Usage:/);
});
