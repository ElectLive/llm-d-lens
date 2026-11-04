import assert from 'node:assert/strict';
import test from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { AuthContext } from '../../pages/auth/AuthContext.js';
import { canAccessView, VIEW_PERMISSIONS } from '../../pages/auth/permissions.js';
import LeftNavigation from '../ui/LeftNavigation.jsx';
import { firstAccessibleView, NAVIGATION_GROUPS, resolveView, viewTitle } from './views.js';

test('loading pages use their own navigation or detail title', () => {
  for (const item of NAVIGATION_GROUPS.flatMap(group => group.items)) {
    assert.equal(viewTitle(item.view), item.label);
  }
  assert.equal(viewTitle('welcome/issues'), 'Issues');
  assert.equal(viewTitle('optimization-evaluate-new'), 'New Benchmark Task');
  assert.equal(viewTitle('optimization-evaluation-details'), 'Benchmark details');
  assert.equal(viewTitle('optimization-simulate-details'), 'Simulation task details');
  assert.equal(viewTitle(''), 'Lens');
});

test('landing page follows the first accessible sidebar entry, including custom grants', () => {
  for (const [permissions, expected] of [
    [['*:*:*'], 'welcome'],
    [['deployment:run:read', 'cluster:cluster:read'], 'welcome'],
    [['deployment:run:read'], 'optimization-deployments'],
    [['model-service:group:read', 'model-service:usage:read'], 'admin/model-service'],
    [['evaluate:run:read', 'model-service:usage:read'], 'optimization-evaluate'],
    [['model-service:usage:read', 'model-service:token:manage', 'model-service:inference:use'], 'model-service'],
    [['model-service:token:manage'], 'api-keys'],
    [['cluster:cluster:read', 'user:user:read'], 'welcome'],
    [['user:user:read'], 'admin/users'],
    [['playground:chat:use'], 'playground'],
    [['ai-provider:provider:read', 'model-service:usage:read'], 'usage'],
    [[], ''],
  ]) {
    assert.equal(firstAccessibleView(view => canAccessView(permissions, view)), expected);
  }
});

test('sidebar preserves every destination once in the requested group and order', () => {
  assert.deepEqual(NAVIGATION_GROUPS.map(group => [group.title || '', group.items.map(item => [item.label, item.view])]), [
    ['', [['Welcome', 'welcome']]],
    ['Inference', [['Deployments', 'optimization-deployments'], ['Model Services', 'admin/model-service'], ['Model Catalog', 'model-market']]],
    ['Performance', [['Evaluation', 'optimization-evaluate'], ['Trace Replayer', 'optimization-simulate']]],
    ['Infrastructure', [['Clusters', 'clusters'], ['Observability', 'cluster-monitoring-stack'], ['Storage', 'storage-management'], ['Model Cache', 'model-cache']]],
    ['My Resources', [['Services', 'model-service'], ['API Keys', 'api-keys'], ['Usage', 'usage']]],
    ['Administration', [
      ['Model Usage', 'admin/usage'], ['External Providers', 'ai-providers'],
      ['Users', 'admin/users'], ['Groups', 'admin/groups'], ['Roles', 'admin/roles'], ['Identity Providers', 'admin/identity-providers'],
      ['Master Key', 'admin/master-key'], ['Audit Log', 'admin/audit'], ['Sessions', 'admin/sessions'],
    ]],
    ['', [['Lens Assistant', 'playground']]],
  ]);
  const views = NAVIGATION_GROUPS.flatMap(group => group.items.map(item => item.view));
  assert.equal(views.length, 23);
  assert.equal(new Set(views).size, views.length);
});

test('sidebar renders the administration accordions and fixed footer controls', (t) => {
  const originalStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: () => null,
      setItem() {},
      removeItem() {},
    },
  });
  t.after(() => {
    if (originalStorage) Object.defineProperty(globalThis, 'localStorage', originalStorage);
    else delete globalThis.localStorage;
  });
  const html = renderToStaticMarkup(
    createElement(AuthContext.Provider, {
      value: {
        canView: view => canAccessView(['*:*:*'], view),
        principal: { username: 'sidebar-test' },
        logout() {},
      },
    }, createElement(LeftNavigation, { currentView: 'welcome', isExpanded: true })),
  );
  assert.match(html, /id="nav-group-administration"/);
  assert.match(html, />Platform</);
  assert.match(html, />Model Usage</);
  assert.match(html, /id="navigation-footer"/);
  assert.match(html, /Account menu/);
  assert.match(html, />Account</);
  assert.match(html, /aria-label="Lens Assistant"/);
  assert.match(html, /role="tooltip"[^>]*>Lens Assistant<\/span>/);
  assert.doesNotMatch(html.match(/<button[^>]*aria-label="Lens Assistant"[\s\S]*?<\/button>/)[0], />Lens Assistant</);
  assert.ok(html.indexOf('aria-label="Account menu"') < html.indexOf('aria-label="Lens Assistant"'));
  assert.match(html, /Collapse sidebar/);
  assert.doesNotMatch(html, /Switch to (light|dark) mode/);

  const collapsed = renderToStaticMarkup(
    createElement(AuthContext.Provider, {
      value: {
        canView: view => canAccessView(['*:*:*'], view),
        principal: { username: 'sidebar-test' },
        logout() {},
      },
    }, createElement(LeftNavigation, { currentView: 'playground', isExpanded: false })),
  );
  assert.match(collapsed, /aria-label="Account menu"/);
  assert.match(collapsed, /aria-label="Lens Assistant"/);
  assert.doesNotMatch(collapsed, />Account</);
  assert.match(collapsed, /role="tooltip"[^>]*>Lens Assistant<\/span>/);
  const activeAssistant = collapsed.match(/<button[^>]*aria-label="Lens Assistant"[\s\S]*?<\/button>/)[0];
  assert.match(activeAssistant, /aria-current="page"/);
  assert.match(activeAssistant, /bg-sky-500/);
  assert.ok(collapsed.indexOf('aria-label="Account menu"') < collapsed.indexOf('aria-label="Lens Assistant"'));
});

test('every sidebar entry can be the first accessible destination without a separate priority list', () => {
  for (const item of NAVIGATION_GROUPS.flatMap(group => group.items)) {
    assert.equal(firstAccessibleView(view => view === item.view), item.view);
    assert.equal(resolveView(item.view), item.view);
  }
});

test('default URLs use the accessible landing page but supported deep links remain intact', () => {
  const fallback = firstAccessibleView(view => canAccessView(['model-service:usage:read'], view));
  for (const view of [null, '', 'unknown']) assert.equal(resolveView(view, fallback), 'usage');
  assert.equal(resolveView('admin/model-service', fallback), 'admin/model-service');
  assert.equal(resolveView(null, ''), '');
});

test('legacy configuration URLs resolve to the shared evaluation wizard', () => {
  assert.equal(resolveView('optimization-explore'), 'optimization-evaluate-new');
  assert.equal(resolveView('optimization-evaluate-new'), 'optimization-evaluate-new');
});

test('legacy configuration URLs use the destination permissions', () => {
  const view = resolveView('optimization-explore');
  assert.equal(canAccessView(['configuration:artifact:read'], view), false);
  assert.equal(canAccessView(['deployment:run:create'], view), true);
  assert.equal(canAccessView(['evaluate:workflow:create'], view), true);
  assert.equal(canAccessView(['evaluate:run:create'], view), true);
});

test('retired optimization URLs use the accessible landing page and have no permission entries', () => {
  const fallback = firstAccessibleView(view => canAccessView(['model-service:usage:read'], view));
  for (const view of ['optimization-deploy', 'optimization-workspace', 'opt-define', 'opt-search', 'opt-deploy', 'opt-benchmark', 'opt-performance']) {
    assert.equal(resolveView(view, fallback), 'usage');
    assert.equal(resolveView(view, ''), '');
    assert.equal(Object.hasOwn(VIEW_PERMISSIONS, view), false);
  }
});

test('active detail pages and existing aliases remain available', () => {
  for (const view of ['optimization-evaluate-new', 'optimization-evaluation-details', 'optimization-simulate-details', 'welcome/issues']) {
    assert.equal(resolveView(view, 'usage'), view);
  }
  assert.equal(resolveView('home'), 'clusters');
  assert.equal(resolveView('dashboard'), 'welcome');
  assert.equal(resolveView('dashboard/alerts'), 'welcome/issues');
  assert.equal(resolveView('simulation'), 'optimization-simulate');
  assert.equal(resolveView('unknown'), 'model-market');
});
