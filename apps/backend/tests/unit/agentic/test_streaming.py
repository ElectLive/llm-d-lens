"""Route-level contracts for shared planning SSE delivery."""

import asyncio
from types import SimpleNamespace

import pytest

from llm_d_lens.agentic import router
from llm_d_lens.agentic.facts import PlanningEvidence
from llm_d_lens.auth.contracts import Principal, RoleBinding
from llm_d_lens.core.exceptions import ForbiddenError


async def response_for(monkeypatch, mode, work):
    monkeypatch.setattr(router, "service", SimpleNamespace(create=work, refine=work, get=lambda _id: None))
    if mode == "create":
        return await router.stream_agentic_deployment(SimpleNamespace(), None)
    return await router.stream_refine_agentic_deployment("run-1", SimpleNamespace(), None)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["create", "refine"])
async def test_progress_precedes_one_complete_event(monkeypatch, mode):
    async def work(*args, on_progress, principal_id, principal):
        assert principal_id is None
        assert principal is None
        await on_progress({"phase": "planning"})
        return SimpleNamespace(model_dump=lambda **kwargs: {"id": "run-1"})

    response = await response_for(monkeypatch, mode, work)
    assert response.media_type == "text/event-stream"
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["X-Accel-Buffering"] == "no"
    assert [chunk async for chunk in response.body_iterator] == [
        'event: progress\ndata: {"phase": "planning"}\n\n',
        'event: complete\ndata: {"id": "run-1"}\n\n',
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,error,expected",
    [
        ("create", ValueError("invalid"), "invalid"),
        ("refine", ValueError("invalid"), "invalid"),
        ("refine", KeyError("missing"), "'missing'"),
        ("create", KeyError("missing"), "Unable to generate an Agentic recommendation."),
        ("refine", RuntimeError("internal detail"), "Unable to recalculate the Agentic recommendation."),
    ],
)
async def test_error_contracts_remain_route_specific(monkeypatch, mode, error, expected):
    async def work(*args, **kwargs):
        raise error

    response = await response_for(monkeypatch, mode, work)
    chunks = [chunk async for chunk in response.body_iterator]
    assert len(chunks) == 1
    assert chunks[0].startswith("event: error\n")
    assert expected in chunks[0]
    assert "internal detail" not in chunks[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["create", "refine"])
async def test_closing_stream_cancels_and_awaits_planning_work(monkeypatch, mode):
    stopped = asyncio.Event()

    async def work(*args, on_progress, **kwargs):
        try:
            await on_progress({"phase": "planning"})
            await asyncio.Event().wait()
        finally:
            stopped.set()

    response = await response_for(monkeypatch, mode, work)
    await anext(response.body_iterator)
    await response.body_iterator.aclose()
    assert stopped.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["create", "refine"])
async def test_stream_preserves_the_actual_principal_for_public_rag(monkeypatch, mode):
    principal = Principal("user-1", "admin", bindings=(RoleBinding("admin"),))

    async def work(*args, on_progress, principal_id, principal):
        assert principal_id == "user-1"
        assert principal.username == "admin"
        return SimpleNamespace(model_dump=lambda **kwargs: {"id": "run-1"})

    monkeypatch.setattr(router, "service", SimpleNamespace(create=work, refine=work, get=lambda _id: None))
    request = SimpleNamespace(state=SimpleNamespace(principal=principal))
    if mode == "create":
        response = await router.stream_agentic_deployment(SimpleNamespace(), request)
    else:
        response = await router.stream_refine_agentic_deployment("run-1", SimpleNamespace(), request)
    assert "event: complete" in "".join([chunk async for chunk in response.body_iterator])


@pytest.mark.asyncio
async def test_public_rag_plans_are_not_an_alternate_unauthorized_corpus_read(monkeypatch):
    enriched = SimpleNamespace(
        id="enriched",
        planning_evidence=[PlanningEvidence(id="public-rag:query", source="public-benchmark-rag", status="available")],
    )
    ordinary = SimpleNamespace(id="ordinary", planning_evidence=[])
    monkeypatch.setattr(router, "service", SimpleNamespace(list=lambda: [enriched, ordinary], get=lambda _id: enriched))
    reader = SimpleNamespace(state=SimpleNamespace(principal=Principal("reader", "reader")))
    assert await router.list_agentic_deployments(reader) == [ordinary]
    with pytest.raises(ForbiddenError):
        await router.get_agentic_deployment("enriched", reader)
    admin = SimpleNamespace(
        state=SimpleNamespace(principal=Principal("admin", "admin", bindings=(RoleBinding("admin"),)))
    )
    assert await router.get_agentic_deployment("enriched", admin) is enriched
    assert await router.list_agentic_deployments(admin) == [enriched, ordinary]


@pytest.mark.asyncio
async def test_internal_case_plans_use_case_authorization_without_public_corpus_permission(monkeypatch):
    case = {"origins": ["agentic:evaluate"], "experiment_ids": ["benchmark-1"]}
    run = SimpleNamespace(
        id="internal",
        planning_evidence=[
            PlanningEvidence(
                id="configuration-cases:query",
                source="configuration-cases",
                status="available",
                details={"cases": [case]},
            )
        ],
    )
    principal = Principal("reader", "reader")
    request = SimpleNamespace(state=SimpleNamespace(principal=principal))
    checked = []

    def authorize_cases(cases, actual_principal):
        assert actual_principal is principal
        checked.append(cases)

    monkeypatch.setattr(router, "assert_case_access", authorize_cases)
    monkeypatch.setattr(router, "service", SimpleNamespace(list=lambda: [run], get=lambda _id: run))
    assert await router.list_agentic_deployments(request) == [run]
    assert await router.get_agentic_deployment("internal", request) is run
    assert checked == [(case,), (case,)]
