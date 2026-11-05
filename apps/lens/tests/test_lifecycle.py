import pytest

from lens import inventory, lifecycle
from lens.catalog import find_stack
from lens.cli import main
from lens.errors import ConflictError, PolicyError
from lens.kube import Kube
from lens.plan import Options, make_plan
from lens.preflight import preflight
from lens.sources import resolve_source

from .fakes import FakeCluster
from .test_preflight import STACK, prerequisites


@pytest.fixture
def cluster():
    cluster = FakeCluster()
    prerequisites(cluster)
    return cluster


def plan_for(llmrepo, cluster, instance="r1", stack=STACK):
    planned = make_plan(
        find_stack(resolve_source(str(llmrepo)), stack),
        Options(namespace="team-a", instance=instance),
        runner=cluster,
    )
    preflight(Kube(runner=cluster), planned)
    return planned


def test_standup_records_then_applies(llmrepo, cluster):
    plan = plan_for(llmrepo, cluster)
    result = lifecycle.standup(Kube(runner=cluster), plan, actor="alice")
    assert result["applied"] == 2 and result["pruned"] == []
    record = inventory.require(Kube(runner=cluster), "team-a", "r1")
    assert record["phase"] == "Applied" and record["management"] == {"mode": "llmd"}
    assert record["stack"]["id"] == STACK and record["planDigest"] == plan.digest
    assert {(item["kind"], item["name"]) for item in record["objects"]} == {
        ("Service", "model-a"),
        ("Deployment", "model-a"),
    }
    assert all(item["uid"] for item in record["objects"])
    deployment = cluster.get("team-a", "deployments.apps", "model-a")
    assert deployment["metadata"]["labels"]["lens.llm-d.ai/instance"] == "r1"
    assert "lens.llm-d.ai/instance" not in deployment["spec"]["template"]["metadata"]["labels"]
    # The record went in before the objects.
    applies = [call for call in cluster.calls if call[0] == "apply" and "--dry-run=server" not in call]
    assert len(applies) == 3


def test_a_second_standup_updates_and_prunes(llmrepo, cluster):
    lifecycle.standup(Kube(runner=cluster), plan_for(llmrepo, cluster))
    service = llmrepo / "examples/model-a/base/model-server/kustomization.yaml"
    service.write_text(service.read_text().replace("  - service.yaml\n", ""))
    again = plan_for(llmrepo, cluster)
    assert not again.blocking and again.cluster["objects"] == {"create": 0, "update": 1, "foreign": 0}
    assert again.cluster["prune"] == ["Service/model-a"]
    assert [f.message for f in again.findings if f.code == "prune"] == [
        "standup deletes 1 object(s) of instance r1 the stack no longer renders: Service/model-a"
    ]
    result = lifecycle.standup(Kube(runner=cluster), again)
    assert result["pruned"] == ["Service/model-a"]
    assert cluster.get("team-a", "services", "model-a") is None
    record = inventory.require(Kube(runner=cluster), "team-a", "r1")
    assert [item["kind"] for item in record["objects"]] == ["Deployment"]


def test_a_pruned_object_someone_else_took_over_is_left_alone(llmrepo, cluster):
    lifecycle.standup(Kube(runner=cluster), plan_for(llmrepo, cluster))
    cluster.put("team-a", {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "model-a"}, "spec": {}})
    service = llmrepo / "examples/model-a/base/model-server/kustomization.yaml"
    service.write_text(service.read_text().replace("  - service.yaml\n", ""))
    again = plan_for(llmrepo, cluster)
    assert again.cluster["prune"] == [] and "prune" not in {f.code for f in again.findings}
    result = lifecycle.standup(Kube(runner=cluster), again)
    assert result["pruned"] == [] and cluster.get("team-a", "services", "model-a") is not None


def test_standup_refuses_a_changed_plan_and_blocking_findings(llmrepo, cluster):
    plan = plan_for(llmrepo, cluster)
    with pytest.raises(ConflictError, match="changed since the plan was reviewed"):
        lifecycle.standup(Kube(runner=cluster), plan, expected_digest="0" * 64)
    cluster.objects.pop(("team-a", "secrets", "hf-token"))
    with pytest.raises(PolicyError, match="blocking"):
        lifecycle.standup(Kube(runner=cluster), plan_for(llmrepo, cluster))
    assert inventory.read(Kube(runner=cluster), "team-a", "r1") is None


def test_one_instance_runs_one_stack(llmrepo, cluster):
    lifecycle.standup(Kube(runner=cluster), plan_for(llmrepo, cluster))
    other = plan_for(llmrepo, cluster, stack="examples/model-c/plain")
    [finding] = [finding for finding in other.blocking if finding.code == "stack-differs"]
    assert finding.message == f"instance r1 runs stack {STACK}, not examples/model-c/plain"
    with pytest.raises(PolicyError, match="blocking"):
        lifecycle.standup(Kube(runner=cluster), other)
    other.policy.findings[:] = [finding for finding in other.findings if finding.code != "stack-differs"]
    with pytest.raises(ConflictError, match="runs stack"):
        lifecycle.standup(Kube(runner=cluster), other)


def test_standup_command(llmrepo, cluster, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("lens.commands.RUNNER", cluster)
    base = ["--state-dir", str(tmp_path), "standup", STACK, "--source", str(llmrepo), "--name", "r1"]
    assert main(base) == 0
    assert "Stood up r1 in team-a (2 objects" in capsys.readouterr().out
    cluster.objects.pop(("team-a", "secrets", "hf-token"))
    assert main(base) == 3
    assert "BLOCKED" in capsys.readouterr().out


def test_standup_takes_the_digest_plan_prints(llmrepo, cluster):
    plan = plan_for(llmrepo, cluster)
    with pytest.raises(ConflictError, match="changed since the plan was reviewed"):
        lifecycle.standup(Kube(runner=cluster), plan, expected_digest=plan.digest[:8])
    result = lifecycle.standup(Kube(runner=cluster), plan, expected_digest=plan.digest[:16].upper())
    assert result["digest"] == plan.digest


def copy_of_the_stack(llmrepo, *, without_service=False):
    """A second overlay of the same base, with the same object names."""
    target = llmrepo / "examples/model-a/copy"
    target.mkdir(parents=True)
    (target / "kustomization.yaml").write_text(
        "apiVersion: kustomize.config.k8s.io/v1beta1\nkind: Kustomization\n"
        "resources:\n  - ../base\n"
        "replicas:\n  - name: model-a\n    count: 1\n"
        + (
            "patches:\n  - patch: |-\n      $patch: delete\n      apiVersion: v1\n      kind: Service\n"
            "      metadata:\n        name: model-a\n"
            if without_service
            else ""
        )
    )
    return "examples/model-a/copy"


def switch_plan(llmrepo, cluster, stack):
    planned = make_plan(
        find_stack(resolve_source(str(llmrepo)), stack),
        Options(namespace="team-a", instance="r1", switch_stack=True),
        runner=cluster,
    )
    preflight(Kube(runner=cluster), planned)
    return planned


def test_switch_stack_moves_an_instance_in_place(llmrepo, cluster):
    lifecycle.standup(Kube(runner=cluster), plan_for(llmrepo, cluster))
    before = cluster.get("team-a", "deployments.apps", "model-a")
    uid, version = before["metadata"]["uid"], before["metadata"]["resourceVersion"]
    other = copy_of_the_stack(llmrepo, without_service=True)
    planned = switch_plan(llmrepo, cluster, other)
    assert not planned.blocking
    assert planned.cluster["switchStack"] == {"from": STACK, "to": other}
    assert planned.cluster["prune"] == ["Service/model-a"]
    assert {f.code for f in planned.findings} >= {"stack-switched", "prune"}
    result = lifecycle.standup(Kube(runner=cluster), planned)
    assert result["switchedFrom"] == STACK and result["pruned"] == ["Service/model-a"]
    after = cluster.get("team-a", "deployments.apps", "model-a")
    assert after["metadata"]["uid"] == uid and after["spec"] == before["spec"]
    assert int(after["metadata"]["resourceVersion"]) > int(version)
    record = inventory.require(Kube(runner=cluster), "team-a", "r1")
    assert record["stack"]["id"] == other and record["previousStack"]["id"] == STACK
    # A later standup of the new stack needs no flag and keeps where the instance came from.
    again = plan_for(llmrepo, cluster, stack=other)
    assert not again.blocking and "switchStack" not in again.cluster
    lifecycle.standup(Kube(runner=cluster), again)
    assert inventory.require(Kube(runner=cluster), "team-a", "r1")["previousStack"]["id"] == STACK
    # And the old path is now the other one.
    back = plan_for(llmrepo, cluster)
    assert [f.code for f in back.blocking] == ["stack-differs"]


def test_switch_stack_is_refused_during_a_rollout_and_for_an_imported_instance(llmrepo, cluster):
    lifecycle.standup(Kube(runner=cluster), plan_for(llmrepo, cluster))
    other = copy_of_the_stack(llmrepo)
    record = inventory.require(Kube(runner=cluster), "team-a", "r1")
    inventory.write(Kube(runner=cluster), {**record, "rollout": {"phase": "Rolling"}})
    with pytest.raises(ConflictError, match="rollout in progress"):
        lifecycle.standup(Kube(runner=cluster), switch_plan(llmrepo, cluster, other))
    inventory.write(Kube(runner=cluster), {**record, "management": {"mode": "imported", "manager": "kubectl"}})
    planned = switch_plan(llmrepo, cluster, other)
    assert "instance-imported" in {f.code for f in planned.blocking}
    with pytest.raises(PolicyError):
        lifecycle.standup(Kube(runner=cluster), planned)


def test_switch_stack_command(llmrepo, cluster, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("lens.commands.RUNNER", cluster)
    state = ["--state-dir", str(tmp_path)]
    assert main([*state, "standup", STACK, "--source", str(llmrepo), "--name", "r1"]) == 0
    other = copy_of_the_stack(llmrepo)
    capsys.readouterr()
    assert main([*state, "plan", other, "--source", str(llmrepo), "--name", "r1"]) == 3
    assert "Pass --switch-stack to move it to this stack in place" in capsys.readouterr().out
    assert main([*state, "standup", other, "--source", str(llmrepo), "--name", "r1", "--switch-stack"]) == 0
    out = capsys.readouterr().out
    assert f"  moved from stack {STACK}" in out and "pruned" not in out
    assert main([*state, "status", "r1"]) == 0
    assert f"  stack {other} (moved from {STACK})" in capsys.readouterr().out
