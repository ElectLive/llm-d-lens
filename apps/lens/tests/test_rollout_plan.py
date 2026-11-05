import json

import pytest

from lens.catalog import find_stack
from lens.cli import main
from lens.importing import import_stack
from lens.kube import Kube
from lens.plan import Options, make_plan
from lens.preflight import preflight
from lens.render import render
from lens.rollouts.plan import plan_rollout
from lens.sources import resolve_source

from .conftest import committed_copy, git
from .fakes import FakeCluster

STACK = "guides/demo-pd"
NS = "llm-d-demo-pd"
POOL = "llm-d.ai/in-pool"
DS = "disaggregatedsets.disaggregatedset.x-k8s.io"


@pytest.fixture
def rollrepo(tmp_path):
    return committed_copy(tmp_path / "rollrepo")


@pytest.fixture
def cluster(rollrepo):
    """The guide deployed with its own commands and imported, every group serving."""
    cluster = FakeCluster(namespaces=(NS,))
    for document in rendered(rollrepo, cluster):
        cluster.put(NS, document)
    cluster.settle()
    import_stack(Kube(runner=cluster), stack_plan(rollrepo, cluster, existing="import"), manager="kubectl")
    return cluster


def rendered(repo, cluster):
    return render(find_stack(resolve_source(str(repo)), STACK), runner=cluster, namespace=NS).documents


def stack_plan(repo, cluster, existing="refuse"):
    stack = find_stack(resolve_source(str(repo)), STACK)
    plan = make_plan(stack, Options(namespace=NS, instance="pd"), runner=cluster, label=False)
    preflight(Kube(runner=cluster), plan, existing=existing)
    return plan


def new_image(repo, image="v2"):
    path = repo / "guides/demo-pd/modelserver/gpu/vllm/base/disaggregatedset.yaml"
    path.write_text(path.read_text().replace("vllm:v1", f"vllm:{image}"))
    git(repo, "commit", "-q", "-am", image)


def rollout(repo, cluster, **options):
    return plan_rollout(
        Kube(runner=cluster), stack_plan(repo, cluster, "import"), floors={}, pool_label=POOL, **options
    )


def test_an_image_change_plans_an_in_place_roll(rollrepo, cluster):
    new_image(rollrepo)
    planned = rollout(rollrepo, cluster)
    summary = planned.summary()
    assert not planned.blocking
    assert summary["objects"] == {"DisaggregatedSet/model-p": "changed"} and summary["unchanged"] == 1
    [item] = summary["sets"]
    assert item["change"] == "roll" and item["steps"] == 3
    assert item["servingDuringAStep"] == {"prefill": 2, "decode": 2}
    assert any("image" in path for path in item["differences"])
    assert summary["revision"] == planned.before.target


def test_a_replicas_change_is_a_scale_not_a_roll(rollrepo, cluster):
    path = rollrepo / "guides/demo-pd/modelserver/gpu/vllm/base/disaggregatedset.yaml"
    path.write_text(path.read_text().replace("replicas: 3", "replicas: 4"))
    [item] = rollout(rollrepo, cluster).summary()["sets"]
    assert item["change"] == "scale" and item["steps"] == 0


def test_nothing_to_do(rollrepo, cluster):
    planned = rollout(rollrepo, cluster)
    assert not planned.blocking and [f.code for f in planned.findings] == ["nothing-to-do"]


def test_a_floor_a_step_would_break_blocks(rollrepo, cluster):
    new_image(rollrepo)
    planned = plan_rollout(
        Kube(runner=cluster), stack_plan(rollrepo, cluster, "import"), floors={"prefill": 3}, pool_label=POOL
    )
    assert [f.code for f in planned.blocking] == ["below-floor"]


def test_a_step_that_leaves_a_role_with_no_group_serving_blocks_unless_accepted(rollrepo, cluster):
    """A role of one group with maxSurge 0 serves nothing while its group is replaced."""
    path = rollrepo / "guides/demo-pd/modelserver/gpu/vllm/base/disaggregatedset.yaml"
    path.write_text(path.read_text().replace("replicas: 3", "replicas: 1"))
    git(rollrepo, "commit", "-q", "-am", "one decode group")
    documents = rendered(rollrepo, cluster)
    cluster.put(NS, next(doc for doc in documents if doc["kind"] == "DisaggregatedSet"))  # what runs
    cluster.settle()
    new_image(rollrepo)
    planned = rollout(rollrepo, cluster)
    assert planned.change.roll.serving_during() == {"prefill": 2, "decode": 0}
    assert [f.code for f in planned.blocking] == ["no-group-serving"]
    accepted = rollout(rollrepo, cluster, allow_outage=True)
    assert not accepted.blocking and "no-group-serving" in [f.code for f in accepted.findings]


def test_a_roll_in_progress_or_a_group_out_of_the_pool_blocks(rollrepo, cluster):
    pod = next(item for key, item in cluster.objects.items() if key[1] == "pods" and "-decode-" in key[2])
    pod["metadata"]["labels"][POOL] = "false"
    new_image(rollrepo)
    assert [f.code for f in rollout(rollrepo, cluster).blocking] == ["not-settled"]
    pod["metadata"]["labels"][POOL] = "true"
    live = cluster.get(NS, DS, "model-p")
    live["spec"]["roles"][0]["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]["image"] = "x"
    cluster.roll_step(NS, "model-p")
    assert "roll-in-progress" in [f.code for f in rollout(rollrepo, cluster).blocking]


def test_rollouts_plan_command_from_a_pull_request_ref(rollrepo, cluster, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("lens.commands.RUNNER", cluster)
    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(bare))
    git(rollrepo, "push", "-q", str(bare), "main")
    new_image(rollrepo)
    git(rollrepo, "push", "-q", str(bare), "HEAD:refs/pull/9/head")
    argv = ["--state-dir", str(tmp_path), "rollouts", "plan", STACK, "-n", NS, "--name", "pd"]
    source = ["--source", f"file://{bare}#refs/pull/9/head"]
    assert main([*argv, *source]) == 0
    out = capsys.readouterr().out
    assert "model-p: roll" in out and "3 step(s)" in out and "Result ready to start" in out
    assert main(["-o", "json", *argv, *source, "--floor", "decode=3"]) == 3
    assert json.loads(capsys.readouterr().out)["blocking"] == 1
