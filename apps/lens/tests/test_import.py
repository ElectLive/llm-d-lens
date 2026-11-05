import copy
import json

import pytest

from lens import inventory, lifecycle
from lens.catalog import find_stack
from lens.cli import main
from lens.errors import ConflictError, NotFoundError, PolicyError
from lens.importing import compare, drift, import_stack
from lens.kube import Kube
from lens.plan import Options, make_plan
from lens.preflight import preflight
from lens.render import render
from lens.sources import resolve_source
from lens.status import instance_status

from .fakes import FakeCluster
from .test_preflight import STACK


def deployed_with_kubectl_apply(llmrepo, cluster, namespace="model-a-serving"):
    """Apply the stack the way a team does: `kubectl apply -k` of the overlay, no llmd labels."""
    cluster.namespaces.add(namespace)
    for document in render(find_stack(resolve_source(str(llmrepo)), STACK), runner=cluster).documents:
        live = copy.deepcopy(document)
        live["metadata"].setdefault("annotations", {})["kubectl.kubernetes.io/last-applied-configuration"] = "{}"
        live.setdefault("spec", {}).setdefault("revisionHistoryLimit", 10)  # a server default
        cluster.put(namespace, live)
    cluster.settle()
    for name in ("hf-token",):
        cluster.put(namespace, {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": name}})
    cluster.put(namespace, {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": "model-a"}})
    cluster.put(
        namespace,
        {"apiVersion": "v1", "kind": "PersistentVolumeClaim", "metadata": {"name": "model-a-cache"}},
    )


def import_plan(llmrepo, cluster, namespace="model-a-serving"):
    stack = find_stack(resolve_source(str(llmrepo)), STACK)
    plan = make_plan(stack, Options(namespace=namespace, instance="team-model"), runner=cluster, label=False)
    preflight(Kube(runner=cluster), plan, existing="import")
    return plan


@pytest.fixture
def cluster():
    return FakeCluster(namespaces=())


def test_import_records_running_objects_and_changes_nothing_else(llmrepo, cluster):
    deployed_with_kubectl_apply(llmrepo, cluster)
    before = copy.deepcopy(cluster.objects)
    plan = import_plan(llmrepo, cluster)
    assert not plan.blocking and plan.cluster["objects"]["foreign"] == 2
    result = import_stack(Kube(runner=cluster), plan, manager="kubectl")
    assert result == {"instance": "team-model", "namespace": "model-a-serving", "imported": 2, "manager": "kubectl"}
    changed = {key for key in cluster.objects if cluster.objects[key] != before.get(key)}
    assert changed == {("model-a-serving", "configmaps", "lens-instance-team-model")}
    record = inventory.require(Kube(runner=cluster), "model-a-serving", "team-model")
    assert record["management"] == {"mode": "imported", "manager": "kubectl"}
    assert all(item["uid"] for item in record["objects"])


def test_an_imported_stack_reports_its_health(llmrepo, cluster):
    deployed_with_kubectl_apply(llmrepo, cluster)
    import_stack(Kube(runner=cluster), import_plan(llmrepo, cluster), manager="kubectl")
    cluster.get("model-a-serving", "deployments.apps", "model-a")["status"] = {
        "replicas": 1,
        "readyReplicas": 1,
        "updatedReplicas": 1,
    }
    status = instance_status(Kube(runner=cluster), "model-a-serving", "team-model")
    assert status["health"] == "Ready" and status["pods"]["ready"] == status["pods"]["total"] == 1
    assert status["management"]["mode"] == "imported"


def test_privileged_findings_do_not_block_an_import(llmrepo, cluster):
    deployment = llmrepo / "examples/model-a/base/model-server/deployment.yaml"
    deployment.write_text(
        deployment.read_text().replace(
            "        - name: vllm\n", "        - name: vllm\n          securityContext: {privileged: true}\n"
        )
    )
    deployed_with_kubectl_apply(llmrepo, cluster)
    plan = import_plan(llmrepo, cluster)
    assert not plan.blocking
    assert [f.severity for f in plan.findings if f.code == "privileged"] == ["info"]


def test_an_import_needs_every_object_to_run(llmrepo, cluster):
    deployed_with_kubectl_apply(llmrepo, cluster)
    cluster.objects.pop(("model-a-serving", "services", "model-a"))
    plan = import_plan(llmrepo, cluster)
    with pytest.raises(NotFoundError, match="1 object"):
        import_stack(Kube(runner=cluster), plan, manager="kubectl")
    assert compare(Kube(runner=cluster), plan)[0] == {
        "object": "Service/model-a",
        "state": "missing",
        "differences": [],
    }


def test_drift_ignores_server_defaults_and_finds_real_changes(llmrepo, cluster):
    deployed_with_kubectl_apply(llmrepo, cluster)
    plan = import_plan(llmrepo, cluster)
    assert {item["state"] for item in compare(Kube(runner=cluster), plan)} == {"matches"}
    cluster.get("model-a-serving", "deployments.apps", "model-a")["spec"]["replicas"] = 4
    report = {item["object"]: item for item in compare(Kube(runner=cluster), plan)}
    assert report["Deployment/model-a"]["differences"] == ["spec.replicas"]
    assert drift({"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}], "c": 1}) == ["a[1].b"]


def test_the_pull_secret_openshift_adds_to_a_service_account_is_not_drift(llmrepo, cluster):
    deployed_with_kubectl_apply(llmrepo, cluster)
    plan = import_plan(llmrepo, cluster)
    account = {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": "model-a"}}
    plan.documents.append({**account, "imagePullSecrets": [{"name": "registry-pull"}]})
    # OpenShift adds `<account>-dockercfg-<suffix>` to every ServiceAccount; a stack cannot name it.
    live = {**account, "imagePullSecrets": [{"name": "registry-pull"}, {"name": "model-a-dockercfg-4qjxr"}]}
    cluster.put("model-a-serving", live)
    report = {item["object"]: item for item in compare(Kube(runner=cluster), plan)}
    assert report["ServiceAccount/model-a"] == {
        "object": "ServiceAccount/model-a",
        "state": "matches",
        "differences": [],
    }
    live["imagePullSecrets"] = [{"name": "model-a-dockercfg-4qjxr"}]
    cluster.put("model-a-serving", live)
    report = {item["object"]: item for item in compare(Kube(runner=cluster), plan)}
    assert report["ServiceAccount/model-a"]["differences"] == ["imagePullSecrets"]


def test_an_imported_instance_is_never_stood_up_or_imported_twice(llmrepo, cluster):
    deployed_with_kubectl_apply(llmrepo, cluster)
    kube = Kube(runner=cluster)
    import_stack(kube, import_plan(llmrepo, cluster), manager="helm")
    with pytest.raises(ConflictError, match="already recorded"):
        import_stack(kube, import_plan(llmrepo, cluster), manager="helm")
    stack = find_stack(resolve_source(str(llmrepo)), STACK)
    standup_plan = make_plan(stack, Options(namespace="model-a-serving", instance="team-model"), runner=cluster)
    preflight(kube, standup_plan)
    assert {f.code for f in standup_plan.blocking} == {"instance-imported"}
    with pytest.raises(PolicyError):
        lifecycle.standup(kube, standup_plan)


def test_teardown_of_an_imported_stack(llmrepo, cluster):
    deployed_with_kubectl_apply(llmrepo, cluster)
    kube = Kube(runner=cluster)
    import_stack(kube, import_plan(llmrepo, cluster), manager="kubectl")
    assert lifecycle.teardown(kube, "model-a-serving", "team-model", forget=True)["forgot"]
    assert cluster.get("model-a-serving", "deployments.apps", "model-a") is not None
    import_stack(kube, import_plan(llmrepo, cluster), manager="kubectl")
    result = lifecycle.teardown(kube, "model-a-serving", "team-model")
    assert sorted(result["deleted"]) == ["Deployment/model-a", "Service/model-a"]


def test_import_command(llmrepo, cluster, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("lens.commands.RUNNER", cluster)
    deployed_with_kubectl_apply(llmrepo, cluster)
    base = ["--state-dir", str(tmp_path), "import", STACK, "--source", str(llmrepo), "-n", "model-a-serving"]
    assert main([*base, "--name", "team-model", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "matches   Deployment/model-a" in out
    assert inventory.read(Kube(runner=cluster), "model-a-serving", "team-model") is None
    assert main(["-o", "json", *base, "--name", "team-model"]) == 0
    assert json.loads(capsys.readouterr().out)["imported"] == 2
