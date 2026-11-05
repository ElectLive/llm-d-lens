import json

from lens.catalog import find_stack
from lens.cli import main
from lens.plan import Options, make_plan
from lens.process import subprocess_runner
from lens.sources import resolve_source

STACK = "examples/model-a/cluster-a"


def test_a_plan_digest_follows_what_would_be_applied(llmrepo):
    stack = find_stack(resolve_source(str(llmrepo)), STACK)
    first = make_plan(stack, Options(namespace="team-a", instance="r1"), runner=subprocess_runner)
    again = make_plan(stack, Options(namespace="team-a", instance="r1"), runner=subprocess_runner)
    elsewhere = make_plan(stack, Options(namespace="team-b", instance="r1"), runner=subprocess_runner)
    assert first.digest == again.digest != elsewhere.digest
    assert first.objects() == [
        {"apiVersion": "v1", "kind": "Service", "name": "model-a"},
        {"apiVersion": "apps/v1", "kind": "Deployment", "name": "model-a"},
    ]


def test_offline_plan_output_and_exit_codes(llmrepo, tmp_path, capsys):
    base = ["--state-dir", str(tmp_path), "plan", STACK, "--source", str(llmrepo), "--offline"]
    assert main([*base, "-n", "team-a"]) == 0
    out = capsys.readouterr().out
    assert "Target team-a/model-a-cluster-a" in out and "Result ready to stand up" in out
    assert "Secret/hf-token" in out and "PersistentVolumeClaim/model-a-cache" in out
    assert "nvidia.com/gpu x1/pod" in out
    assert main([*base]) == 2
    assert "no target namespace" in capsys.readouterr().err
    assert main([*base, "-n", "Bad_NS"]) == 2


def test_a_plan_writes_the_manifest_and_the_administrator_parts(llmrepo, tmp_path, capsys):
    overlay = llmrepo / STACK
    (overlay / "priority.yaml").write_text(
        "apiVersion: scheduling.k8s.io/v1\nkind: PriorityClass\nmetadata: {name: serving}\nvalue: 1000\n"
    )
    kustomization = overlay / "kustomization.yaml"
    kustomization.write_text(kustomization.read_text().replace("- ../base", "- ../base\n  - priority.yaml"))
    manifest, admin = tmp_path / "out.yaml", tmp_path / "admin.yaml"
    argv = [
        "--state-dir",
        str(tmp_path),
        "-o",
        "json",
        "plan",
        STACK,
        "--source",
        str(llmrepo),
        "-n",
        "team-a",
    ]
    assert main([*argv, "--offline", "--manifest", str(manifest), "--admin-yaml", str(admin)]) == 3
    summary = json.loads(capsys.readouterr().out)
    assert summary["adminParts"] == [{"object": "PriorityClass/serving", "reason": "cluster-scoped"}]
    assert "kind: PriorityClass" in admin.read_text() and "PriorityClass" not in manifest.read_text()
    assert main([*argv, "--offline", "--skip-admin-parts"]) == 0
