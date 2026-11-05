import pytest

from lens.catalog import default_name, find_stack, list_stacks
from lens.cli import main
from lens.errors import NotFoundError, UsageError
from lens.process import subprocess_runner
from lens.sources import resolve_source


def test_the_guides_are_the_stacks(llmrepo):
    source = resolve_source(str(llmrepo))
    assert [stack.id for stack in list_stacks(source, subprocess_runner)] == [
        "guides/demo-baseline",
        "guides/demo-pd",
    ]
    assert {stack.kind for stack in list_stacks(source, subprocess_runner)} == {"guide"}


def test_listing_by_prefix(llmrepo):
    source = resolve_source(str(llmrepo))
    assert list_stacks(source, subprocess_runner, "examples") == []
    assert [stack.id for stack in list_stacks(source, subprocess_runner, "guides/demo-pd/")] == ["guides/demo-pd"]


def test_a_stack_is_a_guide_or_any_path_with_manifests(llmrepo):
    source = resolve_source(str(llmrepo))
    assert find_stack(source, "guides/demo-baseline").kind == "guide"
    assert find_stack(source, "demo-baseline").id == "guides/demo-baseline"  # a guide by its name
    overlay = find_stack(source, "guides/demo-baseline/modelserver/gpu/vllm/base")
    assert overlay.kind == "kustomize"
    assert find_stack(source, "examples/model-a/cluster-a").kind == "kustomize"
    plain = find_stack(source, "examples/model-c/plain/")
    assert (plain.id, plain.kind) == ("examples/model-c/plain", "manifests")
    single = find_stack(source, "examples/model-c/plain/deployment.yaml")
    assert single.kind == "manifests"
    with pytest.raises(NotFoundError):
        find_stack(source, "guides/missing")
    with pytest.raises(UsageError, match="escapes"):
        find_stack(source, "../outside")
    with pytest.raises(UsageError, match="no Kubernetes manifests"):
        find_stack(source, "charts/router/Chart.yaml")


def test_default_names_are_short_dns_labels(llmrepo):
    source = resolve_source(str(llmrepo))
    assert default_name(find_stack(source, "guides/demo-baseline")) == "demo-baseline"
    assert default_name(find_stack(source, "examples/model-a/cluster-a")) == "model-a-cluster-a"
    long = find_stack(source, "examples/model-c/plain")
    long = type(long)(id="examples/" + "x" * 60 + "/cluster.v1.2", kind="kustomize", source=source)
    name = default_name(long)
    assert len(name) <= 40 and name.endswith(name[-8:]) and name.islower()


def test_stacks_command(llmrepo, tmp_path, capsys):
    assert main(["--state-dir", str(tmp_path / "s"), "stacks", "--source", str(llmrepo)]) == 0
    out = capsys.readouterr().out
    assert "2 guide(s)" in out and "guides/demo-baseline" in out and "guides/demo-pd" in out
