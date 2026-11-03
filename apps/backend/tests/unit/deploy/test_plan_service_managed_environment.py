"""Deployment-run creation rejects user-configured Hugging Face environment."""

import pytest
import yaml

from llm_d_lens.deploy import plan_service as plan_service_module
from llm_d_lens.deploy.contracts import (
    DeployableConfiguration,
    DeploymentRunCreateRequest,
    RuntimeBinding,
)
from llm_d_lens.deploy.plan_service import DeploymentPlanService
from llm_d_lens.infrastructure.artifacts.manifests import configuration_checksum


class _Store:
    def create_run(self, run):
        return run


class _Planner:
    def plan(self, configurations, *, run_id, provenance):
        return []


def _configuration(content):
    return DeployableConfiguration(
        type="baseline",
        format="helm",
        provider_ref="baseline-vllm",
        checksum="checksum",
        content=content,
    )


def _service():
    return DeploymentPlanService(_Store(), _Planner())


def test_create_run_rejects_admin_managed_custom_environment():
    request = DeploymentRunCreateRequest(
        configurations=[
            _configuration(
                {"customParameters": [{"target": "both", "kind": "environment", "name": "HF_HOME", "value": "/models"}]}
            )
        ]
    )
    with pytest.raises(ValueError, match="platform administrator"):
        _service().create_run(request)


def test_create_run_rejects_admin_managed_runtime_binding_environment():
    request = DeploymentRunCreateRequest(
        configurations=[_configuration({"runtime": {"image": "img:tag"}})],
        runtime_binding=RuntimeBinding(environment={"HF_TOKEN": "secret"}),
    )
    with pytest.raises(ValueError, match="platform administrator"):
        _service().create_run(request)


def test_create_run_allows_other_custom_environment():
    request = DeploymentRunCreateRequest(
        configurations=[
            _configuration(
                {
                    "customParameters": [
                        {"target": "both", "kind": "environment", "name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"}
                    ]
                }
            )
        ]
    )
    assert _service().create_run(request) is not None


@pytest.mark.parametrize(
    "override,expected",
    [
        (None, {"mode": "named", "name": "saved"}),
        ({"mode": "auto"}, {"mode": "auto"}),
        ({"mode": "node-default"}, {"mode": "node-default"}),
        ({"mode": "named", "name": "override"}, {"mode": "named", "name": "override"}),
    ],
)
def test_runtime_policy_is_preserved_per_case_and_request_override_wins(override, expected):
    configuration = _configuration({"model": {"name": "org/model"}, "runtime": {"image": "img:tag"}})
    configuration.checksum = configuration_checksum(configuration.content)
    configuration.provenance["runtime_class"] = {"mode": "named", "name": "saved"}
    request = DeploymentRunCreateRequest(
        configurations=[configuration],
        runtime_binding=RuntimeBinding(runtime_class=override),
    )
    run = DeploymentPlanService(_Store()).create_run(request)
    assert run.cases[0].create_request.deployment_policy.value["runtime_class"] == expected
    assert configuration.provenance["runtime_class"] == {"mode": "named", "name": "saved"}
    assert run.source_configurations[0].checksum == configuration.checksum


def test_runtime_binding_patches_disaggregatedset_model_server_environment(monkeypatch):
    document = {
        "apiVersion": "disaggregatedset.x-k8s.io/v1",
        "kind": "DisaggregatedSet",
        "metadata": {"name": "pd-disagg"},
        "spec": {
            "roles": [
                {
                    "name": "prefill",
                    "spec": {
                        "leaderWorkerTemplate": {
                            "workerTemplate": {
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "modelserver",
                                            "env": [{"name": "HTTP_PROXY", "value": "http://old:1"}],
                                        }
                                    ]
                                }
                            }
                        }
                    },
                }
            ]
        },
    }
    manifest = yaml.safe_dump(document, sort_keys=False)
    configuration = _configuration(
        {
            "runtime": {"image": "img:tag"},
            "officialGuide": {"renderedManifest": manifest, "manifestChecksum": "old"},
        }
    )
    configuration.checksum = configuration_checksum(configuration.content)
    monkeypatch.setattr(
        plan_service_module,
        "validate_configuration_manifest_content",
        lambda *_args, **_kwargs: (manifest, "source", None, None),
    )
    request = DeploymentRunCreateRequest(
        configurations=[configuration],
        runtime_binding=RuntimeBinding(environment={"HTTP_PROXY": "http://new:2"}),
    )

    run = DeploymentPlanService(_Store(), _Planner()).create_run(request)

    bound = yaml.safe_load(run.source_configurations[0].content["officialGuide"]["renderedManifest"])
    environment = bound["spec"]["roles"][0]["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0][
        "env"
    ]
    assert environment == [{"name": "HTTP_PROXY", "value": "http://new:2"}]
