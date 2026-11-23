"""The live workflow must retry the Pages deploy once when the first attempt fails (issue #4)."""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "live.yml"


def _job() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["publish"]


def test_first_deploy_may_fail_and_is_retried_after_a_wait():
    steps = _job()["steps"]
    deploys = [i for i, s in enumerate(steps) if str(s.get("uses", "")).startswith("actions/deploy-pages@")]
    assert len(deploys) == 2
    first, second = steps[deploys[0]], steps[deploys[1]]

    assert first.get("id") == "deploy"
    assert first.get("continue-on-error") is True
    assert "if" not in first

    assert second.get("id") == "deploy_retry"
    assert second.get("if") == "steps.deploy.outcome == 'failure'"
    assert "continue-on-error" not in second

    waits = steps[deploys[0] + 1:deploys[1]]
    assert len(waits) == 1
    assert waits[0].get("if") == "steps.deploy.outcome == 'failure'"
    assert "sleep" in waits[0].get("run", "")


def test_environment_url_falls_back_to_the_retry():
    url = _job()["environment"]["url"]
    assert "steps.deploy.outputs.page_url" in url
    assert "steps.deploy_retry.outputs.page_url" in url
