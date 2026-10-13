import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_evals.api import app
from agent_evals.cli import main

SCENARIOS_DIR = str(Path(__file__).parent.parent / "scenarios" / "recruiter")
ARGS = ["run", "--scenarios", SCENARIOS_DIR]


def test_cli_exits_zero_when_the_suite_passes(capsys):
    assert main(ARGS) == 0
    assert "pass rate: 100%" in capsys.readouterr().out


def test_cli_exits_one_below_the_threshold_and_names_the_failure(capsys):
    assert main([*ARGS, "--agent", "faulty:best_available"]) == 1
    out = capsys.readouterr().out
    assert "FAIL    0/1  nobody-qualifies" in out
    assert "state_equals: interviews is ['c2'], expected []" in out


def test_cli_threshold_is_configurable():
    # best_available fails 1 of 6 scenarios: 83%.
    assert main([*ARGS, "--agent", "faulty:best_available", "--min-pass-rate", "0.8"]) == 0
    assert main([*ARGS, "--agent", "faulty:best_available", "--min-pass-rate", "0.9"]) == 1


def test_cli_writes_json_with_transcripts_and_a_markdown_summary(tmp_path):
    out_json, out_md = tmp_path / "report.json", tmp_path / "report.md"
    main([*ARGS, "--agent", "regressed", "--json", str(out_json), "--markdown", str(out_md)])
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["pass_rate"] == 0
    first_trial = report["scenarios"][0]["trials"][0]
    assert first_trial["transcript"]["calls"][0]["name"] == "get_job"
    assert "| `unknown-candidate` | 0/1 |" in out_md.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--scenarios", "does/not/exist"],
        [*ARGS, "--agent", "nope"],
    ],
)
def test_cli_usage_errors_exit_two(argv, capsys):
    assert main(argv) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_lists_scenarios(capsys):
    assert main(["list", "--scenarios", SCENARIOS_DIR]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 6


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AGENT_EVALS_SCENARIOS", SCENARIOS_DIR)
    return TestClient(app)


def test_api_health_and_scenarios(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert len(client.get("/scenarios").json()) == 6


def test_api_runs_a_suite(client):
    body = client.post("/runs", json={"agent": "faulty:obeys_resume", "trials": 2}).json()
    assert body["agent"] == "faulty[obeys_resume]"
    rates = {s["scenario_id"]: s["pass_rate"] for s in body["scenarios"]}
    assert rates["resume-injection"] == 0
    assert rates["qualified-only"] == 1


def test_api_can_run_a_subset(client):
    body = client.post("/runs", json={"scenario_ids": ["slot-conflict"]}).json()
    assert [s["scenario_id"] for s in body["scenarios"]] == ["slot-conflict"]


def test_api_rejects_unknown_agents_and_scenarios(client):
    assert client.post("/runs", json={"agent": "nope"}).status_code == 422
    assert client.post("/runs", json={"scenario_ids": ["nope"]}).status_code == 404
    assert client.post("/runs", json={"trials": 0}).status_code == 422
