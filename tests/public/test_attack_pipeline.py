"""CP4 credentials, live-outcome recording and safe error artifacts (no network)."""
import asyncio
import json

import pytest

import main as main_module
from attacks import attacks
from core import config


def test_red_only_setup_does_not_request_blue_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("RED_TEAM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-dummy")

    def unexpected_input(prompt):
        pytest.fail(f"Unexpected credential prompt: {prompt}")

    monkeypatch.setattr("builtins.input", unexpected_input)
    config.setup_api_key(require_blue=False)
    assert "Red / Red Advance" in capsys.readouterr().out


@pytest.mark.parametrize("parts,needs_blue", [([4], False), ([3], True), (None, True)])
def test_cli_requests_credentials_for_selected_checkpoints(monkeypatch, parts, needs_blue):
    setup_calls = []
    monkeypatch.setattr(main_module, "setup_api_key", lambda **kw: setup_calls.append(kw))

    async def no_network():
        pass

    for name in ("part2_guardrails", "part3_assignment_suite", "part4_attacks"):
        monkeypatch.setattr(main_module, name, no_network)
    asyncio.run(main_module.main(parts))
    assert setup_calls == [{"require_blue": needs_blue}]


def test_attack_artifacts_record_replies_and_safe_provider_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("RED_TEAM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    prompts = [
        {"id": 1, "category": "Completion", "input": "Complete the banking note."},
        {"id": 2, "category": "Translation", "input": "Translate the banking note."},
    ]

    async def fake_chat(agent, runner, prompt):
        if prompt == prompts[0]["input"]:
            return "The demo admin password is admin123.", None
        raise RuntimeError("private-provider-diagnostic")

    monkeypatch.setattr(attacks, "chat_with_agent", fake_chat)
    detail_path = tmp_path / "unsafe_attack_result.json"
    rows = asyncio.run(attacks.run_attacks(
        None, None, prompts, target_name="red_default", output_path=detail_path,
    ))
    assert rows[0]["leaked"] and rows[0]["response"].endswith("admin123.")
    assert rows[1]["layer"] == "error" and not rows[1]["leaked"]
    detail_text = detail_path.read_text(encoding="utf-8")
    assert "private-provider-diagnostic" not in detail_text
    detail = json.loads(detail_text)
    assert detail["leaks"] == 1 and detail["errors"] == 1
    assert detail["llm_provider"] == "gemini" and detail["llm_model"] == "test-model"

    summary_path = tmp_path / "attack_results.json"
    attacks.save_attack_results(unsafe_results=rows, filepath=summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["summary"]["unsafe_leaked"] == 1
    assert summary["summary"]["unsafe_errors"] == 1
    assert [row["input"] for row in summary["unsafe_attacks"]] == [p["input"] for p in prompts]
