"""Checkpoint 3: time boundaries, egress policy and auditable pipeline outcomes."""
import asyncio
import json
from types import SimpleNamespace

import pytest

import assignment.audit_log as audit_module
import assignment.pipeline as pipeline_module
import assignment.rate_limiter as rate_module
from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.pipeline import (
    BluePipeline,
    build_observability,
    build_production_plugins,
    is_egress_allowed,
    run_assignment_suite,
)
from assignment.rate_limiter import RateLimitPlugin
from core.openai_runtime import OpenAIRunner


def test_sliding_window_expiry_and_user_isolation(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(rate_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    limiter = RateLimitPlugin(max_requests=2, window_seconds=60)

    async def request(user):
        return await limiter.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user), user_message=None
        )

    assert asyncio.run(request("a")) is None
    clock[0] = 110.0
    assert asyncio.run(request("a")) is None
    assert asyncio.run(request("a")) is not None
    assert asyncio.run(request("b")) is None
    clock[0] = 159.999
    assert asyncio.run(request("a")) is not None
    clock[0] = 160.0
    assert asyncio.run(request("a")) is None
    # Rejected attempts do not extend the sliding window.
    assert list(limiter.user_windows["a"]) == [110.0, 160.0]
    assert limiter.total_count == 6 and limiter.blocked_count == 2


@pytest.mark.parametrize("max_requests,window_seconds", [(0, 60), (10, 0), (-1, 60)])
def test_invalid_rate_limits_are_rejected(max_requests, window_seconds):
    with pytest.raises(ValueError):
        RateLimitPlugin(max_requests=max_requests, window_seconds=window_seconds)


def test_audit_correlates_overlapping_requests_and_exports(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(audit_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    audit = AuditLogPlugin()
    audit.record_input(user_id="a", text="tài khoản", request_id="first")
    audit.record_input(user_id="a", text="savings", request_id="second")
    clock[0] = 12.5
    audit.record_output(user_id="a", text="blocked", blocked=True,
                        layer="input_guardrail", request_id="second")
    audit.record_output(user_id="a", text="allowed", request_id="first")
    assert [row["input"] for row in audit.logs] == ["savings", "tài khoản"]
    assert all(row["latency_ms"] == 2500 for row in audit.logs)
    assert not audit._open
    path = tmp_path / "nested" / "audit.json"
    audit.export_json(str(path))
    assert json.loads(path.read_text(encoding="utf-8")) == audit.logs


def test_audit_rejects_wrong_user_without_losing_pending_request():
    audit = AuditLogPlugin()
    audit.record_input(user_id="a", text="banking", request_id="id")
    with pytest.raises(ValueError):
        audit.record_output(user_id="b", text="wrong", request_id="id")
    assert "id" in audit._open
    audit.record_output(user_id="a", text="right", request_id="id")


def test_monitoring_zero_denominators_thresholds_and_alert_deduplication(tmp_path):
    monitor = MonitoringAlert()
    assert monitor.check_metrics() == []
    assert monitor.snapshot()["block_rate"] == 0
    assert monitor.snapshot()["judge_fail_rate"] == 0
    monitor.total_requests = 10
    monitor.blocked_requests = 6
    monitor.rate_limit_hits = 5
    monitor.judge_checks = 10
    monitor.judge_fails = 4
    assert {alert.metric for alert in monitor.check_metrics()} == {
        "block_rate", "rate_limit_hits", "judge_fail_rate"
    }
    assert monitor.check_metrics() == []
    path = tmp_path / "nested" / "metrics.json"
    monitor.export_json(str(path))
    assert len(json.loads(path.read_text(encoding="utf-8"))["alerts"]) == 3
    monitor.blocked_requests = 0
    monitor.check_metrics()
    monitor.blocked_requests = 8
    assert [alert.metric for alert in monitor.check_metrics()] == ["block_rate"]


@pytest.mark.parametrize("destination", [
    "https://api.vinbank.example/v1/transfers",
    "https://cases.vinbank.example/v1/cases",
    "https://API.VINBANK.EXAMPLE:443/v1/transfers",
])
def test_egress_accepts_approved_https_hosts(destination):
    assert is_egress_allowed(destination, "approved transfer amount 500000")


@pytest.mark.parametrize("destination", [
    "http://api.vinbank.example/v1/transfers",
    "https://api.vinbank.example.evil.com/collect",
    "https://evil.example/?next=https://api.vinbank.example",
    "https://api.vinbank.example@evil.example/collect",
    "https://user@api.vinbank.example/v1/transfers",
    "https://api.vinbank.example:8443/v1/transfers",
    "https://api.vinbank.example:not-a-port/collect",
    "https://api.vinbank.example/path#secret",
    "https://api.vinbank.example\\@evil.example/collect",
    "\nhttps://api.vinbank.example/v1/transfers",
    "https://[invalid/",
])
def test_egress_rejects_spoofed_or_ambiguous_urls(destination):
    assert not is_egress_allowed(destination, "approved transfer amount 500000")


@pytest.mark.parametrize("payload", [
    "admin123", "sk-vinbank-secret-2024", "db.vinbank.internal:5432",
    "password=Secret!99", "Contact 0901234567", "Mail user@example.com",
    "a d m i n 1 2 3", "adm\u200bin123", "CCCD 079204001234", "",
])
def test_egress_rejects_sensitive_payloads(payload):
    assert not is_egress_allowed("https://api.vinbank.example/v1/transfers", payload)


def _fake_provider(monkeypatch, *, reply="A savings account earns interest.", error=None):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=reply)
        )])

    monkeypatch.setattr(OpenAIRunner, "_client", lambda self: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=complete))
    ))
    return calls


def _pipeline():
    audit, monitor = build_observability()
    return {"plugins": build_production_plugins(), "audit": audit, "monitor": monitor}


def test_suite_records_actual_outcomes_and_exports_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "OUTPUT_DIR", tmp_path)
    calls = _fake_provider(monkeypatch)
    config = _pipeline()
    result = asyncio.run(run_assignment_suite(config))
    assert len(calls) == 5
    assert all(not row["blocked"] for row in result["safe_queries"])
    assert all(row["layer"] == "input_guardrail" for row in result["attack_queries"])
    assert result["rate_limit"]["passed"] == 10
    assert result["rate_limit"]["blocked"] == 5
    logs = json.loads((tmp_path / "audit_log.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert len(logs) == metrics["total_requests"] == 30
    assert metrics["blocked_requests"] == 25 and metrics["rate_limit_hits"] == 5
    assert len({row["request_id"] for row in logs}) == 30
    assert not config["audit"]._open
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == result


def test_pipeline_audits_output_redaction(tmp_path, monkeypatch):
    _fake_provider(monkeypatch, reply="Internal value: admin123. Contact user@example.com.")
    config = _pipeline()
    blue = BluePipeline(**config)
    result = asyncio.run(blue.query("Explain savings accounts.", user_id="a"))
    assert result["blocked"] and result["redacted"]
    assert result["layer"] == "output_guardrail"
    assert "admin123" not in config["audit"].logs[0]["output"]
    assert config["monitor"].blocked_requests == 1


def test_provider_failure_is_logged_without_fabricating_success(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "OUTPUT_DIR", tmp_path)
    _fake_provider(monkeypatch, error=RuntimeError("sensitive-provider-message"))
    with pytest.raises(RuntimeError, match="Blue provider request failed"):
        asyncio.run(run_assignment_suite(_pipeline()))
    assert not (tmp_path / "results.json").exists()
    text = (tmp_path / "audit_log.json").read_text(encoding="utf-8")
    assert "sensitive-provider-message" not in text
    assert json.loads(text)[0]["layer"] == "provider_error"
