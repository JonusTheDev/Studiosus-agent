"""Tests for AIAgent._tool_severity_gate — the opt-in severe-tier approval gate.

The method references no instance attributes, so a bare sentinel stands in for
``self``; we exercise the gate logic directly without building a full agent.
"""

import json
from unittest.mock import patch

import pytest

from run_agent import AIAgent


class _Sentinel:
    """Stand-in for ``self`` — the gate uses no agent attributes."""


def _gate(function_name, *, approval_callback=None):
    return AIAgent._tool_severity_gate(
        _Sentinel(), function_name, approval_callback=approval_callback
    )


def _cfg(*, enabled, overrides=None):
    approvals = {"severity_tiers": enabled}
    if overrides is not None:
        approvals["severity_overrides"] = overrides
    return {"approvals": approvals}


@pytest.fixture(autouse=True)
def _no_env_flag(monkeypatch):
    # The conftest autouse fixture already strips HERMES_SECURITY; be explicit
    # so config, not a stray env flag, controls enablement in these tests.
    monkeypatch.delenv("HERMES_SECURITY", raising=False)


# --- disabled: always a no-op --------------------------------------------


def test_flag_off_is_noop_even_for_severe():
    with patch("hermes_cli.config.load_config_readonly", return_value=_cfg(enabled=False)):
        assert _gate("delegate_task") is None


def test_flag_off_default_config_is_noop():
    # No approvals block at all -> disabled.
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert _gate("delegate_task") is None


# --- enabled: tier decides ------------------------------------------------


def test_benign_tool_not_gated_when_enabled():
    with patch("hermes_cli.config.load_config_readonly", return_value=_cfg(enabled=True)):
        assert _gate("web_search") is None


def test_moderate_tool_not_gated_when_enabled():
    with patch("hermes_cli.config.load_config_readonly", return_value=_cfg(enabled=True)):
        assert _gate("write_file") is None


def test_self_gated_severe_tool_is_exempt():
    # terminal/execute_code/process are severe but reach the finer command-level
    # gate on their own — the blanket severity gate must not double-prompt them.
    with patch("hermes_cli.config.load_config_readonly", return_value=_cfg(enabled=True)):
        assert _gate("terminal") is None
        assert _gate("execute_code") is None
        assert _gate("process") is None


def test_severe_non_exempt_tool_approved_proceeds():
    approve = {"approved": True, "message": None}
    with patch("hermes_cli.config.load_config_readonly", return_value=_cfg(enabled=True)), \
         patch("tools.approval.request_tool_approval", return_value=approve) as m:
        assert _gate("delegate_task", approval_callback=lambda *a, **k: "once") is None
        assert m.call_count == 1
        # The tool name and a stable rule_key are threaded through.
        _, kwargs = m.call_args
        assert kwargs["rule_key"] == "severity:delegate_task"


def test_severe_non_exempt_tool_denied_blocks():
    deny = {"approved": False, "message": "user denied"}
    with patch("hermes_cli.config.load_config_readonly", return_value=_cfg(enabled=True)), \
         patch("tools.approval.request_tool_approval", return_value=deny):
        result = _gate("cronjob", approval_callback=lambda *a, **k: "deny")
    assert result is not None
    payload = json.loads(result)
    assert payload["error"] == "user denied"


def test_gate_fails_open_on_machinery_error():
    # If request_tool_approval blows up, do not wedge the turn — proceed.
    with patch("hermes_cli.config.load_config_readonly", return_value=_cfg(enabled=True)), \
         patch("tools.approval.request_tool_approval", side_effect=RuntimeError("boom")):
        assert _gate("delegate_task") is None


# --- config overrides ------------------------------------------------------


def test_override_escalates_benign_tool_into_the_gate():
    deny = {"approved": False, "message": "nope"}
    cfg = _cfg(enabled=True, overrides={"web_search": "severe"})
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
         patch("tools.approval.request_tool_approval", return_value=deny):
        result = _gate("web_search", approval_callback=lambda *a, **k: "deny")
    assert result is not None
    assert json.loads(result)["error"] == "nope"


def test_override_deescalates_severe_tool_out_of_the_gate():
    cfg = _cfg(enabled=True, overrides={"delegate_task": "moderate"})
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
         patch("tools.approval.request_tool_approval") as m:
        assert _gate("delegate_task") is None
        m.assert_not_called()


# --- env override ----------------------------------------------------------


def test_env_flag_forces_enable(monkeypatch):
    monkeypatch.setenv("HERMES_SECURITY", "1")
    deny = {"approved": False, "message": "denied"}
    # Config says disabled, but the env flag forces the gate on.
    with patch("hermes_cli.config.load_config_readonly", return_value=_cfg(enabled=False)), \
         patch("tools.approval.request_tool_approval", return_value=deny):
        result = _gate("delegate_task", approval_callback=lambda *a, **k: "deny")
    assert result is not None
    assert json.loads(result)["error"] == "denied"
