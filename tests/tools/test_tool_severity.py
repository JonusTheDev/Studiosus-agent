"""Tests for tools/tool_severity.py — the tier constants + default map."""

from tools import tool_severity as sev


def test_tier_constants_and_valid_set():
    assert sev.BENIGN == "benign"
    assert sev.MODERATE == "moderate"
    assert sev.SEVERE == "severe"
    assert set(sev.VALID_TIERS) == {"benign", "moderate", "severe"}
    assert sev.UNKNOWN_DEFAULT == sev.MODERATE


def test_normalize_tier_accepts_valid():
    assert sev.normalize_tier("benign") == "benign"
    assert sev.normalize_tier("  SEVERE  ") == "severe"
    assert sev.normalize_tier("Moderate") == "moderate"


def test_normalize_tier_falls_back_on_junk():
    assert sev.normalize_tier("nonsense") == sev.UNKNOWN_DEFAULT
    assert sev.normalize_tier(None) == sev.UNKNOWN_DEFAULT
    assert sev.normalize_tier(123) == sev.UNKNOWN_DEFAULT
    assert sev.normalize_tier("critical", default=sev.SEVERE) == sev.SEVERE


def test_default_severity_spot_checks():
    assert sev.default_tier_for("web_search") == sev.BENIGN
    assert sev.default_tier_for("read_file") == sev.BENIGN
    assert sev.default_tier_for("write_file") == sev.MODERATE
    assert sev.default_tier_for("patch") == sev.MODERATE
    assert sev.default_tier_for("skill_manage") == sev.MODERATE
    assert sev.default_tier_for("delegate_task") == sev.SEVERE
    assert sev.default_tier_for("cronjob") == sev.SEVERE
    assert sev.default_tier_for("ha_call_service") == sev.SEVERE
    assert sev.default_tier_for("computer_use") == sev.SEVERE


def test_unknown_tool_defaults_to_moderate():
    assert sev.default_tier_for("some_unregistered_tool") == sev.MODERATE


def test_self_gated_exempt_are_the_command_level_tools():
    assert sev.SELF_GATED_EXEMPT == frozenset({"terminal", "execute_code", "process"})
    # Exempt tools are still classified severe (they just aren't blanket-gated).
    for name in sev.SELF_GATED_EXEMPT:
        assert sev.default_tier_for(name) == sev.SEVERE


def test_every_mapped_tier_is_valid():
    for name, tier in sev.DEFAULT_SEVERITY.items():
        assert tier in sev.VALID_TIERS, f"{name} has invalid tier {tier!r}"
