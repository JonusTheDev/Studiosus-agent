"""Tool-severity tiers — classify every tool by action severity.

Item 2 of the Studiosus heart's "blessed but unbuilt" list
(``docs/studiosus-heart.md``): three categories by action severity, so that
approval behavior can key off *what kind of action a tool is*, not only whether
a shell command string happens to trip a dangerous-pattern regex.

Three tiers, in ascending severity:

* **benign** — read-only, no side effect anyone would want to confirm
  (``web_search``, ``read_file``, ``vision_analyze`` …).
* **moderate** — local, reversible mutation (``write_file``, ``patch``,
  ``memory``, ``skill_manage``, kanban writes, media generation …). Cost or a
  local edit, but nothing catastrophic or externally irreversible.
* **severe** — execute / build / actuate / irreversible external side effect
  (``terminal``, ``execute_code``, ``delegate_task``, ``ha_call_service``,
  ``cronjob``, ``computer_use``, sending a DM you can't unsend …).

Only the **severe** tier is gated (see ``run_agent.AIAgent._tool_severity_gate``
and ``tools/approval.request_tool_approval``), and only when the operator opts
in via ``approvals.severity_tiers``. The tiers are otherwise pure, deletable
classification — nothing here changes behavior on its own.

This module is data + tiny helpers only. It imports nothing from the agent so
the registry can consult it at import time without a cycle.
"""

from __future__ import annotations

BENIGN = "benign"
MODERATE = "moderate"
SEVERE = "severe"

VALID_TIERS = (BENIGN, MODERATE, SEVERE)

# An unclassified tool (a late-registered MCP tool, a plugin tool, a future
# built-in nobody added here) defaults to MODERATE: never surprise-prompt an
# action of unknown severity when the operator only asked to gate the severe
# tier. This is a deliberate "fail-quiet, not fail-closed" choice, justified by
# the gate being opt-in (default off) — an operator who wants an unknown tool
# gated escalates it explicitly via ``approvals.severity_overrides``.
UNKNOWN_DEFAULT = MODERATE

# Tools that already reach the human-approval gate through their OWN, finer,
# command-level analysis (``terminal``/``process`` via
# ``check_all_command_guards`` + ``DANGEROUS_PATTERNS``; ``execute_code`` via
# ``check_execute_code_guard``). They are severe, but the blanket per-tool
# severity prompt must NOT double-prompt them — the command-level gate is
# strictly better (it distinguishes ``ls`` from ``rm -rf``). So the severity
# gate skips this set. The net-new coverage of the severity gate is therefore
# exactly the *previously ungated* severe tools below. Do NOT "fix" this by
# removing the exemption — that would give every ``terminal ls`` a redundant
# confirmation on top of the command analysis it already passed.
SELF_GATED_EXEMPT = frozenset({"terminal", "execute_code", "process"})


def normalize_tier(value, default: str = UNKNOWN_DEFAULT) -> str:
    """Coerce arbitrary input to a valid tier, falling back to ``default``.

    Mirrors ``tools/approval._normalize_approval_mode``'s tolerance: unknown /
    junk / None becomes the default rather than raising, so a fat-fingered
    ``severity_overrides`` entry degrades safely instead of breaking dispatch.
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if v in VALID_TIERS:
            return v
    return default


# Built-in tool -> tier. Keyed by the exact registered tool name. Anything not
# listed resolves to UNKNOWN_DEFAULT via registry.get_severity.
DEFAULT_SEVERITY: dict[str, str] = {
    # --- benign: read-only / no side effect ------------------------------
    "web_search": BENIGN,
    "web_extract": BENIGN,
    "x_search": BENIGN,
    "read_file": BENIGN,
    "search_files": BENIGN,
    "session_search": BENIGN,
    "vision_analyze": BENIGN,
    "video_analyze": BENIGN,
    "read_terminal": BENIGN,
    "skills_list": BENIGN,
    "skill_view": BENIGN,
    "clarify": BENIGN,
    "todo": BENIGN,
    "kanban_show": BENIGN,
    "kanban_list": BENIGN,
    "kanban_attachments": BENIGN,
    "project_list": BENIGN,
    "ha_get_state": BENIGN,
    "ha_list_entities": BENIGN,
    "ha_list_services": BENIGN,
    "feishu_doc_read": BENIGN,
    "feishu_drive_list_comments": BENIGN,
    "feishu_drive_list_comment_replies": BENIGN,
    "yb_query_group_info": BENIGN,
    "yb_query_group_members": BENIGN,
    "yb_search_sticker": BENIGN,
    # browser reads / passive inspection
    "browser_snapshot": BENIGN,
    "browser_get_images": BENIGN,
    "browser_console": BENIGN,
    "browser_vision": BENIGN,

    # --- moderate: local / reversible mutation, or cost ------------------
    "write_file": MODERATE,
    "patch": MODERATE,
    "memory": MODERATE,
    "skill_manage": MODERATE,
    "close_terminal": MODERATE,
    "kanban_create": MODERATE,
    "kanban_comment": MODERATE,
    "kanban_complete": MODERATE,
    "kanban_block": MODERATE,
    "kanban_unblock": MODERATE,
    "kanban_link": MODERATE,
    "kanban_attach": MODERATE,
    "kanban_attach_url": MODERATE,
    "kanban_heartbeat": MODERATE,
    "project_create": MODERATE,
    "project_switch": MODERATE,
    # media generation — costs money/quota but is not destructive/irreversible
    "image_generate": MODERATE,
    "video_generate": MODERATE,
    "text_to_speech": MODERATE,
    "xai_video_edit": MODERATE,
    "xai_video_extend": MODERATE,
    # browser navigation / low-stakes interaction
    "browser_navigate": MODERATE,
    "browser_back": MODERATE,
    "browser_scroll": MODERATE,
    "browser_press": MODERATE,
    "browser_dialog": MODERATE,
    # external comment writes — low stakes, but a remote side effect
    "feishu_drive_add_comment": MODERATE,
    "feishu_drive_reply_comment": MODERATE,

    # --- severe: execute / actuate / irreversible external side effect ---
    "terminal": SEVERE,        # self-gated (SELF_GATED_EXEMPT) but classified
    "execute_code": SEVERE,    # self-gated
    "process": SEVERE,         # self-gated
    "delegate_task": SEVERE,   # spawns autonomous subagents (cost/resource)
    "cronjob": SEVERE,         # schedules autonomous future runs
    "computer_use": SEVERE,    # controls the whole desktop
    "ha_call_service": SEVERE,  # actuates physical devices
    "discord": SEVERE,         # posts to a server
    "discord_admin": SEVERE,   # server administration
    # browser interactions that mutate remote state (submit forms, purchase,
    # arbitrary devtools)
    "browser_click": SEVERE,
    "browser_type": SEVERE,
    "browser_cdp": SEVERE,
    # sending a message you cannot unsend
    "yb_send_dm": SEVERE,
    "yb_send_sticker": SEVERE,
}


def default_tier_for(name: str) -> str:
    """The built-in default tier for a tool name, before any config override."""
    return DEFAULT_SEVERITY.get(name, UNKNOWN_DEFAULT)
