"""Tests for tools/skill_provenance.py — write-origin ContextVar."""

import contextvars





def test_set_and_get_origin():
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        get_current_write_origin,
    )
    token = set_current_write_origin("background_review")
    try:
        assert get_current_write_origin() == "background_review"
    finally:
        reset_current_write_origin(token)


def test_reset_restores_prior_origin():
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        get_current_write_origin,
    )
    outer = set_current_write_origin("assistant_tool")
    try:
        inner = set_current_write_origin("background_review")
        try:
            assert get_current_write_origin() == "background_review"
        finally:
            reset_current_write_origin(inner)
        assert get_current_write_origin() == "assistant_tool"
    finally:
        reset_current_write_origin(outer)


def test_is_background_review_truthy_only_for_review():
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        is_background_review,
        BACKGROUND_REVIEW,
    )
    for origin, expected in (
        ("foreground", False),
        ("assistant_tool", False),
        ("random_other_value", False),
        (BACKGROUND_REVIEW, True),
    ):
        token = set_current_write_origin(origin)
        try:
            assert is_background_review() is expected, (
                f"is_background_review() wrong for origin={origin!r}"
            )
        finally:
            reset_current_write_origin(token)


def test_empty_origin_falls_back_to_foreground():
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        get_current_write_origin,
    )
    token = set_current_write_origin("")
    try:
        # Empty is coerced to "foreground" at the set() boundary.
        assert get_current_write_origin() == "foreground"
    finally:
        reset_current_write_origin(token)


def test_context_isolation_between_copies():
    """ContextVar scoping: modifications in one copy do not leak out."""
    from tools.skill_provenance import (
        set_current_write_origin,
        get_current_write_origin,
        BACKGROUND_REVIEW,
    )

    # Start at the module default.
    original = get_current_write_origin()

    def _run_in_copy():
        set_current_write_origin(BACKGROUND_REVIEW)
        return get_current_write_origin()

    ctx = contextvars.copy_context()
    inside = ctx.run(_run_in_copy)
    assert inside == BACKGROUND_REVIEW
    # Parent context unaffected.
    assert get_current_write_origin() == original


# ---------------------------------------------------------------------------
# Review kind — independent of write origin (see module docstring in
# skill_provenance.py for why this is a separate signal).
# ---------------------------------------------------------------------------


def test_review_kind_defaults_to_none():
    from tools.skill_provenance import get_current_review_kind
    assert get_current_review_kind() is None


def test_set_and_get_review_kind():
    from tools.skill_provenance import (
        set_current_review_kind,
        reset_current_review_kind,
        get_current_review_kind,
        REVIEW_KIND_SKILL,
    )
    token = set_current_review_kind(REVIEW_KIND_SKILL)
    try:
        assert get_current_review_kind() == REVIEW_KIND_SKILL
    finally:
        reset_current_review_kind(token)


def test_reset_restores_prior_review_kind():
    from tools.skill_provenance import (
        set_current_review_kind,
        reset_current_review_kind,
        get_current_review_kind,
        REVIEW_KIND_SKILL,
        REVIEW_KIND_CURATOR,
    )
    outer = set_current_review_kind(REVIEW_KIND_CURATOR)
    try:
        inner = set_current_review_kind(REVIEW_KIND_SKILL)
        try:
            assert get_current_review_kind() == REVIEW_KIND_SKILL
        finally:
            reset_current_review_kind(inner)
        assert get_current_review_kind() == REVIEW_KIND_CURATOR
    finally:
        reset_current_review_kind(outer)


def test_is_family_covenant_scope_true_only_for_skill_review_fork():
    from tools.skill_provenance import (
        set_current_write_origin, reset_current_write_origin,
        set_current_review_kind, reset_current_review_kind,
        is_family_covenant_scope,
        BACKGROUND_REVIEW, REVIEW_KIND_SKILL, REVIEW_KIND_CURATOR,
    )
    cases = [
        ("foreground", None, False),
        ("assistant_tool", None, False),
        (BACKGROUND_REVIEW, None, False),  # origin set but no review kind bound
        (BACKGROUND_REVIEW, REVIEW_KIND_CURATOR, False),
        (BACKGROUND_REVIEW, REVIEW_KIND_SKILL, True),
        ("foreground", REVIEW_KIND_SKILL, False),  # review kind alone isn't enough
    ]
    for origin, kind, expected in cases:
        wo = set_current_write_origin(origin)
        rk = set_current_review_kind(kind)
        try:
            assert is_family_covenant_scope() is expected, (
                f"wrong for origin={origin!r} kind={kind!r}"
            )
        finally:
            reset_current_review_kind(rk)
            reset_current_write_origin(wo)


def test_review_kind_context_isolation_between_copies():
    from tools.skill_provenance import (
        set_current_review_kind,
        get_current_review_kind,
        REVIEW_KIND_SKILL,
    )
    original = get_current_review_kind()

    def _run_in_copy():
        set_current_review_kind(REVIEW_KIND_SKILL)
        return get_current_review_kind()

    ctx = contextvars.copy_context()
    inside = ctx.run(_run_in_copy)
    assert inside == REVIEW_KIND_SKILL
    assert get_current_review_kind() == original
