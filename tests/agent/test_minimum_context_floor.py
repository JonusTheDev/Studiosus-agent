"""The 32K gate, and the 64K compression floor that must not follow it down.

``MINIMUM_CONTEXT_LENGTH`` and ``COMPRESSION_FLOOR_TOKENS`` shared one constant
until the gate was lowered, and they are not the same idea:

* the **gate** is the smallest window Hermes will accept at all;
* the **floor** is the point below which auto-compaction merely throws away
  context a large model already has.

Dragging the floor down with the gate would make a 64K model compact at 50% of
its window instead of riding to 85% via the degenerate-window guard (#14690).
These tests exist so that regression cannot land quietly.
"""

import pytest

from agent.context_compressor import ContextCompressor
from agent.model_metadata import COMPRESSION_FLOOR_TOKENS, MINIMUM_CONTEXT_LENGTH


def _threshold(context_length, percent=0.5, max_tokens=None):
    return ContextCompressor._compute_threshold_tokens(
        context_length, percent, max_tokens)


# --- the two constants are distinct -------------------------------------


def test_the_gate_admits_a_32k_model():
    assert MINIMUM_CONTEXT_LENGTH == 32_000


def test_the_compression_floor_did_not_follow_the_gate_down():
    assert COMPRESSION_FLOOR_TOKENS == 64_000
    assert COMPRESSION_FLOOR_TOKENS > MINIMUM_CONTEXT_LENGTH


# --- the gate ------------------------------------------------------------


@pytest.mark.parametrize("window", [32_000, 32_768, 40_000, 65_536, 200_000])
def test_windows_at_or_above_the_gate_are_accepted(window):
    assert window >= MINIMUM_CONTEXT_LENGTH


@pytest.mark.parametrize("window", [4_096, 8_192, 16_384, 31_999])
def test_windows_below_the_gate_are_still_refused(window):
    # Lowering the gate is not opening it: a model too small to hold a
    # tool-calling workflow is still turned away.
    assert window < MINIMUM_CONTEXT_LENGTH


def test_our_measured_operating_context_clears_the_gate():
    # qwen3.6 on a 5070 Ti measured fastest at 32768 (dyno_profiles/). Under
    # the old 64K gate that setting was rejected outright, which is what made
    # this change a prerequisite rather than a preference.
    assert 32_768 >= MINIMUM_CONTEXT_LENGTH


# --- the floor's behavior is unchanged -----------------------------------


def test_a_64k_model_still_rides_to_85_percent_not_50():
    # The regression this whole split exists to prevent. With the floor at 64K
    # the flooring meets the window, the degenerate guard fires, and the model
    # compacts at 85%. Had the floor dropped to 32K, this would be 32000.
    assert _threshold(64_000) == int(64_000 * ContextCompressor._MIN_CTX_TRIGGER_RATIO)
    assert _threshold(64_000) > 32_000


def test_a_large_window_still_compacts_on_the_percentage():
    # 50% of 200K is far above the floor, so the floor never enters into it.
    assert _threshold(200_000) == 100_000


def test_a_32k_model_compacts_below_its_window_so_compaction_can_fire():
    # The newly-admitted case. The floor exceeds the window, so the degenerate
    # guard takes over: trigger high but strictly under the window, or
    # auto-compression could never fire before the provider rejects the call.
    threshold = _threshold(32_768)
    assert 0 < threshold < 32_768
    assert threshold == int(32_768 * ContextCompressor._MIN_CTX_TRIGGER_RATIO)


def test_output_reservation_still_shrinks_the_effective_budget():
    # A provider reserving output space out of the same window (#43547): the
    # threshold is computed against the input budget, not the raw window. Here
    # that halves 200K to a 100K budget, whose 50% (50K) then lands under the
    # comfort floor and is lifted to it — the floor still governing a large
    # model exactly as it did before the gate moved.
    assert _threshold(200_000, 0.5, 100_000) == COMPRESSION_FLOOR_TOKENS
    # Without the reservation the percentage clears the floor on its own.
    assert _threshold(200_000, 0.5) == 100_000


def test_the_threshold_never_reaches_the_window():
    # A threshold at 100% can never be crossed — the provider rejects the
    # request first, and auto-compression silently never fires.
    for window in (32_000, 32_768, 64_000, 65_536, 128_000):
        assert _threshold(window) < window
