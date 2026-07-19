"""The dyno bench — measurement, judgment, and the covenants around them.

The live roller is injected, so every judgment path here runs without a GPU or
a reachable Ollama.  See ``.plans/studiosus-heart-strategy.md``.
"""

import json

import pytest

from agent import dyno


def _point(num_ctx, gpu_frac, tok_s):
    return {"num_ctx": num_ctx, "gpu_frac": gpu_frac, "tok_s": tok_s,
            "resident_gb": 10.0, "prompt_tok_s": 200.0, "load_s": 1.0}


# --- judgment ------------------------------------------------------------


def test_in_band_prefers_the_deepest_context_that_holds():
    curve = [_point(32768, 0.99, 80.0), _point(65536, 0.95, 60.0),
             _point(131072, 0.40, 12.0)]
    assert dyno.recommend_num_ctx(curve)["num_ctx"] == 65536


def test_off_band_steers_by_speed_not_residence():
    # Once nothing holds the band, residence stops being a guide: the ancestor
    # measured a forced split wearing far more "residence" at a fraction of the
    # speed.  Here the deeper, more-resident point is five times slower, and
    # the shallower, faster one must win.
    curve = [_point(32768, 0.62, 50.0), _point(65536, 0.89, 10.0)]
    chosen = dyno.recommend_num_ctx(curve)
    assert chosen["num_ctx"] == 32768
    assert chosen["tok_s"] == 50.0


def test_off_band_takes_depth_when_depth_is_free():
    # Within SPEED_TOLERANCE of the fastest, the deeper context wins: depth to
    # ponder is worth having whenever it costs no speed.
    curve = [_point(32768, 0.60, 50.0), _point(65536, 0.55, 49.0)]
    assert dyno.recommend_num_ctx(curve)["num_ctx"] == 65536


def test_ceiling_is_respected():
    curve = [_point(32768, 0.99, 80.0), _point(65536, 0.99, 70.0)]
    assert dyno.recommend_num_ctx(curve, ceiling=32768)["num_ctx"] == 32768


def test_empty_curve_recommends_nothing():
    assert dyno.recommend_num_ctx([]) is None


def test_ties_at_one_context_break_on_measured_speed():
    curve = [_point(65536, 0.95, 40.0), _point(65536, 0.95, 55.0)]
    assert dyno.recommend_num_ctx(curve)["tok_s"] == 55.0


@pytest.mark.parametrize("size, vram, expected", [
    (100, 100, 1.0),
    (100, 62, 0.62),
    (100, 0, 0.0),
    (0, 0, 0.0),        # a model of no size is not 100% resident
    (100, 200, 1.0),    # never above 1.0, whatever the server reports
])
def test_gpu_fraction_is_bounded(size, vram, expected):
    assert dyno.gpu_fraction(size, vram) == expected


def test_check_plan_says_unknown_rather_than_guessing():
    verdict = dyno.check_plan({"model": "m"}, 65536)
    assert verdict["status"] == "unknown"
    assert verdict["recommended_num_ctx"] is None


def test_check_plan_reports_a_spill_against_the_nearest_measured_point():
    profile = {"model": "m", "power": {"curve": [_point(32768, 0.62, 50.0)]}}
    verdict = dyno.check_plan(profile, 30000)
    assert verdict["status"] == "spills"
    assert verdict["gpu_frac"] == 0.62
    assert "32768" in verdict["note"]


# --- the sweep -----------------------------------------------------------


def test_a_point_the_model_refuses_is_skipped_not_recorded_as_zero():
    def measure(model, num_ctx):
        return None if num_ctx > 32768 else {
            "resident_gb": 9.0, "gpu_frac": 0.98, "tok_s": 70.0,
            "prompt_tok_s": 400.0, "load_s": 2.0}

    profile = dyno.run_dyno("m", measure, ctx_points=(32768, 65536))
    assert [p["num_ctx"] for p in profile["power"]["curve"]] == [32768]
    assert profile["power"]["recommended_num_ctx"] == 32768


def test_the_cold_first_roll_is_discarded_not_blended_in():
    # Measured on a 5070 Ti: the first roll after a cleared field ran ~35%
    # slower at every context. Averaging it with warm rolls yields a number
    # that describes neither state.
    speeds = iter([20.0, 50.0, 50.0])

    def measure(model, num_ctx):
        return {"resident_gb": 9.0, "gpu_frac": 0.98, "tok_s": next(speeds),
                "prompt_tok_s": 100.0, "load_s": 1.0}

    power = dyno.run_dyno("m", measure, ctx_points=(32768,), repeats=2, warmup=1)["power"]
    assert power["curve"][0]["tok_s"] == 50.0
    assert power["warmup_rolls"] == 1


def test_warmup_can_be_turned_off_for_an_already_warm_model():
    speeds = iter([20.0, 50.0])

    def measure(model, num_ctx):
        return {"resident_gb": 9.0, "gpu_frac": 0.98, "tok_s": next(speeds),
                "prompt_tok_s": 100.0, "load_s": 1.0}

    power = dyno.run_dyno("m", measure, ctx_points=(32768,), repeats=2, warmup=0)["power"]
    assert power["curve"][0]["tok_s"] == 35.0  # both rolls counted
    assert power["warmup_rolls"] == 0


def test_repeats_are_averaged_and_failed_rolls_dropped():
    speeds = iter([40.0, None, 60.0])

    def measure(model, num_ctx):
        speed = next(speeds)
        if speed is None:
            return None
        return {"resident_gb": 9.0, "gpu_frac": 0.98, "tok_s": speed,
                "prompt_tok_s": 100.0, "load_s": 1.0}

    profile = dyno.run_dyno("m", measure, ctx_points=(32768,), repeats=3, warmup=0)
    point = profile["power"]["curve"][0]
    assert point["tok_s"] == 50.0  # mean of the two that succeeded
    assert point["rolls"] == 2


def test_a_model_that_never_fits_may_not_wear_its_fallback_as_a_fitness():
    def measure(model, num_ctx):
        return {"resident_gb": 25.0, "gpu_frac": 0.62, "tok_s": 30.0,
                "prompt_tok_s": 90.0, "load_s": 8.0}

    power = dyno.run_dyno("big", measure, ctx_points=(32768, 65536))["power"]
    # A recommendation is still made — the least-bad point is useful — but it
    # is explicitly flagged as out of band and not a fit.
    assert power["recommended_num_ctx"] is not None
    assert power["recommended_in_band"] is False
    assert power["fully_fits"] is False


def test_overlapping_rolls_are_reported_as_undecided():
    # Two contexts whose measured ranges overlap have not been separated. The
    # bench must say so rather than let a mean decide by a hair.
    speeds = iter([30.0, 55.0, 35.0, 50.0])

    def measure(model, num_ctx):
        return {"resident_gb": 22.0, "gpu_frac": 0.62, "tok_s": next(speeds),
                "prompt_tok_s": 100.0, "load_s": 5.0}

    power = dyno.run_dyno("m", measure, ctx_points=(32768, 65536),
                          repeats=2, warmup=0)["power"]
    assert power["recommended_within_noise_of"] != []


def test_cleanly_separated_contexts_are_not_flagged():
    speeds = iter([70.0, 72.0, 10.0, 12.0])

    def measure(model, num_ctx):
        return {"resident_gb": 22.0, "gpu_frac": 0.62, "tok_s": next(speeds),
                "prompt_tok_s": 100.0, "load_s": 5.0}

    power = dyno.run_dyno("m", measure, ctx_points=(32768, 65536),
                          repeats=2, warmup=0)["power"]
    assert power["recommended_num_ctx"] == 32768
    assert power["recommended_within_noise_of"] == []


def test_a_single_roll_claims_no_confidence_about_itself():
    def measure(model, num_ctx):
        return {"resident_gb": 22.0, "gpu_frac": 0.62, "tok_s": 40.0,
                "prompt_tok_s": 100.0, "load_s": 5.0}

    power = dyno.run_dyno("m", measure, ctx_points=(32768, 65536),
                          repeats=1, warmup=0)["power"]
    # One roll measures nothing about its own reliability; silence beats a
    # fabricated confidence in either direction.
    assert power["recommended_within_noise_of"] == []
    assert "tok_s_min" not in power["curve"][0]


def test_the_curve_names_the_server_it_was_measured_against():
    def measure(model, num_ctx):
        return {"resident_gb": 9.0, "gpu_frac": 0.98, "tok_s": 70.0,
                "prompt_tok_s": 400.0, "load_s": 2.0}

    power = dyno.run_dyno(
        "m", measure, ctx_points=(32768,), host="http://box:11434",
        gpu={"name": "RTX 5070 Ti", "vram_gb": 16.0},
        server_env={"ollama_version": "0.30.8"},
    )["power"]
    assert power["host"] == "http://box:11434"
    assert power["gpu"]["name"] == "RTX 5070 Ti"
    assert power["server_env"]["ollama_version"] == "0.30.8"


# --- profiles ------------------------------------------------------------


@pytest.mark.parametrize("model, slug", [
    ("qwen3.6:latest", "qwen3.6-latest"),
    ("hf.co/user/Model-GGUF:Q4_K_M", "hf.co-user-model-gguf-q4-k-m"),
])
def test_model_tags_become_filename_safe_slugs(model, slug):
    # Tags carry ':' and '/', both illegal in Windows filenames.
    assert dyno.profile_slug(model) == slug


def test_a_users_own_measurement_wins_over_the_one_we_shipped(tmp_path, monkeypatch):
    seeded = tmp_path / "seeded"
    seeded.mkdir()
    (seeded / "m.json").write_text(json.dumps({"model": "m", "whose": "ours"}))
    monkeypatch.setattr(dyno, "SEEDED_PROFILE_DIR", seeded)

    assert dyno.load_profile("m")["whose"] == "ours"

    dyno.save_profile({"model": "m", "whose": "theirs"})
    assert dyno.load_profile("m")["whose"] == "theirs"


def test_a_choice_overrides_what_the_bench_recommends():
    # The bench measures throughput alone. A person may weigh room-to-think
    # against it, as we did taking 64K where 32K measured faster.
    profile = {"model": "m", "chosen_num_ctx": 65536,
               "power": {"recommended_num_ctx": 32768}}
    assert dyno.operating_num_ctx(profile) == 65536


def test_without_a_choice_the_recommendation_stands():
    profile = {"model": "m", "power": {"recommended_num_ctx": 32768}}
    assert dyno.operating_num_ctx(profile) == 32768


@pytest.mark.parametrize("profile", [None, {}, {"model": "m"}, {"model": "m", "power": {}}])
def test_no_measurement_and_no_choice_is_no_opinion(profile):
    # None must read as "nobody has an opinion", never as a silent default.
    assert dyno.operating_num_ctx(profile) is None


def test_an_unmeasured_model_has_no_profile():
    assert dyno.load_profile("never-measured:v1") is None


def test_a_corrupt_profile_falls_through_instead_of_raising(tmp_path, monkeypatch):
    seeded = tmp_path / "seeded"
    seeded.mkdir()
    (seeded / "m.json").write_text(json.dumps({"model": "m", "whose": "ours"}))
    monkeypatch.setattr(dyno, "SEEDED_PROFILE_DIR", seeded)

    measured = dyno.measured_profile_dir()
    measured.mkdir(parents=True, exist_ok=True)
    (measured / "m.json").write_text("{ truncated")

    assert dyno.load_profile("m")["whose"] == "ours"


# --- clearing the field --------------------------------------------------


def test_clear_field_releases_every_resident_model(monkeypatch):
    monkeypatch.setattr(dyno, "loaded_split", lambda host=None: [
        {"model": "a:latest"}, {"model": "b:latest"}])
    asked = []
    monkeypatch.setattr(dyno, "_http",
                        lambda url, payload=None, timeout=0: asked.append(payload) or {})

    released = dyno.clear_field(host="http://box:11434")
    assert released == ["a:latest", "b:latest"]
    # keep_alive: 0 is the release; anything else leaves the model pinned.
    assert all(p["keep_alive"] == 0 for p in asked)


def test_clear_field_spares_the_flame(monkeypatch):
    monkeypatch.setattr(dyno, "loaded_split", lambda host=None: [
        {"model": "qwen3.6:latest"}, {"model": "stale:latest"}])
    monkeypatch.setattr(dyno, "_http", lambda url, payload=None, timeout=0: {})

    assert dyno.clear_field(keep={"qwen3.6:latest"}) == ["stale:latest"]


def test_a_server_that_refuses_to_unload_never_raises(monkeypatch):
    monkeypatch.setattr(dyno, "loaded_split", lambda host=None: [{"model": "a:latest"}])

    def _boom(*a, **k):
        raise OSError("connection reset")

    monkeypatch.setattr(dyno, "_http", _boom)
    assert dyno.clear_field() == []  # reported as released-nothing, not a crash


def test_an_unreachable_server_reports_absence_not_an_empty_inventory(monkeypatch):
    def _boom(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr(dyno, "_http", _boom)
    # None ("I could not ask") must never be confused with [] ("it has none").
    assert dyno.installed_models() is None
    assert dyno.server_version() is None
    assert dyno.loaded_split() == []


# --- the roller ----------------------------------------------------------


def test_live_measure_computes_speed_from_the_servers_own_timings(monkeypatch):
    monkeypatch.setattr(dyno, "clear_field", lambda host=None: [])
    monkeypatch.setattr(dyno, "generate_raw", lambda *a, **k: {
        "eval_count": 120, "eval_duration": 2_000_000_000,      # 60 tok/s
        "prompt_eval_count": 400, "prompt_eval_duration": 1_000_000_000,  # 400 tok/s
        "load_duration": 3_000_000_000,
    })
    monkeypatch.setattr(dyno, "model_split", lambda model, host=None: (100, 62))

    got = dyno.live_measure(host="http://box:11434", settle_s=0)("m", 32768)
    assert got["tok_s"] == 60.0
    assert got["prompt_tok_s"] == 400.0
    assert got["gpu_frac"] == 0.62
    assert got["load_s"] == 3.0


def test_a_refused_generation_yields_no_point(monkeypatch):
    monkeypatch.setattr(dyno, "clear_field", lambda host=None: [])

    def _boom(*a, **k):
        raise OSError("model requires more system memory")

    monkeypatch.setattr(dyno, "generate_raw", _boom)
    assert dyno.live_measure(settle_s=0)("m", 131072) is None


def test_an_error_payload_yields_no_point(monkeypatch):
    # Ollama answers 200 with an {"error": ...} body for some refusals; that is
    # a skipped point, not a model that generated at 0 tok/s.
    monkeypatch.setattr(dyno, "clear_field", lambda host=None: [])
    monkeypatch.setattr(dyno, "generate_raw",
                        lambda *a, **k: {"error": "context length exceeded"})
    assert dyno.live_measure(settle_s=0)("m", 131072) is None


@pytest.mark.parametrize("raw, expected", [
    ("", "http://localhost:11434"),
    ("box:11434", "http://box:11434"),
    ("http://box:11434/", "http://box:11434"),
    ("box", "http://box:11434"),                 # port supplied
    ("https://remote.example", "https://remote.example:11434"),
    # A server told to listen on every interface exports its BIND address into
    # the client's environment. Dialing it verbatim reaches nothing.
    ("0.0.0.0", "http://127.0.0.1:11434"),
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),
    ("http://0.0.0.0:11434", "http://127.0.0.1:11434"),
])
def test_host_resolution_repairs_what_it_is_given(monkeypatch, raw, expected):
    monkeypatch.setenv("OLLAMA_HOST", raw)
    assert dyno.default_host() == expected


def test_server_env_records_knobs_but_never_the_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_FLASH_ATTENTION", "1")
    monkeypatch.setenv("LLAMA_ARG_FIT_TARGET", "128")
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0")
    monkeypatch.setattr(dyno, "server_version", lambda host=None: "0.30.8")

    got = dyno.server_env()
    assert got["ollama_version"] == "0.30.8"
    assert got["env"]["OLLAMA_FLASH_ATTENTION"] == "1"
    assert got["env"]["LLAMA_ARG_FIT_TARGET"] == "128"
    # The host is where we dialed, not a knob the measurement ran under.
    assert "OLLAMA_HOST" not in got["env"]
