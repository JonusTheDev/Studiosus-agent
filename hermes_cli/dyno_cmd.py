"""``hermes dyno`` implementation — benchmark a local Ollama model's power curve.

Resolves the local endpoint + model, refuses early if the model isn't pulled,
runs the sweep (agent/governor_dyno.py), prints the curve, and saves it so the
Governor (agent/governor.py) can cap num_ctx to the largest context the GPU
actually holds. See ``hermes_cli/subcommands/dyno.py`` for the parser.
"""

from __future__ import annotations


def cmd_dyno(args) -> int:
    from agent import governor, governor_dyno
    from agent.model_metadata import detect_local_server_type, is_local_endpoint

    # ── Resolve endpoint + api_key ──
    base_url = getattr(args, "host", None) or ""
    api_key = ""
    if not base_url:
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider
            runtime = resolve_runtime_provider()
            base_url = (runtime.get("base_url") or "").strip()
            api_key = runtime.get("api_key") or ""
            if not isinstance(api_key, str):
                api_key = ""
        except Exception as exc:
            print(f"Could not resolve the current endpoint: {exc}")
            print("Pass one explicitly, e.g.  --host http://127.0.0.1:11434/v1")
            return 1

    # ── Resolve model ──
    model = getattr(args, "model", None)
    if not model:
        try:
            from hermes_cli.config import load_config
            model_cfg = load_config().get("model")
            if isinstance(model_cfg, dict):
                model = (model_cfg.get("default") or "").strip()
            elif isinstance(model_cfg, str):
                model = model_cfg.strip()
        except Exception:
            model = None
    if not model:
        print("No model given and none configured.")
        print("Usage:  hermes dyno <model> [--host http://127.0.0.1:11434/v1]")
        return 1

    # ── Validate it's a local Ollama endpoint ──
    if not base_url or not is_local_endpoint(base_url):
        print(f"The dyno benchmarks a LOCAL Ollama endpoint; "
              f"'{base_url or '(none)'}' is not local.")
        print("Point model.base_url at your local Ollama, or pass --host.")
        return 1
    try:
        if detect_local_server_type(base_url, api_key=api_key) != "ollama":
            print(f"{base_url} does not look like an Ollama server "
                  f"(no /api/tags). The dyno needs Ollama's native API.")
            return 1
    except Exception as exc:
        print(f"Could not reach {base_url}: {exc}")
        return 1

    # ── Refuse-if-absent up front (before a multi-minute run) ──
    installed = governor.installed_models(base_url, api_key=api_key)
    if installed is not None and not governor._model_installed(model, installed):
        have = ", ".join(sorted(installed)) or "nothing"
        print(f"Model '{model}' is not installed on {base_url} (it has: {have}).")
        print(f"Pull it first, e.g.  ollama pull {model}")
        return 1

    # ── Parse the sweep points ──
    ctx_points = governor_dyno.DEFAULT_CTX_POINTS
    if getattr(args, "ctx", None):
        try:
            ctx_points = tuple(int(x) for x in str(args.ctx).split(",") if x.strip())
        except ValueError:
            print("--ctx must be a comma-separated list of integers, "
                  "e.g. 8192,16384,32768,65536")
            return 1
        if not ctx_points:
            print("--ctx produced no points.")
            return 1

    repeats = max(1, int(getattr(args, "repeats", 1) or 1))
    gpu = governor.read_gpu(use_cache=False)
    server_env = {"ollama_version": governor.server_version(base_url, api_key=api_key)}

    print(f"⚙️  Dyno: benchmarking {model} on {base_url}")
    if gpu and gpu.get("name"):
        print("    GPU: %s (%.0f MB VRAM)" % (gpu.get("name"), gpu.get("vram_total_mb") or 0))
    print("    sweeping num_ctx = %s  (repeats=%d)"
          % (", ".join(str(c) for c in ctx_points), repeats))
    print("    each point loads the model and times a real generation — "
          "this takes a few minutes.\n")

    power = governor_dyno.run_dyno(
        model,
        governor_dyno.live_measure(base_url, api_key=api_key),
        base_url=base_url,
        ctx_points=ctx_points,
        gpu=gpu,
        server_env=server_env,
        repeats=repeats,
        write=not getattr(args, "no_write", False),
    )

    print(governor_dyno.format_power(power, model))
    if not power.get("curve"):
        return 1
    if not getattr(args, "no_write", False):
        print(f"\nSaved. The Governor will now cap num_ctx to this measured curve "
              f"for {model}.")

    if getattr(args, "json", False):
        import json as _json
        print("\n" + _json.dumps(power, indent=2))
    return 0
