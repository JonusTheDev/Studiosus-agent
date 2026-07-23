"""``hermes dyno`` subcommand parser.

Benchmarks a local Ollama model's VRAM power curve for the Governor
(agent/governor.py). Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_dyno_parser(subparsers, *, cmd_dyno: Callable) -> None:
    """Attach the ``dyno`` subcommand to ``subparsers``."""
    dyno_parser = subparsers.add_parser(
        "dyno",
        help="Benchmark a local Ollama model's VRAM power curve (for the Governor)",
        description=(
            "Put a local Ollama model on the rollers: sweep it across context "
            "sizes, measuring how much of it stays on the GPU (Ollama's own "
            "size_vram/size) and how fast it truly generates (from Ollama's "
            "timings). The measured power curve lets the Governor cap num_ctx to "
            "the largest context that holds the GPU — the best your card can "
            "actually run, instead of blindly requesting the full window and "
            "spilling into system RAM. Runs against your configured local "
            "endpoint (or --host). Each point loads the model and times a real "
            "generation, so a full sweep takes a few minutes."
        ),
    )
    dyno_parser.add_argument(
        "model", nargs="?", default=None,
        help="Model to benchmark (default: your configured model)",
    )
    dyno_parser.add_argument(
        "--host", "--base-url", dest="host", default=None,
        help="Ollama endpoint to benchmark (default: the configured base_url)",
    )
    dyno_parser.add_argument(
        "--ctx", default=None,
        help="Comma-separated num_ctx points to sweep (default: 4096,8192,16384,32768,65536,131072)",
    )
    dyno_parser.add_argument(
        "--repeats", type=int, default=1,
        help="Roll each point N times and record the mean — steadier numbers (default: 1)",
    )
    dyno_parser.add_argument(
        "--no-write", action="store_true",
        help="Measure and print, but do not save the curve",
    )
    dyno_parser.add_argument(
        "--json", action="store_true",
        help="Also print the raw power block as JSON",
    )
    dyno_parser.set_defaults(func=cmd_dyno)
