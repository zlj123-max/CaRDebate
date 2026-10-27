from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from CaRDebate.run import run_debate
from CaRDebate.utils import build_cardebate_run_kwargs, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CaRDebate in baseline, all-history, or no-history mode. "
            "Set algo_name in the algorithm YAML; all other experiment "
            "hyperparameters stay YAML-driven."
        )
    )
    parser.add_argument(
        "--config",
        "--algorithm-config",
        dest="algorithm_config",
        required=True,
        help=(
            "CaRDebate algorithm YAML; algo_name must be cardebate, "
            "cardebate_all_history, or cardebate_no_history."
        ),
    )
    parser.add_argument(
        "--common-config",
        dest="common_config",
        required=True,
        help="Common experiment/runtime YAML file.",
    )

    # Operational path controls only. Experiment conditions come from YAML.
    parser.add_argument(
        "--output-file",
        dest="output_file",
        default="debate.log",
        help="Debate log file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default=None,
        help="Directory used to save/resume checkpoints (use a separate directory per mode).",
    )
    parser.add_argument(
        "--model-output-dir",
        dest="model_output_dir",
        default=None,
        help="Directory for per-model JSONL audit logs.",
    )
    parser.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="Print the resolved YAML-derived runtime configuration before launch.",
    )
    return parser


def validate_operational_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    algorithm_path = Path(args.algorithm_config).expanduser()
    if not algorithm_path.exists():
        parser.error(f"Algorithm config not found: {algorithm_path}")

    common_path = Path(args.common_config).expanduser()
    if not common_path.exists():
        parser.error(f"Common config not found: {common_path}")


def _print_runtime_config(runtime_kwargs: Dict[str, Any]) -> None:
    printable = dict(runtime_kwargs)
    print("Resolved CaRDebate runtime configuration (YAML-derived):")
    print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_operational_args(parser, args)

    config = load_config(
        common_config=args.common_config,
        algorithm_config=args.algorithm_config,
    )
    runtime_kwargs = build_cardebate_run_kwargs(config)

    # All mode policy is resolved centrally by CaRDebate.utils.
    mode = runtime_kwargs["debate_config"]["history_mode"]
    print(f"CaRDebate experiment mode: {mode}")

    if args.print_resolved_config:
        _print_runtime_config(runtime_kwargs)

    run_debate(
        output_file=args.output_file,
        checkpoint_dir=args.checkpoint_dir,
        model_output_dir=args.model_output_dir,
        **runtime_kwargs,
    )


if __name__ == "__main__":
    main()
