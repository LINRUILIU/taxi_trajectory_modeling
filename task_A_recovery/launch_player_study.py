from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch two-round Task A player study: dataset 1/8 then 1/16 with global progress numbering."
    )
    parser.add_argument("--cases-per-dataset", type=int, default=40, help="Representative case count per dataset (recommended 30~50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--session-prefix", type=str, default="")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable path")
    parser.add_argument("--script", type=Path, default=Path("task_A_recovery/interactive_game.py"), help="Interactive game script path")
    parser.add_argument("--no-ui", action="store_true", help="Pass --no-ui to each round (for smoke checks)")
    parser.add_argument("--common-extra-args", nargs="*", default=[], help="Extra args passed to both rounds")
    return parser.parse_args()


def _build_command(
    python_exe: Path,
    script: Path,
    dataset: str,
    cases_per_dataset: int,
    seed: int,
    session_name: str,
    round_label: str,
    progress_offset: int,
    progress_total: int,
    no_ui: bool,
    common_extra_args: List[str],
) -> List[str]:
    cmd = [
        str(python_exe),
        str(script),
        "--dataset",
        dataset,
        "--case-pool-size",
        str(cases_per_dataset),
        "--seed",
        str(seed),
        "--session-name",
        session_name,
        "--round-label",
        round_label,
        "--progress-offset",
        str(progress_offset),
        "--progress-total",
        str(progress_total),
    ]
    if no_ui:
        cmd.append("--no-ui")
    cmd.extend(common_extra_args)
    return cmd


def main() -> None:
    args = parse_args()

    if args.cases_per_dataset < 10 or args.cases_per_dataset > 50:
        raise ValueError("--cases-per-dataset must be in [10, 50] for the requested study setup.")

    if not args.script.exists():
        raise FileNotFoundError(f"interactive script not found: {args.script}")

    total_cases = int(args.cases_per_dataset) * 2

    if args.session_prefix.strip():
        prefix = args.session_prefix.strip()
    else:
        prefix = datetime.now().strftime("player_study_%Y%m%d_%H%M%S")

    rounds = [
        {
            "dataset": "8",
            "session_name": f"{prefix}_r8",
            "round_label": f"round_1_of_2_dataset_1_8_{args.cases_per_dataset}cases",
            "offset": 0,
        },
        {
            "dataset": "16",
            "session_name": f"{prefix}_r16",
            "round_label": f"round_2_of_2_dataset_1_16_{args.cases_per_dataset}cases",
            "offset": int(args.cases_per_dataset),
        },
    ]

    print("Player study launch plan:")
    print(f"- cases per dataset: {args.cases_per_dataset}")
    print(f"- total cases: {total_cases}")
    print(f"- seed: {args.seed}")
    print(f"- session prefix: {prefix}")
    print("- order: 1/8 then 1/16")

    for i, info in enumerate(rounds, start=1):
        cmd = _build_command(
            python_exe=args.python,
            script=args.script,
            dataset=str(info["dataset"]),
            cases_per_dataset=int(args.cases_per_dataset),
            seed=int(args.seed),
            session_name=str(info["session_name"]),
            round_label=str(info["round_label"]),
            progress_offset=int(info["offset"]),
            progress_total=total_cases,
            no_ui=bool(args.no_ui),
            common_extra_args=list(args.common_extra_args),
        )

        print("")
        print(f"[Round {i}/2] launching dataset {info['dataset']} ...")
        subprocess.run(cmd, check=True)
        print(f"[Round {i}/2] finished.")

    print("")
    print("All rounds finished.")
    print("Session names:")
    for info in rounds:
        print(f"- {info['session_name']}")


if __name__ == "__main__":
    main()
