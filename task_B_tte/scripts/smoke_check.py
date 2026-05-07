from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def check_pair(input_path: Path, pred_path: Path, label: str = "task_b") -> list[str]:
    issues: list[str] = []
    input_records = load_pickle(input_path)
    pred_records = load_pickle(pred_path)

    if not isinstance(input_records, list):
        return [f"{label}: input object must be list"]
    if not isinstance(pred_records, list):
        return [f"{label}: prediction object must be list"]
    if len(input_records) != len(pred_records):
        return [f"{label}: record count mismatch input={len(input_records)} pred={len(pred_records)}"]

    seen_ids: set[int] = set()
    for idx, (inp, pred) in enumerate(zip(input_records, pred_records)):
        if not isinstance(inp, dict):
            issues.append(f"{label}: input item #{idx} is not dict")
            continue
        if not isinstance(pred, dict):
            issues.append(f"{label}: pred item #{idx} is not dict")
            continue

        if "traj_id" not in inp:
            issues.append(f"{label}: input item #{idx} missing traj_id")
            continue
        if "traj_id" not in pred or "travel_time" not in pred:
            issues.append(f"{label}: pred item #{idx} missing traj_id/travel_time")
            continue

        input_id = int(inp["traj_id"])
        pred_id = int(pred["traj_id"])
        if input_id != pred_id:
            issues.append(f"{label}: traj_id mismatch at index {idx}: input={input_id} pred={pred_id}")

        if pred_id in seen_ids:
            issues.append(f"{label}: duplicate traj_id={pred_id}")
        seen_ids.add(pred_id)

        try:
            travel_time = float(pred["travel_time"])
        except Exception:
            issues.append(f"{label}: travel_time is not numeric for traj_id={pred_id}")
            continue

        if not math.isfinite(travel_time):
            issues.append(f"{label}: non-finite travel_time for traj_id={pred_id}")
        elif travel_time <= 0.0:
            issues.append(f"{label}: non-positive travel_time for traj_id={pred_id}")

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke check Task B predictions without GT")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--label", default="task_b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issues = check_pair(args.input, args.pred, label=args.label)
    if issues:
        print("Smoke check failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Smoke check passed.")


if __name__ == "__main__":
    main()
