from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import load_pickle, save_csv, save_json, save_pickle
from taska.mixing import (
    build_mixed_predictions,
    collect_gap_rows_single_dataset,
    compute_missing_point_metrics,
    gap_key,
    infer_dataset_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix predictions by gap using offline decision rules")
    parser.add_argument("--base-pred", required=True, type=Path)
    parser.add_argument("--new-pred", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["oracle"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = infer_dataset_name(args.input)
    input_records = load_pickle(args.input)
    gt_records = load_pickle(args.gt)
    base_pred = load_pickle(args.base_pred)
    new_pred = load_pickle(args.new_pred)

    base_rows, base_thr = collect_gap_rows_single_dataset(dataset_name, input_records, base_pred, gt_records)
    new_rows, _ = collect_gap_rows_single_dataset(dataset_name, input_records, new_pred, gt_records)
    new_by_key = {gap_key(r): r for r in new_rows}

    choices: List[Dict[str, Any]] = []
    selected_keys: set[tuple[Any, ...]] = set()
    total_base_error = float(sum(r["official_error_sum_m"] for r in base_rows))
    chosen_new_error_sum = 0.0
    chosen_new_gap_count = 0
    chosen_new_missing = 0
    quadrant_selected: Dict[str, int] = {}
    quadrant_total: Dict[str, int] = {}

    for base in base_rows:
        key = gap_key(base)
        new = new_by_key[key]
        quadrant = base["quadrant_dataset"]
        quadrant_total[quadrant] = quadrant_total.get(quadrant, 0) + 1
        choose_new = bool(new["official_mae_m"] < base["official_mae_m"])
        if choose_new:
            selected_keys.add(key)
            chosen_new_error_sum += float(base["official_error_sum_m"])
            chosen_new_gap_count += 1
            chosen_new_missing += int(base["missing_point_count"])
            quadrant_selected[quadrant] = quadrant_selected.get(quadrant, 0) + 1
        choices.append(
            {
                "dataset": dataset_name,
                "traj_id": base["traj_id"],
                "gap_start_idx": base["gap_start_idx"],
                "gap_end_idx": base["gap_end_idx"],
                "gap_size": base["gap_size"],
                "choice": "new" if choose_new else "base",
                "base_official_mae_m": base["official_mae_m"],
                "new_official_mae_m": new["official_mae_m"],
                "delta_official_mae_m": float(new["official_mae_m"] - base["official_mae_m"]),
                "base_shape_symmetric_m": base["shape_symmetric_m"],
                "new_shape_symmetric_m": new["shape_symmetric_m"],
            }
        )

    mixed_pred = build_mixed_predictions(base_pred, new_pred, input_records, selected_keys, dataset_name)
    save_pickle(args.out, mixed_pred)
    mixed_metrics = compute_missing_point_metrics(input_records, mixed_pred, gt_records)
    base_metrics = compute_missing_point_metrics(input_records, base_pred, gt_records)

    quadrant_selection_ratio = {
        quadrant: float(quadrant_selected.get(quadrant, 0) / max(quadrant_total.get(quadrant, 1), 1))
        for quadrant in sorted(quadrant_total.keys())
    }
    report = {
        "dataset": dataset_name,
        "mode": args.mode,
        "base_metrics": base_metrics,
        "mixed_metrics": mixed_metrics,
        "mix_gain_mae_m": float(mixed_metrics["mae"] - base_metrics["mae"]),
        "mix_gain_p95_m": float(mixed_metrics["p95"] - base_metrics["p95"]),
        "chosen_new_gap_count": int(chosen_new_gap_count),
        "chosen_new_missing_point_count": int(chosen_new_missing),
        "chosen_new_error_share": float(chosen_new_error_sum / max(total_base_error, 1e-9)),
        "baseline_quadrant_selection_ratio": quadrant_selection_ratio,
        "dataset_thresholds": base_thr,
    }

    args.report_out.mkdir(parents=True, exist_ok=True)
    save_json(args.report_out / "mix_report.json", report)
    save_csv(args.report_out / "mixed_gap_choices.csv", choices)
    print(f"Mixed prediction saved: {args.out}")
    print(f"Mix report saved: {args.report_out}")


if __name__ == "__main__":
    main()
