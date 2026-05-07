from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import load_pickle, save_csv, save_json, save_pickle
from taska.mixing import build_mixed_predictions, collect_gap_rows_single_dataset, compute_missing_point_metrics, gap_key


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: List[Dict[str, Any]] = []
    for row in rows:
        parsed: Dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                parsed[key] = value
                continue
            txt = value.strip()
            if txt == "":
                parsed[key] = txt
                continue
            try:
                if any(ch in txt for ch in [".", "e", "E"]):
                    parsed[key] = float(txt)
                else:
                    parsed[key] = int(txt)
            except ValueError:
                if txt.lower() == "true":
                    parsed[key] = True
                elif txt.lower() == "false":
                    parsed[key] = False
                else:
                    parsed[key] = txt
        out.append(parsed)
    return out


def _gate_specs(dataset: str) -> tuple[int, Dict[str, Any]]:
    min_gap = 5 if dataset == "1/8" else 9
    gates = {
        "G0": lambda r: int(r["gap_size"]) >= min_gap,
        "G1": lambda r: int(r["gap_size"]) >= min_gap and float(r["detour_ratio"]) <= 2.0,
        "G2": lambda r: int(r["gap_size"]) >= min_gap and float(r["detour_ratio"]) <= 2.0 and float(r["start_snap_m"]) <= 60.0 and float(r["end_snap_m"]) <= 60.0,
        "G3": lambda r: int(r["gap_size"]) >= min_gap and float(r["detour_ratio"]) <= 2.0 and float(r["start_snap_m"]) <= 60.0 and float(r["end_snap_m"]) <= 60.0 and float(r["base_to_route_mean_m"]) <= 50.0,
        "G4": lambda r: int(r["gap_size"]) >= min_gap and float(r["detour_ratio"]) <= 2.0 and float(r["start_snap_m"]) <= 60.0 and float(r["end_snap_m"]) <= 60.0 and float(r["base_to_route_mean_m"]) <= 50.0 and float(r["projection_s_clamped_max_m"]) <= 80.0,
    }
    return min_gap, gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline gate sweep for route projection")
    parser.add_argument("--base-pred", required=True, type=Path)
    parser.add_argument("--new-pred", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--base-gap-metrics", required=True, type=Path)
    parser.add_argument("--new-gap-metrics", required=True, type=Path)
    parser.add_argument("--debug-gap-csv", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=["1/8", "1/16"])
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_records = load_pickle(args.input)
    gt_records = load_pickle(args.gt)
    base_pred = load_pickle(args.base_pred)
    new_pred = load_pickle(args.new_pred)
    base_gap_rows = [r for r in _read_csv(args.base_gap_metrics) if r["dataset"] == args.dataset]
    debug_rows = [r for r in _read_csv(args.debug_gap_csv) if r["dataset"] == args.dataset]
    debug_by_key = {gap_key(r): r for r in debug_rows}
    base_by_key = {gap_key(r): r for r in base_gap_rows}

    total_gap_count = len(base_gap_rows)
    total_missing_points = int(sum(int(r["missing_point_count"]) for r in base_gap_rows))
    total_error = float(sum(float(r["official_error_sum_m"]) for r in base_gap_rows))
    min_gap, gates = _gate_specs(args.dataset)

    for gate_name, gate_fn in gates.items():
        selected_keys = set()
        gate_choices: List[Dict[str, Any]] = []
        selected_error = 0.0
        selected_missing = 0
        for key, base in base_by_key.items():
            dbg = debug_by_key.get(key)
            if dbg is None:
                continue
            use_new = bool(dbg.get("applied_projection")) and gate_fn(dbg)
            if use_new:
                selected_keys.add(key)
                selected_error += float(base["official_error_sum_m"])
                selected_missing += int(base["missing_point_count"])
            gate_choices.append(
                {
                    "dataset": args.dataset,
                    "traj_id": base["traj_id"],
                    "gap_start_idx": base["gap_start_idx"],
                    "gap_end_idx": base["gap_end_idx"],
                    "gap_size": base["gap_size"],
                    "use_new": bool(use_new),
                    "base_quadrant_global": base["quadrant_global"],
                    "debug_detour_ratio": dbg.get("detour_ratio"),
                    "debug_start_snap_m": dbg.get("start_snap_m"),
                    "debug_end_snap_m": dbg.get("end_snap_m"),
                    "debug_base_to_route_mean_m": dbg.get("base_to_route_mean_m"),
                    "debug_projection_s_clamped_max_m": dbg.get("projection_s_clamped_max_m"),
                }
            )

        mixed_pred = build_mixed_predictions(base_pred, new_pred, input_records, selected_keys, args.dataset)
        pred_out = args.out_dir / f"mixed_pred_{gate_name}.pkl"
        save_pickle(pred_out, mixed_pred)
        mixed_gap_rows, _ = collect_gap_rows_single_dataset(args.dataset, input_records, mixed_pred, gt_records)
        mixed_by_key = {gap_key(r): r for r in mixed_gap_rows}
        mixed_metrics = compute_missing_point_metrics(input_records, mixed_pred, gt_records)
        base_metrics = compute_missing_point_metrics(input_records, base_pred, gt_records)

        quadrant_delta = {}
        for quadrant in ("high_official_low_shape", "high_official_high_shape", "low_official_high_shape"):
            base_q = [r for r in base_gap_rows if r["quadrant_global"] == quadrant]
            if not base_q:
                quadrant_delta[quadrant] = 0.0
                continue
            deltas = [float(mixed_by_key[gap_key(r)]["official_mae_m"] - r["official_mae_m"]) for r in base_q]
            quadrant_delta[quadrant] = float(sum(deltas) / len(deltas))

        report = {
            "dataset": args.dataset,
            "gate_name": gate_name,
            "min_gap": min_gap,
            "selected_gap_count": int(len(selected_keys)),
            "selected_missing_point_count": int(selected_missing),
            "selected_gap_share": float(len(selected_keys) / max(total_gap_count, 1)),
            "selected_error_share": float(selected_error / max(total_error, 1e-9)),
            "mixed_global_metrics": mixed_metrics,
            "mae_delta": float(mixed_metrics["mae"] - base_metrics["mae"]),
            "p95_delta": float(mixed_metrics["p95"] - base_metrics["p95"]),
            "baseline_quadrant_mae_delta": quadrant_delta,
        }
        save_json(args.out_dir / f"gate_report_{gate_name}.json", report)
        save_csv(args.out_dir / f"gate_choices_{gate_name}.csv", gate_choices)
    print(f"Gate sweep outputs saved: {args.out_dir}")


if __name__ == "__main__":
    main()
