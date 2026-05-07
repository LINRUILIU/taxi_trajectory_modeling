from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import joblib

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import load_pickle, save_csv, save_json, save_pickle
from taska.selector_features import (
    MODEL_FEATURE_COLUMNS,
    build_selector_decisions,
    build_selector_feature_rows,
    feature_matrix,
    load_feature_spec,
    mix_predictions_with_selector,
    read_typed_csv,
    validate_feature_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix b28 and route projection predictions with a trained selector")
    parser.add_argument("--dataset", required=True, choices=["1/8", "1/16"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--base-pred", required=True, type=Path)
    parser.add_argument("--route-pred", required=True, type=Path)
    parser.add_argument("--debug-csv", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--feature-columns", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--decision-out", required=True, type=Path)
    parser.add_argument("--metadata-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_out = args.metadata_out or args.decision_out.with_name(f"{args.decision_out.stem}.metadata.json")

    input_records = load_pickle(args.input)
    base_pred = load_pickle(args.base_pred)
    route_pred = load_pickle(args.route_pred)
    debug_rows = [r for r in read_typed_csv(args.debug_csv) if str(r["dataset"]) == args.dataset]
    feature_rows = build_selector_feature_rows(args.dataset, input_records, base_pred, route_pred, debug_rows)

    spec = load_feature_spec(args.feature_columns)
    feature_columns = list(spec["feature_columns"])
    validate_feature_columns(MODEL_FEATURE_COLUMNS, feature_columns)
    actual_cols = [c for c in feature_rows[0].keys() if c.startswith("feature_")]
    validate_feature_columns(feature_columns, actual_cols)
    x, matrix_stats = feature_matrix(feature_rows, feature_columns)

    model = joblib.load(args.model)
    probs = model.predict_proba(x)[:, 1]
    selected_keys, decision_rows, selection_meta = build_selector_decisions(feature_rows, probs, args.threshold)
    mixed_pred = mix_predictions_with_selector(args.dataset, input_records, base_pred, route_pred, selected_keys)

    save_pickle(args.out, mixed_pred)
    save_csv(args.decision_out, decision_rows)
    metadata: Dict[str, Any] = {
        "dataset": args.dataset,
        "threshold": float(args.threshold),
        "model_path": str(args.model),
        "feature_columns_path": str(args.feature_columns),
        "feature_schema_hash": spec["feature_schema_hash"],
        "feature_nan_count": matrix_stats["feature_nan_count"],
        "feature_clip_count": matrix_stats["feature_clip_count"],
        "gap_count": selection_meta["gap_count"],
        "selected_gap_count": selection_meta["selected_gap_count"],
        "selection_rate": selection_meta["selection_rate"],
        "selected_missing_point_count": selection_meta["selected_missing_point_count"],
        "prob_mean": selection_meta["prob_mean"],
        "prob_p50": selection_meta["prob_p50"],
        "prob_p90": selection_meta["prob_p90"],
        "prob_p95": selection_meta["prob_p95"],
        "drift_note": _drift_note(selection_meta["selection_rate"], matrix_stats, x.shape[0] * x.shape[1]),
    }
    save_json(metadata_out, metadata)
    print(f"Selector mixed prediction saved: {args.out}")
    print(f"Selector decisions saved: {args.decision_out}")
    print(f"Selector metadata saved: {metadata_out}")


def _drift_note(selection_rate: float, matrix_stats: Dict[str, int], feature_value_count: int) -> str:
    notes = []
    if selection_rate < 0.02 or selection_rate > 0.30:
        notes.append("selection_rate_outside_expected_band")
    if feature_value_count > 0 and matrix_stats["feature_nan_count"] / feature_value_count > 0.10:
        notes.append("feature_nan_rate_high")
    if matrix_stats["feature_clip_count"] > 0:
        notes.append("feature_clip_applied")
    return ",".join(notes) if notes else "ok"


if __name__ == "__main__":
    main()
