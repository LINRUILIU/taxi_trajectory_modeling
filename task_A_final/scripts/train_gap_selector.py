from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`",
    category=UserWarning,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import load_pickle, save_csv, save_json, save_pickle
from taska.mixing import build_mixed_predictions, compute_missing_point_metrics, gap_key
from taska import selector_features as sf


DATASET_CODE = {"1/8": 8, "1/16": 16}
GAP_BUCKET_CODE = {"1-2": 0, "3-4": 1, "5-7": 2, "8+": 3, "1-4": 10, "5-8": 11, "9-15": 12, "16+": 13}
FALLBACK_REASON_CODE = {
    "applied": 0,
    "route_not_found": 1,
    "degenerate_anchor": 2,
    "empty_route": 3,
    "projection_failed": 4,
    "base_nan": 5,
    "other": 99,
}
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
MODEL_FEATURE_COLUMNS = [
    "feature_dataset_encoded",
    "feature_gap_size",
    "feature_gap_bucket_encoded",
    "feature_delta_t_sec",
    "feature_anchor_direct_m",
    "feature_anchor_speed_mps",
    "feature_route_found",
    "feature_applied_projection",
    "feature_route_length_m",
    "feature_detour_ratio",
    "feature_start_snap_m",
    "feature_end_snap_m",
    "feature_max_snap_m",
    "feature_base_to_route_mean_m",
    "feature_base_to_route_p95_m",
    "feature_projection_s_monotonic_violations",
    "feature_projection_s_clamped_mean_m",
    "feature_projection_s_clamped_max_m",
    "feature_route_sample_vs_base_mean_m",
    "feature_fallback_reason_encoded",
    "feature_pred_pair_mean_dist_m",
    "feature_pred_pair_p95_dist_m",
    "feature_pred_pair_max_dist_m",
    "feature_b28_missing_path_len_m",
    "feature_route_missing_path_len_m",
    "feature_pred_path_length_ratio",
    "feature_start_direction_diff_deg",
    "feature_end_direction_diff_deg",
    "feature_max_turn_b28_deg",
    "feature_max_turn_route_deg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leakage-free OOF gap selector for route projection")
    parser.add_argument("--mode", choices=["build-dataset", "oof", "sweep", "train-full", "all"], default="all")
    parser.add_argument("--base-gap-metrics", type=Path, default=ROOT / "runs" / "b28_compat_full" / "analysis" / "gap_metrics.csv")
    parser.add_argument("--route-gap-metrics", type=Path, default=ROOT / "runs" / "route_projection_full" / "analysis" / "gap_metrics.csv")
    parser.add_argument("--debug-8", type=Path, default=ROOT / "runs" / "route_projection_full_debug" / "debug_8.csv")
    parser.add_argument("--debug-16", type=Path, default=ROOT / "runs" / "route_projection_full_debug" / "debug_16.csv")
    parser.add_argument("--base-pred-8", type=Path, default=ROOT / "runs" / "b28_compat_full" / "pred_8.pkl")
    parser.add_argument("--base-pred-16", type=Path, default=ROOT / "runs" / "b28_compat_full" / "pred_16.pkl")
    parser.add_argument("--route-pred-8", type=Path, default=ROOT / "runs" / "route_projection_full" / "pred_8.pkl")
    parser.add_argument("--route-pred-16", type=Path, default=ROOT / "runs" / "route_projection_full" / "pred_16.pkl")
    parser.add_argument("--input-8", type=Path, default=ROOT / "val_input_8.pkl")
    parser.add_argument("--input-16", type=Path, default=ROOT / "val_input_16.pkl")
    parser.add_argument("--gt", type=Path, default=ROOT / "val_gt.pkl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "selector_oof")
    parser.add_argument("--oof-dir", type=Path, default=ROOT / "runs" / "selector_oof")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260501)
    return parser.parse_args()


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
            lower = txt.lower()
            if lower == "true":
                parsed[key] = True
                continue
            if lower == "false":
                parsed[key] = False
                continue
            try:
                if any(ch in txt for ch in [".", "e", "E"]):
                    parsed[key] = float(txt)
                else:
                    parsed[key] = int(txt)
            except ValueError:
                parsed[key] = txt
        out.append(parsed)
    return out


def _ensure_no_duplicate_keys(rows: Sequence[Dict[str, Any]], label: str) -> None:
    seen: set[Tuple[Any, ...]] = set()
    for row in rows:
        key = gap_key(row)
        if key in seen:
            raise ValueError(f"Duplicate gap key in {label}: {key}")
        seen.add(key)


def _polyline_length_m(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return 0.0
    seg = haversine_meters(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
    return float(np.sum(np.asarray(seg, dtype=np.float64)))


def _bearing_deg(a: np.ndarray, b: np.ndarray) -> float:
    lon1 = math.radians(float(a[0]))
    lat1 = math.radians(float(a[1]))
    lon2 = math.radians(float(b[0]))
    lat2 = math.radians(float(b[1]))
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    angle = math.degrees(math.atan2(y, x))
    return (angle + 360.0) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _start_end_bearings(polyline: np.ndarray) -> tuple[float, float]:
    if polyline.shape[0] < 2:
        return 0.0, 0.0
    return _bearing_deg(polyline[0], polyline[1]), _bearing_deg(polyline[-2], polyline[-1])


def _max_turn_deg(polyline: np.ndarray) -> float:
    if polyline.shape[0] < 3:
        return 0.0
    turns: List[float] = []
    for i in range(1, polyline.shape[0] - 1):
        b1 = _bearing_deg(polyline[i - 1], polyline[i])
        b2 = _bearing_deg(polyline[i], polyline[i + 1])
        turns.append(_angle_diff_deg(b1, b2))
    return float(max(turns)) if turns else 0.0


def _prediction_gap_features(
    pred_records: Sequence[Dict[str, Any]],
    input_records: Sequence[Dict[str, Any]],
    dataset_name: str,
) -> Dict[Tuple[Any, ...], Dict[str, float]]:
    pred_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in pred_records}
    out: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    for item in input_records:
        traj_id = int(item["traj_id"])
        mask = np.asarray(item["mask"], dtype=bool)
        known_idx = np.where(mask)[0]
        pred = pred_by_id[traj_id]
        for pos in range(known_idx.size - 1):
            s = int(known_idx[pos])
            e = int(known_idx[pos + 1])
            if e - s <= 1:
                continue
            key = (dataset_name, traj_id, s, e)
            missing_poly = pred[s + 1 : e]
            full_gap_poly = pred[s : e + 1]
            start_dir, end_dir = _start_end_bearings(full_gap_poly)
            out[key] = {
                "missing_path_len_m": _polyline_length_m(missing_poly),
                "start_dir_deg": start_dir,
                "end_dir_deg": end_dir,
                "max_turn_deg": _max_turn_deg(full_gap_poly),
            }
    return out


def _pairwise_gap_distance_features(
    base_pred_records: Sequence[Dict[str, Any]],
    route_pred_records: Sequence[Dict[str, Any]],
    input_records: Sequence[Dict[str, Any]],
    dataset_name: str,
) -> Dict[Tuple[Any, ...], Dict[str, float]]:
    base_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in base_pred_records}
    route_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in route_pred_records}
    base_gap = _prediction_gap_features(base_pred_records, input_records, dataset_name)
    route_gap = _prediction_gap_features(route_pred_records, input_records, dataset_name)
    out: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    for item in input_records:
        traj_id = int(item["traj_id"])
        mask = np.asarray(item["mask"], dtype=bool)
        known_idx = np.where(mask)[0]
        base_pred = base_by_id[traj_id]
        route_pred = route_by_id[traj_id]
        for pos in range(known_idx.size - 1):
            s = int(known_idx[pos])
            e = int(known_idx[pos + 1])
            if e - s <= 1:
                continue
            key = (dataset_name, traj_id, s, e)
            base_missing = base_pred[s + 1 : e]
            route_missing = route_pred[s + 1 : e]
            if base_missing.shape[0] != route_missing.shape[0]:
                raise ValueError(f"Prediction length mismatch on gap {key}")
            if base_missing.shape[0] == 0:
                dists = np.zeros(0, dtype=np.float64)
            else:
                dists = np.asarray(
                    haversine_meters(
                        base_missing[:, 0],
                        base_missing[:, 1],
                        route_missing[:, 0],
                        route_missing[:, 1],
                    ),
                    dtype=np.float64,
                )
            base_feat = base_gap[key]
            route_feat = route_gap[key]
            base_len = float(base_feat["missing_path_len_m"])
            route_len = float(route_feat["missing_path_len_m"])
            out[key] = {
                "feature_pred_pair_mean_dist_m": float(np.mean(dists)) if dists.size else 0.0,
                "feature_pred_pair_p95_dist_m": float(np.percentile(dists, 95)) if dists.size else 0.0,
                "feature_pred_pair_max_dist_m": float(np.max(dists)) if dists.size else 0.0,
                "feature_b28_missing_path_len_m": base_len,
                "feature_route_missing_path_len_m": route_len,
                "feature_pred_path_length_ratio": float(route_len / max(base_len, 1e-6)),
                "feature_start_direction_diff_deg": _angle_diff_deg(base_feat["start_dir_deg"], route_feat["start_dir_deg"]),
                "feature_end_direction_diff_deg": _angle_diff_deg(base_feat["end_dir_deg"], route_feat["end_dir_deg"]),
                "feature_max_turn_b28_deg": float(base_feat["max_turn_deg"]),
                "feature_max_turn_route_deg": float(route_feat["max_turn_deg"]),
            }
    return out


def _make_bucket_spec(values: Sequence[float]) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float64), ["all"]
    uniq = np.unique(arr)
    if uniq.size <= 6:
        edges = np.concatenate([uniq, [uniq[-1] + 1e-6]])
        labels = []
        for i in range(len(edges) - 1):
            labels.append(f"[{edges[i]:.3f}, {edges[i + 1]:.3f})")
        return edges, labels
    quantiles = np.unique(np.quantile(arr, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    if quantiles.size < 2:
        quantiles = np.asarray([arr.min(), arr.max() + 1e-6], dtype=np.float64)
    else:
        quantiles[-1] = quantiles[-1] + 1e-6
    labels = []
    for i in range(len(quantiles) - 1):
        labels.append(f"[{quantiles[i]:.3f}, {quantiles[i + 1]:.3f})")
    return quantiles, labels


def _bucketize(value: float, edges: np.ndarray, labels: list[str]) -> str:
    if not np.isfinite(value):
        return "nan"
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    idx = max(0, min(idx, len(labels) - 1))
    return labels[idx]


def build_dataset(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    base_gap_rows = _read_csv(args.base_gap_metrics)
    route_gap_rows = _read_csv(args.route_gap_metrics)
    debug8_rows = sf.read_typed_csv(args.debug_8)
    debug16_rows = sf.read_typed_csv(args.debug_16)
    _ensure_no_duplicate_keys(base_gap_rows, "base gap metrics")
    _ensure_no_duplicate_keys(route_gap_rows, "route gap metrics")
    sf.ensure_no_duplicate_keys(debug8_rows, "1/8 debug rows")
    sf.ensure_no_duplicate_keys(debug16_rows, "1/16 debug rows")

    base_by_key = {gap_key(r): r for r in base_gap_rows}
    route_by_key = {gap_key(r): r for r in route_gap_rows}

    input8 = load_pickle(args.input_8)
    input16 = load_pickle(args.input_16)
    base_pred8 = load_pickle(args.base_pred_8)
    base_pred16 = load_pickle(args.base_pred_16)
    route_pred8 = load_pickle(args.route_pred_8)
    route_pred16 = load_pickle(args.route_pred_16)
    feature_rows = sf.build_selector_feature_rows("1/8", input8, base_pred8, route_pred8, debug8_rows)
    feature_rows.extend(sf.build_selector_feature_rows("1/16", input16, base_pred16, route_pred16, debug16_rows))

    rows: List[Dict[str, Any]] = []
    for feature_row in feature_rows:
        key = gap_key(feature_row)
        base = base_by_key.get(key)
        route = route_by_key.get(key)
        if base is None or route is None:
            raise ValueError(f"Missing gap metrics for key {key}")
        row = dict(feature_row)
        row["label_use_route_projection"] = int(float(route["official_mae_m"]) + 5.0 < float(base["official_mae_m"]))
        row["label_delta_official_mae_m"] = float(route["official_mae_m"]) - float(base["official_mae_m"])
        row["label_bad_route_projection"] = int(float(route["official_mae_m"]) > float(base["official_mae_m"]) + 10.0)
        row["eval_b28_official_mae_m"] = float(base["official_mae_m"])
        row["eval_route_projection_official_mae_m"] = float(route["official_mae_m"])
        row["eval_b28_shape_symmetric_m"] = float(base["shape_symmetric_m"])
        row["eval_route_projection_shape_symmetric_m"] = float(route["shape_symmetric_m"])
        row["eval_shape_delta_m"] = float(route["shape_symmetric_m"]) - float(base["shape_symmetric_m"])
        row["eval_b28_quadrant_global"] = str(base["quadrant_global"])
        row["eval_b28_quadrant_dataset"] = str(base["quadrant_dataset"])
        row["eval_b28_quadrant_gap_bucket"] = str(base["quadrant_gap_bucket"])
        row["eval_route_quadrant_global"] = str(route["quadrant_global"])
        row["eval_route_quadrant_dataset"] = str(route["quadrant_dataset"])
        row["eval_route_quadrant_gap_bucket"] = str(route["quadrant_gap_bucket"])
        rows.append(row)

    rows.sort(key=lambda r: (r["dataset"], r["traj_id"], r["gap_start_idx"], r["gap_end_idx"]))
    meta = {
        "row_count": len(rows),
        "dataset_counts": {
            dataset: int(sum(1 for r in rows if r["dataset"] == dataset))
            for dataset in sorted({r["dataset"] for r in rows})
        },
    }
    return rows, {"dataset": meta}


def make_fold_assignments(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    traj_ids = np.asarray(sorted({int(r["traj_id"]) for r in rows}), dtype=np.int64)
    splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
    fold_by_traj: Dict[int, int] = {}
    for fold_id, (_, valid_idx) in enumerate(splitter.split(traj_ids)):
        for idx in valid_idx:
            fold_by_traj[int(traj_ids[idx])] = int(fold_id)
    if len(fold_by_traj) != len(traj_ids):
        raise ValueError("Fold assignment incomplete")
    return [{"traj_id": tid, "fold_id": fold_by_traj[tid]} for tid in sorted(fold_by_traj)]


def build_feature_bucket_lift(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    target_cols = [
        "feature_base_to_route_mean_m",
        "feature_detour_ratio",
        "feature_pred_pair_mean_dist_m",
        "feature_projection_s_clamped_max_m",
        "feature_gap_size",
    ]
    out: List[Dict[str, Any]] = []
    for dataset in sorted({str(r["dataset"]) for r in rows}):
        dataset_rows = [r for r in rows if str(r["dataset"]) == dataset]
        for col in target_cols:
            edges, labels = _make_bucket_spec([float(r[col]) for r in dataset_rows])
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for row in dataset_rows:
                bucket = _bucketize(float(row[col]), edges, labels)
                grouped.setdefault(bucket, []).append(row)
            for bucket, bucket_rows in grouped.items():
                total_missing = float(sum(float(r["missing_point_count"]) for r in bucket_rows))
                win_weight = float(sum(float(r["missing_point_count"]) * int(r["label_use_route_projection"]) for r in bucket_rows))
                out.append(
                    {
                        "dataset": dataset,
                        "feature_name": col,
                        "bucket": bucket,
                        "gap_count": len(bucket_rows),
                        "missing_point_count": int(total_missing),
                        "route_win_rate_gap": float(np.mean([int(r["label_use_route_projection"]) for r in bucket_rows])),
                        "route_win_rate_weighted": float(win_weight / max(total_missing, 1e-9)),
                        "mean_delta_official_mae_m": float(np.mean([float(r["label_delta_official_mae_m"]) for r in bucket_rows])),
                        "weighted_delta_official_mae_m": float(
                            sum(float(r["missing_point_count"]) * float(r["label_delta_official_mae_m"]) for r in bucket_rows)
                            / max(total_missing, 1e-9)
                        ),
                    }
                )
    out.sort(key=lambda r: (r["dataset"], r["feature_name"], r["bucket"]))
    return out


def _load_dataset_rows(path: Path) -> List[Dict[str, Any]]:
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"Dataset CSV is empty: {path}")
    return rows


def _load_fold_map(path: Path) -> Dict[int, int]:
    rows = _read_csv(path)
    return {int(r["traj_id"]): int(r["fold_id"]) for r in rows}


def _fit_hist_gb_classifier(
    clf: HistGradientBoostingClassifier,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> HistGradientBoostingClassifier:
    import sklearn.ensemble._hist_gradient_boosting.binning as hgb_binning

    original_parallel = hgb_binning.Parallel
    hgb_binning.Parallel = lambda *args, **kwargs: joblib.Parallel(n_jobs=1)
    try:
        clf.fit(x, y, sample_weight=sample_weight)
    finally:
        hgb_binning.Parallel = original_parallel
    return clf


def run_oof(rows: Sequence[Dict[str, Any]], fold_map: Dict[int, int], args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    predictions: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    feature_cols = list(sf.MODEL_FEATURE_COLUMNS)
    for dataset in ("1/8", "1/16"):
        ds_rows = [r for r in rows if str(r["dataset"]) == dataset]
        x, matrix_stats = sf.feature_matrix(ds_rows, feature_cols)
        y = np.asarray([int(r["label_use_route_projection"]) for r in ds_rows], dtype=np.int32)
        w = np.asarray([float(r["missing_point_count"]) for r in ds_rows], dtype=np.float64)
        folds = np.asarray([int(fold_map[int(r["traj_id"])]) for r in ds_rows], dtype=np.int32)
        probs = np.zeros(len(ds_rows), dtype=np.float64)
        for fold_id in range(args.n_folds):
            train_mask = folds != fold_id
            valid_mask = folds == fold_id
            if not np.any(valid_mask):
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.04,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=args.random_state + fold_id + DATASET_CODE[dataset],
            )
            _fit_hist_gb_classifier(clf, x[train_mask], y[train_mask], w[train_mask])
            probs[valid_mask] = clf.predict_proba(x[valid_mask])[:, 1]
        for row, prob, fold_id in zip(ds_rows, probs, folds):
            pred_row = dict(row)
            pred_row["oof_fold_id"] = int(fold_id)
            pred_row["oof_route_prob"] = float(prob)
            predictions.append(pred_row)
        reports.append(
            {
                "dataset": dataset,
                "row_count": len(ds_rows),
                "positive_rate_gap": float(np.mean(y)),
                "positive_rate_missing_weighted": float(np.sum(w * y) / max(np.sum(w), 1e-9)),
                "prob_mean": float(np.mean(probs)),
                "prob_p95": float(np.percentile(probs, 95)),
                "feature_columns": feature_cols,
                "feature_schema_hash": sf.feature_schema_hash(feature_cols),
                "feature_nan_count": matrix_stats["feature_nan_count"],
                "feature_clip_count": matrix_stats["feature_clip_count"],
                "selection_bias_note": "Threshold tuning on OOF probabilities still has mild selection bias; require gains above acceptance bar before promotion.",
            }
        )
    predictions.sort(key=lambda r: (r["dataset"], r["traj_id"], r["gap_start_idx"], r["gap_end_idx"]))
    return predictions, reports


def _selected_keys_for_threshold(rows: Sequence[Dict[str, Any]], dataset: str, threshold: float) -> set[Tuple[Any, ...]]:
    keys: set[Tuple[Any, ...]] = set()
    for row in rows:
        if str(row["dataset"]) != dataset:
            continue
        if float(row["oof_route_prob"]) > threshold:
            keys.add((row["dataset"], int(row["traj_id"]), int(row["gap_start_idx"]), int(row["gap_end_idx"])))
    return keys


def _threshold_dataset_summary(rows: Sequence[Dict[str, Any]], dataset: str, threshold: float) -> Dict[str, Any]:
    ds_rows = [r for r in rows if str(r["dataset"]) == dataset]
    selected = [r for r in ds_rows if float(r["oof_route_prob"]) > threshold]
    selected_missing = float(sum(float(r["missing_point_count"]) for r in selected))
    good_selected = [r for r in selected if int(r["label_use_route_projection"]) == 1]
    bad_selected = [r for r in selected if int(r["label_bad_route_projection"]) == 1]
    low_hi = [r for r in selected if str(r["eval_b28_quadrant_global"]) == "low_official_high_shape"]
    return {
        "dataset": dataset,
        "threshold": threshold,
        "selected_count": int(len(selected)),
        "selected_missing_count": int(selected_missing),
        "selected_good_count": int(len(good_selected)),
        "selected_bad_count": int(len(bad_selected)),
        "selected_bad_missing_count": int(sum(int(r["missing_point_count"]) for r in bad_selected)),
        "selected_good_error_saved": float(
            sum(
                max(0.0, float(r["eval_b28_official_mae_m"]) - float(r["eval_route_projection_official_mae_m"]))
                * float(r["missing_point_count"])
                for r in good_selected
            )
        ),
        "selected_bad_error_added": float(
            sum(
                max(0.0, float(r["eval_route_projection_official_mae_m"]) - float(r["eval_b28_official_mae_m"]))
                * float(r["missing_point_count"])
                for r in bad_selected
            )
        ),
        "net_error_delta": float(sum(float(r["label_delta_official_mae_m"]) * float(r["missing_point_count"]) for r in selected)),
        "selection_rate": float(len(selected) / max(len(ds_rows), 1)),
        "damage_on_base_low_official_high_shape_mae_delta": float(
            np.mean([float(r["label_delta_official_mae_m"]) for r in low_hi]) if low_hi else 0.0
        ),
        "damage_on_base_low_official_high_shape_error_sum_delta": float(
            sum(float(r["label_delta_official_mae_m"]) * float(r["missing_point_count"]) for r in low_hi)
        ),
    }


def _run_compare(base_analysis: Path, new_analysis: Path, out_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_runs.py"),
            "--base",
            str(base_analysis),
            "--new",
            str(new_analysis),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )


def _build_combined_best_outputs(
    rows: Sequence[Dict[str, Any]],
    best_by_dataset: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    import shutil

    input8 = load_pickle(args.input_8)
    input16 = load_pickle(args.input_16)
    base_pred8 = load_pickle(args.base_pred_8)
    base_pred16 = load_pickle(args.base_pred_16)
    route_pred8 = load_pickle(args.route_pred_8)
    route_pred16 = load_pickle(args.route_pred_16)
    selected8 = _selected_keys_for_threshold(rows, "1/8", float(best_by_dataset["1/8"]["threshold"]))
    selected16 = _selected_keys_for_threshold(rows, "1/16", float(best_by_dataset["1/16"]["threshold"]))
    mixed8 = build_mixed_predictions(base_pred8, route_pred8, input8, selected8, "1/8")
    mixed16 = build_mixed_predictions(base_pred16, route_pred16, input16, selected16, "1/16")
    save_pickle(args.out_dir / "mixed_oof_pred_8.pkl", mixed8)
    save_pickle(args.out_dir / "mixed_oof_pred_16.pkl", mixed16)

    analysis_dst = args.out_dir / "analysis"
    compare_dst = args.out_dir / "compare_vs_b28"
    if analysis_dst.exists():
        shutil.rmtree(analysis_dst)
    if compare_dst.exists():
        shutil.rmtree(compare_dst)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_analyze.py"),
            "--input-8",
            str(args.input_8),
            "--input-16",
            str(args.input_16),
            "--pred-8",
            str(args.out_dir / "mixed_oof_pred_8.pkl"),
            "--pred-16",
            str(args.out_dir / "mixed_oof_pred_16.pkl"),
            "--gt",
            str(args.gt),
            "--out-dir",
            str(analysis_dst),
        ],
        check=True,
    )
    _run_compare(ROOT / "runs" / "b28_compat_full" / "analysis", analysis_dst, compare_dst)


def run_threshold_sweep(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    input8 = load_pickle(args.input_8)
    input16 = load_pickle(args.input_16)
    gt = load_pickle(args.gt)
    base_pred8 = load_pickle(args.base_pred_8)
    base_pred16 = load_pickle(args.base_pred_16)
    route_pred8 = load_pickle(args.route_pred_8)
    route_pred16 = load_pickle(args.route_pred_16)
    base_metrics8 = compute_missing_point_metrics(input8, base_pred8, gt)
    base_metrics16 = compute_missing_point_metrics(input16, base_pred16, gt)
    summaries: List[Dict[str, Any]] = []
    by_dataset: Dict[str, List[Dict[str, Any]]] = {"1/8": [], "1/16": []}
    threshold_root = args.out_dir / "thresholds"
    threshold_root.mkdir(parents=True, exist_ok=True)

    for threshold in THRESHOLDS:
        threshold_tag = f"t{int(round(threshold * 100)):02d}"
        threshold_dir = threshold_root / threshold_tag
        selected8 = _selected_keys_for_threshold(rows, "1/8", threshold)
        selected16 = _selected_keys_for_threshold(rows, "1/16", threshold)
        mixed8 = build_mixed_predictions(base_pred8, route_pred8, input8, selected8, "1/8")
        mixed16 = build_mixed_predictions(base_pred16, route_pred16, input16, selected16, "1/16")
        save_pickle(threshold_dir / "pred_8.pkl", mixed8)
        save_pickle(threshold_dir / "pred_16.pkl", mixed16)
        analysis_dir = threshold_dir / "analysis"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_analyze.py"),
                "--input-8",
                str(args.input_8),
                "--input-16",
                str(args.input_16),
                "--pred-8",
                str(threshold_dir / "pred_8.pkl"),
                "--pred-16",
                str(threshold_dir / "pred_16.pkl"),
                "--gt",
                str(args.gt),
                "--out-dir",
                str(analysis_dir),
            ],
            check=True,
        )
        compare_dir = threshold_dir / "compare_vs_b28"
        _run_compare(ROOT / "runs" / "b28_compat_full" / "analysis", analysis_dir, compare_dir)
        global_metrics = json.loads((analysis_dir / "global_metrics.json").read_text(encoding="utf-8"))
        for dataset, base_metrics in (("1/8", base_metrics8), ("1/16", base_metrics16)):
            summary = _threshold_dataset_summary(rows, dataset, threshold)
            mixed_metrics = global_metrics[dataset]
            summary["mae_delta"] = float(mixed_metrics["mae"] - base_metrics["mae"])
            summary["rmse_delta"] = float(mixed_metrics["rmse"] - base_metrics["rmse"])
            summary["p95_delta"] = float(mixed_metrics["p95"] - base_metrics["p95"])
            summary["analysis_dir"] = str(analysis_dir)
            summary["compare_dir"] = str(compare_dir)
            summaries.append(summary)
            by_dataset[dataset].append(summary)

    best_by_dataset: Dict[str, Any] = {}
    for dataset, dataset_rows in by_dataset.items():
        chosen = sorted(
            dataset_rows,
            key=lambda r: (
                r["p95_delta"] > 0.0,
                r["mae_delta"],
                abs(r["damage_on_base_low_official_high_shape_mae_delta"]),
                r["selected_bad_error_added"],
                abs(r["selection_rate"] - 0.15),
            ),
        )[0]
        best_by_dataset[dataset] = {
            "threshold": float(chosen["threshold"]),
            "mae_delta": float(chosen["mae_delta"]),
            "p95_delta": float(chosen["p95_delta"]),
            "selection_bias_note": "Best threshold chosen from OOF sweep; treat as mildly optimistic until full-val fixed-threshold validation.",
        }

    _build_combined_best_outputs(rows, best_by_dataset, args)
    return summaries, {"best_by_dataset": best_by_dataset}


def _load_frozen_thresholds(oof_dir: Path) -> Dict[str, float]:
    report_path = oof_dir / "threshold_selection_report.json"
    if report_path.exists():
        data = json.loads(report_path.read_text(encoding="utf-8"))
        best = data.get("best_by_dataset", {})
        if isinstance(best, dict) and "1/8" in best and "1/16" in best:
            return {"1/8": float(best["1/8"]["threshold"]), "1/16": float(best["1/16"]["threshold"])}
    return {"1/8": 0.50, "1/16": 0.50}


def _load_expected_selection_rates(oof_dir: Path, thresholds: Dict[str, float]) -> Dict[str, Any]:
    summary_path = oof_dir / "threshold_sweep_summary.csv"
    out: Dict[str, Any] = {}
    if not summary_path.exists():
        return out
    rows = _read_csv(summary_path)
    for dataset, threshold in thresholds.items():
        match = [r for r in rows if str(r["dataset"]) == dataset and abs(float(r["threshold"]) - float(threshold)) < 1e-9]
        if not match:
            continue
        row = match[0]
        out[dataset] = {
            "selection_rate": float(row["selection_rate"]),
            "selected_count": int(row["selected_count"]),
            "selected_missing_count": int(row["selected_missing_count"]),
            "recommended_selection_rate_range": [0.05, 0.25],
        }
    return out


def run_train_full(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    feature_cols = list(sf.MODEL_FEATURE_COLUMNS)
    thresholds = _load_frozen_thresholds(args.oof_dir)
    expected_selection = _load_expected_selection_rates(args.oof_dir, thresholds)
    spec = {
        "feature_columns": feature_cols,
        "feature_schema_hash": sf.feature_schema_hash(feature_cols),
        "training_script": "scripts/train_gap_selector.py",
        "thresholds": thresholds,
    }
    save_json(args.out_dir / "feature_columns.json", spec)
    threshold_config = {
        "1/8": thresholds["1/8"],
        "1/16": thresholds["1/16"],
        "source": "selector_oof_threshold_sweep",
        "note": "Chosen from OOF sweep; final models trained on full validation data.",
    }
    save_json(args.out_dir / "threshold_config.json", threshold_config)

    report: Dict[str, Any] = {
        "mode": "train-full",
        "feature_columns": feature_cols,
        "feature_schema_hash": spec["feature_schema_hash"],
        "model_params": {
            "max_iter": 200,
            "learning_rate": 0.04,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.05,
            "random_state": args.random_state,
        },
        "thresholds": thresholds,
        "expected_selection_stats_from_oof": expected_selection,
        "datasets": {},
        "sanity_only_note": "Full-val training artifacts are for fixed-threshold deployment only; do not treat any full-val metrics as unbiased evaluation.",
    }

    for dataset in ("1/8", "1/16"):
        ds_rows = [r for r in rows if str(r["dataset"]) == dataset]
        actual_cols = [c for c in ds_rows[0].keys() if c.startswith("feature_")]
        sf.validate_feature_columns(feature_cols, actual_cols)
        x, matrix_stats = sf.feature_matrix(ds_rows, feature_cols)
        y = np.asarray([int(r["label_use_route_projection"]) for r in ds_rows], dtype=np.int32)
        w = np.asarray([float(r["missing_point_count"]) for r in ds_rows], dtype=np.float64)
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=args.random_state + sf.DATASET_CODE[dataset],
        )
        _fit_hist_gb_classifier(clf, x, y, w)
        probs = clf.predict_proba(x)[:, 1]
        suffix = "8" if dataset == "1/8" else "16"
        joblib.dump(clf, args.out_dir / f"selector_{suffix}.joblib")
        report["datasets"][dataset] = {
            "row_count": len(ds_rows),
            "positive_rate_gap": float(np.mean(y)),
            "positive_rate_missing_weighted": float(np.sum(w * y) / max(np.sum(w), 1e-9)),
            "prob_mean": float(np.mean(probs)),
            "prob_p95": float(np.percentile(probs, 95)),
            "feature_nan_count": matrix_stats["feature_nan_count"],
            "feature_clip_count": matrix_stats["feature_clip_count"],
        }

    save_json(
        args.out_dir / "selector_config.json",
        {
            "strategy": "selector_mix",
            "model_paths": {
                "1/8": str(args.out_dir / "selector_8.joblib"),
                "1/16": str(args.out_dir / "selector_16.joblib"),
            },
            "feature_columns_path": str(args.out_dir / "feature_columns.json"),
            "threshold_config_path": str(args.out_dir / "threshold_config.json"),
            "expected_selection_stats_from_oof": expected_selection,
        },
    )
    save_json(args.out_dir / "train_full_report.json", report)
    return report


def _validate_feature_namespace(rows: Sequence[Dict[str, Any]]) -> None:
    bad = [key for key in rows[0].keys() if key.startswith(("label_", "eval_")) and key in sf.MODEL_FEATURE_COLUMNS]
    if bad:
        raise ValueError(f"Feature leakage columns present in model feature list: {bad}")


def main() -> None:
    args = parse_args()
    if args.mode == "train-full" and args.out_dir.name == "selector_oof":
        args.out_dir = ROOT / "runs" / "selector_full"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"build-dataset", "all"}:
        rows, meta = build_dataset(args)
        _validate_feature_namespace(rows)
        save_csv(args.out_dir / "gap_selector_dataset.csv", rows)
        folds = make_fold_assignments(rows, args)
        save_csv(args.out_dir / "fold_assignments.csv", folds)
        save_csv(args.out_dir / "feature_bucket_lift.csv", build_feature_bucket_lift(rows))
        save_json(args.out_dir / "dataset_build_report.json", meta["dataset"])
        print(f"Dataset built: {args.out_dir / 'gap_selector_dataset.csv'}")

    dataset_base_dir = args.out_dir if args.mode != "train-full" else args.oof_dir
    dataset_path = dataset_base_dir / "gap_selector_dataset.csv"
    fold_path = dataset_base_dir / "fold_assignments.csv"
    rows = _load_dataset_rows(dataset_path)
    _validate_feature_namespace(rows)
    fold_map = _load_fold_map(fold_path)

    if args.mode in {"oof", "all"}:
        predictions, reports = run_oof(rows, fold_map, args)
        save_csv(args.out_dir / "selector_oof_gap_predictions.csv", predictions)
        for report in reports:
            dataset_slug = report["dataset"].replace("/", "_")
            save_json(args.out_dir / f"report_{dataset_slug}.json", report)
        print(f"OOF predictions saved: {args.out_dir / 'selector_oof_gap_predictions.csv'}")

    if args.mode in {"sweep", "all"}:
        prediction_path = args.out_dir / "selector_oof_gap_predictions.csv"
        pred_rows = _load_dataset_rows(prediction_path)
        summaries, best = run_threshold_sweep(pred_rows, args)
        save_csv(args.out_dir / "threshold_sweep_summary.csv", summaries)
        save_json(args.out_dir / "threshold_selection_report.json", best)
        print(f"Threshold sweep saved: {args.out_dir / 'threshold_sweep_summary.csv'}")

    if args.mode == "train-full":
        report = run_train_full(rows, args)
        print(f"Full-val selector artifacts saved: {args.out_dir}")


if __name__ == "__main__":
    main()
