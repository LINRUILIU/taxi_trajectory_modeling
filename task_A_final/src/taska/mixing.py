from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .geo import haversine_meters
from .metrics import extract_gap_infos, missing_metrics
from .shape_metrics import shape_symmetric_m


def infer_dataset_name(path: Path) -> str:
    stem = path.stem.lower()
    if "16" in stem:
        return "1/16"
    if "8" in stem:
        return "1/8"
    return "unknown"


def gap_size_bucket(dataset_name: str, gap_size: int) -> str:
    if dataset_name == "1/8":
        if gap_size <= 2:
            return "1-2"
        if gap_size <= 4:
            return "3-4"
        if gap_size <= 7:
            return "5-7"
        return "8+"
    if gap_size <= 4:
        return "1-4"
    if gap_size <= 8:
        return "5-8"
    if gap_size <= 15:
        return "9-15"
    return "16+"


def gap_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (row["dataset"], int(row["traj_id"]), int(row["gap_start_idx"]), int(row["gap_end_idx"]))


def compute_thresholds(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {"official_threshold_m": 0.0, "shape_threshold_m": 0.0}
    return {
        "official_threshold_m": float(np.percentile([r["official_mae_m"] for r in rows], 75)),
        "shape_threshold_m": float(np.percentile([r["shape_symmetric_m"] for r in rows], 75)),
    }


def quadrant_label(official: float, shape: float, official_thr: float, shape_thr: float) -> str:
    if official >= official_thr and shape >= shape_thr:
        return "high_official_high_shape"
    if official >= official_thr and shape < shape_thr:
        return "high_official_low_shape"
    if official < official_thr and shape >= shape_thr:
        return "low_official_high_shape"
    return "low_official_low_shape"


def collect_gap_rows_single_dataset(
    dataset_name: str,
    input_records: Sequence[Dict[str, Any]],
    pred_records: Sequence[Dict[str, Any]],
    gt_records: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    pred_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in pred_records}
    gt_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in gt_records}
    rows: List[Dict[str, Any]] = []
    for item in input_records:
        tid = int(item["traj_id"])
        mask = np.asarray(item["mask"], dtype=bool)
        ts = np.asarray(item["timestamps"], dtype=np.float64)
        pred = pred_by_id[tid]
        gt = gt_by_id[tid]
        gaps, _ = extract_gap_infos(mask=mask, timestamps=ts)
        for gap in gaps:
            idx = gap.missing_indices
            pred_missing = pred[idx]
            gt_missing = gt[idx]
            errs = np.asarray(
                haversine_meters(pred_missing[:, 0], pred_missing[:, 1], gt_missing[:, 0], gt_missing[:, 1]),
                dtype=np.float64,
            )
            official_mae = float(np.mean(errs)) if errs.size else 0.0
            rows.append(
                {
                    "dataset": dataset_name,
                    "traj_id": tid,
                    "gap_start_idx": int(gap.start_idx),
                    "gap_end_idx": int(gap.end_idx),
                    "gap_size": int(gap.missing_count),
                    "missing_point_count": int(gap.missing_count),
                    "official_mae_m": official_mae,
                    "official_rmse_m": float(np.sqrt(np.mean(np.square(errs)))) if errs.size else 0.0,
                    "shape_symmetric_m": float(
                        shape_symmetric_m(
                            gt_missing,
                            pred_missing,
                            gt[gap.start_idx : gap.end_idx + 1],
                            pred[gap.start_idx : gap.end_idx + 1],
                        )
                    ),
                    "gap_bucket": gap_size_bucket(dataset_name, int(gap.missing_count)),
                    "official_error_sum_m": float(official_mae * int(gap.missing_count)),
                }
            )
    thresholds = compute_thresholds(rows)
    for row in rows:
        row["quadrant_dataset"] = quadrant_label(
            row["official_mae_m"],
            row["shape_symmetric_m"],
            thresholds["official_threshold_m"],
            thresholds["shape_threshold_m"],
        )
    return rows, thresholds


def build_mixed_predictions(
    base_pred_records: Sequence[Dict[str, Any]],
    new_pred_records: Sequence[Dict[str, Any]],
    input_records: Sequence[Dict[str, Any]],
    selected_gap_keys: set[Tuple[Any, ...]],
    dataset_name: str,
) -> List[Dict[str, Any]]:
    base_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64).copy() for x in base_pred_records}
    new_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in new_pred_records}
    out: List[Dict[str, Any]] = []
    for item in input_records:
        tid = int(item["traj_id"])
        mixed = base_by_id[tid]
        mask = np.asarray(item["mask"], dtype=bool)
        ts = np.asarray(item["timestamps"], dtype=np.float64)
        gaps, _ = extract_gap_infos(mask=mask, timestamps=ts)
        for gap in gaps:
            key = (dataset_name, tid, int(gap.start_idx), int(gap.end_idx))
            if key in selected_gap_keys:
                mixed[gap.start_idx + 1 : gap.end_idx] = new_by_id[tid][gap.start_idx + 1 : gap.end_idx]
        mixed[mask] = np.asarray(item["coords"], dtype=np.float64)[mask]
        out.append({"traj_id": tid, "coords": mixed.astype(np.float32)})
    return out


def compute_missing_point_metrics(
    input_records: Sequence[Dict[str, Any]],
    pred_records: Sequence[Dict[str, Any]],
    gt_records: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    pred_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in pred_records}
    gt_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in gt_records}
    all_errs: List[np.ndarray] = []
    total_missing = 0
    for item in input_records:
        tid = int(item["traj_id"])
        mask = np.asarray(item["mask"], dtype=bool)
        metrics = missing_metrics(pred_by_id[tid], gt_by_id[tid], mask)
        total_missing += int(metrics["total_missing"])
        missing = ~mask
        if np.any(missing):
            pred_m = pred_by_id[tid][missing]
            gt_m = gt_by_id[tid][missing]
            errs = np.asarray(haversine_meters(pred_m[:, 0], pred_m[:, 1], gt_m[:, 0], gt_m[:, 1]), dtype=np.float64)
            all_errs.append(errs)
    arr = np.concatenate(all_errs) if all_errs else np.empty(0, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mae": math.nan, "rmse": math.nan, "p75": math.nan, "p95": math.nan}
    return {
        "count": int(arr.size),
        "total_missing": int(total_missing),
        "mae": float(np.mean(arr)),
        "rmse": float(np.sqrt(np.mean(np.square(arr)))),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }
