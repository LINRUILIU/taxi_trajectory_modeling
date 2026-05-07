from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .geo import haversine_meters


@dataclass(frozen=True)
class GapInfo:
    start_idx: int
    end_idx: int
    missing_indices: np.ndarray
    delta_t_sec: float
    known_start_order: int
    known_end_order: int

    @property
    def missing_count(self) -> int:
        return int(self.missing_indices.size)


def build_id_map(records: Sequence[Dict], coords_key: str = "coords") -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for item in records:
        out[int(item["traj_id"])] = item
    return out


def haversine_meters_vector(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    return np.asarray(haversine_meters(lon1, lat1, lon2, lat2), dtype=np.float64)


def extract_gap_infos(mask: np.ndarray, timestamps: np.ndarray) -> Tuple[List[GapInfo], np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    known_idx = np.where(mask)[0]
    gaps: List[GapInfo] = []
    for known_pos in range(known_idx.size - 1):
        s = int(known_idx[known_pos])
        e = int(known_idx[known_pos + 1])
        if e - s <= 1:
            continue
        miss = np.arange(s + 1, e, dtype=np.int64)
        gaps.append(
            GapInfo(
                start_idx=s,
                end_idx=e,
                missing_indices=miss,
                delta_t_sec=float(max(0.0, timestamps[e] - timestamps[s])),
                known_start_order=int(known_pos),
                known_end_order=int(known_pos + 1),
            )
        )
    return gaps, known_idx.astype(np.int64)


def missing_metrics(pred_coords: np.ndarray, gt_coords: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    missing = ~np.asarray(mask, dtype=bool)
    total_missing = int(np.sum(missing))
    if total_missing == 0:
        return {
            "count": 0,
            "total_missing": 0,
            "evaluated_missing": 0,
            "unfilled_missing": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "p75": math.nan,
            "p95": math.nan,
        }
    pred_m = np.asarray(pred_coords, dtype=np.float64)[missing]
    gt_m = np.asarray(gt_coords, dtype=np.float64)[missing]
    valid = np.isfinite(pred_m[:, 0]) & np.isfinite(pred_m[:, 1]) & np.isfinite(gt_m[:, 0]) & np.isfinite(gt_m[:, 1])
    if not np.any(valid):
        return {
            "count": 0,
            "total_missing": total_missing,
            "evaluated_missing": 0,
            "unfilled_missing": total_missing,
            "mae": math.nan,
            "rmse": math.nan,
            "p75": math.nan,
            "p95": math.nan,
        }
    errs = haversine_meters_vector(pred_m[valid, 0], pred_m[valid, 1], gt_m[valid, 0], gt_m[valid, 1])
    return {
        "count": int(errs.size),
        "total_missing": total_missing,
        "evaluated_missing": int(np.sum(valid)),
        "unfilled_missing": int(total_missing - np.sum(valid)),
        "mae": float(np.mean(errs)),
        "rmse": float(np.sqrt(np.mean(np.square(errs)))),
        "p75": float(np.percentile(errs, 75)),
        "p95": float(np.percentile(errs, 95)),
    }


def global_metrics(error_values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(error_values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mae": math.nan, "rmse": math.nan, "p75": math.nan, "p95": math.nan}
    return {
        "count": int(arr.size),
        "mae": float(np.mean(arr)),
        "rmse": float(np.sqrt(np.mean(np.square(arr)))),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }
