from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .geo import haversine_meters, point_distance_m
from .metrics import extract_gap_infos
from .mixing import build_mixed_predictions, gap_key, gap_size_bucket

DATASET_CODE = {"1/8": 8, "1/16": 16}
GAP_BUCKET_CODE = {"1-2": 0, "3-4": 1, "5-7": 2, "8+": 3, "1-4": 10, "5-8": 11, "9-15": 12, "16+": 13}
FALLBACK_REASON_CODE = {
    "applied": 0,
    "route_not_found": 1,
    "degenerate_anchor": 2,
    "empty_route": 3,
    "projection_failed": 4,
    "base_nan": 5,
    "unknown": 6,
    "not_triggered": 7,
    "snap_far": 8,
    "no_route": 9,
    "monotonic_bad": 10,
    "projection_far": 11,
    "other": 99,
}
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


def read_typed_csv(path: Path) -> List[Dict[str, Any]]:
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


def ensure_no_duplicate_keys(rows: Sequence[Dict[str, Any]], label: str) -> None:
    seen: set[Tuple[Any, ...]] = set()
    for row in rows:
        key = gap_key(row)
        if key in seen:
            raise ValueError(f"Duplicate gap key in {label}: {key}")
        seen.add(key)


def feature_schema_hash(feature_columns: Sequence[str]) -> str:
    payload = json.dumps(list(feature_columns), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_feature_columns(expected: Sequence[str], actual: Sequence[str]) -> None:
    if list(expected) != list(actual):
        raise ValueError(
            "Feature column mismatch.\n"
            f"Expected: {list(expected)}\n"
            f"Actual:   {list(actual)}"
        )


def load_feature_spec(path: Path) -> Dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid feature spec at {path}")
    cols = spec.get("feature_columns")
    if not isinstance(cols, list) or not all(isinstance(x, str) for x in cols):
        raise ValueError(f"feature_columns missing or invalid in {path}")
    expected_hash = spec.get("feature_schema_hash")
    actual_hash = feature_schema_hash(cols)
    if expected_hash is not None and str(expected_hash) != actual_hash:
        raise ValueError(f"Feature schema hash mismatch in {path}")
    spec["feature_schema_hash"] = actual_hash
    return spec


def feature_matrix(rows: Sequence[Dict[str, Any]], feature_columns: Sequence[str]) -> tuple[np.ndarray, Dict[str, int]]:
    if not rows:
        return np.empty((0, len(feature_columns)), dtype=np.float64), {"feature_nan_count": 0, "feature_clip_count": 0}
    validate_feature_columns(feature_columns, [c for c in feature_columns])
    matrix = np.asarray([[float(row[col]) for col in feature_columns] for row in rows], dtype=np.float64)
    nan_count = int(np.isnan(matrix).sum())
    posinf = np.isposinf(matrix)
    neginf = np.isneginf(matrix)
    clip_count = int(posinf.sum() + neginf.sum())
    if clip_count:
        matrix = matrix.copy()
        matrix[posinf] = 1e9
        matrix[neginf] = -1e9
    return matrix, {"feature_nan_count": nan_count, "feature_clip_count": clip_count}


def build_selector_feature_rows(
    dataset_name: str,
    input_records: Sequence[Dict[str, Any]],
    base_pred_records: Sequence[Dict[str, Any]],
    route_pred_records: Sequence[Dict[str, Any]],
    debug_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ensure_no_duplicate_keys(debug_rows, f"{dataset_name} debug rows")
    debug_by_key = {gap_key(r): r for r in debug_rows}
    pair_features = _pairwise_gap_distance_features(base_pred_records, route_pred_records, input_records, dataset_name)
    rows: List[Dict[str, Any]] = []
    for item in input_records:
        traj_id = int(item["traj_id"])
        coords = np.asarray(item["coords"], dtype=np.float64)
        mask = np.asarray(item["mask"], dtype=bool)
        timestamps = np.asarray(item["timestamps"], dtype=np.float64)
        gaps, _ = extract_gap_infos(mask=mask, timestamps=timestamps)
        for gap in gaps:
            key = (dataset_name, traj_id, int(gap.start_idx), int(gap.end_idx))
            dbg = debug_by_key.get(key)
            if dbg is None:
                raise ValueError(f"Missing debug row for {key}")
            pair = pair_features.get(key)
            if pair is None:
                raise ValueError(f"Missing pairwise feature row for {key}")
            anchor_direct_m = float(point_distance_m(coords[gap.start_idx], coords[gap.end_idx]))
            gap_bucket = gap_size_bucket(dataset_name, int(gap.missing_count))
            start_snap = _to_float(dbg.get("start_snap_m"))
            end_snap = _to_float(dbg.get("end_snap_m"))
            row = {
                "dataset": dataset_name,
                "traj_id": traj_id,
                "gap_start_idx": int(gap.start_idx),
                "gap_end_idx": int(gap.end_idx),
                "gap_size": int(gap.missing_count),
                "missing_point_count": int(gap.missing_count),
                "feature_dataset_encoded": DATASET_CODE[dataset_name],
                "feature_gap_size": int(gap.missing_count),
                "feature_gap_bucket_encoded": GAP_BUCKET_CODE[gap_bucket],
                "feature_delta_t_sec": float(gap.delta_t_sec),
                "feature_anchor_direct_m": anchor_direct_m,
                "feature_anchor_speed_mps": float(anchor_direct_m / max(float(gap.delta_t_sec), 1e-6)),
                "feature_route_found": int(bool(dbg.get("route_found", False))),
                "feature_applied_projection": int(bool(dbg.get("applied_projection", False))),
                "feature_route_length_m": _to_float(dbg.get("route_length_m")),
                "feature_detour_ratio": _to_float(dbg.get("detour_ratio")),
                "feature_start_snap_m": start_snap,
                "feature_end_snap_m": end_snap,
                "feature_max_snap_m": _nanmax(start_snap, end_snap),
                "feature_base_to_route_mean_m": _to_float(dbg.get("base_to_route_mean_m")),
                "feature_base_to_route_p95_m": _to_float(dbg.get("base_to_route_p95_m")),
                "feature_projection_s_monotonic_violations": int(dbg.get("projection_s_monotonic_violations", 0)),
                "feature_projection_s_clamped_mean_m": _to_float(dbg.get("projection_s_clamped_mean_m")),
                "feature_projection_s_clamped_max_m": _to_float(dbg.get("projection_s_clamped_max_m")),
                "feature_route_sample_vs_base_mean_m": _to_float(dbg.get("route_sample_vs_base_mean_m")),
                "feature_fallback_reason_encoded": FALLBACK_REASON_CODE.get(str(dbg.get("fallback_reason", "other")), FALLBACK_REASON_CODE["other"]),
                "route_found": bool(dbg.get("route_found", False)),
                "applied_projection": bool(dbg.get("applied_projection", False)),
                "fallback_reason": str(dbg.get("fallback_reason", "")),
                "detour_ratio": _to_float(dbg.get("detour_ratio")),
                "base_to_route_mean_m": _to_float(dbg.get("base_to_route_mean_m")),
                "projection_s_clamped_max_m": _to_float(dbg.get("projection_s_clamped_max_m")),
            }
            row.update(pair)
            row["pred_pair_mean_dist_m"] = row["feature_pred_pair_mean_dist_m"]
            rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["traj_id"], r["gap_start_idx"], r["gap_end_idx"]))
    return rows


def build_selector_decisions(
    feature_rows: Sequence[Dict[str, Any]],
    probabilities: Sequence[float],
    threshold: float,
) -> tuple[set[Tuple[Any, ...]], List[Dict[str, Any]], Dict[str, Any]]:
    if len(feature_rows) != len(probabilities):
        raise ValueError("Feature/probability length mismatch")
    selected_keys: set[Tuple[Any, ...]] = set()
    decision_rows: List[Dict[str, Any]] = []
    probs = np.asarray(probabilities, dtype=np.float64)
    selected_missing_point_count = 0
    for row, prob in zip(feature_rows, probs):
        selected = bool(prob > threshold)
        key = (row["dataset"], int(row["traj_id"]), int(row["gap_start_idx"]), int(row["gap_end_idx"]))
        if selected:
            selected_keys.add(key)
            selected_missing_point_count += int(row["missing_point_count"])
        decision_rows.append(
            {
                "dataset": row["dataset"],
                "traj_id": int(row["traj_id"]),
                "gap_start_idx": int(row["gap_start_idx"]),
                "gap_end_idx": int(row["gap_end_idx"]),
                "gap_size": int(row["gap_size"]),
                "missing_point_count": int(row["missing_point_count"]),
                "selected": selected,
                "route_probability": float(prob),
                "threshold": float(threshold),
                "route_found": bool(row["route_found"]),
                "applied_projection": bool(row["applied_projection"]),
                "fallback_reason": str(row["fallback_reason"]),
                "detour_ratio": _to_float(row["detour_ratio"]),
                "base_to_route_mean_m": _to_float(row["base_to_route_mean_m"]),
                "projection_s_clamped_max_m": _to_float(row["projection_s_clamped_max_m"]),
                "pred_pair_mean_dist_m": _to_float(row["pred_pair_mean_dist_m"]),
            }
        )
    metadata = {
        "gap_count": len(feature_rows),
        "selected_gap_count": len(selected_keys),
        "selection_rate": float(len(selected_keys) / max(len(feature_rows), 1)),
        "selected_missing_point_count": int(selected_missing_point_count),
        "prob_mean": float(np.mean(probs)) if probs.size else math.nan,
        "prob_p50": float(np.percentile(probs, 50)) if probs.size else math.nan,
        "prob_p90": float(np.percentile(probs, 90)) if probs.size else math.nan,
        "prob_p95": float(np.percentile(probs, 95)) if probs.size else math.nan,
    }
    return selected_keys, decision_rows, metadata


def mix_predictions_with_selector(
    dataset_name: str,
    input_records: Sequence[Dict[str, Any]],
    base_pred_records: Sequence[Dict[str, Any]],
    route_pred_records: Sequence[Dict[str, Any]],
    selected_keys: set[Tuple[Any, ...]],
) -> List[Dict[str, Any]]:
    return build_mixed_predictions(base_pred_records, route_pred_records, input_records, selected_keys, dataset_name)


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def _nanmax(a: float, b: float) -> float:
    if np.isnan(a):
        return b
    if np.isnan(b):
        return a
    return float(max(a, b))


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


def _prediction_gap_shape_features(
    pred_records: Sequence[Dict[str, Any]],
    input_records: Sequence[Dict[str, Any]],
    dataset_name: str,
) -> Dict[Tuple[Any, ...], Dict[str, float]]:
    pred_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in pred_records}
    out: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    for item in input_records:
        traj_id = int(item["traj_id"])
        mask = np.asarray(item["mask"], dtype=bool)
        timestamps = np.asarray(item["timestamps"], dtype=np.float64)
        pred = pred_by_id[traj_id]
        gaps, _ = extract_gap_infos(mask=mask, timestamps=timestamps)
        for gap in gaps:
            key = (dataset_name, traj_id, int(gap.start_idx), int(gap.end_idx))
            missing_poly = pred[gap.start_idx + 1 : gap.end_idx]
            full_gap_poly = pred[gap.start_idx : gap.end_idx + 1]
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
    base_gap = _prediction_gap_shape_features(base_pred_records, input_records, dataset_name)
    route_gap = _prediction_gap_shape_features(route_pred_records, input_records, dataset_name)
    out: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    for item in input_records:
        traj_id = int(item["traj_id"])
        mask = np.asarray(item["mask"], dtype=bool)
        timestamps = np.asarray(item["timestamps"], dtype=np.float64)
        base_pred = base_by_id[traj_id]
        route_pred = route_by_id[traj_id]
        gaps, _ = extract_gap_infos(mask=mask, timestamps=timestamps)
        for gap in gaps:
            key = (dataset_name, traj_id, int(gap.start_idx), int(gap.end_idx))
            base_missing = base_pred[gap.start_idx + 1 : gap.end_idx]
            route_missing = route_pred[gap.start_idx + 1 : gap.end_idx]
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
