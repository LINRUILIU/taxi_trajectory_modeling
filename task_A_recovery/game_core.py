from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


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


@dataclass
class CaseData:
    traj_id: int
    timestamps: np.ndarray
    input_coords: np.ndarray
    mask: np.ndarray
    gt_coords: np.ndarray
    pred23_coords: np.ndarray
    pred28_coords: np.ndarray
    gaps: List[GapInfo]
    known_indices: np.ndarray
    mae_b23: float
    mae_b28: float


@dataclass
class CasePoolResult:
    case_ids: List[int]
    metrics_table: List[Dict[str, float]]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_id_map(records: Sequence[Dict], coords_key: str = "coords") -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for item in records:
        tid = int(item["traj_id"])
        out[tid] = item
    return out


def haversine_meters_vector(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    r = 6371000.0
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)

    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    return r * c


def _finite_mask(coords: np.ndarray) -> np.ndarray:
    return np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])


def _missing_error_vector(pred_coords: np.ndarray, gt_coords: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    missing = ~np.asarray(mask, dtype=bool)
    total_missing = int(np.sum(missing))
    if total_missing == 0:
        return np.empty(0, dtype=np.float64), 0, 0

    pred_m = np.asarray(pred_coords, dtype=np.float64)[missing]
    gt_m = np.asarray(gt_coords, dtype=np.float64)[missing]
    valid = _finite_mask(pred_m) & _finite_mask(gt_m)
    if not np.any(valid):
        return np.empty(0, dtype=np.float64), total_missing, 0

    errs = haversine_meters_vector(
        pred_m[valid, 0],
        pred_m[valid, 1],
        gt_m[valid, 0],
        gt_m[valid, 1],
    )
    return errs.astype(np.float64), total_missing, int(np.sum(valid))


def error_metrics_from_vector(errors: np.ndarray, total_missing: int, evaluated_missing: int) -> Dict[str, float]:
    if errors.size == 0:
        return {
            "count": 0,
            "total_missing": int(total_missing),
            "evaluated_missing": int(evaluated_missing),
            "unfilled_missing": int(max(0, total_missing - evaluated_missing)),
            "mae": math.nan,
            "rmse": math.nan,
            "p75": math.nan,
            "p95": math.nan,
        }

    return {
        "count": int(errors.size),
        "total_missing": int(total_missing),
        "evaluated_missing": int(evaluated_missing),
        "unfilled_missing": int(max(0, total_missing - evaluated_missing)),
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "p75": float(np.percentile(errors, 75)),
        "p95": float(np.percentile(errors, 95)),
    }


def missing_metrics(pred_coords: np.ndarray, gt_coords: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    errs, total_missing, evaluated_missing = _missing_error_vector(pred_coords, gt_coords, mask)
    return error_metrics_from_vector(errs, total_missing=total_missing, evaluated_missing=evaluated_missing)


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
        missing_indices = np.arange(s + 1, e, dtype=np.int64)
        delta_t_sec = float(max(0.0, timestamps[e] - timestamps[s]))
        gaps.append(
            GapInfo(
                start_idx=s,
                end_idx=e,
                missing_indices=missing_indices,
                delta_t_sec=delta_t_sec,
                known_start_order=int(known_pos),
                known_end_order=int(known_pos + 1),
            )
        )

    return gaps, known_idx.astype(np.int64)


def _compute_traj_mae(pred_coords: np.ndarray, gt_coords: np.ndarray, mask: np.ndarray) -> float:
    errors, _, _ = _missing_error_vector(pred_coords, gt_coords, mask)
    if errors.size == 0:
        return math.nan
    return float(np.mean(errors))


def _stratified_sample_by_score(
    rows: List[Dict[str, float]],
    target_count: int,
    seed: int,
) -> List[int]:
    if not rows:
        return []

    rows_sorted = sorted(rows, key=lambda x: x["mae_combo"])
    if target_count <= 0 or target_count >= len(rows_sorted):
        return [int(r["traj_id"]) for r in rows_sorted]

    n = len(rows_sorted)
    i1 = n // 3
    i2 = 2 * n // 3
    bins = [rows_sorted[:i1], rows_sorted[i1:i2], rows_sorted[i2:]]

    base = target_count // 3
    alloc = [base, base, base]
    for i in range(target_count - base * 3):
        alloc[i] += 1

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    leftovers: List[int] = []

    for b, k in zip(bins, alloc):
        ids = [int(x["traj_id"]) for x in b]
        if not ids:
            continue
        if len(ids) <= k:
            selected.extend(ids)
        else:
            chosen_idx = set(int(x) for x in rng.choice(len(ids), size=k, replace=False))
            for idx, tid in enumerate(ids):
                if idx in chosen_idx:
                    selected.append(tid)
                else:
                    leftovers.append(tid)

    if len(selected) < target_count:
        need = target_count - len(selected)
        if leftovers:
            if len(leftovers) <= need:
                selected.extend(leftovers)
            else:
                chosen_idx = rng.choice(len(leftovers), size=need, replace=False)
                selected.extend([leftovers[int(i)] for i in chosen_idx])

    unique = []
    seen = set()
    for tid in selected:
        if tid not in seen:
            seen.add(tid)
            unique.append(tid)

    if len(unique) < target_count:
        for row in rows_sorted:
            tid = int(row["traj_id"])
            if tid not in seen:
                unique.append(tid)
                seen.add(tid)
            if len(unique) >= target_count:
                break

    return unique[:target_count]


def build_case_pool(
    input_records: Sequence[Dict],
    gt_records: Sequence[Dict],
    pred23_records: Sequence[Dict],
    pred28_records: Sequence[Dict],
    target_count: int,
    seed: int,
) -> CasePoolResult:
    input_by_id = build_id_map(input_records)
    gt_by_id = build_id_map(gt_records)
    p23_by_id = build_id_map(pred23_records)
    p28_by_id = build_id_map(pred28_records)

    common_ids = sorted(set(input_by_id.keys()) & set(gt_by_id.keys()) & set(p23_by_id.keys()) & set(p28_by_id.keys()))

    rows: List[Dict[str, float]] = []
    for tid in common_ids:
        inp = input_by_id[tid]
        gt = gt_by_id[tid]
        p23 = p23_by_id[tid]
        p28 = p28_by_id[tid]

        mask = np.asarray(inp["mask"], dtype=bool)
        if int(np.sum(~mask)) <= 0:
            continue

        gt_coords = np.asarray(gt["coords"], dtype=np.float64)
        p23_coords = np.asarray(p23["coords"], dtype=np.float64)
        p28_coords = np.asarray(p28["coords"], dtype=np.float64)

        mae23 = _compute_traj_mae(p23_coords, gt_coords, mask)
        mae28 = _compute_traj_mae(p28_coords, gt_coords, mask)
        if not (np.isfinite(mae23) and np.isfinite(mae28)):
            continue

        rows.append(
            {
                "traj_id": float(tid),
                "mae_b23": float(mae23),
                "mae_b28": float(mae28),
                "mae_combo": float(0.5 * (mae23 + mae28)),
                "missing_count": float(np.sum(~mask)),
            }
        )

    case_ids = _stratified_sample_by_score(rows=rows, target_count=target_count, seed=seed)

    rows_out: List[Dict[str, float]] = []
    row_by_id = {int(r["traj_id"]): r for r in rows}
    for tid in case_ids:
        row = row_by_id[tid]
        rows_out.append(
            {
                "traj_id": float(tid),
                "mae_b23": float(row["mae_b23"]),
                "mae_b28": float(row["mae_b28"]),
                "mae_combo": float(row["mae_combo"]),
                "missing_count": float(row["missing_count"]),
            }
        )

    return CasePoolResult(case_ids=case_ids, metrics_table=rows_out)


def build_case_data(
    traj_id: int,
    input_by_id: Dict[int, Dict],
    gt_by_id: Dict[int, Dict],
    pred23_by_id: Dict[int, Dict],
    pred28_by_id: Dict[int, Dict],
) -> CaseData:
    inp = input_by_id[traj_id]
    gt = gt_by_id[traj_id]
    p23 = pred23_by_id[traj_id]
    p28 = pred28_by_id[traj_id]

    timestamps = np.asarray(inp["timestamps"], dtype=np.float64)
    input_coords = np.asarray(inp["coords"], dtype=np.float64)
    mask = np.asarray(inp["mask"], dtype=bool)
    gt_coords = np.asarray(gt["coords"], dtype=np.float64)
    pred23_coords = np.asarray(p23["coords"], dtype=np.float64)
    pred28_coords = np.asarray(p28["coords"], dtype=np.float64)

    if not (
        input_coords.shape == gt_coords.shape == pred23_coords.shape == pred28_coords.shape
        and input_coords.shape[0] == timestamps.shape[0]
        and input_coords.shape[0] == mask.shape[0]
    ):
        raise ValueError(f"shape mismatch on traj_id={traj_id}")

    gaps, known_indices = extract_gap_infos(mask=mask, timestamps=timestamps)

    mae23 = _compute_traj_mae(pred23_coords, gt_coords, mask)
    mae28 = _compute_traj_mae(pred28_coords, gt_coords, mask)

    return CaseData(
        traj_id=int(traj_id),
        timestamps=timestamps,
        input_coords=input_coords,
        mask=mask,
        gt_coords=gt_coords,
        pred23_coords=pred23_coords,
        pred28_coords=pred28_coords,
        gaps=gaps,
        known_indices=known_indices,
        mae_b23=float(mae23),
        mae_b28=float(mae28),
    )


def init_player_prediction(input_coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pred = np.asarray(input_coords, dtype=np.float64).copy()
    pred[~np.asarray(mask, dtype=bool)] = np.nan
    return pred


def _deduplicate_polyline(polyline: np.ndarray) -> np.ndarray:
    if polyline.shape[0] <= 1:
        return polyline

    keep = [0]
    for i in range(1, polyline.shape[0]):
        if np.hypot(polyline[i, 0] - polyline[keep[-1], 0], polyline[i, 1] - polyline[keep[-1], 1]) > 1e-12:
            keep.append(i)

    return polyline[np.asarray(keep, dtype=np.int64)]


def sample_polyline_by_arclength(polyline: np.ndarray, num_points: int) -> np.ndarray:
    if num_points <= 0:
        return np.empty((0, 2), dtype=np.float64)

    p = np.asarray(polyline, dtype=np.float64)
    if p.shape[0] == 0:
        return np.empty((num_points, 2), dtype=np.float64)

    p = _deduplicate_polyline(p)
    if p.shape[0] == 1:
        return np.repeat(p, repeats=num_points, axis=0)

    seg_lens = haversine_meters_vector(
        p[:-1, 0],
        p[:-1, 1],
        p[1:, 0],
        p[1:, 1],
    ).astype(np.float64)

    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total < 1e-9:
        return np.repeat(p[:1], repeats=num_points, axis=0)

    t = np.linspace(0.0, total, num_points, dtype=np.float64)
    out_lon = np.interp(t, cum, p[:, 0])
    out_lat = np.interp(t, cum, p[:, 1])
    return np.stack([out_lon, out_lat], axis=1)


def fill_gap_with_stroke(
    start_anchor: np.ndarray,
    end_anchor: np.ndarray,
    stroke_points: Optional[np.ndarray],
    missing_count: int,
) -> np.ndarray:
    if missing_count <= 0:
        return np.empty((0, 2), dtype=np.float64)

    if stroke_points is None or stroke_points.size == 0:
        polyline = np.stack([start_anchor, end_anchor], axis=0).astype(np.float64)
        sampled = sample_polyline_by_arclength(polyline=polyline, num_points=missing_count + 2)
        return sampled[1:-1].astype(np.float64)

    stroke = np.asarray(stroke_points, dtype=np.float64)
    finite = _finite_mask(stroke)
    stroke = stroke[finite]
    if stroke.size == 0:
        polyline = np.stack([start_anchor, end_anchor], axis=0).astype(np.float64)
        sampled = sample_polyline_by_arclength(polyline=polyline, num_points=missing_count + 2)
        return sampled[1:-1].astype(np.float64)

    # Player only needs to draw a rough curve; we always anchor with known points,
    # then uniformly resample to match the missing-point count.
    polyline = np.vstack([
        np.asarray(start_anchor, dtype=np.float64).reshape(1, 2),
        stroke,
        np.asarray(end_anchor, dtype=np.float64).reshape(1, 2),
    ])
    sampled = sample_polyline_by_arclength(polyline=polyline, num_points=missing_count + 2)
    return sampled[1:-1].astype(np.float64)


def apply_gap_fill(pred_coords: np.ndarray, gap: GapInfo, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (gap.missing_count, 2):
        raise ValueError(
            f"gap fill shape mismatch, expected {(gap.missing_count, 2)} but got {tuple(points.shape)}"
        )
    pred_coords[gap.missing_indices] = points


def evaluate_case(case: CaseData, player_pred_coords: np.ndarray) -> Dict:
    m_player = missing_metrics(player_pred_coords, case.gt_coords, case.mask)
    m_b23 = missing_metrics(case.pred23_coords, case.gt_coords, case.mask)
    m_b28 = missing_metrics(case.pred28_coords, case.gt_coords, case.mask)

    gap_rows = []
    for gap_idx, gap in enumerate(case.gaps):
        if gap.missing_count <= 0:
            continue

        g_gt = case.gt_coords[gap.missing_indices]
        g_player = player_pred_coords[gap.missing_indices]
        g_b23 = case.pred23_coords[gap.missing_indices]
        g_b28 = case.pred28_coords[gap.missing_indices]

        e_player, _, _ = _missing_error_vector(
            pred_coords=np.vstack([g_player]),
            gt_coords=np.vstack([g_gt]),
            mask=np.zeros(g_player.shape[0], dtype=bool),
        )
        e_b23, _, _ = _missing_error_vector(
            pred_coords=np.vstack([g_b23]),
            gt_coords=np.vstack([g_gt]),
            mask=np.zeros(g_b23.shape[0], dtype=bool),
        )
        e_b28, _, _ = _missing_error_vector(
            pred_coords=np.vstack([g_b28]),
            gt_coords=np.vstack([g_gt]),
            mask=np.zeros(g_b28.shape[0], dtype=bool),
        )

        gap_rows.append(
            {
                "gap_index": int(gap_idx),
                "start_idx": int(gap.start_idx),
                "end_idx": int(gap.end_idx),
                "missing_count": int(gap.missing_count),
                "delta_t_sec": float(gap.delta_t_sec),
                "mae_player": float(np.mean(e_player)) if e_player.size > 0 else math.nan,
                "mae_b23": float(np.mean(e_b23)) if e_b23.size > 0 else math.nan,
                "mae_b28": float(np.mean(e_b28)) if e_b28.size > 0 else math.nan,
            }
        )

    out = {
        "traj_id": int(case.traj_id),
        "missing_count": int(np.sum(~case.mask)),
        "player": m_player,
        "baseline23_e5": m_b23,
        "baseline28_turncurve": m_b28,
        "delta_player_vs_b23_mae": float(m_player["mae"] - m_b23["mae"]) if np.isfinite(m_player["mae"]) else math.nan,
        "delta_player_vs_b28_mae": float(m_player["mae"] - m_b28["mae"]) if np.isfinite(m_player["mae"]) else math.nan,
        "gap_metrics": gap_rows,
    }
    return out


def case_completion_status(case: CaseData, player_pred_coords: np.ndarray) -> Dict[str, int]:
    missing = ~case.mask
    total_missing = int(np.sum(missing))
    pred_missing = np.asarray(player_pred_coords, dtype=np.float64)[missing]
    filled = int(np.sum(_finite_mask(pred_missing))) if total_missing > 0 else 0
    return {
        "total_missing": total_missing,
        "filled_missing": filled,
        "unfilled_missing": int(max(0, total_missing - filled)),
    }
