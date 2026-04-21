import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def haversine_meters(lon1, lat1, lon2, lat2):
    """Vectorized Haversine distance in meters."""
    r = 6371000.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def _safe_interpolate_1d(x_all: np.ndarray, x_known: np.ndarray, y_known: np.ndarray) -> np.ndarray:
    if len(x_known) == 0:
        return np.full_like(x_all, np.nan, dtype=np.float64)
    if len(x_known) == 1:
        return np.full_like(x_all, y_known[0], dtype=np.float64)
    return np.interp(x_all, x_known, y_known)


def interpolate_traj_linear(timestamps: np.ndarray, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill missing coordinates with linear interpolation over timestamps.

    coords shape: (N, 2), columns are [lon, lat].
    mask shape: (N,), True for observed points.
    """
    x_all = timestamps.astype(np.float64)
    known_idx = np.where(mask)[0]
    x_known = x_all[known_idx]
    lon_known = coords[known_idx, 0].astype(np.float64)
    lat_known = coords[known_idx, 1].astype(np.float64)

    lon_full = _safe_interpolate_1d(x_all, x_known, lon_known)
    lat_full = _safe_interpolate_1d(x_all, x_known, lat_known)
    pred = np.stack([lon_full, lat_full], axis=1)

    # Keep observed points exactly unchanged.
    pred[mask] = coords[mask]
    return pred.astype(np.float32)


def build_predictions(records: Iterable[Dict]) -> List[Dict]:
    outputs: List[Dict] = []
    for rec in records:
        traj_id = rec["traj_id"]
        timestamps = np.asarray(rec["timestamps"])
        coords = np.asarray(rec["coords"], dtype=np.float64)
        mask = np.asarray(rec["mask"], dtype=bool)

        pred_coords = interpolate_traj_linear(timestamps=timestamps, coords=coords, mask=mask)
        outputs.append({"traj_id": traj_id, "coords": pred_coords})
    return outputs


def evaluate_missing_only(pred_records: Iterable[Dict], input_records: Iterable[Dict], gt_records: Iterable[Dict]) -> Dict[str, float]:
    pred_by_id = {item["traj_id"]: np.asarray(item["coords"], dtype=np.float64) for item in pred_records}
    input_by_id = {item["traj_id"]: item for item in input_records}
    gt_by_id = {item["traj_id"]: np.asarray(item["coords"], dtype=np.float64) for item in gt_records}

    dists = []
    for traj_id, pred_coords in pred_by_id.items():
        input_item = input_by_id[traj_id]
        mask = np.asarray(input_item["mask"], dtype=bool)
        missing = ~mask
        if not np.any(missing):
            continue

        gt_coords = gt_by_id[traj_id]
        dist = haversine_meters(
            lon1=pred_coords[missing, 0],
            lat1=pred_coords[missing, 1],
            lon2=gt_coords[missing, 0],
            lat2=gt_coords[missing, 1],
        )
        dists.append(dist)

    if not dists:
        return {"mae": math.nan, "rmse": math.nan, "count": 0}

    all_dists = np.concatenate(dists)
    mae = float(np.mean(np.abs(all_dists)))
    rmse = float(np.sqrt(np.mean(all_dists**2)))
    return {"mae": mae, "rmse": rmse, "count": int(all_dists.size)}


def parse_args():
    parser = argparse.ArgumentParser(description="Task A baseline: linear interpolation for trajectory recovery.")
    parser.add_argument("--input", required=True, type=Path, help="Path to val/test input .pkl")
    parser.add_argument("--output", required=True, type=Path, help="Path to output prediction .pkl")
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Optional ground-truth path for evaluation (e.g., val_gt.pkl)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_records = load_pickle(args.input)
    pred_records = build_predictions(input_records)
    save_pickle(args.output, pred_records)
    print(f"Saved predictions: {args.output} (n={len(pred_records)})")

    if args.gt is not None:
        gt_records = load_pickle(args.gt)
        metrics = evaluate_missing_only(pred_records, input_records, gt_records)
        print(
            "Missing-point evaluation | "
            f"count={metrics['count']} | "
            f"MAE={metrics['mae']:.4f} m | "
            f"RMSE={metrics['rmse']:.4f} m"
        )


if __name__ == "__main__":
    main()
