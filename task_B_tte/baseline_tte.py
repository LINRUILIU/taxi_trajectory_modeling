import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from map_feature_utils import MAP_FEATURE_NAMES, build_map_runtime, extract_map_features_for_coords
except ImportError:
    from task_B_tte.map_feature_utils import MAP_FEATURE_NAMES, build_map_runtime, extract_map_features_for_coords


SECONDS_PER_DAY = 24 * 3600

FEATURE_NAMES = [
    "n_points",
    "path_len_m",
    "direct_dist_m",
    "detour_ratio",
    "mean_step_m",
    "std_step_m",
    "p50_step_m",
    "p90_step_m",
    "max_step_m",
    "mean_turn_deg",
    "p90_turn_deg",
    "max_turn_deg",
    "turn_gt30_ratio",
    "turn_gt60_ratio",
    "turn_gt90_ratio",
    "bbox_w_m",
    "bbox_h_m",
    "bbox_diag_m",
    "bbox_area_km2",
    "net_heading_sin",
    "net_heading_cos",
    "start_lon",
    "start_lat",
    "end_lon",
    "end_lat",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "minute_of_day_norm",
    "log1p_path_len",
] + MAP_FEATURE_NAMES


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def haversine_meters(lon1, lat1, lon2, lat2):
    r = 6371000.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return r * c


def bearing_deg(lon1, lat1, lon2, lat2):
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)
    dlon = lon2r - lon1r

    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    out = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0

    near_zero = (np.abs(x) < 1e-12) & (np.abs(y) < 1e-12)
    if np.any(near_zero):
        out = out.astype(np.float64)
        out[near_zero] = np.nan
    return out


def angle_diff_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def _safe_stat(arr: np.ndarray, fn, default: float = 0.0) -> float:
    if arr.size == 0:
        return float(default)
    val = float(fn(arr))
    if not np.isfinite(val):
        return float(default)
    return val


def _departure_timestamp(rec: Dict) -> int:
    if "departure_timestamp" in rec:
        return int(rec["departure_timestamp"])
    if "timestamps" in rec and len(rec["timestamps"]) > 0:
        return int(rec["timestamps"][0])
    return 0


def extract_trip_features(
    coords: np.ndarray,
    departure_ts: int,
    map_runtime: Optional[Dict] = None,
    map_query_radius_m: float = 120.0,
    map_max_points: int = 12,
    map_near_threshold_1_m: float = 20.0,
    map_near_threshold_2_m: float = 50.0,
) -> np.ndarray:
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"coords must be [N,2], got shape={pts.shape}")

    n = pts.shape[0]
    if n == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    lons = pts[:, 0]
    lats = pts[:, 1]

    start_lon, start_lat = float(lons[0]), float(lats[0])
    end_lon, end_lat = float(lons[-1]), float(lats[-1])

    if n >= 2:
        step = haversine_meters(lons[:-1], lats[:-1], lons[1:], lats[1:]).astype(np.float64)
        path_len = float(np.sum(step))
        direct = float(haversine_meters(start_lon, start_lat, end_lon, end_lat))

        bearings = bearing_deg(lons[:-1], lats[:-1], lons[1:], lats[1:]).astype(np.float64)
        if bearings.size >= 2:
            turn = angle_diff_deg(bearings[1:], bearings[:-1])
            turn = turn[np.isfinite(turn)]
        else:
            turn = np.empty(0, dtype=np.float64)

        mean_step = _safe_stat(step, np.mean)
        std_step = _safe_stat(step, np.std)
        p50_step = _safe_stat(step, lambda x: np.percentile(x, 50))
        p90_step = _safe_stat(step, lambda x: np.percentile(x, 90))
        max_step = _safe_stat(step, np.max)

        mean_turn = _safe_stat(turn, np.mean)
        p90_turn = _safe_stat(turn, lambda x: np.percentile(x, 90))
        max_turn = _safe_stat(turn, np.max)
        turn_gt30_ratio = float(np.mean(turn > 30.0)) if turn.size else 0.0
        turn_gt60_ratio = float(np.mean(turn > 60.0)) if turn.size else 0.0
        turn_gt90_ratio = float(np.mean(turn > 90.0)) if turn.size else 0.0

        net_heading = float(bearing_deg(np.array([start_lon]), np.array([start_lat]), np.array([end_lon]), np.array([end_lat]))[0])
        if np.isfinite(net_heading):
            net_heading_sin = float(math.sin(math.radians(net_heading)))
            net_heading_cos = float(math.cos(math.radians(net_heading)))
        else:
            net_heading_sin = 0.0
            net_heading_cos = 1.0
    else:
        path_len = 0.0
        direct = 0.0
        mean_step = 0.0
        std_step = 0.0
        p50_step = 0.0
        p90_step = 0.0
        max_step = 0.0
        mean_turn = 0.0
        p90_turn = 0.0
        max_turn = 0.0
        turn_gt30_ratio = 0.0
        turn_gt60_ratio = 0.0
        turn_gt90_ratio = 0.0
        net_heading_sin = 0.0
        net_heading_cos = 1.0

    detour_ratio = path_len / max(direct, 1e-6)

    min_lon, max_lon = float(np.min(lons)), float(np.max(lons))
    min_lat, max_lat = float(np.min(lats)), float(np.max(lats))
    center_lat = 0.5 * (min_lat + max_lat)
    meter_per_lon = 111320.0 * max(0.2, math.cos(math.radians(center_lat)))
    meter_per_lat = 111320.0
    bbox_w_m = max(0.0, (max_lon - min_lon) * meter_per_lon)
    bbox_h_m = max(0.0, (max_lat - min_lat) * meter_per_lat)
    bbox_diag_m = float(math.hypot(bbox_w_m, bbox_h_m))
    bbox_area_km2 = (bbox_w_m * bbox_h_m) / 1_000_000.0

    sec_day = int(departure_ts % SECONDS_PER_DAY)
    hour = sec_day / 3600.0
    minute_of_day_norm = sec_day / float(SECONDS_PER_DAY)
    dow = int((departure_ts // SECONDS_PER_DAY + 4) % 7)

    hour_angle = 2.0 * math.pi * (hour / 24.0)
    dow_angle = 2.0 * math.pi * (dow / 7.0)

    feat_base = np.array(
        [
            float(n),
            path_len,
            direct,
            float(detour_ratio),
            mean_step,
            std_step,
            p50_step,
            p90_step,
            max_step,
            mean_turn,
            p90_turn,
            max_turn,
            turn_gt30_ratio,
            turn_gt60_ratio,
            turn_gt90_ratio,
            bbox_w_m,
            bbox_h_m,
            bbox_diag_m,
            bbox_area_km2,
            net_heading_sin,
            net_heading_cos,
            start_lon,
            start_lat,
            end_lon,
            end_lat,
            math.sin(hour_angle),
            math.cos(hour_angle),
            math.sin(dow_angle),
            math.cos(dow_angle),
            1.0 if dow >= 5 else 0.0,
            minute_of_day_norm,
            math.log1p(path_len),
        ],
        dtype=np.float64,
    )

    map_feat = extract_map_features_for_coords(
        coords=pts,
        map_runtime=map_runtime,
        query_radius_m=map_query_radius_m,
        max_points=map_max_points,
        near_threshold_1_m=map_near_threshold_1_m,
        near_threshold_2_m=map_near_threshold_2_m,
    )

    feat = np.concatenate([feat_base, np.asarray(map_feat, dtype=np.float64)], axis=0)
    feat[~np.isfinite(feat)] = 0.0
    return feat.astype(np.float32)


def _travel_time_from_timestamps(rec: Dict) -> Optional[float]:
    ts = rec.get("timestamps")
    if ts is None or len(ts) < 2:
        return None
    tt = float(int(ts[-1]) - int(ts[0]))
    if not np.isfinite(tt):
        return None
    return tt


def _sample_weight(y_sec: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones_like(y_sec, dtype=np.float64)

    if mode == "short-boost":
        w = np.ones_like(y_sec, dtype=np.float64)
        w[y_sec <= 600.0] = 1.8
        w[(y_sec > 600.0) & (y_sec <= 1800.0)] = 1.2
        return w

    raise ValueError(f"Unknown weight mode: {mode}")


def build_supervised_dataset(
    records: Sequence[Dict],
    min_travel_time: float,
    max_travel_time: float,
    min_speed_kmh: float,
    max_speed_kmh: float,
    map_runtime: Optional[Dict],
    map_query_radius_m: float,
    map_max_points: int,
    map_near_threshold_1_m: float,
    map_near_threshold_2_m: float,
    limit: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    xs: List[np.ndarray] = []
    ys: List[float] = []
    stats = {
        "total": 0,
        "used": 0,
        "drop_invalid_tt": 0,
        "drop_tt_range": 0,
        "drop_speed_range": 0,
        "drop_short_coords": 0,
    }

    n_take = len(records) if limit <= 0 else min(limit, len(records))
    for rec in records[:n_take]:
        stats["total"] += 1

        coords = np.asarray(rec["coords"], dtype=np.float64)
        if coords.shape[0] < 2:
            stats["drop_short_coords"] += 1
            continue

        tt = _travel_time_from_timestamps(rec)
        if tt is None or tt <= 0.0:
            stats["drop_invalid_tt"] += 1
            continue

        if tt < min_travel_time or tt > max_travel_time:
            stats["drop_tt_range"] += 1
            continue

        dep = _departure_timestamp(rec)
        feat = extract_trip_features(
            coords=coords,
            departure_ts=dep,
            map_runtime=map_runtime,
            map_query_radius_m=map_query_radius_m,
            map_max_points=map_max_points,
            map_near_threshold_1_m=map_near_threshold_1_m,
            map_near_threshold_2_m=map_near_threshold_2_m,
        )

        path_len = float(feat[FEATURE_NAMES.index("path_len_m")])
        speed_kmh = path_len / tt * 3.6
        if speed_kmh < min_speed_kmh or speed_kmh > max_speed_kmh:
            stats["drop_speed_range"] += 1
            continue

        xs.append(feat)
        ys.append(tt)
        stats["used"] += 1

    if not xs:
        raise ValueError("No samples available after filtering. Relax cleaning thresholds.")

    x_arr = np.vstack(xs).astype(np.float32)
    y_arr = np.asarray(ys, dtype=np.float64)
    return x_arr, y_arr, stats


def build_inference_dataset(
    records: Sequence[Dict],
    map_runtime: Optional[Dict],
    map_query_radius_m: float,
    map_max_points: int,
    map_near_threshold_1_m: float,
    map_near_threshold_2_m: float,
    limit: int = 0,
) -> Tuple[np.ndarray, List[int]]:
    xs: List[np.ndarray] = []
    ids: List[int] = []

    n_take = len(records) if limit <= 0 else min(limit, len(records))
    for rec in records[:n_take]:
        traj_id = int(rec["traj_id"])
        coords = np.asarray(rec["coords"], dtype=np.float64)
        dep = _departure_timestamp(rec)
        feat = extract_trip_features(
            coords=coords,
            departure_ts=dep,
            map_runtime=map_runtime,
            map_query_radius_m=map_query_radius_m,
            map_max_points=map_max_points,
            map_near_threshold_1_m=map_near_threshold_1_m,
            map_near_threshold_2_m=map_near_threshold_2_m,
        )
        xs.append(feat)
        ids.append(traj_id)

    if not xs:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), []
    return np.vstack(xs).astype(np.float32), ids


def build_model(model_type: str, seed: int, ridge_alpha: float, hgb_max_depth: int, hgb_max_iter: int):
    if model_type == "linear":
        return make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha, random_state=seed))

    if model_type == "hgb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=hgb_max_depth,
            max_iter=hgb_max_iter,
            min_samples_leaf=40,
            l2_regularization=0.05,
            random_state=seed,
        )

    raise ValueError(f"Unknown model type: {model_type}")


def _forward_target(y_sec: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y_sec
    if mode == "log1p":
        return np.log1p(y_sec)
    raise ValueError(f"Unknown target transform: {mode}")


def _inverse_target(y_model: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y_model
    if mode == "log1p":
        return np.expm1(y_model)
    raise ValueError(f"Unknown target transform: {mode}")


def _align_feature_matrix_to_model(x: np.ndarray, model) -> np.ndarray:
    expected = getattr(model, "n_features_in_", None)
    if expected is None:
        return x

    cur = int(x.shape[1])
    exp = int(expected)
    if cur == exp:
        return x
    if cur > exp:
        return x[:, :exp]

    pad = np.zeros((x.shape[0], exp - cur), dtype=x.dtype)
    return np.concatenate([x, pad], axis=1)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)

    return {
        "count": int(y_true.size),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


def evaluate_by_buckets(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    out: Dict[str, Dict[str, float]] = {}
    buckets = {
        "short_le_10m": y_true <= 600.0,
        "medium_10m_30m": (y_true > 600.0) & (y_true <= 1800.0),
        "long_gt_30m": y_true > 1800.0,
    }

    for name, mask in buckets.items():
        if not np.any(mask):
            out[name] = {"count": 0, "mae": math.nan, "rmse": math.nan, "mape": math.nan}
            continue
        out[name] = evaluate_regression(y_true[mask], y_pred[mask])
    return out


def fit_and_predict(
    x_train: np.ndarray,
    y_train_sec: np.ndarray,
    x_pred: Optional[np.ndarray],
    model_type: str,
    target_transform: str,
    weight_mode: str,
    seed: int,
    ridge_alpha: float,
    hgb_max_depth: int,
    hgb_max_iter: int,
):
    model = build_model(
        model_type=model_type,
        seed=seed,
        ridge_alpha=ridge_alpha,
        hgb_max_depth=hgb_max_depth,
        hgb_max_iter=hgb_max_iter,
    )

    y_train_model = _forward_target(y_train_sec, mode=target_transform)
    w_train = _sample_weight(y_train_sec, mode=weight_mode)

    model.fit(x_train, y_train_model, **{"sample_weight": w_train})

    y_train_hat_model = model.predict(x_train)
    y_train_hat = _inverse_target(y_train_hat_model, mode=target_transform)
    y_train_hat = np.clip(y_train_hat, 1.0, None)

    y_pred = None
    if x_pred is not None:
        y_pred_model = model.predict(x_pred)
        y_pred = _inverse_target(y_pred_model, mode=target_transform)
        y_pred = np.clip(y_pred, 1.0, None)

    return model, y_train_hat, y_pred


def train_command(args: argparse.Namespace) -> None:
    map_runtime = None
    if args.osm is not None:
        map_runtime = build_map_runtime(
            osm_path=args.osm,
            cache_path=args.map_cache,
            force_rebuild=args.map_force_rebuild,
            cell_size_deg=args.map_cell_size_deg,
        )
        print(
            "[mapfeat] enabled | "
            f"segments={int(map_runtime['road_segments']['lon1'].size)} | "
            f"query_radius_m={args.map_query_radius_m} | max_points={args.map_max_points}"
        )

    train_records = load_pickle(args.train)
    x_train, y_train, ds_stats = build_supervised_dataset(
        records=train_records,
        min_travel_time=args.min_travel_time,
        max_travel_time=args.max_travel_time,
        min_speed_kmh=args.min_speed_kmh,
        max_speed_kmh=args.max_speed_kmh,
        map_runtime=map_runtime,
        map_query_radius_m=args.map_query_radius_m,
        map_max_points=args.map_max_points,
        map_near_threshold_1_m=args.map_near_threshold_1_m,
        map_near_threshold_2_m=args.map_near_threshold_2_m,
        limit=args.train_limit,
    )

    x_val = None
    val_ids: Optional[List[int]] = None
    y_val = None
    if args.val_input is not None:
        val_in = load_pickle(args.val_input)
        x_val, val_ids = build_inference_dataset(
            records=val_in,
            map_runtime=map_runtime,
            map_query_radius_m=args.map_query_radius_m,
            map_max_points=args.map_max_points,
            map_near_threshold_1_m=args.map_near_threshold_1_m,
            map_near_threshold_2_m=args.map_near_threshold_2_m,
            limit=args.val_limit,
        )
    if args.val_gt is not None:
        val_gt = load_pickle(args.val_gt)
        y_val = np.asarray([float(item["travel_time"]) for item in val_gt[: (len(val_gt) if args.val_limit <= 0 else args.val_limit)]], dtype=np.float64)

    model, y_train_hat, y_val_hat = fit_and_predict(
        x_train=x_train,
        y_train_sec=y_train,
        x_pred=x_val,
        model_type=args.model_type,
        target_transform=args.target_transform,
        weight_mode=args.weight_mode,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
        hgb_max_depth=args.hgb_max_depth,
        hgb_max_iter=args.hgb_max_iter,
    )

    train_metrics = evaluate_regression(y_train, y_train_hat)
    train_buckets = evaluate_by_buckets(y_train, y_train_hat)

    summary: Dict[str, Dict] = {
        "dataset_stats": ds_stats,
        "train": train_metrics,
        "train_buckets": train_buckets,
    }

    if y_val_hat is not None and y_val is not None:
        if len(y_val) != len(y_val_hat):
            raise ValueError(f"val size mismatch: gt={len(y_val)} pred={len(y_val_hat)}")
        val_metrics = evaluate_regression(y_val, y_val_hat)
        val_buckets = evaluate_by_buckets(y_val, y_val_hat)
        summary["val"] = val_metrics
        summary["val_buckets"] = val_buckets

        if args.val_pred_out is not None and val_ids is not None:
            pred_rows = [{"traj_id": int(tid), "travel_time": float(pred)} for tid, pred in zip(val_ids, y_val_hat)]
            save_pickle(args.val_pred_out, pred_rows)
            print(f"Saved val predictions: {args.val_pred_out} (n={len(pred_rows)})")

    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "model_type": args.model_type,
        "target_transform": args.target_transform,
        "weight_mode": args.weight_mode,
        "seed": args.seed,
        "ridge_alpha": args.ridge_alpha,
        "hgb_max_depth": args.hgb_max_depth,
        "hgb_max_iter": args.hgb_max_iter,
        "cleaning": {
            "min_travel_time": args.min_travel_time,
            "max_travel_time": args.max_travel_time,
            "min_speed_kmh": args.min_speed_kmh,
            "max_speed_kmh": args.max_speed_kmh,
        },
        "map_features": {
            "enabled": args.osm is not None,
            "osm_path": str(args.osm) if args.osm is not None else None,
            "map_cache": str(args.map_cache) if args.map_cache is not None else None,
            "map_cell_size_deg": args.map_cell_size_deg,
            "map_query_radius_m": args.map_query_radius_m,
            "map_max_points": args.map_max_points,
            "map_near_threshold_1_m": args.map_near_threshold_1_m,
            "map_near_threshold_2_m": args.map_near_threshold_2_m,
        },
        "summary": summary,
    }

    save_pickle(args.model_out, artifact)
    print(f"Saved model artifact: {args.model_out}")

    if args.metrics_out is not None:
        save_json(args.metrics_out, summary)
        print(f"Saved metrics json: {args.metrics_out}")

    print("\n=== Train Metrics ===")
    print(json.dumps(train_metrics, ensure_ascii=False, indent=2))
    if "val" in summary:
        print("\n=== Val Metrics ===")
        print(json.dumps(summary["val"], ensure_ascii=False, indent=2))


def predict_command(args: argparse.Namespace) -> None:
    artifact = load_pickle(args.model_in)
    model = artifact["model"]
    target_transform = artifact["target_transform"]

    map_cfg = artifact.get("map_features", {})
    map_enabled = bool(map_cfg.get("enabled", False))
    map_runtime = None
    if map_enabled:
        osm_path = args.osm if args.osm is not None else (
            Path(map_cfg["osm_path"]) if map_cfg.get("osm_path") else None
        )
        if osm_path is None:
            raise ValueError("Model requires map features. Provide --osm or ensure osm path is stored in artifact.")

        cache_path = args.map_cache
        if cache_path is None and map_cfg.get("map_cache"):
            cache_path = Path(map_cfg["map_cache"])

        cell_size = args.map_cell_size_deg if args.map_cell_size_deg is not None else float(map_cfg.get("map_cell_size_deg", 0.001))
        map_runtime = build_map_runtime(
            osm_path=Path(osm_path),
            cache_path=cache_path,
            force_rebuild=args.map_force_rebuild,
            cell_size_deg=cell_size,
        )

    map_query_radius_m = float(map_cfg.get("map_query_radius_m", 120.0))
    map_max_points = int(map_cfg.get("map_max_points", 12))
    map_near_threshold_1_m = float(map_cfg.get("map_near_threshold_1_m", 20.0))
    map_near_threshold_2_m = float(map_cfg.get("map_near_threshold_2_m", 50.0))

    recs = load_pickle(args.input)
    x, ids = build_inference_dataset(
        records=recs,
        map_runtime=map_runtime,
        map_query_radius_m=map_query_radius_m,
        map_max_points=map_max_points,
        map_near_threshold_1_m=map_near_threshold_1_m,
        map_near_threshold_2_m=map_near_threshold_2_m,
        limit=args.limit,
    )
    x = _align_feature_matrix_to_model(x, model)
    y_model = model.predict(x)
    y_pred = _inverse_target(np.asarray(y_model, dtype=np.float64), mode=target_transform)
    y_pred = np.clip(y_pred, 1.0, None)

    out_rows = [{"traj_id": int(tid), "travel_time": float(tt)} for tid, tt in zip(ids, y_pred)]
    save_pickle(args.output, out_rows)
    print(f"Saved predictions: {args.output} (n={len(out_rows)})")


def evaluate_command(args: argparse.Namespace) -> None:
    pred = load_pickle(args.pred)
    gt = load_pickle(args.gt)

    pred_by_id = {int(item["traj_id"]): float(item["travel_time"]) for item in pred}
    y_true = []
    y_pred = []
    for item in gt:
        tid = int(item["traj_id"])
        if tid not in pred_by_id:
            continue
        y_true.append(float(item["travel_time"]))
        y_pred.append(float(pred_by_id[tid]))

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    metrics = evaluate_regression(y_true_arr, y_pred_arr)
    buckets = evaluate_by_buckets(y_true_arr, y_pred_arr)

    print("=== Global ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n=== Buckets ===")
    print(json.dumps(buckets, ensure_ascii=False, indent=2))

    if args.metrics_out is not None:
        save_json(args.metrics_out, {"global": metrics, "buckets": buckets})
        print(f"Saved metrics json: {args.metrics_out}")


def check_align_command(args: argparse.Namespace) -> None:
    taskb = load_pickle(args.taskb_input)
    ds15 = load_pickle(args.ds15_val)

    n = min(len(taskb), len(ds15))
    if args.limit > 0:
        n = min(n, args.limit)

    mismatch_dep = 0
    mismatch_len = 0
    mismatch_first = 0
    mismatch_last = 0

    for i in range(n):
        a = taskb[i]
        b = ds15[i]

        dep_a = int(a["departure_timestamp"])
        dep_b = int(b["timestamps"][0])
        if dep_a != dep_b:
            mismatch_dep += 1

        ca = np.asarray(a["coords"], dtype=np.float64)
        cb = np.asarray(b["coords"], dtype=np.float64)
        if ca.shape[0] != cb.shape[0]:
            mismatch_len += 1
            continue

        if ca.shape[0] > 0:
            if float(np.max(np.abs(ca[0] - cb[0]))) > 1e-9:
                mismatch_first += 1
            if float(np.max(np.abs(ca[-1] - cb[-1]))) > 1e-9:
                mismatch_last += 1

    print(
        json.dumps(
            {
                "checked": n,
                "mismatch_departure": mismatch_dep,
                "mismatch_length": mismatch_len,
                "mismatch_first_point": mismatch_first,
                "mismatch_last_point": mismatch_last,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Task B baseline: trip-level travel time estimation")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train model and optionally evaluate on val")
    train.add_argument("--train", type=Path, default=Path("data_ds15/train.pkl"))
    train.add_argument("--model-out", type=Path, default=Path("task_B_tte/model_baseline_hgb.pkl"))
    train.add_argument("--metrics-out", type=Path, default=Path("task_B_tte/metrics_baseline_hgb.json"))

    train.add_argument("--val-input", type=Path, default=Path("task_B_tte/val_input.pkl"))
    train.add_argument("--val-gt", type=Path, default=Path("task_B_tte/val_gt.pkl"))
    train.add_argument("--val-pred-out", type=Path, default=Path("task_B_tte/pred_val_baseline_hgb.pkl"))

    train.add_argument("--train-limit", type=int, default=0)
    train.add_argument("--val-limit", type=int, default=0)

    train.add_argument("--model-type", choices=["linear", "hgb"], default="hgb")
    train.add_argument("--target-transform", choices=["none", "log1p"], default="log1p")
    train.add_argument("--weight-mode", choices=["none", "short-boost"], default="short-boost")

    train.add_argument("--min-travel-time", type=float, default=60.0)
    train.add_argument("--max-travel-time", type=float, default=7200.0)
    train.add_argument("--min-speed-kmh", type=float, default=3.0)
    train.add_argument("--max-speed-kmh", type=float, default=120.0)

    train.add_argument("--seed", type=int, default=20260420)
    train.add_argument("--ridge-alpha", type=float, default=3.0)
    train.add_argument("--hgb-max-depth", type=int, default=8)
    train.add_argument("--hgb-max-iter", type=int, default=450)

    train.add_argument("--osm", type=Path, default=None)
    train.add_argument("--map-cache", type=Path, default=Path("task_B_tte/map_segments_cache.pkl"))
    train.add_argument("--map-force-rebuild", action="store_true")
    train.add_argument("--map-cell-size-deg", type=float, default=0.001)
    train.add_argument("--map-query-radius-m", type=float, default=120.0)
    train.add_argument("--map-max-points", type=int, default=14)
    train.add_argument("--map-near-threshold-1-m", type=float, default=20.0)
    train.add_argument("--map-near-threshold-2-m", type=float, default=50.0)

    pred = sub.add_parser("predict", help="Predict on an input pkl")
    pred.add_argument("--model-in", type=Path, required=True)
    pred.add_argument("--input", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    pred.add_argument("--limit", type=int, default=0)
    pred.add_argument("--osm", type=Path, default=None)
    pred.add_argument("--map-cache", type=Path, default=None)
    pred.add_argument("--map-force-rebuild", action="store_true")
    pred.add_argument("--map-cell-size-deg", type=float, default=None)

    eva = sub.add_parser("evaluate", help="Evaluate prediction against gt")
    eva.add_argument("--pred", type=Path, required=True)
    eva.add_argument("--gt", type=Path, required=True)
    eva.add_argument("--metrics-out", type=Path, default=None)

    chk = sub.add_parser("check-align", help="Check whether task_B val_input aligns with ds15 val")
    chk.add_argument("--taskb-input", type=Path, default=Path("task_B_tte/val_input.pkl"))
    chk.add_argument("--ds15-val", type=Path, default=Path("data_ds15/val.pkl"))
    chk.add_argument("--limit", type=int, default=2000)

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "check-align":
        check_align_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
