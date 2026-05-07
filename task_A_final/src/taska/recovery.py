from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
)

from .interpolation import interpolate_traj
from .io_utils import load_pickle, load_text_config, pathify, save_csv, save_pickle
from .legacy_bridge import load_legacy_recovery_module
from .metrics import extract_gap_infos
from .route_projection import route_projection_fill, route_s_fill
from .geo import haversine_meters, point_distance_m, polyline_lengths_m
from .selector_features import build_selector_decisions, build_selector_feature_rows, feature_matrix, load_feature_spec, mix_predictions_with_selector, read_typed_csv, validate_feature_columns


def _records_from_arrays(input_records: List[Dict[str, Any]], pred_arrays: Dict[int, np.ndarray]) -> List[Dict[str, Any]]:
    out = []
    for item in input_records:
        tid = int(item["traj_id"])
        out.append({"traj_id": tid, "coords": np.asarray(pred_arrays[tid], dtype=np.float32)})
    return out


def _predict_pchip_only(input_records: List[Dict[str, Any]], mode: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pred_arrays: Dict[int, np.ndarray] = {}
    for item in input_records:
        tid = int(item["traj_id"])
        pred_arrays[tid] = interpolate_traj(
            timestamps=np.asarray(item["timestamps"], dtype=np.float64),
            coords=np.asarray(item["coords"], dtype=np.float64),
            mask=np.asarray(item["mask"], dtype=bool),
            mode=mode,
        )
    return _records_from_arrays(input_records, pred_arrays), {"predict_impl": mode, "segments_projected": 0, "segments_fallback": 0}


def _build_route_polyline(graph: Dict[str, Any], path_nodes: List[int], start_coord: np.ndarray, end_coord: np.ndarray) -> np.ndarray:
    pts = [np.asarray(start_coord, dtype=np.float64)]
    for nid in path_nodes:
        idx = graph["id_to_idx"].get(int(nid))
        if idx is None:
            continue
        pts.append(np.array([graph["lons"][idx], graph["lats"][idx]], dtype=np.float64))
    pts.append(np.asarray(end_coord, dtype=np.float64))
    poly = np.vstack(pts)
    keep = [0]
    for i in range(1, poly.shape[0]):
        if not np.allclose(poly[i], poly[keep[-1]]):
            keep.append(i)
    return poly[keep]


def _infer_dataset_name(input_path: Path) -> str:
    name = input_path.stem.lower()
    if "16" in name:
        return "1/16"
    if "8" in name:
        return "1/8"
    return "unknown"


def _empty_debug_row(dataset_name: str, traj_id: int, gap, fallback_reason: str) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "traj_id": int(traj_id),
        "gap_start_idx": int(gap.start_idx),
        "gap_end_idx": int(gap.end_idx),
        "gap_size": int(gap.missing_count),
        "route_found": False,
        "route_length_m": math.nan,
        "anchor_direct_m": math.nan,
        "detour_ratio": math.nan,
        "start_snap_m": math.nan,
        "end_snap_m": math.nan,
        "base_to_route_mean_m": math.nan,
        "base_to_route_p95_m": math.nan,
        "projection_s_monotonic_violations": 0,
        "projection_s_clamped_mean_m": math.nan,
        "projection_s_clamped_max_m": math.nan,
        "route_sample_vs_base_mean_m": math.nan,
        "fallback_reason": fallback_reason,
        "applied_projection": False,
    }


def _apply_route_mode(
    input_records: List[Dict[str, Any]],
    config: Dict[str, Any],
    dataset_name: str,
    debug_out_path: Path | None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    legacy = load_legacy_recovery_module()
    map_cfg = config.get("map", {})
    routing_cfg = config.get("routing", {})
    route_cfg = config.get("route_projection", {})
    interp_mode = str(config.get("base_interpolation", "pchip"))
    graph = legacy.load_or_build_graph(
        osm_path=pathify(map_cfg.get("osm_path", "map")),
        cache_path=pathify(map_cfg.get("cache_path", "task_A_final/caches/map_graph_cache.pkl")),
        cell_size_deg=float(map_cfg.get("cell_size_deg", 0.0015)),
        force_rebuild=bool(map_cfg.get("force_rebuild", False)),
    )
    pred_arrays: Dict[int, np.ndarray] = {}
    debug_rows: List[Dict[str, Any]] = []
    stats = {
        "predict_impl": str(config.get("strategy", {}).get("name", "route_projection")),
        "segments_projected": 0,
        "segments_fallback": 0,
        "projection_raw_backtracks": 0,
    }
    for item in input_records:
        coords = np.asarray(item["coords"], dtype=np.float64)
        mask = np.asarray(item["mask"], dtype=bool)
        timestamps = np.asarray(item["timestamps"], dtype=np.float64)
        pred = interpolate_traj(timestamps=timestamps, coords=coords, mask=mask, mode=interp_mode).astype(np.float64)
        gaps, _ = extract_gap_infos(mask=mask, timestamps=timestamps)
        for gap in gaps:
            threshold = int(route_cfg.get("min_gap", 5))
            if gap.missing_count < threshold:
                debug_rows.append(_empty_debug_row(dataset_name, int(item["traj_id"]), gap, "not_triggered"))
                continue
            start = gap.start_idx
            end = gap.end_idx
            anchor_direct_m = float(point_distance_m(coords[start], coords[end]))
            debug_row = _empty_debug_row(dataset_name, int(item["traj_id"]), gap, "unknown")
            debug_row["anchor_direct_m"] = anchor_direct_m
            node_a, dist_a = legacy.nearest_node_in_radius(
                lon=float(coords[start, 0]),
                lat=float(coords[start, 1]),
                graph=graph,
                radius_m=float(map_cfg.get("candidate_radius_m", 180.0)),
            )
            node_b, dist_b = legacy.nearest_node_in_radius(
                lon=float(coords[end, 0]),
                lat=float(coords[end, 1]),
                graph=graph,
                radius_m=float(map_cfg.get("candidate_radius_m", 180.0)),
            )
            debug_row["start_snap_m"] = float(dist_a) if np.isfinite(dist_a) else math.nan
            debug_row["end_snap_m"] = float(dist_b) if np.isfinite(dist_b) else math.nan
            if node_a is None or node_b is None:
                stats["segments_fallback"] += 1
                debug_row["fallback_reason"] = "snap_far"
                debug_rows.append(debug_row)
                continue
            path_nodes = legacy.get_route_path(
                u=int(node_a),
                v=int(node_b),
                graph=graph,
                path_cache={},
                astar_max_expansions=int(routing_cfg.get("astar_max_expansions", 7000)),
                astar_road_class_weight=float(routing_cfg.get("astar_road_class_weight", 1.0)),
                astar_turn_penalty_m=float(routing_cfg.get("astar_turn_penalty_m", 55.0)),
                astar_turn_angle_threshold_deg=float(routing_cfg.get("astar_turn_angle_threshold_deg", 55.0)),
            )
            if not path_nodes or len(path_nodes) < 2:
                stats["segments_fallback"] += 1
                debug_row["fallback_reason"] = "no_route"
                debug_rows.append(debug_row)
                continue
            route_polyline = _build_route_polyline(graph, path_nodes, coords[start], coords[end])
            if route_polyline.shape[0] < 2:
                stats["segments_fallback"] += 1
                debug_row["fallback_reason"] = "no_route"
                debug_rows.append(debug_row)
                continue
            route_length_m = float(polyline_lengths_m(route_polyline)[-1])
            debug_row["route_found"] = True
            debug_row["route_length_m"] = route_length_m
            debug_row["detour_ratio"] = float(route_length_m / max(anchor_direct_m, 1e-6))
            base_seg = pred[start + 1 : end]
            try:
                if str(config.get("strategy", {}).get("name")) == "route_s":
                    filled, seg_stats = route_s_fill(base_seg, route_polyline, beta=float(route_cfg.get("beta", 1.0)))
                else:
                    filled, seg_stats = route_projection_fill(base_seg, route_polyline)
            except Exception:
                stats["segments_fallback"] += 1
                debug_row["fallback_reason"] = "monotonic_bad"
                debug_rows.append(debug_row)
                continue
            route_sample_vs_base = np.asarray(
                haversine_meters(base_seg[:, 0], base_seg[:, 1], filled[:, 0], filled[:, 1]),
                dtype=np.float64,
            ) if base_seg.size else np.empty(0, dtype=np.float64)
            debug_row["base_to_route_mean_m"] = float(seg_stats.get("projection_mean_dist_m", math.nan))
            debug_row["base_to_route_p95_m"] = float(seg_stats.get("projection_p95_dist_m", math.nan))
            debug_row["projection_s_monotonic_violations"] = int(seg_stats.get("projection_raw_backtracks", 0))
            debug_row["projection_s_clamped_mean_m"] = float(seg_stats.get("projection_clamped_mean_m", math.nan))
            debug_row["projection_s_clamped_max_m"] = float(seg_stats.get("projection_clamped_max_m", math.nan))
            debug_row["route_sample_vs_base_mean_m"] = float(np.mean(route_sample_vs_base)) if route_sample_vs_base.size else 0.0
            max_projection_dist = float(route_cfg.get("max_projection_dist_m", 250.0))
            if seg_stats.get("projection_max_dist_m", 0.0) > max_projection_dist:
                stats["segments_fallback"] += 1
                debug_row["fallback_reason"] = "projection_far"
                debug_rows.append(debug_row)
                continue
            pred[start + 1 : end] = filled
            stats["segments_projected"] += 1
            stats["projection_raw_backtracks"] += int(seg_stats.get("projection_raw_backtracks", 0))
            stats["last_snap_dist_mean_m"] = float(0.5 * (dist_a + dist_b))
            debug_row["fallback_reason"] = "applied"
            debug_row["applied_projection"] = True
            debug_rows.append(debug_row)
        pred[mask] = coords[mask]
        pred_arrays[int(item["traj_id"])] = pred.astype(np.float32)
    if debug_out_path is not None:
        save_csv(debug_out_path, debug_rows)
        stats["debug_gap_csv"] = str(debug_out_path)
        stats["debug_gap_rows"] = len(debug_rows)
    return _records_from_arrays(input_records, pred_arrays), stats


def _run_legacy_wrapper(config_path: Path, input_path: Path, output_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    script = root / "task_A_recovery" / "baseline2_hmm_map_recovery.py"
    wrapper_cfg = config.get("legacy_wrapper", {})
    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--map",
        str(pathify(wrapper_cfg.get("map_path", "map"))),
        "--cache",
        str(pathify(wrapper_cfg.get("cache_path", "task_A_final/caches/map_graph_cache.pkl"))),
    ]
    for arg in wrapper_cfg.get("args", []):
        cmd.append(str(arg))
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return {
        "predict_impl": "legacy_wrapper",
        "wrapper_command": cmd,
        "wrapper_runtime_sec": time.time() - start,
        "wrapper_stdout_tail": proc.stdout[-2000:],
        "wrapper_stderr_tail": proc.stderr[-2000:],
    }


def _load_nested_config(path_str: str | Path) -> Dict[str, Any]:
    cfg_path = pathify(path_str)
    if cfg_path is None:
        raise ValueError("Nested config path is required")
    config = load_text_config(cfg_path)
    config["_config_path"] = str(cfg_path.resolve())
    return config


def _predict_selector_mix(config: Dict[str, Any], input_path: Path, output_path: Path, debug_out_path: Path | None) -> Dict[str, Any]:
    dataset_name = _infer_dataset_name(input_path)
    base_cfg = _load_nested_config(config.get("base", {}).get("config"))
    route_cfg = _load_nested_config(config.get("route_candidate", {}).get("config"))
    fallback_cfg_path = config.get("fallback", {}).get("config")
    fallback_enabled = bool(config.get("fallback", {}).get("enabled", False))
    try:
        selector_cfg = config.get("selector", {})
        feature_spec = load_feature_spec(pathify(selector_cfg.get("feature_columns")))
        threshold = float(selector_cfg.get("threshold", 0.50))
        model = joblib.load(pathify(selector_cfg.get("model")))

        precomputed_base_path = pathify(config.get("base", {}).get("pred_path"))
        precomputed_route_path = pathify(config.get("route_candidate", {}).get("pred_path"))
        precomputed_debug_path = pathify(config.get("route_candidate", {}).get("debug_csv"))
        if precomputed_base_path is not None and precomputed_route_path is not None and precomputed_debug_path is not None:
            input_records = load_pickle(input_path)
            base_pred = load_pickle(precomputed_base_path)
            route_pred = load_pickle(precomputed_route_path)
            debug_rows = [r for r in read_typed_csv(precomputed_debug_path) if str(r["dataset"]) == dataset_name]
            feature_rows = build_selector_feature_rows(dataset_name, input_records, base_pred, route_pred, debug_rows)
            feature_columns = list(feature_spec["feature_columns"])
            actual_cols = [c for c in feature_rows[0].keys() if c.startswith("feature_")]
            validate_feature_columns(feature_columns, actual_cols)
            x, matrix_stats = feature_matrix(feature_rows, feature_columns)
            probs = model.predict_proba(x)[:, 1]
            selected_keys, decision_rows, selection_meta = build_selector_decisions(feature_rows, probs, threshold)
            mixed_pred = mix_predictions_with_selector(dataset_name, input_records, base_pred, route_pred, selected_keys)
            save_pickle(output_path, mixed_pred)
            if debug_out_path is not None:
                save_csv(debug_out_path, decision_rows)
            return {
                "predict_impl": "selector_mix",
                "dataset": dataset_name,
                "threshold": threshold,
                "feature_schema_hash": feature_spec["feature_schema_hash"],
                "feature_nan_count": matrix_stats["feature_nan_count"],
                "feature_clip_count": matrix_stats["feature_clip_count"],
                "selected_gap_count": selection_meta["selected_gap_count"],
                "selection_rate": selection_meta["selection_rate"],
                "selected_missing_point_count": selection_meta["selected_missing_point_count"],
                "prob_mean": selection_meta["prob_mean"],
                "prob_p95": selection_meta["prob_p95"],
                "decision_csv": str(debug_out_path) if debug_out_path is not None else None,
                "used_precomputed_candidates": True,
            }

        with tempfile.TemporaryDirectory(prefix="selector_mix_", dir=str(output_path.parent)) as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            base_pred_path = tmpdir / f"base_{dataset_name.replace('/', '_')}.pkl"
            route_pred_path = tmpdir / f"route_{dataset_name.replace('/', '_')}.pkl"
            route_debug_path = tmpdir / f"route_debug_{dataset_name.replace('/', '_')}.csv"

            base_stats = predict_records(base_cfg, input_path=input_path, output_path=base_pred_path, debug_out_path=None)
            route_stats = predict_records(route_cfg, input_path=input_path, output_path=route_pred_path, debug_out_path=route_debug_path)

            input_records = load_pickle(input_path)
            base_pred = load_pickle(base_pred_path)
            route_pred = load_pickle(route_pred_path)
            debug_rows = [r for r in read_typed_csv(route_debug_path) if str(r["dataset"]) == dataset_name]
            feature_rows = build_selector_feature_rows(dataset_name, input_records, base_pred, route_pred, debug_rows)
            feature_columns = list(feature_spec["feature_columns"])
            actual_cols = [c for c in feature_rows[0].keys() if c.startswith("feature_")]
            validate_feature_columns(feature_columns, actual_cols)
            x, matrix_stats = feature_matrix(feature_rows, feature_columns)
            probs = model.predict_proba(x)[:, 1]
            selected_keys, decision_rows, selection_meta = build_selector_decisions(feature_rows, probs, threshold)
            mixed_pred = mix_predictions_with_selector(dataset_name, input_records, base_pred, route_pred, selected_keys)
            save_pickle(output_path, mixed_pred)
            if debug_out_path is not None:
                save_csv(debug_out_path, decision_rows)
            return {
                "predict_impl": "selector_mix",
                "dataset": dataset_name,
                "threshold": threshold,
                "feature_schema_hash": feature_spec["feature_schema_hash"],
                "feature_nan_count": matrix_stats["feature_nan_count"],
                "feature_clip_count": matrix_stats["feature_clip_count"],
                "selected_gap_count": selection_meta["selected_gap_count"],
                "selection_rate": selection_meta["selection_rate"],
                "selected_missing_point_count": selection_meta["selected_missing_point_count"],
                "prob_mean": selection_meta["prob_mean"],
                "prob_p95": selection_meta["prob_p95"],
                "base_stats": base_stats,
                "route_stats": route_stats,
                "decision_csv": str(debug_out_path) if debug_out_path is not None else None,
                "fallback_triggered": False,
            }
    except Exception as exc:
        if not fallback_enabled or not fallback_cfg_path:
            raise
        fallback_cfg = _load_nested_config(fallback_cfg_path)
        fallback_stats = predict_records(fallback_cfg, input_path=input_path, output_path=output_path, debug_out_path=None)
        return {
            "predict_impl": "selector_mix",
            "dataset": dataset_name,
            "fallback_triggered": True,
            "fallback_reason": repr(exc),
            "fallback_stats": fallback_stats,
        }


def predict_records(config: Dict[str, Any], input_path: Path, output_path: Path, debug_out_path: Path | None = None) -> Dict[str, Any]:
    input_records = load_pickle(input_path)
    strategy = str(config.get("strategy", {}).get("name", "pchip_only"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if strategy == "pchip_only":
        pred_records, stats = _predict_pchip_only(input_records, mode=str(config.get("base_interpolation", "pchip")))
        save_pickle(output_path, pred_records)
        return stats
    if strategy == "b28_compat":
        return _run_legacy_wrapper(Path(config["_config_path"]), input_path, output_path, config)
    if strategy in {"route_projection", "route_s"}:
        effective_debug = debug_out_path
        if effective_debug is None:
            route_cfg = config.get("route_projection", {})
            if route_cfg.get("debug_out_path"):
                effective_debug = pathify(route_cfg.get("debug_out_path"))
        pred_records, stats = _apply_route_mode(
            input_records,
            config,
            dataset_name=_infer_dataset_name(input_path),
            debug_out_path=effective_debug,
        )
        save_pickle(output_path, pred_records)
        return stats
    if strategy == "selector_mix":
        return _predict_selector_mix(config=config, input_path=input_path, output_path=output_path, debug_out_path=debug_out_path)
    raise ValueError(f"Unsupported strategy: {strategy}")
