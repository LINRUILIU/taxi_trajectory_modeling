from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .io_utils import load_pickle, save_json, save_text
from .legacy_bridge import load_legacy_analyze_module
from .metrics import build_id_map, extract_gap_infos, global_metrics, missing_metrics
from .plotting import plot_gap_distribution, plot_interval_distribution, plot_length_distribution, plot_official_vs_shape_scatter
from .shape_metrics import shape_symmetric_m


def _gap_size_bucket(dataset_name: str, gap_size: int) -> str:
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


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _gap_row(dataset_name: str, traj_id: int, gap, pred: np.ndarray, gt: np.ndarray) -> Dict:
    idx = gap.missing_indices
    pred_missing = pred[idx]
    gt_missing = gt[idx]
    errs = np.asarray(
        np.sqrt(np.square(0.0) + np.square(0.0)), dtype=np.float64
    )
    from .geo import haversine_meters

    errs = np.asarray(haversine_meters(pred_missing[:, 0], pred_missing[:, 1], gt_missing[:, 0], gt_missing[:, 1]), dtype=np.float64)
    official_mae = float(np.mean(errs)) if errs.size else 0.0
    gt_poly = gt[gap.start_idx : gap.end_idx + 1]
    pred_poly = pred[gap.start_idx : gap.end_idx + 1]
    shape = shape_symmetric_m(gt_missing, pred_missing, gt_poly, pred_poly)
    return {
        "dataset": dataset_name,
        "traj_id": int(traj_id),
        "gap_start_idx": int(gap.start_idx),
        "gap_end_idx": int(gap.end_idx),
        "gap_size": int(gap.missing_count),
        "missing_point_count": int(gap.missing_count),
        "official_mae_m": official_mae,
        "official_rmse_m": float(np.sqrt(np.mean(np.square(errs)))) if errs.size else 0.0,
        "shape_symmetric_m": float(shape),
        "delta_t_sec": float(gap.delta_t_sec),
        "gap_bucket": _gap_size_bucket(dataset_name, int(gap.missing_count)),
        "official_error_sum_m": float(official_mae * int(gap.missing_count)),
    }


def _collect_gap_rows(dataset_name: str, input_records: Iterable[Dict], pred_records: Iterable[Dict], gt_records: Iterable[Dict]) -> Tuple[List[Dict], List[Dict], List[float]]:
    pred_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in pred_records}
    gt_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in gt_records}
    gap_rows: List[Dict] = []
    traj_rows: List[Dict] = []
    official_values: List[float] = []
    for item in input_records:
        tid = int(item["traj_id"])
        mask = np.asarray(item["mask"], dtype=bool)
        ts = np.asarray(item["timestamps"], dtype=np.float64)
        gaps, _ = extract_gap_infos(mask=mask, timestamps=ts)
        pred = pred_by_id[tid]
        gt = gt_by_id[tid]
        traj_gap_rows = [_gap_row(dataset_name, tid, gap, pred, gt) for gap in gaps]
        gap_rows.extend(traj_gap_rows)
        official_values.extend([r["official_mae_m"] for r in traj_gap_rows])
        if traj_gap_rows:
            traj_rows.append(
                {
                    "dataset": dataset_name,
                    "traj_id": tid,
                    "num_gaps": len(traj_gap_rows),
                    "official_mae_m": float(np.mean([r["official_mae_m"] for r in traj_gap_rows])),
                    "shape_symmetric_m": float(np.mean([r["shape_symmetric_m"] for r in traj_gap_rows])),
                    "max_gap_size": int(max(r["gap_size"] for r in traj_gap_rows)),
                }
            )
    return gap_rows, traj_rows, official_values


def _collect_point_level_metrics(input_records: Iterable[Dict], pred_records: Iterable[Dict], gt_records: Iterable[Dict]) -> Dict[str, float]:
    pred_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in pred_records}
    gt_by_id = {int(x["traj_id"]): np.asarray(x["coords"], dtype=np.float64) for x in gt_records}
    all_errs: List[np.ndarray] = []
    total_missing = 0
    evaluated_missing = 0
    for item in input_records:
        tid = int(item["traj_id"])
        mask = np.asarray(item["mask"], dtype=bool)
        metric = missing_metrics(pred_by_id[tid], gt_by_id[tid], mask)
        total_missing += int(metric["total_missing"])
        evaluated_missing += int(metric["evaluated_missing"])
        missing = ~mask
        if np.any(missing):
            pred_m = pred_by_id[tid][missing]
            gt_m = gt_by_id[tid][missing]
            valid = np.isfinite(pred_m[:, 0]) & np.isfinite(pred_m[:, 1]) & np.isfinite(gt_m[:, 0]) & np.isfinite(gt_m[:, 1])
            if np.any(valid):
                from .geo import haversine_meters

                errs = np.asarray(
                    haversine_meters(pred_m[valid, 0], pred_m[valid, 1], gt_m[valid, 0], gt_m[valid, 1]),
                    dtype=np.float64,
                )
                all_errs.append(errs)
    if not all_errs:
        return {"count": 0, "mae": math.nan, "rmse": math.nan, "p75": math.nan, "p95": math.nan}
    arr = np.concatenate(all_errs)
    return {
        "count": int(arr.size),
        "total_missing": int(total_missing),
        "evaluated_missing": int(evaluated_missing),
        "mae": float(np.mean(arr)),
        "rmse": float(np.sqrt(np.mean(np.square(arr)))),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def _quadrant_label(official: float, shape: float, official_thr: float, shape_thr: float) -> str:
    if official >= official_thr and shape >= shape_thr:
        return "high_official_high_shape"
    if official >= official_thr and shape < shape_thr:
        return "high_official_low_shape"
    if official < official_thr and shape >= shape_thr:
        return "low_official_high_shape"
    return "low_official_low_shape"


def _compute_thresholds(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {"official_threshold_m": 0.0, "shape_threshold_m": 0.0}
    official_thr = float(np.percentile([r["official_mae_m"] for r in rows], 75))
    shape_thr = float(np.percentile([r["shape_symmetric_m"] for r in rows], 75))
    return {"official_threshold_m": official_thr, "shape_threshold_m": shape_thr}


def _assign_quadrants(rows: List[Dict]) -> tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[Tuple[str, str], Dict[str, float]]]:
    global_thr = _compute_thresholds(rows)
    by_dataset_thr: Dict[str, Dict[str, float]] = {}
    by_bucket_thr: Dict[Tuple[str, str], Dict[str, float]] = {}

    for dataset in sorted({r["dataset"] for r in rows}):
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        by_dataset_thr[dataset] = _compute_thresholds(dataset_rows)

    for key in sorted({(r["dataset"], r["gap_bucket"]) for r in rows}):
        bucket_rows = [r for r in rows if (r["dataset"], r["gap_bucket"]) == key]
        by_bucket_thr[key] = _compute_thresholds(bucket_rows)

    for row in rows:
        row["quadrant_global"] = _quadrant_label(
            row["official_mae_m"],
            row["shape_symmetric_m"],
            global_thr["official_threshold_m"],
            global_thr["shape_threshold_m"],
        )
        d_thr = by_dataset_thr[row["dataset"]]
        row["quadrant_dataset"] = _quadrant_label(
            row["official_mae_m"],
            row["shape_symmetric_m"],
            d_thr["official_threshold_m"],
            d_thr["shape_threshold_m"],
        )
        b_thr = by_bucket_thr[(row["dataset"], row["gap_bucket"])]
        row["quadrant_gap_bucket"] = _quadrant_label(
            row["official_mae_m"],
            row["shape_symmetric_m"],
            b_thr["official_threshold_m"],
            b_thr["shape_threshold_m"],
        )
    return global_thr, by_dataset_thr, by_bucket_thr


def _bucket_summary(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        key = (row["dataset"], row["gap_bucket"])
        grouped.setdefault(key, []).append(row)
    out: List[Dict] = []
    for (dataset, bucket), bucket_rows in grouped.items():
        official = np.asarray([r["official_mae_m"] for r in bucket_rows], dtype=np.float64)
        shape = np.asarray([r["shape_symmetric_m"] for r in bucket_rows], dtype=np.float64)
        out.append(
            {
                "dataset": dataset,
                "gap_bucket": bucket,
                "count": int(len(bucket_rows)),
                "official_mae_m": float(np.mean(official)),
                "official_p95_m": float(np.percentile(official, 95)),
                "shape_symmetric_m": float(np.mean(shape)),
            }
        )
    out.sort(key=lambda x: (x["dataset"], x["gap_bucket"]))
    return out


def _quadrant_stats(rows: List[Dict], label_field: str) -> Dict[str, Dict[str, float]]:
    total_error = float(sum(r["official_error_sum_m"] for r in rows))
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row[label_field]].append(row)
    out: Dict[str, Dict[str, float]] = {}
    for quadrant, q_rows in grouped.items():
        error_sum = float(sum(r["official_error_sum_m"] for r in q_rows))
        missing_points = int(sum(int(r["missing_point_count"]) for r in q_rows))
        gap_count = int(len(q_rows))
        out[quadrant] = {
            "gap_count": gap_count,
            "missing_point_count": missing_points,
            "official_error_sum_m": error_sum,
            "official_error_share": float(error_sum / max(total_error, 1e-9)),
            "official_mae_mean_m": float(np.mean([r["official_mae_m"] for r in q_rows])) if q_rows else 0.0,
            "shape_symmetric_mean_m": float(np.mean([r["shape_symmetric_m"] for r in q_rows])) if q_rows else 0.0,
        }
    for quadrant in (
        "high_official_high_shape",
        "high_official_low_shape",
        "low_official_high_shape",
        "low_official_low_shape",
    ):
        out.setdefault(
            quadrant,
            {
                "gap_count": 0,
                "missing_point_count": 0,
                "official_error_sum_m": 0.0,
                "official_error_share": 0.0,
                "official_mae_mean_m": 0.0,
                "shape_symmetric_mean_m": 0.0,
            },
        )
    return out


def _review_queues(rows: List[Dict], out_dir: Path) -> Dict[str, int]:
    queues = {
        "review_queue_path_wrong.csv": [r for r in rows if r["quadrant_global"] == "high_official_high_shape"],
        "review_queue_phase_wrong.csv": [r for r in rows if r["quadrant_global"] == "high_official_low_shape"],
        "review_queue_metric_cheat.csv": [r for r in rows if r["quadrant_global"] == "low_official_high_shape"],
    }
    dataset_map = {"1/8": "8", "1/16": "16"}
    counts: Dict[str, int] = {}
    for name, queue_rows in queues.items():
        _write_csv(out_dir / name, queue_rows)
        counts[name] = len(queue_rows)
        stem = name.removesuffix(".csv")
        for dataset, suffix in dataset_map.items():
            dataset_rows = [r for r in queue_rows if r["dataset"] == dataset]
            dataset_name = f"{stem}_{suffix}.csv"
            _write_csv(out_dir / dataset_name, dataset_rows)
            counts[dataset_name] = len(dataset_rows)
    return counts


def _build_quadrant_outputs(
    rows: List[Dict],
    global_thr: Dict[str, float],
    by_dataset_thr: Dict[str, Dict[str, float]],
    by_bucket_thr: Dict[Tuple[str, str], Dict[str, float]],
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    global_summary: Dict[str, Any] = {
        "official_threshold_m": global_thr["official_threshold_m"],
        "shape_threshold_m": global_thr["shape_threshold_m"],
        "quadrants": _quadrant_stats(rows, "quadrant_global"),
    }
    dataset_summary: Dict[str, Any] = {}
    for dataset, thr in by_dataset_thr.items():
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        dataset_summary[dataset] = {
            "official_threshold_m": thr["official_threshold_m"],
            "shape_threshold_m": thr["shape_threshold_m"],
            "quadrants": _quadrant_stats(dataset_rows, "quadrant_dataset"),
        }
    bucket_summary: Dict[str, Any] = {}
    for (dataset, gap_bucket), thr in by_bucket_thr.items():
        bucket_rows = [r for r in rows if r["dataset"] == dataset and r["gap_bucket"] == gap_bucket]
        bucket_summary.setdefault(dataset, {})
        bucket_summary[dataset][gap_bucket] = {
            "official_threshold_m": thr["official_threshold_m"],
            "shape_threshold_m": thr["shape_threshold_m"],
            "quadrants": _quadrant_stats(bucket_rows, "quadrant_gap_bucket"),
        }
    return global_summary, dataset_summary, bucket_summary


def analyze_predictions(
    input8_path: Path,
    input16_path: Path,
    pred8_path: Path,
    pred16_path: Path,
    gt_path: Path,
    out_dir: Path,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    input8 = load_pickle(input8_path)
    input16 = load_pickle(input16_path)
    pred8 = load_pickle(pred8_path)
    pred16 = load_pickle(pred16_path)
    gt = load_pickle(gt_path)

    gap_rows8, traj_rows8, _ = _collect_gap_rows("1/8", input8, pred8, gt)
    gap_rows16, traj_rows16, _ = _collect_gap_rows("1/16", input16, pred16, gt)
    all_gap_rows = gap_rows8 + gap_rows16
    global_thr, by_dataset_thr, by_bucket_thr = _assign_quadrants(all_gap_rows)

    global_out = {
        "1/8": _collect_point_level_metrics(input8, pred8, gt),
        "1/16": _collect_point_level_metrics(input16, pred16, gt),
    }
    root = Path(__file__).resolve().parents[3]
    map_path = root / "map"
    if map_path.exists():
        legacy_analyze = load_legacy_analyze_module()
        road_segments = legacy_analyze.load_or_build_road_segments(
            osm_path=map_path,
            cache_path=root / "task_A_final" / "caches" / "map_roads_overlay_cache.pkl",
            force_rebuild=False,
        )
        seg_index = legacy_analyze.build_road_segment_grid_index(
            road_segments=road_segments,
            cell_size_deg=0.002,
        )
        global_out["1/8"].update(
            legacy_analyze.compute_topology_violation_metrics(
                input_records=input8,
                pred_records=pred8,
                road_segments=road_segments,
                seg_index=seg_index,
                violation_threshold_m=35.0,
                max_eval_points=30000,
                seed=42,
            )
        )
        global_out["1/16"].update(
            legacy_analyze.compute_topology_violation_metrics(
                input_records=input16,
                pred_records=pred16,
                road_segments=road_segments,
                seg_index=seg_index,
                violation_threshold_m=35.0,
                max_eval_points=30000,
                seed=43,
            )
        )
    save_json(out_dir / "global_metrics.json", global_out)

    gap_rows_all = gap_rows8 + gap_rows16
    traj_rows_all = traj_rows8 + traj_rows16
    _write_csv(out_dir / "gap_metrics.csv", gap_rows_all)
    _write_csv(out_dir / "trajectory_metrics.csv", traj_rows_all)

    bucket_rows = _bucket_summary(gap_rows_all)
    _write_csv(out_dir / "bucket_summary.csv", bucket_rows)
    plot_official_vs_shape_scatter(gap_rows_all, out_dir / "official_vs_shape_scatter_gap.png")

    quadrant_summary_global, quadrant_summary_by_dataset, quadrant_summary_by_gap_bucket = _build_quadrant_outputs(
        gap_rows_all,
        global_thr,
        by_dataset_thr,
        by_bucket_thr,
    )
    queue_counts = _review_queues(gap_rows_all, out_dir)
    quadrant_summary = {
        "official_threshold_m": quadrant_summary_global["official_threshold_m"],
        "shape_threshold_m": quadrant_summary_global["shape_threshold_m"],
    }
    for quadrant, stats in quadrant_summary_global["quadrants"].items():
        quadrant_summary[quadrant] = stats["gap_count"]
    quadrant_summary.update(queue_counts)
    save_json(out_dir / "quadrant_summary.json", quadrant_summary)
    save_json(out_dir / "quadrant_summary_global.json", quadrant_summary_global)
    save_json(out_dir / "quadrant_summary_by_dataset.json", quadrant_summary_by_dataset)
    save_json(out_dir / "quadrant_summary_by_gap_bucket.json", quadrant_summary_by_gap_bucket)

    decision_lines = [
        "# Gap-Level Decision Summary",
        "",
        f"- 1/8 official MAE: {global_out['1/8']['mae']:.4f} m",
        f"- 1/16 official MAE: {global_out['1/16']['mae']:.4f} m",
        f"- Global official threshold: {quadrant_summary_global['official_threshold_m']:.4f} m",
        f"- Global shape threshold: {quadrant_summary_global['shape_threshold_m']:.4f} m",
        f"- high_official_high_shape: {quadrant_summary_global['quadrants']['high_official_high_shape']['gap_count']}",
        f"- high_official_low_shape: {quadrant_summary_global['quadrants']['high_official_low_shape']['gap_count']}",
        f"- low_official_high_shape: {quadrant_summary_global['quadrants']['low_official_high_shape']['gap_count']}",
        f"- low_official_low_shape: {quadrant_summary_global['quadrants']['low_official_low_shape']['gap_count']}",
        "",
        "## Suggested Focus",
        "- `high_official_high_shape`: prioritize path/routing errors.",
        "- `high_official_low_shape`: prioritize phase/timing errors.",
        "- `low_official_high_shape`: inspect metric-cheat or unrealistic path geometry.",
    ]
    save_text(out_dir / "decision_summary.md", "\n".join(decision_lines) + "\n")
    return {
        "global_metrics": global_out,
        "quadrant_summary": quadrant_summary,
        "output_files": [
            out_dir / "global_metrics.json",
            out_dir / "bucket_summary.csv",
            out_dir / "trajectory_metrics.csv",
            out_dir / "gap_metrics.csv",
            out_dir / "official_vs_shape_scatter_gap.png",
            out_dir / "quadrant_summary.json",
            out_dir / "quadrant_summary_global.json",
            out_dir / "quadrant_summary_by_dataset.json",
            out_dir / "quadrant_summary_by_gap_bucket.json",
            out_dir / "review_queue_path_wrong.csv",
            out_dir / "review_queue_phase_wrong.csv",
            out_dir / "review_queue_metric_cheat.csv",
            out_dir / "review_queue_path_wrong_8.csv",
            out_dir / "review_queue_path_wrong_16.csv",
            out_dir / "review_queue_phase_wrong_8.csv",
            out_dir / "review_queue_phase_wrong_16.csv",
            out_dir / "review_queue_metric_cheat_8.csv",
            out_dir / "review_queue_metric_cheat_16.csv",
            out_dir / "decision_summary.md",
        ],
    }


def run_static_eda(input8_path: Path, input16_path: Path, out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    input8 = load_pickle(input8_path)
    input16 = load_pickle(input16_path)
    lengths8 = np.asarray([len(item["coords"]) for item in input8], dtype=np.int32)
    lengths16 = np.asarray([len(item["coords"]) for item in input16], dtype=np.int32)
    dt8 = np.concatenate([np.diff(np.asarray(item["timestamps"], dtype=np.int64)) for item in input8]).astype(np.float64)
    dt16 = np.concatenate([np.diff(np.asarray(item["timestamps"], dtype=np.int64)) for item in input16]).astype(np.float64)
    gap_rows8: List[Dict] = []
    gap_rows16: List[Dict] = []
    segment_counts = {"1/8": 0, "1/16": 0}
    for dataset_name, input_records, collector in (("1/8", input8, gap_rows8), ("1/16", input16, gap_rows16)):
        for item in input_records:
            mask = np.asarray(item["mask"], dtype=bool)
            ts = np.asarray(item["timestamps"], dtype=np.float64)
            gaps, _ = extract_gap_infos(mask=mask, timestamps=ts)
            segment_counts[dataset_name] += len(gaps)
            for gap in gaps:
                for _ in gap.missing_indices:
                    collector.append({"dataset": dataset_name, "gap_size": int(gap.missing_count)})

    plot_length_distribution(lengths8, lengths16, out_dir / "length_distribution.png")
    plot_interval_distribution(dt8, dt16, out_dir / "interval_distribution.png")
    plot_gap_distribution(gap_rows8, gap_rows16, out_dir / "missing_point_weighted_gap_distribution.png")

    summary = {
        "1/8": {"num_traj": len(input8), "mean_length": float(np.mean(lengths8)), "num_gap_points": int(sum(r["gap_size"] for r in gap_rows8))},
        "1/16": {"num_traj": len(input16), "mean_length": float(np.mean(lengths16)), "num_gap_points": int(sum(r["gap_size"] for r in gap_rows16))},
        "gap_segment_distribution": segment_counts,
    }
    save_json(out_dir / "dataset_summary.json", summary)
    save_json(out_dir / "gap_segment_distribution.json", segment_counts)
    return {"dataset_summary": summary}
