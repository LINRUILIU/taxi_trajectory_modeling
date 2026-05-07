from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .analyze import analyze_predictions
from .io_utils import load_pickle, save_csv, save_json, save_text
from .legacy_bridge import load_legacy_analyze_module
from .metrics import extract_gap_infos


CASE_CATEGORY_ORDER = ["good_case", "bad_case", "metric_cheat", "path_wrong"]
OVERLAP_MEAN_THRESHOLD_M = 12.0
OVERLAP_MAX_THRESHOLD_M = 28.0
GT_COLOR = "#1f77b4"
PRED_COLOR = "#d62728"
SHARE_GT_COLOR = "#4b5563"
SHARE_PRED_COLOR = "#111827"


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    import csv

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                    continue
                txt = value.strip()
                if txt == "":
                    parsed[key] = txt
                    continue
                try:
                    if any(ch in txt for ch in [".", "e", "E"]):
                        parsed[key] = float(txt)
                    else:
                        parsed[key] = int(txt)
                except ValueError:
                    parsed[key] = txt
            rows.append(parsed)
    return rows


def _case_key(row: Dict[str, Any]) -> Tuple[str, int, int, int]:
    return (str(row["dataset"]), int(row["traj_id"]), int(row["gap_start_idx"]), int(row["gap_end_idx"]))


def _load_overlay(map_path: Path) -> tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, Any]], Any]:
    if not map_path.exists():
        return None, None, None
    legacy_analyze = load_legacy_analyze_module()
    root = Path(__file__).resolve().parents[3]
    road_segments = legacy_analyze.load_or_build_road_segments(
        osm_path=map_path,
        cache_path=root / "task_A_final" / "caches" / "map_roads_overlay_cache.pkl",
        force_rebuild=False,
    )
    seg_index = legacy_analyze.build_road_segment_grid_index(
        road_segments=road_segments,
        cell_size_deg=0.002,
    )
    return road_segments, seg_index, legacy_analyze


def _gap_topology_stats(
    coords: np.ndarray,
    missing_indices: np.ndarray,
    road_segments: Optional[Dict[str, np.ndarray]],
    seg_index: Optional[Dict[str, Any]],
    legacy_analyze: Any,
) -> Tuple[float, float]:
    if road_segments is None or seg_index is None or legacy_analyze is None or missing_indices.size == 0:
        return math.nan, math.nan
    dists: List[float] = []
    for idx in missing_indices:
        lon = float(coords[int(idx), 0])
        lat = float(coords[int(idx), 1])
        dists.append(
            float(
                legacy_analyze.nearest_road_distance_m(
                    lon=lon,
                    lat=lat,
                    road_segments=road_segments,
                    seg_index=seg_index,
                    search_radius_m=250.0,
                )
            )
        )
    if not dists:
        return math.nan, math.nan
    arr = np.asarray(dists, dtype=np.float64)
    return float(np.mean(arr)), float(np.percentile(arr, 95))


def _choose_first_unique(
    rows: Sequence[Dict[str, Any]],
    used_keys: set[Tuple[str, int, int, int]],
) -> Optional[Dict[str, Any]]:
    for row in rows:
        key = _case_key(row)
        if key not in used_keys:
            used_keys.add(key)
            return row
    return None


def _select_case_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used_keys: set[Tuple[str, int, int, int]] = set()
    for dataset in ("1/8", "1/16"):
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        good = sorted(
            [r for r in dataset_rows if r["quadrant_global"] == "low_official_low_shape"],
            key=lambda r: (float(r["official_mae_m"]), float(r["shape_symmetric_m"])),
        )
        bad = sorted(
            dataset_rows,
            key=lambda r: (float(r["official_mae_m"]), float(r["shape_symmetric_m"])),
            reverse=True,
        )
        metric_cheat = sorted(
            [r for r in dataset_rows if r["quadrant_global"] == "low_official_high_shape"],
            key=lambda r: (float(r["shape_symmetric_m"]), -float(r["official_mae_m"])),
            reverse=True,
        )
        path_wrong = sorted(
            [r for r in dataset_rows if r["quadrant_global"] == "high_official_high_shape"],
            key=lambda r: (float(r["shape_symmetric_m"]), float(r["official_mae_m"])),
            reverse=True,
        )
        category_map = {
            "good_case": good if good else bad,
            "bad_case": bad,
            "metric_cheat": metric_cheat if metric_cheat else bad,
            "path_wrong": path_wrong if path_wrong else bad,
        }
        for category in CASE_CATEGORY_ORDER:
            picked = _choose_first_unique(category_map[category], used_keys)
            if picked is None:
                continue
            row = dict(picked)
            row["case_category"] = category
            selected.append(row)
    return selected


def _expand_bounds(xy: np.ndarray, pad_ratio: float = 0.08) -> Tuple[float, float, float, float]:
    finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    if not np.any(finite):
        return -1.0, 1.0, -1.0, 1.0
    xy = xy[finite]
    x_min = float(np.min(xy[:, 0]))
    x_max = float(np.max(xy[:, 0]))
    y_min = float(np.min(xy[:, 1]))
    y_max = float(np.max(xy[:, 1]))
    dx = max(x_max - x_min, 1e-4)
    dy = max(y_max - y_min, 1e-4)
    pad_x = dx * pad_ratio
    pad_y = dy * pad_ratio
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def _draw_overlay_window(
    ax: Any,
    road_segments: Optional[Dict[str, np.ndarray]],
    legacy_analyze: Any,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    label: Optional[str],
) -> None:
    if road_segments is None or legacy_analyze is None:
        return
    legacy_analyze.draw_road_overlay(
        ax=ax,
        road_segments=road_segments,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        color="#bec6cd",
        alpha=0.6,
        linewidth=0.65,
        max_segments=6500,
        label=label,
    )


def _segment_alignment_stats(gt_seg: np.ndarray, pred_seg: np.ndarray) -> Tuple[float, float]:
    from .geo import haversine_meters

    if gt_seg.shape != pred_seg.shape or gt_seg.size == 0:
        return math.inf, math.inf
    dists = np.asarray(
        haversine_meters(gt_seg[:, 0], gt_seg[:, 1], pred_seg[:, 0], pred_seg[:, 1]),
        dtype=np.float64,
    )
    if dists.size == 0:
        return 0.0, 0.0
    return float(np.mean(dists)), float(np.max(dists))


def _plot_gap_segment(
    ax: Any,
    gt_seg: np.ndarray,
    pred_seg: np.ndarray,
    gt_missing: np.ndarray,
    pred_missing: np.ndarray,
    label_prefix: str = "",
    show_labels: bool = True,
) -> bool:
    mean_dist_m, max_dist_m = _segment_alignment_stats(gt_seg, pred_seg)
    merged_view = mean_dist_m <= OVERLAP_MEAN_THRESHOLD_M and max_dist_m <= OVERLAP_MAX_THRESHOLD_M
    gt_label = f"{label_prefix}Actual path".strip()
    pred_label = f"{label_prefix}Predicted path".strip()
    if merged_view:
        ax.plot(
            gt_seg[:, 0],
            gt_seg[:, 1],
            color=SHARE_GT_COLOR,
            linewidth=1.7,
            alpha=0.98,
            label=gt_label if show_labels else None,
            zorder=4,
        )
        ax.plot(
            pred_seg[:, 0],
            pred_seg[:, 1],
            color=SHARE_PRED_COLOR,
            linewidth=1.5,
            alpha=0.82,
            label=pred_label if show_labels else None,
            zorder=5,
        )
    else:
        ax.plot(
            gt_seg[:, 0],
            gt_seg[:, 1],
            color=GT_COLOR,
            linewidth=1.25,
            alpha=0.95,
            label=gt_label if show_labels else None,
            zorder=4,
        )
        ax.plot(
            pred_seg[:, 0],
            pred_seg[:, 1],
            color=PRED_COLOR,
            linewidth=1.15,
            alpha=0.95,
            label=pred_label if show_labels else None,
            zorder=5,
        )

    ax.scatter(
        gt_missing[:, 0],
        gt_missing[:, 1],
        s=14,
        facecolors="white",
        edgecolors=GT_COLOR if not merged_view else "#6b7280",
        linewidths=0.95,
        label=None,
        zorder=6,
    )
    ax.scatter(
        pred_missing[:, 0],
        pred_missing[:, 1],
        s=16,
        marker="x",
        color=PRED_COLOR if not merged_view else "#4b5563",
        linewidths=0.95,
        label=None,
        zorder=7,
    )
    return merged_view


def _scatter_all_pred_gt_points(
    ax: Any,
    gt: np.ndarray,
    pred: np.ndarray,
    gap_mask: np.ndarray,
    merged_view: bool,
    show_labels: bool,
) -> None:
    non_gap = ~gap_mask

    gt_gray = "#6b7280"
    pred_gray = "#4b5563"
    gt_gap = GT_COLOR
    pred_gap = PRED_COLOR

    valid_gt_non_gap = non_gap & np.isfinite(gt[:, 0]) & np.isfinite(gt[:, 1])
    valid_pred_non_gap = non_gap & np.isfinite(pred[:, 0]) & np.isfinite(pred[:, 1])
    valid_gt_gap = gap_mask & np.isfinite(gt[:, 0]) & np.isfinite(gt[:, 1])
    valid_pred_gap = gap_mask & np.isfinite(pred[:, 0]) & np.isfinite(pred[:, 1])

    ax.scatter(
        gt[valid_gt_non_gap, 0],
        gt[valid_gt_non_gap, 1],
        s=8,
        facecolors="white",
        edgecolors=gt_gray,
        linewidths=0.65,
        alpha=0.8,
        label="Actual points" if show_labels else None,
        zorder=2.8,
    )
    ax.scatter(
        pred[valid_pred_non_gap, 0],
        pred[valid_pred_non_gap, 1],
        s=9,
        marker="x",
        color=pred_gray if merged_view else "#7b8794",
        linewidths=0.7,
        alpha=0.8,
        label="Predicted points" if show_labels else None,
        zorder=2.9,
    )
    ax.scatter(
        gt[valid_gt_gap, 0],
        gt[valid_gt_gap, 1],
        s=13,
        facecolors="white",
        edgecolors=gt_gap,
        linewidths=0.9,
        alpha=0.95,
        label="Actual gap points" if show_labels else None,
        zorder=6.8,
    )
    ax.scatter(
        pred[valid_pred_gap, 0],
        pred[valid_pred_gap, 1],
        s=14,
        marker="x",
        color=pred_gap if not merged_view else pred_gray,
        linewidths=0.9,
        alpha=0.95,
        label="Predicted gap points" if show_labels else None,
        zorder=6.9,
    )


def _draw_case_panel(
    ax: Any,
    inp_coords: np.ndarray,
    mask: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    gap_start: int,
    gap_end: int,
    miss_idx: np.ndarray,
    road_segments: Optional[Dict[str, np.ndarray]],
    legacy_analyze: Any,
    legend_labels: bool,
) -> bool:
    known_xy = inp_coords[mask]
    gt_seg = gt[gap_start : gap_end + 1]
    pred_seg = pred[gap_start : gap_end + 1]
    gt_missing = gt[miss_idx]
    pred_missing = pred[miss_idx]

    full_xy = np.vstack([gt, pred, inp_coords])
    x_min, x_max, y_min, y_max = _expand_bounds(full_xy, pad_ratio=0.05)
    _draw_overlay_window(
        ax=ax,
        road_segments=road_segments,
        legacy_analyze=legacy_analyze,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        label="Road network" if legend_labels else None,
    )

    ax.plot(gt[:, 0], gt[:, 1], color="#a8b4c4", linewidth=0.7, alpha=0.65, zorder=1)
    ax.plot(pred[:, 0], pred[:, 1], color="#d6dde7", linewidth=0.65, alpha=0.65, zorder=1)
    merged_view = _plot_gap_segment(
        ax=ax,
        gt_seg=gt_seg,
        pred_seg=pred_seg,
        gt_missing=gt_missing,
        pred_missing=pred_missing,
        show_labels=legend_labels,
    )
    gap_mask = np.zeros(len(gt), dtype=bool)
    gap_mask[gap_start : gap_end + 1] = True
    _scatter_all_pred_gt_points(
        ax=ax,
        gt=gt,
        pred=pred,
        gap_mask=gap_mask,
        merged_view=merged_view,
        show_labels=legend_labels,
    )
    ax.scatter(
        [gt_seg[0, 0], gt_seg[-1, 0]],
        [gt_seg[0, 1], gt_seg[-1, 1]],
        s=22,
        color="#111827",
        marker="o",
        label="Gap anchors" if legend_labels else None,
        zorder=8,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.2)
    ax.ticklabel_format(style="plain", useOffset=False)

    return merged_view


def _plot_single_case(
    out_path: Path,
    dataset_name: str,
    row: Dict[str, Any],
    input_by_dataset: Dict[str, Dict[int, Dict[str, Any]]],
    pred_by_dataset: Dict[str, Dict[int, np.ndarray]],
    gt_by_id: Dict[int, np.ndarray],
    road_segments: Optional[Dict[str, np.ndarray]],
    legacy_analyze: Any,
    case_topology_mean_m: float,
) -> None:
    traj_id = int(row["traj_id"])
    gap_start = int(row["gap_start_idx"])
    gap_end = int(row["gap_end_idx"])
    inp = input_by_dataset[dataset_name][traj_id]
    mask = np.asarray(inp["mask"], dtype=bool)
    inp_coords = np.asarray(inp["coords"], dtype=np.float64)
    pred = pred_by_dataset[dataset_name][traj_id]
    gt = gt_by_id[traj_id]
    miss_idx = np.arange(gap_start + 1, gap_end, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    merged_view = _draw_case_panel(
        ax=ax,
        inp_coords=inp_coords,
        mask=mask,
        gt=gt,
        pred=pred,
        gap_start=gap_start,
        gap_end=gap_end,
        miss_idx=miss_idx,
        road_segments=road_segments,
        legacy_analyze=legacy_analyze,
        legend_labels=True,
    )

    title = (
        f"{row['case_category']} | {dataset_name} | traj {traj_id} | gap {gap_start}->{gap_end}\n"
        f"MAE {float(row['official_mae_m']):.2f}m | road {case_topology_mean_m:.2f}m | shape {float(row['shape_symmetric_m']):.2f}m"
        f"{' | overlap view' if merged_view else ''}"
    )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def _build_summary_text(run_name: str, global_metrics: Dict[str, Any], case_rows: Sequence[Dict[str, Any]]) -> str:
    selected_counts: Dict[str, int] = {}
    for row in case_rows:
        selected_counts[row["dataset"]] = selected_counts.get(row["dataset"], 0) + 1
    lines = [
        f"# Unified Analysis Summary: {run_name}",
        "",
        "## Global Metrics",
        f"- 1/8: MAE={global_metrics['1/8']['mae']:.4f} m, RMSE={global_metrics['1/8']['rmse']:.4f} m, P95={global_metrics['1/8']['p95']:.4f} m, Topology={global_metrics['1/8'].get('topology_violation_rate', math.nan) * 100.0:.2f}%",
        f"- 1/16: MAE={global_metrics['1/16']['mae']:.4f} m, RMSE={global_metrics['1/16']['rmse']:.4f} m, P95={global_metrics['1/16']['p95']:.4f} m, Topology={global_metrics['1/16'].get('topology_violation_rate', math.nan) * 100.0:.2f}%",
        "",
        "## Interpretation",
        "- `shape_symmetric_m` is used as the reference path-similarity metric; lower is better.",
        "- `metric_cheat` cases indicate low official MAE but still visibly mismatched geometry.",
        "- `path_wrong` cases indicate both official error and route geometry are poor and deserve manual review first.",
        "",
        "## Case Gallery",
        f"- Selected cases: {len(case_rows)} total, with 1/8={selected_counts.get('1/8', 0)} and 1/16={selected_counts.get('1/16', 0)}.",
        "- Each case figure overlays road network, known points, actual missing points, predicted missing points, actual path, and predicted path.",
    ]
    return "\n".join(lines) + "\n"


def run_unified_analysis(
    input8_path: Path,
    input16_path: Path,
    pred8_path: Path,
    pred16_path: Path,
    gt_path: Path,
    out_dir: Path,
    run_name: str,
    map_path: Optional[Path] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_result = analyze_predictions(
        input8_path=input8_path,
        input16_path=input16_path,
        pred8_path=pred8_path,
        pred16_path=pred16_path,
        gt_path=gt_path,
        out_dir=out_dir,
    )

    input8 = load_pickle(input8_path)
    input16 = load_pickle(input16_path)
    pred8 = load_pickle(pred8_path)
    pred16 = load_pickle(pred16_path)
    gt = load_pickle(gt_path)
    gt_by_id = {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in gt}
    input_by_dataset = {
        "1/8": {int(item["traj_id"]): item for item in input8},
        "1/16": {int(item["traj_id"]): item for item in input16},
    }
    pred_by_dataset = {
        "1/8": {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in pred8},
        "1/16": {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in pred16},
    }

    gap_rows = _read_csv_rows(out_dir / "gap_metrics.csv")
    selected_rows = _select_case_rows(gap_rows)

    road_segments = None
    seg_index = None
    legacy_analyze = None
    if map_path is not None:
        road_segments, seg_index, legacy_analyze = _load_overlay(map_path)

    overview_rows: List[Dict[str, Any]] = []
    case_gallery_dir = out_dir / "case_gallery"
    for row in selected_rows:
        dataset_name = str(row["dataset"])
        traj_id = int(row["traj_id"])
        inp = input_by_dataset[dataset_name][traj_id]
        mask = np.asarray(inp["mask"], dtype=bool)
        timestamps = np.asarray(inp["timestamps"], dtype=np.float64)
        gaps, _ = extract_gap_infos(mask=mask, timestamps=timestamps)
        matched_gap = None
        for gap in gaps:
            if int(gap.start_idx) == int(row["gap_start_idx"]) and int(gap.end_idx) == int(row["gap_end_idx"]):
                matched_gap = gap
                break
        if matched_gap is None:
            continue

        pred = pred_by_dataset[dataset_name][traj_id]
        topo_mean_m, topo_p95_m = _gap_topology_stats(
            coords=pred,
            missing_indices=np.asarray(matched_gap.missing_indices, dtype=np.int64),
            road_segments=road_segments,
            seg_index=seg_index,
            legacy_analyze=legacy_analyze,
        )
        filename = f"case_{dataset_name.replace('/', '_')}_traj{traj_id}_gap{int(row['gap_start_idx'])}_{int(row['gap_end_idx'])}.png"
        out_path = case_gallery_dir / filename
        _plot_single_case(
            out_path=out_path,
            dataset_name=dataset_name,
            row=row,
            input_by_dataset=input_by_dataset,
            pred_by_dataset=pred_by_dataset,
            gt_by_id=gt_by_id,
            road_segments=road_segments,
            legacy_analyze=legacy_analyze,
            case_topology_mean_m=topo_mean_m,
        )
        overview_rows.append(
            {
                "run_name": run_name,
                "dataset": dataset_name,
                "case_category": row["case_category"],
                "traj_id": traj_id,
                "gap_start_idx": int(row["gap_start_idx"]),
                "gap_end_idx": int(row["gap_end_idx"]),
                "gap_size": int(row["gap_size"]),
                "official_mae_m": float(row["official_mae_m"]),
                "official_rmse_m": float(row["official_rmse_m"]),
                "shape_symmetric_m": float(row["shape_symmetric_m"]),
                "topology_nearest_road_mean_m": topo_mean_m,
                "topology_nearest_road_p95_m": topo_p95_m,
                "quadrant_global": row["quadrant_global"],
                "image_path": str(out_path),
            }
        )

    save_csv(case_gallery_dir / "case_overview.csv", overview_rows)
    save_text(out_dir / "summary.md", _build_summary_text(run_name, analysis_result["global_metrics"], overview_rows))
    save_json(
        out_dir / "unified_analysis_manifest.json",
        {
            "run_name": run_name,
            "case_categories": CASE_CATEGORY_ORDER,
            "selected_case_count": len(overview_rows),
            "case_overview_csv": str(case_gallery_dir / "case_overview.csv"),
            "summary_md": str(out_dir / "summary.md"),
        },
    )
    output_files = list(analysis_result["output_files"])
    output_files.extend([out_dir / "summary.md", out_dir / "unified_analysis_manifest.json", case_gallery_dir / "case_overview.csv"])
    output_files.extend([Path(row["image_path"]) for row in overview_rows])
    return {
        "run_name": run_name,
        "global_metrics": analysis_result["global_metrics"],
        "case_overview_rows": overview_rows,
        "output_files": output_files,
    }
