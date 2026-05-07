from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .io_utils import load_pickle, save_csv, save_json, save_text
from .legacy_bridge import load_legacy_analyze_module
from .metrics import extract_gap_infos
from .unified_analysis import (
    GT_COLOR,
    PRED_COLOR,
    _case_key,
    _draw_overlay_window,
    _expand_bounds,
    _gap_topology_stats,
    _load_overlay,
    _read_csv_rows,
    run_unified_analysis,
)


VERSION_ORDER = ["baseline1", "baseline23e5", "b28_compat", "final"]
DATASET_LABELS = {"1/8": "dataset8", "1/16": "dataset16"}
CASE_CATEGORIES = ["improvement_showcase", "topology_rescue", "remaining_hard_case"]


def _parse_version_specs(items: Sequence[str]) -> Dict[str, Tuple[Path, Path]]:
    versions: Dict[str, Tuple[Path, Path]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid version spec: {item}")
        name, payload = item.split("=", 1)
        parts = [p.strip() for p in payload.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Version spec must be name=pred8,pred16: {item}")
        versions[name.strip()] = (Path(parts[0]), Path(parts[1]))
    return versions


def _plot_metric_trend(rows: Sequence[Dict[str, Any]], metric_key: str, title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = list(range(len(VERSION_ORDER)))
    for dataset in ("1/8", "1/16"):
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        values = [next(r[metric_key] for r in dataset_rows if r["version"] == version) for version in VERSION_ORDER]
        ax.plot(x, values, marker="o", linewidth=2, label=dataset)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(VERSION_ORDER, rotation=15, ha="right")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def _build_metrics_table(version_outputs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for version in VERSION_ORDER:
        analysis_dir = Path(version_outputs[version]["analysis_dir"])
        global_metrics = version_outputs[version]["global_metrics"]
        gap_rows = _read_csv_rows(analysis_dir / "gap_metrics.csv")
        for dataset in ("1/8", "1/16"):
            dataset_gap_rows = [r for r in gap_rows if r["dataset"] == dataset]
            shape_mean = float(np.mean([float(r["shape_symmetric_m"]) for r in dataset_gap_rows])) if dataset_gap_rows else math.nan
            gm = global_metrics[dataset]
            rows.append(
                {
                    "version": version,
                    "dataset": dataset,
                    "mae_m": float(gm["mae"]),
                    "rmse_m": float(gm["rmse"]),
                    "p95_m": float(gm["p95"]),
                    "topology_violation_rate": float(gm.get("topology_violation_rate", math.nan)),
                    "shape_symmetric_mean_m": shape_mean,
                }
            )
    return rows


def _paired_rows(version_outputs: Dict[str, Dict[str, Any]], version: str) -> Dict[Tuple[str, int, int, int], Dict[str, Any]]:
    analysis_dir = Path(version_outputs[version]["analysis_dir"])
    return {_case_key(r): r for r in _read_csv_rows(analysis_dir / "gap_metrics.csv")}


def _select_longitudinal_cases(version_outputs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_b23 = _paired_rows(version_outputs, "baseline23e5")
    base_b28 = _paired_rows(version_outputs, "b28_compat")
    final_rows = _paired_rows(version_outputs, "final")
    selected: List[Dict[str, Any]] = []
    used_keys: set[Tuple[str, int, int, int]] = set()

    for dataset in ("1/8", "1/16"):
        candidates: List[Dict[str, Any]] = []
        for key, final_row in final_rows.items():
            if key[0] != dataset or key not in base_b23 or key not in base_b28:
                continue
            b23 = base_b23[key]
            b28 = base_b28[key]
            candidates.append(
                {
                    "dataset": dataset,
                    "traj_id": int(final_row["traj_id"]),
                    "gap_start_idx": int(final_row["gap_start_idx"]),
                    "gap_end_idx": int(final_row["gap_end_idx"]),
                    "gap_size": int(final_row["gap_size"]),
                    "final_official_mae_m": float(final_row["official_mae_m"]),
                    "final_shape_symmetric_m": float(final_row["shape_symmetric_m"]),
                    "delta_vs_b23_mae_m": float(final_row["official_mae_m"]) - float(b23["official_mae_m"]),
                    "delta_vs_b28_mae_m": float(final_row["official_mae_m"]) - float(b28["official_mae_m"]),
                    "delta_vs_b23_shape_m": float(final_row["shape_symmetric_m"]) - float(b23["shape_symmetric_m"]),
                    "delta_vs_b28_shape_m": float(final_row["shape_symmetric_m"]) - float(b28["shape_symmetric_m"]),
                }
            )

        improvement = sorted(
            [
                r
                for r in candidates
                if r["delta_vs_b23_mae_m"] < 0.0 and r["delta_vs_b28_mae_m"] < 0.0
            ],
            key=lambda r: (r["delta_vs_b23_mae_m"] + r["delta_vs_b28_mae_m"], r["final_official_mae_m"]),
        )
        hard = sorted(
            candidates,
            key=lambda r: (r["final_official_mae_m"], r["final_shape_symmetric_m"]),
            reverse=True,
        )

        picked_improve = next((r for r in improvement if _case_key(r) not in used_keys), None)
        if picked_improve is not None:
            picked_improve["case_category"] = "improvement_showcase"
            used_keys.add(_case_key(picked_improve))
            selected.append(picked_improve)

        topology_pool = []
        for row in candidates:
            key = _case_key(row)
            if key in used_keys:
                continue
            if row["delta_vs_b28_mae_m"] <= 5.0:
                topology_pool.append(row)
        topology_pool.sort(key=lambda r: (r["delta_vs_b28_shape_m"], r["delta_vs_b28_mae_m"]))
        picked_topology = topology_pool[0] if topology_pool else next((r for r in hard if _case_key(r) not in used_keys), None)
        if picked_topology is not None:
            picked_topology["case_category"] = "topology_rescue"
            used_keys.add(_case_key(picked_topology))
            selected.append(picked_topology)

        picked_hard = next((r for r in hard if _case_key(r) not in used_keys), None)
        if picked_hard is not None:
            picked_hard["case_category"] = "remaining_hard_case"
            used_keys.add(_case_key(picked_hard))
            selected.append(picked_hard)

    return selected


def _plot_case_comparison(
    out_path: Path,
    case_row: Dict[str, Any],
    version_outputs: Dict[str, Dict[str, Any]],
    input_by_dataset: Dict[str, Dict[int, Dict[str, Any]]],
    gt_by_id: Dict[int, np.ndarray],
    road_segments: Optional[Dict[str, np.ndarray]],
    seg_index: Optional[Dict[str, Any]],
    legacy_analyze: Any,
) -> Dict[str, Any]:
    dataset = str(case_row["dataset"])
    traj_id = int(case_row["traj_id"])
    gap_start = int(case_row["gap_start_idx"])
    gap_end = int(case_row["gap_end_idx"])
    inp = input_by_dataset[dataset][traj_id]
    mask = np.asarray(inp["mask"], dtype=bool)
    timestamps = np.asarray(inp["timestamps"], dtype=np.float64)
    inp_coords = np.asarray(inp["coords"], dtype=np.float64)
    gaps, _ = extract_gap_infos(mask=mask, timestamps=timestamps)
    matched_gap = next(g for g in gaps if int(g.start_idx) == gap_start and int(g.end_idx) == gap_end)
    miss_idx = np.asarray(matched_gap.missing_indices, dtype=np.int64)
    gt = gt_by_id[traj_id]
    version_colors = {
        "baseline1": "#94a3b8",
        "baseline23e5": "#7c3aed",
        "b28_compat": "#f59e0b",
        "final": "#d62728",
    }
    version_labels = {
        "baseline1": "baseline1 pred",
        "baseline23e5": "baseline23e5 pred",
        "b28_compat": "b28_compat pred",
        "final": "final pred",
    }
    pred_paths = {}
    for version in VERSION_ORDER:
        pred_by_dataset = version_outputs[version]["pred_by_dataset"]
        pred_paths[version] = pred_by_dataset[dataset][traj_id]

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    summary_rows: List[Dict[str, Any]] = []
    full_xy = [gt, inp_coords]
    full_xy.extend(pred_paths[version] for version in VERSION_ORDER)
    x_min, x_max, y_min, y_max = _expand_bounds(np.vstack(full_xy), pad_ratio=0.05)
    _draw_overlay_window(
        ax=ax,
        road_segments=road_segments,
        legacy_analyze=legacy_analyze,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        label="Road network",
    )
    known_xy = inp_coords[mask]
    ax.plot(gt[:, 0], gt[:, 1], color=GT_COLOR, linewidth=1.25, alpha=0.9, label="GT", zorder=4)
    gap_mask = np.zeros(len(gt), dtype=bool)
    gap_mask[gap_start : gap_end + 1] = True
    gt_non_gap = (~gap_mask) & np.isfinite(gt[:, 0]) & np.isfinite(gt[:, 1])
    gt_gap = gap_mask & np.isfinite(gt[:, 0]) & np.isfinite(gt[:, 1])
    ax.scatter(gt[gt_non_gap, 0], gt[gt_non_gap, 1], s=8, facecolors="white", edgecolors="#6b7280", linewidths=0.65, alpha=0.78, zorder=2.8)
    ax.scatter(gt[gt_gap, 0], gt[gt_gap, 1], s=12, facecolors="white", edgecolors=GT_COLOR, linewidths=0.9, alpha=0.95, zorder=7)
    ax.scatter(
        [gt[gap_start, 0], gt[gap_end, 0]],
        [gt[gap_start, 1], gt[gap_end, 1]],
        s=22,
        color="#111827",
        marker="o",
        zorder=8,
    )
    for version in VERSION_ORDER:
        pred = pred_paths[version]
        topo_mean_m, topo_p95_m = _gap_topology_stats(pred, miss_idx, road_segments, seg_index, legacy_analyze)
        gap_rows = version_outputs[version]["gap_rows"]
        row = gap_rows[_case_key(case_row)]

        ax.plot(
            pred[:, 0],
            pred[:, 1],
            color=version_colors[version],
            linewidth=0.82 if version != "final" else 1.05,
            alpha=0.8 if version != "final" else 0.95,
            label=version.replace("_", " "),
            zorder=3 if version != "final" else 5,
        )
        pred_non_gap = (~gap_mask) & np.isfinite(pred[:, 0]) & np.isfinite(pred[:, 1])
        pred_gap = gap_mask & np.isfinite(pred[:, 0]) & np.isfinite(pred[:, 1])
        ax.scatter(
            pred[pred_non_gap, 0],
            pred[pred_non_gap, 1],
            s=8,
            marker="x",
            color="#5b6470",
            linewidths=0.65,
            alpha=0.68,
            zorder=2.9 if version != "final" else 3.2,
        )
        ax.scatter(
            pred[pred_gap, 0],
            pred[pred_gap, 1],
            s=11,
            marker="x",
            color=version_colors[version],
            linewidths=0.9,
            zorder=6 if version != "final" else 7,
        )
        summary_rows.append(
            {
                "version": version,
                "dataset": dataset,
                "traj_id": traj_id,
                "gap_start_idx": gap_start,
                "gap_end_idx": gap_end,
                "official_mae_m": float(row["official_mae_m"]),
                "shape_symmetric_m": float(row["shape_symmetric_m"]),
                "topology_nearest_road_mean_m": topo_mean_m,
                "topology_nearest_road_p95_m": topo_p95_m,
            }
        )

    title_lines = [f"{case_row['case_category']} | {dataset} | traj {traj_id} | gap {gap_start}->{gap_end}"]
    for version in VERSION_ORDER:
        row = version_outputs[version]["gap_rows"][_case_key(case_row)]
        title_lines.append(
            f"{version}: MAE {float(row['official_mae_m']):.1f}m | shape {float(row['shape_symmetric_m']):.1f}m"
        )
    ax.set_title("\n".join(title_lines), fontsize=10.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.2)
    ax.ticklabel_format(style="plain", useOffset=False)
    traj_handles = [
        Line2D([0], [0], color="#c7cfd8", linewidth=0.8, label="Road"),
        Line2D([0], [0], color=GT_COLOR, linewidth=1.25, label="GT"),
    ]
    traj_handles.extend(
        Line2D([0], [0], color=version_colors[version], linewidth=0.82 if version != "final" else 1.05, label=version.replace("_", " "))
        for version in VERSION_ORDER
    )
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor=GT_COLOR, markeredgewidth=0.95, markersize=5.5, label="GT gap pts"),
        Line2D([0], [0], marker="x", linestyle="None", color="#444444", markersize=5.5, label="Pred gap pts"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="#6b7280", markeredgewidth=0.75, markersize=4.8, label="GT non-gap pts"),
        Line2D([0], [0], marker="x", linestyle="None", color="#5b6470", markersize=4.8, label="Pred non-gap pts"),
        Line2D([0], [0], marker="o", linestyle="None", color="#111827", markersize=6, label="Anchors"),
    ]
    legend1 = ax.legend(handles=traj_handles, loc="upper left", fontsize=8, framealpha=0.92)
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, loc="lower right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=195)
    plt.close(fig)
    return {
        "dataset": dataset,
        "case_category": case_row["case_category"],
        "traj_id": traj_id,
        "gap_start_idx": gap_start,
        "gap_end_idx": gap_end,
        "image_path": str(out_path),
        "versions": summary_rows,
    }


def _build_summary_md(metric_rows: Sequence[Dict[str, Any]], case_rows: Sequence[Dict[str, Any]]) -> str:
    def _metric(version: str, dataset: str, key: str) -> float:
        return next(float(r[key]) for r in metric_rows if r["version"] == version and r["dataset"] == dataset)

    lines = [
        "# Longitudinal Analysis Summary",
        "",
        "## Final Takeaway",
        f"- `final` (selector mix) improves over `b28_compat` on MAE from {_metric('b28_compat', '1/8', 'mae_m'):.2f}m to {_metric('final', '1/8', 'mae_m'):.2f}m on 1/8, and from {_metric('b28_compat', '1/16', 'mae_m'):.2f}m to {_metric('final', '1/16', 'mae_m'):.2f}m on 1/16.",
        f"- Topology violation also drops from {_metric('b28_compat', '1/8', 'topology_violation_rate') * 100.0:.2f}% to {_metric('final', '1/8', 'topology_violation_rate') * 100.0:.2f}% on 1/8, and from {_metric('b28_compat', '1/16', 'topology_violation_rate') * 100.0:.2f}% to {_metric('final', '1/16', 'topology_violation_rate') * 100.0:.2f}% on 1/16.",
        f"- Mean reference path-similarity (`shape_symmetric_m`) also improves from {_metric('b28_compat', '1/8', 'shape_symmetric_mean_m'):.2f}m to {_metric('final', '1/8', 'shape_symmetric_mean_m'):.2f}m on 1/8, and from {_metric('b28_compat', '1/16', 'shape_symmetric_mean_m'):.2f}m to {_metric('final', '1/16', 'shape_symmetric_mean_m'):.2f}m on 1/16.",
        "",
        "## Reading Guide",
        "- `baseline1` is the no-road-network geometric lower bound.",
        "- `baseline23e5` captures the first strong jump from HMM plus topology awareness.",
        "- `b28_compat` is the final-pipeline anchor for legacy baseline28 behavior.",
        "- `final` is the promoted selector-based release candidate.",
        "",
        "## Case Pool",
        f"- Selected longitudinal cases: {len(case_rows)}",
        "- Categories cover improvement showcase, topology rescue, and remaining hard cases for both 1/8 and 1/16.",
    ]
    return "\n".join(lines) + "\n"


def run_longitudinal_analysis(
    version_specs: Sequence[str],
    input8_path: Path,
    input16_path: Path,
    gt_path: Path,
    out_dir: Path,
    map_path: Optional[Path] = None,
) -> Dict[str, Any]:
    versions = _parse_version_specs(version_specs)
    missing = [name for name in VERSION_ORDER if name not in versions]
    if missing:
        raise ValueError(f"Missing required versions: {', '.join(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    version_outputs: Dict[str, Dict[str, Any]] = {}
    for version in VERSION_ORDER:
        pred8_path, pred16_path = versions[version]
        version_analysis_dir = out_dir / "version_analyses" / version
        result = run_unified_analysis(
            input8_path=input8_path,
            input16_path=input16_path,
            pred8_path=pred8_path,
            pred16_path=pred16_path,
            gt_path=gt_path,
            out_dir=version_analysis_dir,
            run_name=version,
            map_path=map_path,
        )
        version_outputs[version] = {
            "analysis_dir": str(version_analysis_dir),
            "global_metrics": result["global_metrics"],
            "pred_by_dataset": {
                "1/8": {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in load_pickle(pred8_path)},
                "1/16": {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in load_pickle(pred16_path)},
            },
            "gap_rows": {_case_key(r): r for r in _read_csv_rows(version_analysis_dir / "gap_metrics.csv")},
        }

    metric_rows = _build_metrics_table(version_outputs)
    save_csv(out_dir / "longitudinal_metrics.csv", metric_rows)
    _plot_metric_trend(metric_rows, "mae_m", "Longitudinal MAE Trend", "MAE (m)", out_dir / "mae_trend.png")
    _plot_metric_trend(metric_rows, "topology_violation_rate", "Longitudinal Topology Trend", "Topology violation rate", out_dir / "topology_trend.png")
    _plot_metric_trend(metric_rows, "shape_symmetric_mean_m", "Longitudinal Path Similarity Trend", "Mean shape symmetric distance (m)", out_dir / "shape_trend.png")

    gt = load_pickle(gt_path)
    gt_by_id = {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in gt}
    input8 = load_pickle(input8_path)
    input16 = load_pickle(input16_path)
    input_by_dataset = {
        "1/8": {int(item["traj_id"]): item for item in input8},
        "1/16": {int(item["traj_id"]): item for item in input16},
    }
    road_segments = None
    seg_index = None
    legacy_analyze = None
    if map_path is not None:
        road_segments, seg_index, legacy_analyze = _load_overlay(map_path)

    case_candidates = _select_longitudinal_cases(version_outputs)
    case_summaries: List[Dict[str, Any]] = []
    cases_dir = out_dir / "cases"
    for case_row in case_candidates:
        dataset_suffix = DATASET_LABELS[str(case_row["dataset"])]
        image_path = cases_dir / f"{case_row['case_category']}_{dataset_suffix}.png"
        case_summaries.append(
            _plot_case_comparison(
                out_path=image_path,
                case_row=case_row,
                version_outputs=version_outputs,
                input_by_dataset=input_by_dataset,
                gt_by_id=gt_by_id,
                road_segments=road_segments,
                seg_index=seg_index,
                legacy_analyze=legacy_analyze,
            )
        )

    flat_case_rows: List[Dict[str, Any]] = []
    for item in case_summaries:
        for version_row in item["versions"]:
            flat_case_rows.append(
                {
                    "case_category": item["case_category"],
                    "dataset": item["dataset"],
                    "traj_id": item["traj_id"],
                    "gap_start_idx": item["gap_start_idx"],
                    "gap_end_idx": item["gap_end_idx"],
                    "image_path": item["image_path"],
                    **version_row,
                }
            )
    save_csv(out_dir / "longitudinal_case_overview.csv", flat_case_rows)
    save_text(out_dir / "longitudinal_summary.md", _build_summary_md(metric_rows, case_summaries))
    save_json(
        out_dir / "longitudinal_manifest.json",
        {
            "version_order": VERSION_ORDER,
            "case_categories": CASE_CATEGORIES,
            "metric_table": str(out_dir / "longitudinal_metrics.csv"),
            "case_overview": str(out_dir / "longitudinal_case_overview.csv"),
            "summary": str(out_dir / "longitudinal_summary.md"),
        },
    )
    return {
        "metric_rows": metric_rows,
        "case_summaries": case_summaries,
        "output_files": [
            out_dir / "longitudinal_metrics.csv",
            out_dir / "mae_trend.png",
            out_dir / "topology_trend.png",
            out_dir / "shape_trend.png",
            out_dir / "longitudinal_case_overview.csv",
            out_dir / "longitudinal_summary.md",
            out_dir / "longitudinal_manifest.json",
        ]
        + [Path(item["image_path"]) for item in case_summaries],
    }
