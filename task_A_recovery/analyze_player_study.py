from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analyze_recovery import draw_road_overlay
from game_core import missing_metrics


Array = np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze two-round Task A player study submissions and export MAE/RMSE + montage plots."
    )

    parser.add_argument("--session-dir-8", type=Path, required=True, help="Round-1 session directory for dataset 1/8")
    parser.add_argument("--session-dir-16", type=Path, required=True, help="Round-2 session directory for dataset 1/16")

    parser.add_argument("--input-8", type=Path, default=Path("task_A_recovery/val_input_8.pkl"))
    parser.add_argument("--input-16", type=Path, default=Path("task_A_recovery/val_input_16.pkl"))
    parser.add_argument("--gt", type=Path, default=Path("task_A_recovery/val_gt.pkl"))

    parser.add_argument("--pred23-8", type=Path, default=Path("task_A_recovery/pred_hmm_val_8_b23_e5_gapaware.pkl"))
    parser.add_argument("--pred23-16", type=Path, default=Path("task_A_recovery/pred_hmm_val_16_b23_e5_gapaware.pkl"))
    parser.add_argument("--pred28-8", type=Path, default=Path("task_A_recovery/pred_hmm_val_8_b28_turncurve.pkl"))
    parser.add_argument("--pred28-16", type=Path, default=Path("task_A_recovery/pred_hmm_val_16_b28_turncurve.pkl"))

    parser.add_argument("--map-cache", type=Path, default=Path("task_A_recovery/map_roads_overlay_cache.pkl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-montage-cases", type=int, default=24)
    parser.add_argument("--montage-cols", type=int, default=4)
    parser.add_argument("--map-max-segments", type=int, default=3000)

    return parser.parse_args()


def load_pickle(path: Path):
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def build_id_map(records: Sequence[dict]) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for item in records:
        out[int(item["traj_id"])] = item
    return out


def _finite_rows(arr: Array) -> Array:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    ok = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[ok]


def _bbox_from_points(points: Iterable[Array], margin_ratio: float = 0.08) -> Tuple[float, float, float, float]:
    valid = [_finite_rows(p) for p in points]
    valid = [v for v in valid if v.size > 0]
    if not valid:
        return 0.0, 1.0, 0.0, 1.0

    merged = np.vstack(valid)
    x_min = float(np.min(merged[:, 0]))
    x_max = float(np.max(merged[:, 0]))
    y_min = float(np.min(merged[:, 1]))
    y_max = float(np.max(merged[:, 1]))

    dx = max(1e-6, x_max - x_min)
    dy = max(1e-6, y_max - y_min)
    x_pad = dx * margin_ratio
    y_pad = dy * margin_ratio

    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _as_prediction_records(source: Path) -> List[dict]:
    if source.is_file():
        obj = load_pickle(source)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict) and ("traj_id" in x) and ("coords" in x)]
        raise ValueError(f"unsupported prediction pkl format: {source}")

    if not source.is_dir():
        raise FileNotFoundError(f"session path not found: {source}")

    primary = source / "player_predictions_saved_cases.pkl"
    if primary.exists():
        obj = load_pickle(primary)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict) and ("traj_id" in x) and ("coords" in x)]

    cases_dir = source / "cases"
    records: List[dict] = []
    if cases_dir.exists() and cases_dir.is_dir():
        for child in sorted(cases_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            pred_file = child / "player_pred.pkl"
            if not pred_file.exists():
                continue
            obj = load_pickle(pred_file)
            if isinstance(obj, list) and obj:
                rec0 = obj[0]
                if isinstance(rec0, dict) and ("traj_id" in rec0) and ("coords" in rec0):
                    records.append({"traj_id": int(rec0["traj_id"]), "coords": rec0["coords"]})

    return records


def _ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _weighted_mae(rows: Sequence[dict], key: str) -> float:
    num = 0.0
    den = 0
    for r in rows:
        v = float(r[key])
        n = int(r["missing_count"])
        if np.isfinite(v) and n > 0:
            num += v * n
            den += n
    if den <= 0:
        return float("nan")
    return float(num / den)


def _weighted_rmse(rows: Sequence[dict], key: str) -> float:
    num = 0.0
    den = 0
    for r in rows:
        v = float(r[key])
        n = int(r["missing_count"])
        if np.isfinite(v) and n > 0:
            num += (v * v) * n
            den += n
    if den <= 0:
        return float("nan")
    return float(math.sqrt(num / den))


def _mean(rows: Sequence[dict], key: str) -> float:
    if not rows:
        return float("nan")
    arr = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    return float(np.nanmean(arr))


def _write_case_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = [
        "dataset",
        "traj_id",
        "missing_count",
        "player_mae_m",
        "player_rmse_m",
        "b23_mae_m",
        "b23_rmse_m",
        "b28_mae_m",
        "b28_rmse_m",
        "player_minus_b23_mae_m",
        "player_minus_b28_mae_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _plot_case_overlay(
    ax,
    row: dict,
    road_segments: Optional[Dict[str, Array]],
    map_max_segments: int,
) -> None:
    truth_xy = row["truth_xy"]
    player_xy = row["player_xy"]
    b23_xy = row["b23_xy"]
    b28_xy = row["b28_xy"]
    known_xy = row["known_xy"]

    x_min, x_max, y_min, y_max = _bbox_from_points([truth_xy, player_xy, b23_xy, b28_xy, known_xy], margin_ratio=0.08)

    if road_segments is not None:
        draw_road_overlay(
            ax=ax,
            road_segments=road_segments,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            color="#c3c8cf",
            alpha=0.42,
            linewidth=0.55,
            max_segments=max(500, int(map_max_segments)),
            label="Road",
        )

    ax.plot(truth_xy[:, 0], truth_xy[:, 1], color="#1f77b4", linewidth=1.4, label="GT", zorder=3)
    ax.plot(player_xy[:, 0], player_xy[:, 1], color="#d62728", linewidth=1.3, label="Player", zorder=4)
    ax.plot(b23_xy[:, 0], b23_xy[:, 1], color="#2ca02c", linewidth=1.0, linestyle="--", label="B23", zorder=3)
    ax.plot(b28_xy[:, 0], b28_xy[:, 1], color="#ff7f0e", linewidth=1.0, linestyle="-.", label="B28", zorder=3)
    ax.scatter(known_xy[:, 0], known_xy[:, 1], s=8, color="#111111", alpha=0.65, zorder=5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22, linewidth=0.4)

    ax.set_title(
        (
            f"d={row['dataset']} tid={row['traj_id']}\n"
            f"P={row['player_mae_m']:.1f} B23={row['b23_mae_m']:.1f} B28={row['b28_mae_m']:.1f}"
        ),
        fontsize=8,
    )


def _write_montage(
    rows: Sequence[dict],
    out_png: Path,
    road_segments: Optional[Dict[str, Array]],
    map_max_segments: int,
    max_cases: int,
    cols: int,
) -> None:
    picked = list(rows[: max(0, int(max_cases))])
    if not picked:
        return

    n = len(picked)
    cols = max(1, int(cols))
    rows_n = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.2, rows_n * 3.8), squeeze=False)
    for i in range(rows_n * cols):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        if i >= n:
            ax.axis("off")
            continue
        _plot_case_overlay(ax=ax, row=picked[i], road_segments=road_segments, map_max_segments=map_max_segments)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _metrics_block(rows: Sequence[dict]) -> dict:
    return {
        "case_count": int(len(rows)),
        "missing_points_total": int(sum(int(r["missing_count"]) for r in rows)),
        "player_mae_case_mean_m": _mean(rows, "player_mae_m"),
        "player_rmse_case_mean_m": _mean(rows, "player_rmse_m"),
        "b23_mae_case_mean_m": _mean(rows, "b23_mae_m"),
        "b23_rmse_case_mean_m": _mean(rows, "b23_rmse_m"),
        "b28_mae_case_mean_m": _mean(rows, "b28_mae_m"),
        "b28_rmse_case_mean_m": _mean(rows, "b28_rmse_m"),
        "player_mae_weighted_m": _weighted_mae(rows, "player_mae_m"),
        "player_rmse_weighted_m": _weighted_rmse(rows, "player_rmse_m"),
        "b23_mae_weighted_m": _weighted_mae(rows, "b23_mae_m"),
        "b23_rmse_weighted_m": _weighted_rmse(rows, "b23_rmse_m"),
        "b28_mae_weighted_m": _weighted_mae(rows, "b28_mae_m"),
        "b28_rmse_weighted_m": _weighted_rmse(rows, "b28_rmse_m"),
        "delta_player_minus_b23_mae_case_mean_m": _mean(rows, "player_minus_b23_mae_m"),
        "delta_player_minus_b28_mae_case_mean_m": _mean(rows, "player_minus_b28_mae_m"),
    }


def _write_summary_md(path: Path, overall: dict, by_dataset: Dict[str, dict]) -> None:
    lines: List[str] = []
    lines.append("# Player Study Summary")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- case_count: {overall['case_count']}")
    lines.append(f"- missing_points_total: {overall['missing_points_total']}")
    lines.append(f"- player_mae_weighted_m: {overall['player_mae_weighted_m']:.4f}")
    lines.append(f"- player_rmse_weighted_m: {overall['player_rmse_weighted_m']:.4f}")
    lines.append(f"- b23_mae_weighted_m: {overall['b23_mae_weighted_m']:.4f}")
    lines.append(f"- b23_rmse_weighted_m: {overall['b23_rmse_weighted_m']:.4f}")
    lines.append(f"- b28_mae_weighted_m: {overall['b28_mae_weighted_m']:.4f}")
    lines.append(f"- b28_rmse_weighted_m: {overall['b28_rmse_weighted_m']:.4f}")
    lines.append(f"- delta_player_minus_b23_mae_case_mean_m: {overall['delta_player_minus_b23_mae_case_mean_m']:.4f}")
    lines.append(f"- delta_player_minus_b28_mae_case_mean_m: {overall['delta_player_minus_b28_mae_case_mean_m']:.4f}")
    lines.append("")
    lines.append("Interpretation: negative delta means player MAE is lower than baseline.")
    lines.append("")

    lines.append("## By Dataset")
    lines.append("")
    for ds in ("8", "16"):
        block = by_dataset.get(ds)
        if block is None:
            continue
        lines.append(f"### dataset={ds}")
        lines.append("")
        lines.append(f"- case_count: {block['case_count']}")
        lines.append(f"- missing_points_total: {block['missing_points_total']}")
        lines.append(f"- player_mae_weighted_m: {block['player_mae_weighted_m']:.4f}")
        lines.append(f"- player_rmse_weighted_m: {block['player_rmse_weighted_m']:.4f}")
        lines.append(f"- b23_mae_weighted_m: {block['b23_mae_weighted_m']:.4f}")
        lines.append(f"- b28_mae_weighted_m: {block['b28_mae_weighted_m']:.4f}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_dataset_case_rows(
    dataset: str,
    player_records: Sequence[dict],
    input_by_id: Dict[int, dict],
    gt_by_id: Dict[int, dict],
    pred23_by_id: Dict[int, dict],
    pred28_by_id: Dict[int, dict],
) -> List[dict]:
    rows: List[dict] = []

    for rec in player_records:
        traj_id = int(rec["traj_id"])
        if traj_id not in input_by_id:
            continue
        if traj_id not in gt_by_id or traj_id not in pred23_by_id or traj_id not in pred28_by_id:
            continue

        inp = input_by_id[traj_id]
        gt = gt_by_id[traj_id]
        p23 = pred23_by_id[traj_id]
        p28 = pred28_by_id[traj_id]

        mask = np.asarray(inp["mask"], dtype=bool)
        known_xy = np.asarray(inp["coords"], dtype=np.float64)[mask]

        player_xy = np.asarray(rec["coords"], dtype=np.float64)
        truth_xy = np.asarray(gt["coords"], dtype=np.float64)
        b23_xy = np.asarray(p23["coords"], dtype=np.float64)
        b28_xy = np.asarray(p28["coords"], dtype=np.float64)

        if player_xy.shape != truth_xy.shape or truth_xy.ndim != 2 or truth_xy.shape[1] != 2:
            continue
        if b23_xy.shape != truth_xy.shape or b28_xy.shape != truth_xy.shape:
            continue

        m_player = missing_metrics(player_xy, truth_xy, mask)
        m_b23 = missing_metrics(b23_xy, truth_xy, mask)
        m_b28 = missing_metrics(b28_xy, truth_xy, mask)

        rows.append(
            {
                "dataset": dataset,
                "traj_id": int(traj_id),
                "missing_count": int(m_player.get("total_missing", int(np.sum(~mask)))),
                "player_mae_m": float(m_player["mae"]),
                "player_rmse_m": float(m_player["rmse"]),
                "b23_mae_m": float(m_b23["mae"]),
                "b23_rmse_m": float(m_b23["rmse"]),
                "b28_mae_m": float(m_b28["mae"]),
                "b28_rmse_m": float(m_b28["rmse"]),
                "player_minus_b23_mae_m": float(m_player["mae"] - m_b23["mae"]),
                "player_minus_b28_mae_m": float(m_player["mae"] - m_b28["mae"]),
                "truth_xy": truth_xy,
                "player_xy": player_xy,
                "b23_xy": b23_xy,
                "b28_xy": b28_xy,
                "known_xy": known_xy,
            }
        )

    rows.sort(key=lambda r: int(r["traj_id"]))
    return rows


def main() -> None:
    args = parse_args()

    for p, name in [
        (args.input_8, "input_8"),
        (args.input_16, "input_16"),
        (args.gt, "gt"),
        (args.pred23_8, "pred23_8"),
        (args.pred23_16, "pred23_16"),
        (args.pred28_8, "pred28_8"),
        (args.pred28_16, "pred28_16"),
    ]:
        _ensure_file_exists(p, name)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    input8_by_id = build_id_map(load_pickle(args.input_8))
    input16_by_id = build_id_map(load_pickle(args.input_16))
    gt_by_id = build_id_map(load_pickle(args.gt))

    pred23_8_by_id = build_id_map(load_pickle(args.pred23_8))
    pred23_16_by_id = build_id_map(load_pickle(args.pred23_16))
    pred28_8_by_id = build_id_map(load_pickle(args.pred28_8))
    pred28_16_by_id = build_id_map(load_pickle(args.pred28_16))

    player8_records = _as_prediction_records(args.session_dir_8)
    player16_records = _as_prediction_records(args.session_dir_16)

    rows8 = _prepare_dataset_case_rows(
        dataset="8",
        player_records=player8_records,
        input_by_id=input8_by_id,
        gt_by_id=gt_by_id,
        pred23_by_id=pred23_8_by_id,
        pred28_by_id=pred28_8_by_id,
    )
    rows16 = _prepare_dataset_case_rows(
        dataset="16",
        player_records=player16_records,
        input_by_id=input16_by_id,
        gt_by_id=gt_by_id,
        pred23_by_id=pred23_16_by_id,
        pred28_by_id=pred28_16_by_id,
    )

    all_rows = list(rows8) + list(rows16)
    all_rows.sort(key=lambda r: (int(r["dataset"]), int(r["traj_id"])))

    csv_rows = []
    for r in all_rows:
        csv_rows.append({k: v for k, v in r.items() if isinstance(v, (int, float, str, np.floating, np.integer))})

    _write_case_csv(args.out_dir / "case_metrics.csv", csv_rows)

    by_dataset = {
        "8": _metrics_block(rows8),
        "16": _metrics_block(rows16),
    }
    overall = _metrics_block(all_rows)

    global_metrics = {
        "overall": overall,
        "by_dataset": by_dataset,
        "session_dir_8": str(args.session_dir_8),
        "session_dir_16": str(args.session_dir_16),
    }
    (args.out_dir / "global_metrics.json").write_text(
        json.dumps(global_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_summary_md(args.out_dir / "summary.md", overall=overall, by_dataset=by_dataset)

    road_segments: Optional[Dict[str, Array]] = None
    if args.map_cache.exists():
        obj = load_pickle(args.map_cache)
        if isinstance(obj, dict) and all(k in obj for k in ["lon1", "lat1", "lon2", "lat2", "min_lon", "max_lon", "min_lat", "max_lat"]):
            road_segments = {k: np.asarray(obj[k]) for k in ["lon1", "lat1", "lon2", "lat2", "min_lon", "max_lon", "min_lat", "max_lat"]}

    _write_montage(
        rows=all_rows,
        out_png=args.out_dir / "case_montage_all.png",
        road_segments=road_segments,
        map_max_segments=args.map_max_segments,
        max_cases=args.max_montage_cases,
        cols=args.montage_cols,
    )
    _write_montage(
        rows=rows8,
        out_png=args.out_dir / "case_montage_dataset8.png",
        road_segments=road_segments,
        map_max_segments=args.map_max_segments,
        max_cases=args.max_montage_cases,
        cols=args.montage_cols,
    )
    _write_montage(
        rows=rows16,
        out_png=args.out_dir / "case_montage_dataset16.png",
        road_segments=road_segments,
        map_max_segments=args.map_max_segments,
        max_cases=args.max_montage_cases,
        cols=args.montage_cols,
    )

    print("analysis completed")
    print(f"- out_dir: {args.out_dir}")
    print(f"- submitted cases dataset8: {len(rows8)}")
    print(f"- submitted cases dataset16: {len(rows16)}")
    print(f"- submitted cases total: {len(all_rows)}")


if __name__ == "__main__":
    main()
