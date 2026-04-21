import argparse
import csv
import json
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SECONDS_PER_DAY = 24 * 3600


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


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


def _safe_float(x, default=math.nan) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return {
            "count": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "mape": math.nan,
            "bias": math.nan,
            "p50_abs_error": math.nan,
            "p90_abs_error": math.nan,
            "p95_abs_error": math.nan,
        }

    residual = y_pred - y_true
    abs_err = np.abs(residual)
    ape = abs_err / np.maximum(np.abs(y_true), 1e-6) * 100.0

    return {
        "count": int(y_true.size),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mape": float(np.mean(ape)),
        "bias": float(np.mean(residual)),
        "p50_abs_error": float(np.percentile(abs_err, 50)),
        "p90_abs_error": float(np.percentile(abs_err, 90)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
    }


def align_pred_gt(pred_records: Sequence[Dict], gt_records: Sequence[Dict]) -> Dict:
    pred_by_id = {int(item["traj_id"]): _safe_float(item["travel_time"]) for item in pred_records}
    gt_by_id = {int(item["traj_id"]): _safe_float(item["travel_time"]) for item in gt_records}

    traj_ids: List[int] = []
    y_true: List[float] = []
    y_pred: List[float] = []
    missing_pred: List[int] = []

    for item in gt_records:
        tid = int(item["traj_id"])
        if tid not in pred_by_id:
            missing_pred.append(tid)
            continue
        pred_v = pred_by_id[tid]
        gt_v = _safe_float(item["travel_time"])
        if not np.isfinite(pred_v) or not np.isfinite(gt_v):
            continue
        traj_ids.append(tid)
        y_true.append(gt_v)
        y_pred.append(pred_v)

    extra_pred = [tid for tid in pred_by_id.keys() if tid not in gt_by_id]

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred_arr - y_true_arr
    abs_err = np.abs(residual)
    ape = abs_err / np.maximum(np.abs(y_true_arr), 1e-6) * 100.0

    return {
        "traj_ids": traj_ids,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "residual": residual,
        "abs_err": abs_err,
        "ape": ape,
        "coverage": {
            "gt_count": len(gt_records),
            "pred_count": len(pred_records),
            "matched_count": int(y_true_arr.size),
            "missing_pred_count": len(missing_pred),
            "extra_pred_count": len(extra_pred),
            "missing_pred_preview": missing_pred[:20],
            "extra_pred_preview": extra_pred[:20],
        },
    }


def _departure_hour(ts: int) -> int:
    sec_day = int(ts % SECONDS_PER_DAY)
    return int(sec_day // 3600)


def compute_context_by_id(input_records: Sequence[Dict]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for rec in input_records:
        tid = int(rec["traj_id"])
        coords = np.asarray(rec["coords"], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            continue

        n_points = int(coords.shape[0])
        if n_points >= 2:
            step = haversine_meters(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
            path_len_m = float(np.sum(step))
            direct_dist_m = float(haversine_meters(coords[0, 0], coords[0, 1], coords[-1, 0], coords[-1, 1]))
        else:
            path_len_m = 0.0
            direct_dist_m = 0.0

        dep_ts = int(rec.get("departure_timestamp", 0))
        hour = _departure_hour(dep_ts)

        out[tid] = {
            "hour": float(hour),
            "n_points": float(n_points),
            "path_len_km": path_len_m / 1000.0,
            "direct_dist_km": direct_dist_m / 1000.0,
            "detour_ratio": path_len_m / max(direct_dist_m, 1e-6),
        }
    return out


def bucket_travel_time(tt_sec: float) -> str:
    if tt_sec < 600.0:
        return "00-10m"
    if tt_sec < 1200.0:
        return "10-20m"
    if tt_sec < 1800.0:
        return "20-30m"
    if tt_sec < 2700.0:
        return "30-45m"
    return "45m+"


def bucket_hour(hour: float) -> str:
    h = int(hour)
    if h < 6:
        return "00-06"
    if h < 10:
        return "06-10"
    if h < 17:
        return "10-17"
    if h < 20:
        return "17-20"
    return "20-24"


def bucket_path_km(path_km: float) -> str:
    if path_km < 8.0:
        return "<8km"
    if path_km < 12.0:
        return "8-12km"
    if path_km < 16.0:
        return "12-16km"
    if path_km < 22.0:
        return "16-22km"
    return "22km+"


def bucket_points(n_points: float) -> str:
    n = int(n_points)
    if n <= 80:
        return "<=80"
    if n <= 120:
        return "81-120"
    if n <= 160:
        return "121-160"
    if n <= 200:
        return "161-200"
    return ">200"


def bucket_detour(detour_ratio: float) -> str:
    if detour_ratio < 1.15:
        return "<1.15"
    if detour_ratio < 1.30:
        return "1.15-1.30"
    if detour_ratio < 1.50:
        return "1.30-1.50"
    return ">=1.50"


def summarize_bucket(group_name: str, labels: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray) -> List[Dict]:
    rows: List[Dict] = []
    labels_arr = np.asarray(labels)
    total = max(1, y_true.size)

    ordered_labels = sorted(list(set(labels)))
    for label in ordered_labels:
        m = labels_arr == label
        if not np.any(m):
            continue
        met = compute_metrics(y_true[m], y_pred[m])
        rows.append(
            {
                "group": group_name,
                "bucket": label,
                "count": int(np.sum(m)),
                "share": float(np.sum(m) / total),
                "mae": met["mae"],
                "rmse": met["rmse"],
                "mape": met["mape"],
                "bias": met["bias"],
                "p50_abs_error": met["p50_abs_error"],
                "p90_abs_error": met["p90_abs_error"],
                "p95_abs_error": met["p95_abs_error"],
            }
        )
    return rows


def write_bucket_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "bucket",
        "count",
        "share",
        "mae",
        "rmse",
        "mape",
        "bias",
        "p50_abs_error",
        "p90_abs_error",
        "p95_abs_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_top_cases_csv(
    path: Path,
    traj_ids: Sequence[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    abs_err: np.ndarray,
    ape: np.ndarray,
    context_by_id: Optional[Dict[int, Dict[str, float]]],
    top_k: int,
) -> None:
    idx = np.argsort(-abs_err)
    idx = idx[: max(1, top_k)]

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "traj_id",
        "travel_time_true",
        "travel_time_pred",
        "abs_error",
        "ape_percent",
        "hour",
        "n_points",
        "path_len_km",
        "direct_dist_km",
        "detour_ratio",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, i in enumerate(idx, start=1):
            tid = int(traj_ids[i])
            ctx = context_by_id.get(tid, {}) if context_by_id is not None else {}
            w.writerow(
                {
                    "rank": rank,
                    "traj_id": tid,
                    "travel_time_true": float(y_true[i]),
                    "travel_time_pred": float(y_pred[i]),
                    "abs_error": float(abs_err[i]),
                    "ape_percent": float(ape[i]),
                    "hour": _safe_float(ctx.get("hour"), default=math.nan),
                    "n_points": _safe_float(ctx.get("n_points"), default=math.nan),
                    "path_len_km": _safe_float(ctx.get("path_len_km"), default=math.nan),
                    "direct_dist_km": _safe_float(ctx.get("direct_dist_km"), default=math.nan),
                    "detour_ratio": _safe_float(ctx.get("detour_ratio"), default=math.nan),
                }
            )


def _sample_for_plot(x: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.size
    if n <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def plot_scatter_pred_vs_gt(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, max_points: int, seed: int) -> None:
    x, y = _sample_for_plot(y_true, y_pred, max_points=max_points, seed=seed)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(x, y, s=8, alpha=0.3)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    ax.set_title("Prediction vs Ground Truth")
    ax.set_xlabel("Ground Truth Travel Time (s)")
    ax.set_ylabel("Predicted Travel Time (s)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error_hist(abs_err: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(abs_err, bins=50, alpha=0.85)
    ax.set_title("Absolute Error Distribution")
    ax.set_xlabel("Absolute Error (s)")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_residual_vs_true(y_true: np.ndarray, residual: np.ndarray, out_path: Path, max_points: int, seed: int) -> None:
    x, y = _sample_for_plot(y_true, residual, max_points=max_points, seed=seed)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(x, y, s=8, alpha=0.3)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_title("Residual vs Ground Truth")
    ax.set_xlabel("Ground Truth Travel Time (s)")
    ax.set_ylabel("Residual (pred - true, s)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bucket_metrics(rows: Sequence[Dict], group_name: str, out_path: Path) -> None:
    items = [r for r in rows if r["group"] == group_name]
    if not items:
        return

    labels = [str(r["bucket"]) for r in items]
    mae = [float(r["mae"]) for r in items]
    rmse = [float(r["rmse"]) for r in items]
    mape = [float(r["mape"]) for r in items]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(x - width, mae, width=width, label="MAE")
    ax.bar(x, rmse, width=width, label="RMSE")
    ax.bar(x + width, mape, width=width, label="MAPE(%)")
    ax.set_title(f"Bucket Metrics: {group_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _score_grade(mae: float, rmse: float, mape: float) -> str:
    if mape <= 1.5 and mae <= 20 and rmse <= 30:
        return "strong"
    if mape <= 2.0 and mae <= 30 and rmse <= 45:
        return "good"
    if mape <= 3.0 and mae <= 45 and rmse <= 65:
        return "fair"
    return "weak"


def _load_reference_metrics(reference_path: Optional[Path]) -> Optional[Dict[str, float]]:
    if reference_path is None:
        return None
    if not reference_path.exists():
        return None
    with reference_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if "global" in obj:
        g = obj["global"]
        return {
            "mae": _safe_float(g.get("mae")),
            "rmse": _safe_float(g.get("rmse")),
            "mape": _safe_float(g.get("mape")),
        }

    if "val" in obj:
        g = obj["val"]
        return {
            "mae": _safe_float(g.get("mae")),
            "rmse": _safe_float(g.get("rmse")),
            "mape": _safe_float(g.get("mape")),
        }

    if all(k in obj for k in ("mae", "rmse", "mape")):
        return {
            "mae": _safe_float(obj.get("mae")),
            "rmse": _safe_float(obj.get("rmse")),
            "mape": _safe_float(obj.get("mape")),
        }

    return None


def write_decision_summary(
    out_path: Path,
    global_metrics: Dict[str, float],
    coverage: Dict,
    bucket_rows: Sequence[Dict],
    pred_path: Path,
    gt_path: Path,
    input_path: Optional[Path],
    reference_metrics: Optional[Dict[str, float]],
) -> None:
    grade = _score_grade(
        mae=float(global_metrics["mae"]),
        rmse=float(global_metrics["rmse"]),
        mape=float(global_metrics["mape"]),
    )

    lines: List[str] = []
    lines.append("# Task B Analysis Summary")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Prediction: {pred_path}")
    lines.append(f"- Ground Truth: {gt_path}")
    if input_path is not None:
        lines.append(f"- Input (for bucket/context): {input_path}")
    lines.append("")

    lines.append("## Official Metrics")
    lines.append("| count | MAE(s) | RMSE(s) | MAPE(%) | bias(s) | P90_abs_err(s) | P95_abs_err(s) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        "| {count} | {mae:.4f} | {rmse:.4f} | {mape:.4f} | {bias:.4f} | {p90:.4f} | {p95:.4f} |".format(
            count=int(global_metrics["count"]),
            mae=float(global_metrics["mae"]),
            rmse=float(global_metrics["rmse"]),
            mape=float(global_metrics["mape"]),
            bias=float(global_metrics["bias"]),
            p90=float(global_metrics["p90_abs_error"]),
            p95=float(global_metrics["p95_abs_error"]),
        )
    )
    lines.append("")

    lines.append("## Coverage")
    lines.append(
        "- matched={matched} / gt={gt} / pred={pred} / missing_pred={miss} / extra_pred={extra}".format(
            matched=coverage["matched_count"],
            gt=coverage["gt_count"],
            pred=coverage["pred_count"],
            miss=coverage["missing_pred_count"],
            extra=coverage["extra_pred_count"],
        )
    )
    lines.append("")

    if reference_metrics is not None:
        d_mae = float(global_metrics["mae"]) - float(reference_metrics["mae"])
        d_rmse = float(global_metrics["rmse"]) - float(reference_metrics["rmse"])
        d_mape = float(global_metrics["mape"]) - float(reference_metrics["mape"])
        lines.append("## Delta vs Reference")
        lines.append(f"- Delta MAE: {d_mae:+.4f} s")
        lines.append(f"- Delta RMSE: {d_rmse:+.4f} s")
        lines.append(f"- Delta MAPE: {d_mape:+.4f} %")
        lines.append("")

    lines.append("## Judgment")
    lines.append(f"- Heuristic grade: {grade}")

    travel_rows = [r for r in bucket_rows if r["group"] == "travel_time_bucket"]
    if travel_rows:
        worst = max(travel_rows, key=lambda r: float(r["mape"]))
        best = min(travel_rows, key=lambda r: float(r["mape"]))
        lines.append(
            f"- Worst travel-time bucket by MAPE: {worst['bucket']} (MAPE={float(worst['mape']):.4f}%, count={int(worst['count'])})"
        )
        lines.append(
            f"- Best travel-time bucket by MAPE: {best['bucket']} (MAPE={float(best['mape']):.4f}%, count={int(best['count'])})"
        )

    lines.append("")
    lines.append("## Next Actions")
    lines.append("- Keep this file as the milestone log anchor for future runs.")
    lines.append("- Compare new runs via --reference-metrics and milestone trend plots.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _read_milestone_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_milestone_rows(csv_path: Path, rows: Sequence[Dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "order",
        "timestamp",
        "milestone",
        "count",
        "mae",
        "rmse",
        "mape",
        "analysis_dir",
        "pred_file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def append_milestone(
    csv_path: Path,
    milestone_name: str,
    metrics: Dict[str, float],
    analysis_dir: Path,
    pred_path: Path,
) -> List[Dict]:
    old_rows = _read_milestone_rows(csv_path)
    order = len(old_rows) + 1
    row = {
        "order": order,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "milestone": milestone_name,
        "count": int(metrics["count"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "mape": float(metrics["mape"]),
        "analysis_dir": str(analysis_dir),
        "pred_file": str(pred_path),
    }
    rows = old_rows + [row]
    _write_milestone_rows(csv_path, rows)
    return rows


def plot_milestone_trend(rows: Sequence[Dict], out_path: Path) -> None:
    if not rows:
        return

    labels = [str(r["milestone"]) for r in rows]
    x = np.arange(len(labels))
    mae = np.asarray([_safe_float(r["mae"]) for r in rows], dtype=np.float64)
    rmse = np.asarray([_safe_float(r["rmse"]) for r in rows], dtype=np.float64)
    mape = np.asarray([_safe_float(r["mape"]) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.4), sharex=True)

    axes[0].plot(x, mae, marker="o", linewidth=2)
    axes[0].set_ylabel("MAE(s)")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(x, rmse, marker="s", linewidth=2)
    axes[1].set_ylabel("RMSE(s)")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    axes[2].plot(x, mape, marker="^", linewidth=2)
    axes[2].set_ylabel("MAPE(%)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=22, ha="right")
    axes[2].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Task B Milestone Trend")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_milestone_summary(rows: Sequence[Dict], out_path: Path) -> None:
    if not rows:
        return

    def best_by(key: str) -> Dict:
        return min(rows, key=lambda r: _safe_float(r[key], default=float("inf")))

    best_mae = best_by("mae")
    best_rmse = best_by("rmse")
    best_mape = best_by("mape")

    lines: List[str] = []
    lines.append("# Task B Milestone Summary")
    lines.append("")
    lines.append(f"- Best MAE: {best_mae['milestone']} ({_safe_float(best_mae['mae']):.4f} s)")
    lines.append(f"- Best RMSE: {best_rmse['milestone']} ({_safe_float(best_rmse['rmse']):.4f} s)")
    lines.append(f"- Best MAPE: {best_mape['milestone']} ({_safe_float(best_mape['mape']):.4f} %)" )
    lines.append("")
    lines.append("## Table")
    lines.append("")
    lines.append("| order | milestone | MAE(s) | RMSE(s) | MAPE(%) | count | timestamp |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")

    for r in rows:
        lines.append(
            "| {order} | {milestone} | {mae:.4f} | {rmse:.4f} | {mape:.4f} | {count} | {ts} |".format(
                order=int(_safe_float(r["order"], default=0)),
                milestone=r["milestone"],
                mae=_safe_float(r["mae"]),
                rmse=_safe_float(r["rmse"]),
                mape=_safe_float(r["mape"]),
                count=int(_safe_float(r["count"], default=0)),
                ts=r.get("timestamp", ""),
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Task B analysis and milestone tracking")
    p.add_argument("--pred", type=Path, required=True, help="Prediction pkl ({traj_id, travel_time})")
    p.add_argument("--gt", type=Path, required=True, help="Ground-truth pkl ({traj_id, travel_time})")
    p.add_argument("--input", type=Path, default=None, help="Optional input pkl (coords + departure_timestamp)")
    p.add_argument("--output-dir", type=Path, required=True, help="Analysis output directory")
    p.add_argument("--reference-metrics", type=Path, default=None, help="Optional reference metrics json for delta")
    p.add_argument("--top-k-cases", type=int, default=30)
    p.add_argument("--max-plot-points", type=int, default=12000)
    p.add_argument("--seed", type=int, default=20260420)

    p.add_argument("--milestone-name", type=str, default=None, help="Optional milestone name to append history")
    p.add_argument(
        "--milestone-csv",
        type=Path,
        default=Path("task_B_tte/analysis_outputs_milestones/milestone_metrics.csv"),
        help="Milestone csv path",
    )
    p.add_argument(
        "--milestone-summary-md",
        type=Path,
        default=Path("task_B_tte/analysis_outputs_milestones/milestone_summary.md"),
        help="Milestone summary markdown",
    )
    p.add_argument(
        "--milestone-trend-png",
        type=Path,
        default=Path("task_B_tte/analysis_outputs_milestones/milestone_trend.png"),
        help="Milestone trend figure",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    pred_records = load_pickle(args.pred)
    gt_records = load_pickle(args.gt)

    aligned = align_pred_gt(pred_records=pred_records, gt_records=gt_records)
    y_true = aligned["y_true"]
    y_pred = aligned["y_pred"]
    residual = aligned["residual"]
    abs_err = aligned["abs_err"]
    ape = aligned["ape"]
    traj_ids = aligned["traj_ids"]

    global_metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    coverage = aligned["coverage"]

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    context_by_id = None
    bucket_rows: List[Dict] = []

    travel_labels = [bucket_travel_time(v) for v in y_true]
    bucket_rows.extend(summarize_bucket("travel_time_bucket", travel_labels, y_true, y_pred))

    if args.input is not None:
        input_records = load_pickle(args.input)
        context_by_id = compute_context_by_id(input_records)

        hours = []
        path_km = []
        n_points = []
        detours = []
        valid_mask = []

        for tid in traj_ids:
            ctx = context_by_id.get(int(tid))
            if ctx is None:
                hours.append(math.nan)
                path_km.append(math.nan)
                n_points.append(math.nan)
                detours.append(math.nan)
                valid_mask.append(False)
                continue
            hours.append(_safe_float(ctx["hour"]))
            path_km.append(_safe_float(ctx["path_len_km"]))
            n_points.append(_safe_float(ctx["n_points"]))
            detours.append(_safe_float(ctx["detour_ratio"]))
            valid_mask.append(True)

        hours_arr = np.asarray(hours, dtype=np.float64)
        path_arr = np.asarray(path_km, dtype=np.float64)
        npts_arr = np.asarray(n_points, dtype=np.float64)
        det_arr = np.asarray(detours, dtype=np.float64)
        vm = np.asarray(valid_mask, dtype=bool)

        if np.any(vm):
            hour_labels = [bucket_hour(v) for v in hours_arr[vm]]
            bucket_rows.extend(summarize_bucket("departure_hour_bucket", hour_labels, y_true[vm], y_pred[vm]))

            path_labels = [bucket_path_km(v) for v in path_arr[vm]]
            bucket_rows.extend(summarize_bucket("path_length_bucket", path_labels, y_true[vm], y_pred[vm]))

            point_labels = [bucket_points(v) for v in npts_arr[vm]]
            bucket_rows.extend(summarize_bucket("n_points_bucket", point_labels, y_true[vm], y_pred[vm]))

            detour_labels = [bucket_detour(v) for v in det_arr[vm]]
            bucket_rows.extend(summarize_bucket("detour_bucket", detour_labels, y_true[vm], y_pred[vm]))

    global_metrics_path = output_dir / "global_metrics.json"
    bucket_csv_path = output_dir / "bucket_summary.csv"
    decision_md_path = output_dir / "decision_summary.md"
    top_case_csv_path = output_dir / "top_error_cases.csv"

    reference_metrics = _load_reference_metrics(args.reference_metrics)

    save_json(
        global_metrics_path,
        {
            "global": global_metrics,
            "coverage": coverage,
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "pred": str(args.pred),
                "gt": str(args.gt),
                "input": str(args.input) if args.input is not None else None,
            },
        },
    )

    write_bucket_csv(bucket_csv_path, bucket_rows)
    write_top_cases_csv(
        path=top_case_csv_path,
        traj_ids=traj_ids,
        y_true=y_true,
        y_pred=y_pred,
        abs_err=abs_err,
        ape=ape,
        context_by_id=context_by_id,
        top_k=args.top_k_cases,
    )
    write_decision_summary(
        out_path=decision_md_path,
        global_metrics=global_metrics,
        coverage=coverage,
        bucket_rows=bucket_rows,
        pred_path=args.pred,
        gt_path=args.gt,
        input_path=args.input,
        reference_metrics=reference_metrics,
    )

    plot_scatter_pred_vs_gt(
        y_true=y_true,
        y_pred=y_pred,
        out_path=output_dir / "scatter_pred_vs_gt.png",
        max_points=args.max_plot_points,
        seed=args.seed,
    )
    plot_error_hist(abs_err=abs_err, out_path=output_dir / "abs_error_hist.png")
    plot_residual_vs_true(
        y_true=y_true,
        residual=residual,
        out_path=output_dir / "residual_vs_gt.png",
        max_points=args.max_plot_points,
        seed=args.seed,
    )

    for group_name in [
        "travel_time_bucket",
        "departure_hour_bucket",
        "path_length_bucket",
        "n_points_bucket",
        "detour_bucket",
    ]:
        plot_bucket_metrics(bucket_rows, group_name=group_name, out_path=output_dir / f"bucket_{group_name}.png")

    if args.milestone_name:
        milestone_rows = append_milestone(
            csv_path=args.milestone_csv,
            milestone_name=args.milestone_name,
            metrics=global_metrics,
            analysis_dir=output_dir,
            pred_path=args.pred,
        )
        write_milestone_summary(rows=milestone_rows, out_path=args.milestone_summary_md)
        plot_milestone_trend(rows=milestone_rows, out_path=args.milestone_trend_png)

    print("=== Global Metrics ===")
    print(json.dumps(global_metrics, ensure_ascii=False, indent=2))
    print("=== Coverage ===")
    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    print(f"Saved: {global_metrics_path}")
    print(f"Saved: {bucket_csv_path}")
    print(f"Saved: {decision_md_path}")
    print(f"Saved: {top_case_csv_path}")
    if args.milestone_name:
        print(f"Milestone appended: {args.milestone_csv} :: {args.milestone_name}")


if __name__ == "__main__":
    main()
