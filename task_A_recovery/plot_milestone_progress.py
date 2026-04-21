from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_MILESTONES = [
    ("baseline23_mainroad_full", Path("task_A_recovery/analysis_outputs_baseline23_mainroad_full")),
    ("baseline23_e5", Path("task_A_recovery/analysis_outputs_baseline23_e5")),
    ("baseline25_speedaware_cont", Path("task_A_recovery/analysis_outputs_baseline25_speedaware_cont")),
    ("baseline26_pchip_smooth", Path("task_A_recovery/analysis_outputs_baseline26_pchip_smooth")),
    ("baseline27_bind", Path("task_A_recovery/analysis_outputs_baseline27_bind")),
    ("baseline28_turncurve", Path("task_A_recovery/analysis_outputs_baseline28_turncurve")),
]


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_milestones(raw_items: list[str], skip_defaults: bool) -> list[tuple[str, Path]]:
    milestones: list[tuple[str, Path]] = []
    if not skip_defaults:
        milestones.extend(DEFAULT_MILESTONES)

    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --milestone format: {item}. Expected name=dir")
        name, path_str = item.split("=", 1)
        name = name.strip()
        path = Path(path_str.strip())
        if not name:
            raise ValueError(f"Invalid --milestone name in: {item}")
        milestones.append((name, path))

    if not milestones:
        raise ValueError("No milestones configured. Use defaults or pass --milestone entries.")
    return milestones


def load_global_metrics(metrics_path_or_dir: Path) -> dict[str, Any]:
    metrics_path = metrics_path_or_dir
    if metrics_path_or_dir.is_dir():
        metrics_path = metrics_path_or_dir / "global_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_rows(milestones: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (name, metrics_dir) in enumerate(milestones):
        metrics = load_global_metrics(metrics_dir)
        for ratio in ("1/8", "1/16"):
            part = metrics.get(ratio, {})
            rows.append(
                {
                    "order": idx,
                    "milestone": name,
                    "ratio": ratio,
                    "count": int(part.get("count", 0) or 0),
                    "mae": _to_float(part.get("mae")),
                    "rmse": _to_float(part.get("rmse")),
                    "p75": _to_float(part.get("p75")),
                    "p95": _to_float(part.get("p95")),
                    "topology_violation_rate": _to_float(part.get("topology_violation_rate")),
                    "topology_eval_points": int(part.get("topology_eval_points", 0) or 0),
                    "source_dir": str(metrics_dir),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    fieldnames = [
        "order",
        "milestone",
        "ratio",
        "count",
        "mae",
        "rmse",
        "p75",
        "p95",
        "topology_violation_rate",
        "topology_eval_points",
        "source_dir",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rows_by_ratio(rows: list[dict[str, Any]], ratio: str) -> list[dict[str, Any]]:
    return sorted((r for r in rows if r["ratio"] == ratio), key=lambda r: r["order"])


def plot_mae_rmse(rows: list[dict[str, Any]], out_png: Path) -> None:
    grouped_8 = _rows_by_ratio(rows, "1/8")
    grouped_16 = _rows_by_ratio(rows, "1/16")
    labels = [r["milestone"] for r in grouped_8]
    x = list(range(len(labels)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, ratio, grouped in ((axes[0], "1/8", grouped_8), (axes[1], "1/16", grouped_16)):
        ax.plot(x, [r["mae"] for r in grouped], marker="o", linewidth=2, label="MAE")
        ax.plot(x, [r["rmse"] for r in grouped], marker="s", linewidth=2, label="RMSE")
        ax.set_title(f"{ratio} missing-point error trend")
        ax.set_ylabel("distance (m)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_topology(rows: list[dict[str, Any]], out_png: Path) -> None:
    grouped_8 = _rows_by_ratio(rows, "1/8")
    grouped_16 = _rows_by_ratio(rows, "1/16")
    labels = [r["milestone"] for r in grouped_8]
    x = list(range(len(labels)))
    y8 = [r["topology_violation_rate"] * 100.0 for r in grouped_8]
    y16 = [r["topology_violation_rate"] * 100.0 for r in grouped_16]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(x, y8, marker="o", linewidth=2, label="1/8")
    ax.plot(x, y16, marker="s", linewidth=2, label="1/16")
    ax.set_title("Topology violation rate trend")
    ax.set_ylabel("violation rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_markdown(rows: list[dict[str, Any]], out_md: Path) -> None:
    grouped_8 = _rows_by_ratio(rows, "1/8")
    grouped_16 = _rows_by_ratio(rows, "1/16")
    ratio_rank = {"1/8": 0, "1/16": 1}

    def _best(grouped: list[dict[str, Any]], key: str) -> dict[str, Any]:
        return min(grouped, key=lambda r: r[key])

    def _delta(base: float, cur: float) -> float:
        if not math.isfinite(base) or abs(base) < 1e-12:
            return math.nan
        return (cur - base) / base * 100.0

    lines: list[str] = []
    lines.append("# Milestone Longitudinal Summary")
    lines.append("")

    for ratio, grouped in (("1/8", grouped_8), ("1/16", grouped_16)):
        base = grouped[0]
        best_mae = _best(grouped, "mae")
        best_rmse = _best(grouped, "rmse")
        best_topo = _best(grouped, "topology_violation_rate")

        lines.append(f"## {ratio}")
        lines.append(
            f"- Best MAE: {best_mae['milestone']} ({best_mae['mae']:.4f} m, "
            f"{_delta(base['mae'], best_mae['mae']):.2f}% vs {base['milestone']})"
        )
        lines.append(
            f"- Best RMSE: {best_rmse['milestone']} ({best_rmse['rmse']:.4f} m, "
            f"{_delta(base['rmse'], best_rmse['rmse']):.2f}% vs {base['milestone']})"
        )
        lines.append(
            f"- Best Topology (lower is better): {best_topo['milestone']} "
            f"({best_topo['topology_violation_rate'] * 100.0:.2f}%)"
        )
        lines.append("")

    lines.append("## Full Table")
    lines.append("")
    lines.append("| milestone | ratio | MAE(m) | RMSE(m) | P75(m) | P95(m) | Topology(%) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: (r["order"], ratio_rank.get(r["ratio"], 99))):
        lines.append(
            "| {milestone} | {ratio} | {mae:.4f} | {rmse:.4f} | {p75:.4f} | {p95:.4f} | {topo:.2f} |".format(
                milestone=row["milestone"],
                ratio=row["ratio"],
                mae=row["mae"],
                rmse=row["rmse"],
                p75=row["p75"],
                p95=row["p95"],
                topo=row["topology_violation_rate"] * 100.0,
            )
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build milestone longitudinal comparison charts from global_metrics.json files")
    parser.add_argument(
        "--milestone",
        action="append",
        default=[],
        help="Custom milestone in format name=analysis_dir (can be repeated)",
    )
    parser.add_argument("--skip-defaults", action="store_true", help="Use only user-provided milestones")
    parser.add_argument("--out_dir", type=Path, default=Path("task_A_recovery/analysis_outputs_milestones"))
    args = parser.parse_args()

    milestones = parse_milestones(args.milestone, args.skip_defaults)
    rows = build_rows(milestones)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "milestone_metrics.csv"
    out_md = args.out_dir / "milestone_summary.md"
    out_err_png = args.out_dir / "milestone_mae_rmse.png"
    out_topo_png = args.out_dir / "milestone_topology.png"

    write_csv(rows, out_csv)
    write_markdown(rows, out_md)
    plot_mae_rmse(rows, out_err_png)
    plot_topology(rows, out_topo_png)

    print("Milestone longitudinal outputs generated:")
    print(f"- {out_csv}")
    print(f"- {out_md}")
    print(f"- {out_err_png}")
    print(f"- {out_topo_png}")


if __name__ == "__main__":
    main()
