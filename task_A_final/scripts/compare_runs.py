from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: List[Dict[str, Any]] = []
    for row in rows:
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
        out.append(parsed)
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _gap_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (row["dataset"], row["traj_id"], row["gap_start_idx"], row["gap_end_idx"])


def _pair_gap_rows(base_rows: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new_by_key = {_gap_key(r): r for r in new_rows}
    paired: List[Dict[str, Any]] = []
    for base in base_rows:
        key = _gap_key(base)
        if key not in new_by_key:
            continue
        new = new_by_key[key]
        paired.append(
            {
                "dataset": base["dataset"],
                "traj_id": base["traj_id"],
                "gap_start_idx": base["gap_start_idx"],
                "gap_end_idx": base["gap_end_idx"],
                "gap_size": base["gap_size"],
                "missing_point_count": base["missing_point_count"],
                "gap_bucket": base["gap_bucket"],
                "base_official_mae_m": base["official_mae_m"],
                "new_official_mae_m": new["official_mae_m"],
                "delta_official_mae_m": float(new["official_mae_m"] - base["official_mae_m"]),
                "base_shape_symmetric_m": base["shape_symmetric_m"],
                "new_shape_symmetric_m": new["shape_symmetric_m"],
                "delta_shape_symmetric_m": float(new["shape_symmetric_m"] - base["shape_symmetric_m"]),
                "base_quadrant_global": base["quadrant_global"],
                "new_quadrant_global": new["quadrant_global"],
                "base_quadrant_dataset": base["quadrant_dataset"],
                "new_quadrant_dataset": new["quadrant_dataset"],
                "base_quadrant_gap_bucket": base["quadrant_gap_bucket"],
                "new_quadrant_gap_bucket": new["quadrant_gap_bucket"],
                "base_official_error_sum_m": base["official_error_sum_m"],
                "new_official_error_sum_m": new["official_error_sum_m"],
                "delta_official_error_sum_m": float(new["official_error_sum_m"] - base["official_error_sum_m"]),
            }
        )
    return paired


def _global_delta(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dataset in sorted(base.keys()):
        out[dataset] = {}
        for metric, base_value in base[dataset].items():
            new_value = new.get(dataset, {}).get(metric)
            if isinstance(base_value, (int, float)) and isinstance(new_value, (int, float)):
                out[dataset][metric] = {
                    "base": base_value,
                    "new": new_value,
                    "delta": float(new_value - base_value),
                }
    return out


def _bucket_delta(base_rows: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new_by_key = {(r["dataset"], r["gap_bucket"]): r for r in new_rows}
    out: List[Dict[str, Any]] = []
    for base in base_rows:
        key = (base["dataset"], base["gap_bucket"])
        if key not in new_by_key:
            continue
        new = new_by_key[key]
        out.append(
            {
                "dataset": base["dataset"],
                "gap_bucket": base["gap_bucket"],
                "base_count": base["count"],
                "new_count": new["count"],
                "base_official_mae_m": base["official_mae_m"],
                "new_official_mae_m": new["official_mae_m"],
                "delta_official_mae_m": float(new["official_mae_m"] - base["official_mae_m"]),
                "base_official_p95_m": base["official_p95_m"],
                "new_official_p95_m": new["official_p95_m"],
                "delta_official_p95_m": float(new["official_p95_m"] - base["official_p95_m"]),
                "base_shape_symmetric_m": base["shape_symmetric_m"],
                "new_shape_symmetric_m": new["shape_symmetric_m"],
                "delta_shape_symmetric_m": float(new["shape_symmetric_m"] - base["shape_symmetric_m"]),
            }
        )
    return out


def _quadrant_delta(paired_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        key = (
            row["base_quadrant_global"],
            row["base_quadrant_dataset"],
            row["dataset"],
            row["gap_bucket"],
        )
        grouped[key].append(row)
    out: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        base_error = float(sum(r["base_official_error_sum_m"] for r in rows))
        new_error = float(sum(r["new_official_error_sum_m"] for r in rows))
        gap_count = len(rows)
        missing_point_count = int(sum(int(r["missing_point_count"]) for r in rows))
        out.append(
            {
                "base_quadrant_global": key[0],
                "base_quadrant_dataset": key[1],
                "dataset": key[2],
                "gap_bucket": key[3],
                "gap_count": gap_count,
                "missing_point_count": missing_point_count,
                "base_official_error_sum_m": base_error,
                "new_official_error_sum_m": new_error,
                "delta_official_error_sum_m": float(new_error - base_error),
                "base_official_mae_mean_m": float(sum(r["base_official_mae_m"] for r in rows) / max(gap_count, 1)),
                "new_official_mae_mean_m": float(sum(r["new_official_mae_m"] for r in rows) / max(gap_count, 1)),
                "delta_official_mae_mean_m": float(sum(r["delta_official_mae_m"] for r in rows) / max(gap_count, 1)),
            }
        )
    out.sort(key=lambda r: (r["dataset"], r["gap_bucket"], r["base_quadrant_global"], r["base_quadrant_dataset"]))
    return out


def _summarize_quadrant_focus(rows: List[Dict[str, Any]], quadrant: str) -> tuple[float, int]:
    matched = [r for r in rows if r["base_quadrant_global"] == quadrant]
    if not matched:
        return 0.0, 0
    return float(sum(r["delta_official_mae_m"] for r in matched) / len(matched)), len(matched)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Task A analysis runs")
    parser.add_argument("--base", required=True, type=Path, help="Base analysis directory")
    parser.add_argument("--new", required=True, type=Path, help="New analysis directory")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_global = _load_json(args.base / "global_metrics.json")
    new_global = _load_json(args.new / "global_metrics.json")
    base_gap = _read_csv(args.base / "gap_metrics.csv")
    new_gap = _read_csv(args.new / "gap_metrics.csv")
    base_bucket = _read_csv(args.base / "bucket_summary.csv")
    new_bucket = _read_csv(args.new / "bucket_summary.csv")

    paired = _pair_gap_rows(base_gap, new_gap)
    bucket_delta = _bucket_delta(base_bucket, new_bucket)
    quadrant_delta = _quadrant_delta(paired)
    global_delta = _global_delta(base_global, new_global)

    paired_sorted_improved = sorted(paired, key=lambda r: (r["delta_official_mae_m"], r["delta_shape_symmetric_m"]))
    paired_sorted_worsened = sorted(paired, key=lambda r: (r["delta_official_mae_m"], r["delta_shape_symmetric_m"]), reverse=True)

    _write_csv(args.out_dir / "paired_gap_delta.csv", paired)
    _write_csv(args.out_dir / "bucket_delta.csv", bucket_delta)
    _write_csv(args.out_dir / "quadrant_delta.csv", quadrant_delta)
    _write_csv(args.out_dir / "top_improved_gaps.csv", paired_sorted_improved[:200])
    _write_csv(args.out_dir / "top_worsened_gaps.csv", paired_sorted_worsened[:200])
    (args.out_dir / "global_delta.json").write_text(json.dumps(global_delta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    hi_low_mean, hi_low_count = _summarize_quadrant_focus(paired, "high_official_low_shape")
    hi_hi_mean, hi_hi_count = _summarize_quadrant_focus(paired, "high_official_high_shape")
    low_hi_mean, low_hi_count = _summarize_quadrant_focus(paired, "low_official_high_shape")
    long_16 = [r for r in bucket_delta if r["dataset"] == "1/16" and r["gap_bucket"] == "9-15"]
    long_16_line = "n/a"
    if long_16:
        row = long_16[0]
        long_16_line = f"{row['delta_official_mae_m']:.4f}m MAE, {row['delta_official_p95_m']:.4f}m P95"

    lines = [
        "# Run Comparison Summary",
        "",
        f"- Base: {args.base}",
        f"- New: {args.new}",
        f"- Paired gaps: {len(paired)}",
        "",
        "## Global Delta",
        f"- 1/8 MAE delta: {global_delta['1/8']['mae']['delta']:.4f} m",
        f"- 1/16 MAE delta: {global_delta['1/16']['mae']['delta']:.4f} m",
        f"- 1/8 P95 delta: {global_delta['1/8']['p95']['delta']:.4f} m",
        f"- 1/16 P95 delta: {global_delta['1/16']['p95']['delta']:.4f} m",
        "",
        "## Baseline Quadrant Focus",
        f"- high_official_low_shape: mean official MAE delta {hi_low_mean:.4f} m across {hi_low_count} gaps",
        f"- high_official_high_shape: mean official MAE delta {hi_hi_mean:.4f} m across {hi_hi_count} gaps",
        f"- low_official_high_shape: mean official MAE delta {low_hi_mean:.4f} m across {low_hi_count} gaps",
        f"- 1/16 gap bucket 9-15: {long_16_line}",
    ]
    (args.out_dir / "compare_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison outputs saved: {args.out_dir}")


if __name__ == "__main__":
    main()
