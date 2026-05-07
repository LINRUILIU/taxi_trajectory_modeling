from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, List


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _priority_key(row: Dict[str, Any]) -> tuple:
    dataset_rank = 0 if row.get("dataset") == "1/16" else 1
    gap_size = -int(float(row.get("gap_size", 0) or 0))
    official = -float(row.get("official_mae_m", 0.0) or 0.0)
    stable = hashlib.md5(
        f"{row.get('dataset')}|{row.get('traj_id')}|{row.get('gap_start_idx')}|{row.get('gap_end_idx')}".encode("utf-8")
    ).hexdigest()
    return (dataset_rank, gap_size, official, stable)


def _sample_rows(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return sorted(rows, key=_priority_key)[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample review queues for manual inspection")
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("review_queue_path_wrong.csv", "sample_path_wrong.csv", 20),
        ("review_queue_phase_wrong.csv", "sample_phase_wrong.csv", 20),
        ("review_queue_metric_cheat.csv", "sample_metric_cheat.csv", 10),
    ]
    for src_name, out_name, limit in configs:
        src = args.analysis_dir / src_name
        rows = _read_csv(src)
        sampled = _sample_rows(rows, limit)
        _write_csv(args.out_dir / out_name, sampled)
    print(f"Sample review queues saved: {args.out_dir}")


if __name__ == "__main__":
    main()
