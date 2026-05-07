from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import load_pickle, load_text_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of Task A experiments")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input-8", required=True, type=Path)
    parser.add_argument("--input-16", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--runs-dir", default=ROOT / "runs", type=Path)
    return parser.parse_args()


def _python(script: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script)]


def main() -> None:
    args = parse_args()
    manifest = load_text_config(args.manifest)
    rows = []
    for exp in manifest.get("experiments", []):
        name = exp["name"]
        run_dir = args.runs_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        pred8 = run_dir / "pred_8.pkl"
        pred16 = run_dir / "pred_16.pkl"
        analysis_dir = run_dir / "analysis"
        subprocess.run(_python("run_predict.py") + ["--config", str(Path(exp["config_8"])), "--input", str(args.input_8), "--out", str(pred8)], check=True)
        subprocess.run(_python("run_predict.py") + ["--config", str(Path(exp["config_16"])), "--input", str(args.input_16), "--out", str(pred16)], check=True)
        subprocess.run(
            _python("run_analyze.py")
            + [
                "--input-8",
                str(args.input_8),
                "--input-16",
                str(args.input_16),
                "--pred-8",
                str(pred8),
                "--pred-16",
                str(pred16),
                "--gt",
                str(args.gt),
                "--out-dir",
                str(analysis_dir),
            ],
            check=True,
        )
        metrics = load_text_config(analysis_dir / "global_metrics.json")
        quadrant = load_text_config(analysis_dir / "quadrant_summary.json")
        for dataset in ("1/8", "1/16"):
            rows.append(
                {
                    "experiment": name,
                    "dataset": dataset,
                    "mae": metrics[dataset]["mae"],
                    "rmse": metrics[dataset]["rmse"],
                    "p75": metrics[dataset]["p75"],
                    "p95": metrics[dataset]["p95"],
                    "shape_symmetric_m": quadrant["shape_threshold_m"],
                    "high_official_high_shape_count": quadrant.get("high_official_high_shape", 0),
                    "high_official_low_shape_count": quadrant.get("high_official_low_shape", 0),
                    "runtime_sec": 0.0,
                }
            )
    out_csv = args.runs_dir / "experiment_compare.csv"
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Batch summary saved: {out_csv}")


if __name__ == "__main__":
    main()
