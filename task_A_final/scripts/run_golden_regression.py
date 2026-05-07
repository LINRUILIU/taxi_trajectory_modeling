from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import load_pickle, load_text_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golden regression suite")
    parser.add_argument("--manifest", default=ROOT / "golden" / "manifest.yaml", type=Path)
    parser.add_argument("--tmp-dir", default=ROOT / "runs" / "golden_regression", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_text_config(args.manifest)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for case in manifest.get("cases", []):
        out_path = args.tmp_dir / f"{case['name']}.pkl"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_predict.py"),
                "--config",
                str(ROOT / case["config"]),
                "--input",
                str(ROOT / case["input"]),
                "--out",
                str(out_path),
            ],
            check=True,
        )
        pred = load_pickle(out_path)
        expected = load_pickle(ROOT / case["expected_output"])
        if len(pred) != len(expected):
            failures.append(f"{case['name']}: record count mismatch")
            continue
        for p, e in zip(pred, expected):
            if int(p["traj_id"]) != int(e["traj_id"]):
                failures.append(f"{case['name']}: traj_id mismatch")
                break
            import numpy as np

            if not np.allclose(np.asarray(p["coords"], dtype=np.float64), np.asarray(e["coords"], dtype=np.float64), atol=float(case.get("atol", 1e-5))):
                failures.append(f"{case['name']}: coords drifted beyond tolerance")
                break
    if failures:
        print("Golden regression failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Golden regression passed.")


if __name__ == "__main__":
    main()
