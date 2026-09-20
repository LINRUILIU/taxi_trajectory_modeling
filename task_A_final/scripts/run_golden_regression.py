from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import load_pickle, load_text_config

# Absolute per-coordinate tolerance in longitude/latitude degrees.
DEFAULT_ATOL_DEGREES = 1e-5


def _compare_records(pred, expected, label: str, atol: float) -> list[str]:
    if not np.isfinite(atol) or atol < 0:
        return [f"{label}: atol must be finite and non-negative (degrees)"]
    if not isinstance(pred, list) or not isinstance(expected, list):
        return [f"{label}: prediction and expected objects must be lists"]
    if len(pred) != len(expected):
        return [f"{label}: record count mismatch"]
    for idx, (p, e) in enumerate(zip(pred, expected)):
        context = f"{label}: item #{idx}"
        if not isinstance(p, dict) or not isinstance(e, dict):
            return [f"{context}: records must be dicts"]
        if any(key not in record for record in (p, e) for key in ("traj_id", "coords")):
            return [f"{context}: missing traj_id/coords"]
        if p["traj_id"] != e["traj_id"]:
            return [f"{context}: traj_id mismatch"]
        try:
            actual = np.asarray(p["coords"], dtype=np.float64)
            reference = np.asarray(e["coords"], dtype=np.float64)
        except (TypeError, ValueError):
            return [f"{context}: coords must be numeric arrays"]
        # Check shapes before allclose: broadcasting can accept missing rows.
        if actual.shape != reference.shape:
            return [f"{context}: coords shape mismatch"]
        if actual.ndim != 2 or actual.shape[1] != 2:
            return [f"{context}: coords must have shape [N,2]"]
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference)):
            return [f"{context}: non-finite coords"]
        if not np.allclose(actual, reference, rtol=0.0, atol=atol, equal_nan=False):
            return [f"{context}: coords drifted beyond absolute tolerance {atol:g} degrees"]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golden regression suite")
    parser.add_argument("--manifest", default=ROOT / "golden" / "manifest.yaml", type=Path)
    parser.add_argument("--tmp-dir", default=ROOT / "runs" / "golden_regression", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_text_config(args.manifest)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Golden manifest must contain a non-empty cases list")
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for case in cases:
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
        failures.extend(_compare_records(
            pred, expected, case["name"], float(case.get("atol", DEFAULT_ATOL_DEGREES)),
        ))
    if failures:
        print("Golden regression failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Golden regression passed.")


if __name__ == "__main__":
    main()
