from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import load_pickle


def _check_pair(input_path: Path, pred_path: Path, label: str) -> list[str]:
    issues: list[str] = []
    input_records = load_pickle(input_path)
    pred_records = load_pickle(pred_path)
    if not isinstance(pred_records, list):
        issues.append(f"{label}: prediction object must be list")
        return issues
    if len(input_records) != len(pred_records):
        issues.append(f"{label}: record count mismatch input={len(input_records)} pred={len(pred_records)}")
        return issues
    for idx, (inp, pred) in enumerate(zip(input_records, pred_records)):
        if not isinstance(pred, dict):
            issues.append(f"{label}: item #{idx} is not dict")
            continue
        if "traj_id" not in pred or "coords" not in pred:
            issues.append(f"{label}: item #{idx} missing traj_id/coords")
            continue
        if int(inp["traj_id"]) != int(pred["traj_id"]):
            issues.append(f"{label}: traj_id mismatch at index {idx}")
        in_coords = np.asarray(inp["coords"], dtype=np.float64)
        mask = np.asarray(inp["mask"], dtype=bool)
        pred_coords = np.asarray(pred["coords"], dtype=np.float64)
        if pred_coords.shape != in_coords.shape:
            issues.append(f"{label}: coords shape mismatch for traj_id={pred['traj_id']}")
            continue
        if pred_coords.ndim != 2 or pred_coords.shape[1] != 2:
            issues.append(f"{label}: coords must have shape [N,2] for traj_id={pred['traj_id']}")
        if not np.all(np.isfinite(pred_coords)):
            issues.append(f"{label}: non-finite coords for traj_id={pred['traj_id']}")
        if not np.allclose(pred_coords[mask], in_coords[mask], atol=1e-7, equal_nan=False):
            issues.append(f"{label}: known points changed for traj_id={pred['traj_id']}")
        lon_ok = np.all((pred_coords[:, 0] >= -180.0) & (pred_coords[:, 0] <= 180.0))
        lat_ok = np.all((pred_coords[:, 1] >= -90.0) & (pred_coords[:, 1] <= 90.0))
        if not lon_ok or not lat_ok:
            issues.append(f"{label}: out-of-range lon/lat for traj_id={pred['traj_id']}")
        missing_count = int(np.sum(~mask))
        pred_missing = pred_coords[~mask]
        if pred_missing.shape[0] != missing_count:
            issues.append(f"{label}: missing-point count mismatch for traj_id={pred['traj_id']}")
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke check predictions without GT/OSM")
    parser.add_argument("--input-8", required=True, type=Path)
    parser.add_argument("--input-16", required=True, type=Path)
    parser.add_argument("--pred-8", required=True, type=Path)
    parser.add_argument("--pred-16", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issues = []
    issues.extend(_check_pair(args.input_8, args.pred_8, "1/8"))
    issues.extend(_check_pair(args.input_16, args.pred_16, "1/16"))
    if issues:
        print("Smoke check failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Smoke check passed.")


if __name__ == "__main__":
    main()
