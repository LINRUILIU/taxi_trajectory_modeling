from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import load_pickle, save_pickle, save_text


def _subset_records(records, ids):
    keep = set(int(x) for x in ids)
    return [item for item in records if int(item["traj_id"]) in keep]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build golden regression assets")
    parser.add_argument("--input-8", required=True, type=Path)
    parser.add_argument("--input-16", required=True, type=Path)
    parser.add_argument("--expected-pchip-8", required=True, type=Path)
    parser.add_argument("--expected-pchip-16", required=True, type=Path)
    parser.add_argument("--expected-b28-8", required=True, type=Path)
    parser.add_argument("--expected-b28-16", required=True, type=Path)
    parser.add_argument("--out-dir", default=ROOT / "golden", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    input8 = load_pickle(args.input_8)
    input16 = load_pickle(args.input_16)
    ids8 = [int(x["traj_id"]) for x in input8[:8]]
    ids16 = [int(x["traj_id"]) for x in input16[:8]]
    save_pickle(args.out_dir / "golden_input_8.pkl", _subset_records(input8, ids8))
    save_pickle(args.out_dir / "golden_input_16.pkl", _subset_records(input16, ids16))
    save_pickle(args.out_dir / "golden_expected_pchip_8.pkl", _subset_records(load_pickle(args.expected_pchip_8), ids8))
    save_pickle(args.out_dir / "golden_expected_pchip_16.pkl", _subset_records(load_pickle(args.expected_pchip_16), ids16))
    save_pickle(args.out_dir / "golden_expected_b28_8.pkl", _subset_records(load_pickle(args.expected_b28_8), ids8))
    save_pickle(args.out_dir / "golden_expected_b28_16.pkl", _subset_records(load_pickle(args.expected_b28_16), ids16))
    save_text(
        args.out_dir / "manifest.yaml",
        """{
  "cases": [
    {
      "name": "pchip_only_8",
      "config": "configs/exp_pchip_only_8.yaml",
      "input": "golden/golden_input_8.pkl",
      "expected_output": "golden/golden_expected_pchip_8.pkl",
      "atol": 1e-5
    },
    {
      "name": "pchip_only_16",
      "config": "configs/exp_pchip_only_16.yaml",
      "input": "golden/golden_input_16.pkl",
      "expected_output": "golden/golden_expected_pchip_16.pkl",
      "atol": 1e-5
    },
    {
      "name": "b28_compat_8",
      "config": "configs/exp_b28_compat_8.yaml",
      "input": "golden/golden_input_8.pkl",
      "expected_output": "golden/golden_expected_b28_8.pkl",
      "atol": 1e-5
    },
    {
      "name": "b28_compat_16",
      "config": "configs/exp_b28_compat_16.yaml",
      "input": "golden/golden_input_16.pkl",
      "expected_output": "golden/golden_expected_b28_16.pkl",
      "atol": 1e-5
    }
  ]
}
""",
    )
    print(f"Golden assets built under: {args.out_dir}")


if __name__ == "__main__":
    main()
