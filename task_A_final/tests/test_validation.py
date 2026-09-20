from __future__ import annotations

import copy
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from smoke_check import _check_pair
from run_golden_regression import _compare_records


class SmokeValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = [{
            "traj_id": 7,
            "coords": np.array([[108.0, 34.0], [np.nan, np.nan], [108.2, 34.2]]),
            "mask": np.array([True, False, True]),
        }]
        self.pred = [{"traj_id": 7, "coords": np.array([[108.0, 34.0], [108.1, 34.1], [108.2, 34.2]])}]

    def check(self, pred=None):
        self.input_path = self.root / "input.pkl"
        self.pred_path = self.root / "pred.pkl"
        self.input_path.write_bytes(pickle.dumps(self.input))
        self.pred_path.write_bytes(pickle.dumps(self.pred if pred is None else pred))
        return _check_pair(self.input_path, self.pred_path, "sample")

    def test_reconstructed_missing_points_pass(self):
        self.assertEqual(self.check(), [])

    def test_known_longitude_and_latitude_drift_are_rejected(self):
        for axis in (0, 1):
            with self.subTest(axis=axis):
                bad = copy.deepcopy(self.pred)
                bad[0]["coords"][0, axis] += 1e-4
                self.assertIn("known points changed", " ".join(self.check(bad)))

    def test_small_rounding_within_absolute_tolerance_passes(self):
        self.pred[0]["coords"][0, 0] += 5e-8
        self.assertEqual(self.check(), [])

    def test_bad_outputs_are_rejected(self):
        for mutation, expected_message in (
            ("id", "traj_id mismatch"), ("fractional_id", "traj_id mismatch"), ("nan", "non-finite"),
            ("inf", "non-finite"), ("shape", "shape mismatch"),
            ("missing_record", "record count mismatch"), ("extra_record", "record count mismatch"),
            ("missing_coords", "missing traj_id/coords"), ("range", "out-of-range"),
        ):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.pred)
                if mutation == "id":
                    bad[0]["traj_id"] = 8
                elif mutation == "fractional_id":
                    bad[0]["traj_id"] = 7.5
                elif mutation in ("nan", "inf"):
                    bad[0]["coords"][1, 0] = float(mutation)
                elif mutation == "shape":
                    bad[0]["coords"] = bad[0]["coords"][:1]
                elif mutation == "missing_record":
                    bad = []
                elif mutation == "extra_record":
                    bad.append(copy.deepcopy(bad[0]))
                elif mutation == "missing_coords":
                    del bad[0]["coords"]
                elif mutation == "range":
                    bad[0]["coords"][1, 1] = 91.0
                self.assertIn(expected_message, " ".join(self.check(bad)))

    def test_invalid_dimensions_and_mask_fail_cleanly(self):
        self.input[0]["mask"] = np.array([True])
        self.assertIn("mask shape mismatch", " ".join(self.check()))
        self.input[0]["coords"] = np.array([108.0, 34.0])
        self.pred[0]["coords"] = np.array([108.0, 34.0])
        self.assertIn("coords must have shape", " ".join(self.check()))

    def test_cli_exit_code_and_diagnostic(self):
        self.check()
        args = [sys.executable, str(ROOT / "scripts" / "smoke_check.py"),
                "--input-8", str(self.input_path), "--input-16", str(self.input_path),
                "--pred-8", str(self.pred_path), "--pred-16", str(self.pred_path)]
        good = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.pred[0]["coords"][0, 0] += 1e-4
        self.check()
        bad = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertEqual(bad.returncode, 1, bad.stderr)
        self.assertIn("known points changed", bad.stdout)


class GoldenValidationTests(unittest.TestCase):
    def setUp(self):
        self.expected = [{"traj_id": 7, "coords": np.array([[108.0, 34.0], [108.0, 34.0]])}]

    def check(self, pred, atol=1e-5):
        return _compare_records(pred, self.expected, "golden", atol)

    def test_unchanged_and_small_rounding_pass(self):
        good = copy.deepcopy(self.expected)
        self.assertEqual(self.check(good), [])
        good[0]["coords"][0, 0] += 5e-6
        self.assertEqual(self.check(good), [])

    def test_absolute_drift_is_rejected(self):
        bad = copy.deepcopy(self.expected)
        bad[0]["coords"][0, 0] += 1e-4
        self.assertIn("absolute tolerance", " ".join(self.check(bad)))

    def test_broadcastable_missing_rows_are_rejected(self):
        bad = copy.deepcopy(self.expected)
        bad[0]["coords"] = bad[0]["coords"][:1]
        self.assertIn("shape mismatch", " ".join(self.check(bad)))

    def test_nan_and_matching_infinity_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.expected[0]["coords"][0, 0] = value
                self.assertIn("non-finite", " ".join(self.check(copy.deepcopy(self.expected))))

    def test_ids_counts_and_malformed_records_are_rejected(self):
        for pred in ([], self.expected * 2, {}, [None], [{}],
                     [{"traj_id": 8, "coords": self.expected[0]["coords"]}],
                     [{"traj_id": 7, "coords": "bad"}],
                     [{"traj_id": 7.5, "coords": self.expected[0]["coords"]}]):
            with self.subTest(pred=pred):
                self.assertTrue(self.check(pred))

    def test_invalid_tolerances_are_rejected(self):
        for atol in (-1.0, np.nan, np.inf):
            with self.subTest(atol=atol):
                self.assertIn("atol must be", " ".join(self.check(self.expected, atol)))

    def test_empty_manifest_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"cases": []}), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_golden_regression.py"), "--manifest", str(manifest)],
                capture_output=True, text=True, timeout=30,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-empty cases list", result.stderr)


if __name__ == "__main__":
    unittest.main()
