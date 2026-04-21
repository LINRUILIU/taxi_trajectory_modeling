import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CaseConfig:
    name: str
    use_map: bool
    map_query_radius_m: float = 120.0
    map_max_points: int = 14
    map_near_threshold_1_m: float = 20.0
    map_near_threshold_2_m: float = 50.0
    map_cell_size_deg: float = 0.001


def _default_cases() -> List[CaseConfig]:
    return [
        CaseConfig(name="nomap_ref", use_map=False),
        CaseConfig(name="map_r80_p10_t1535", use_map=True, map_query_radius_m=80.0, map_max_points=10, map_near_threshold_1_m=15.0, map_near_threshold_2_m=35.0),
        CaseConfig(name="map_r95_p12_t1540", use_map=True, map_query_radius_m=95.0, map_max_points=12, map_near_threshold_1_m=15.0, map_near_threshold_2_m=40.0),
        CaseConfig(name="map_r120_p14_t2050", use_map=True, map_query_radius_m=120.0, map_max_points=14, map_near_threshold_1_m=20.0, map_near_threshold_2_m=50.0),
        CaseConfig(name="map_r90_p18_t1545", use_map=True, map_query_radius_m=90.0, map_max_points=18, map_near_threshold_1_m=15.0, map_near_threshold_2_m=45.0),
        CaseConfig(name="map_r140_p18_t2060", use_map=True, map_query_radius_m=140.0, map_max_points=18, map_near_threshold_1_m=20.0, map_near_threshold_2_m=60.0),
        CaseConfig(name="map_r160_p22_t2570", use_map=True, map_query_radius_m=160.0, map_max_points=22, map_near_threshold_1_m=25.0, map_near_threshold_2_m=70.0),
        CaseConfig(name="map_r120_p14_t2050_c08", use_map=True, map_query_radius_m=120.0, map_max_points=14, map_near_threshold_1_m=20.0, map_near_threshold_2_m=50.0, map_cell_size_deg=0.0008),
    ]


def _run_case(
    python_exe: str,
    case: CaseConfig,
    train_limit: int,
    val_limit: int,
    hgb_max_iter: int,
    osm_path: Path,
    map_cache: Path,
    out_dir: Path,
) -> Dict:
    model_out = out_dir / f"{case.name}_model.pkl"
    metrics_out = out_dir / f"{case.name}_metrics.json"
    val_pred_out = out_dir / f"{case.name}_pred.pkl"

    cmd = [
        python_exe,
        "task_B_tte/baseline_tte.py",
        "train",
        "--train-limit",
        str(train_limit),
        "--val-limit",
        str(val_limit),
        "--hgb-max-iter",
        str(hgb_max_iter),
        "--model-out",
        str(model_out),
        "--metrics-out",
        str(metrics_out),
        "--val-pred-out",
        str(val_pred_out),
    ]

    if case.use_map:
        cmd.extend(
            [
                "--osm",
                str(osm_path),
                "--map-cache",
                str(map_cache),
                "--map-cell-size-deg",
                str(case.map_cell_size_deg),
                "--map-query-radius-m",
                str(case.map_query_radius_m),
                "--map-max-points",
                str(case.map_max_points),
                "--map-near-threshold-1-m",
                str(case.map_near_threshold_1_m),
                "--map-near-threshold-2-m",
                str(case.map_near_threshold_2_m),
            ]
        )

    print("\n[round2] running case:", case.name)
    print("[round2] cmd:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        return {
            "name": case.name,
            "use_map": case.use_map,
            "status": "failed",
            "returncode": proc.returncode,
            "metrics_file": str(metrics_out),
            "pred_file": str(val_pred_out),
            "model_file": str(model_out),
        }

    with metrics_out.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    val = metrics.get("val", {})

    return {
        "name": case.name,
        "use_map": case.use_map,
        "status": "ok",
        "mae": float(val.get("mae", float("nan"))),
        "rmse": float(val.get("rmse", float("nan"))),
        "mape": float(val.get("mape", float("nan"))),
        "metrics_file": str(metrics_out),
        "pred_file": str(val_pred_out),
        "model_file": str(model_out),
        "map_query_radius_m": case.map_query_radius_m,
        "map_max_points": case.map_max_points,
        "map_near_threshold_1_m": case.map_near_threshold_1_m,
        "map_near_threshold_2_m": case.map_near_threshold_2_m,
        "map_cell_size_deg": case.map_cell_size_deg,
    }


def _write_results_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "use_map",
        "status",
        "mae",
        "rmse",
        "mape",
        "delta_mae_vs_nomap",
        "delta_rmse_vs_nomap",
        "delta_mape_vs_nomap",
        "map_query_radius_m",
        "map_max_points",
        "map_near_threshold_1_m",
        "map_near_threshold_2_m",
        "map_cell_size_deg",
        "metrics_file",
        "pred_file",
        "model_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_summary_md(path: Path, rows: List[Dict], baseline_row: Optional[Dict]) -> None:
    lines: List[str] = []
    lines.append("# Phase3 Round2 Map Tuning Summary")
    lines.append("")

    if baseline_row is not None and baseline_row.get("status") == "ok":
        lines.append(
            "- nomap_ref: MAE={:.4f}, RMSE={:.4f}, MAPE={:.4f}".format(
                baseline_row["mae"], baseline_row["rmse"], baseline_row["mape"]
            )
        )
        lines.append("")

    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("use_map")]
    ok_rows_sorted = sorted(ok_rows, key=lambda r: (r["mape"], r["mae"], r["rmse"]))

    if ok_rows_sorted:
        best = ok_rows_sorted[0]
        lines.append(
            "- best_map_case: {} (MAE={:.4f}, RMSE={:.4f}, MAPE={:.4f})".format(
                best["name"], best["mae"], best["rmse"], best["mape"]
            )
        )
        lines.append("")

    lines.append("## Table")
    lines.append("")
    lines.append("| case | use_map | status | MAE | RMSE | MAPE | ΔMAE_vs_nomap | ΔRMSE_vs_nomap | ΔMAPE_vs_nomap |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {name} | {use_map} | {status} | {mae:.4f} | {rmse:.4f} | {mape:.4f} | {d_mae:.4f} | {d_rmse:.4f} | {d_mape:.4f} |".format(
                name=r.get("name", ""),
                use_map=int(bool(r.get("use_map", False))),
                status=r.get("status", ""),
                mae=float(r.get("mae", float("nan"))),
                rmse=float(r.get("rmse", float("nan"))),
                mape=float(r.get("mape", float("nan"))),
                d_mae=float(r.get("delta_mae_vs_nomap", float("nan"))),
                d_rmse=float(r.get("delta_rmse_vs_nomap", float("nan"))),
                d_mape=float(r.get("delta_mape_vs_nomap", float("nan"))),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase3 round2 map tuning on limited train/val")
    parser.add_argument("--train-limit", type=int, default=24000)
    parser.add_argument("--val-limit", type=int, default=6000)
    parser.add_argument("--hgb-max-iter", type=int, default=260)
    parser.add_argument("--osm", type=Path, default=Path("map"))
    parser.add_argument("--map-cache", type=Path, default=Path("task_B_tte/map_segments_cache.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("task_B_tte/phase3_round2_screen"))
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    cases = _default_cases()

    rows: List[Dict] = []
    for case in cases:
        row = _run_case(
            python_exe=python_exe,
            case=case,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            hgb_max_iter=args.hgb_max_iter,
            osm_path=args.osm,
            map_cache=args.map_cache,
            out_dir=out_dir,
        )
        rows.append(row)

    baseline_row = None
    for r in rows:
        if r.get("name") == "nomap_ref" and r.get("status") == "ok":
            baseline_row = r
            break

    if baseline_row is not None:
        b_mae = baseline_row["mae"]
        b_rmse = baseline_row["rmse"]
        b_mape = baseline_row["mape"]
        for r in rows:
            if r.get("status") != "ok":
                r["delta_mae_vs_nomap"] = float("nan")
                r["delta_rmse_vs_nomap"] = float("nan")
                r["delta_mape_vs_nomap"] = float("nan")
                continue
            r["delta_mae_vs_nomap"] = float(r["mae"] - b_mae)
            r["delta_rmse_vs_nomap"] = float(r["rmse"] - b_rmse)
            r["delta_mape_vs_nomap"] = float(r["mape"] - b_mape)
    else:
        for r in rows:
            r["delta_mae_vs_nomap"] = float("nan")
            r["delta_rmse_vs_nomap"] = float("nan")
            r["delta_mape_vs_nomap"] = float("nan")

    csv_path = out_dir / "round2_results.csv"
    json_path = out_dir / "round2_results.json"
    md_path = out_dir / "round2_summary.md"

    _write_results_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    _write_summary_md(md_path, rows, baseline_row=baseline_row)

    ok_maps = [r for r in rows if r.get("status") == "ok" and r.get("use_map")]
    if ok_maps:
        best = sorted(ok_maps, key=lambda r: (r["mape"], r["mae"], r["rmse"]))[0]
        print("\n[round2] best map case:", best["name"])
        print(
            "[round2] best metrics: MAE={:.4f}, RMSE={:.4f}, MAPE={:.4f}".format(
                best["mae"], best["rmse"], best["mape"]
            )
        )
    else:
        print("\n[round2] no successful map case")

    print(f"[round2] saved csv: {csv_path}")
    print(f"[round2] saved json: {json_path}")
    print(f"[round2] saved summary: {md_path}")


if __name__ == "__main__":
    main()
