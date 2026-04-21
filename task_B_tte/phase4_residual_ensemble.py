import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from baseline_tte import _departure_timestamp, _travel_time_from_timestamps, extract_trip_features
except ImportError:
    from task_B_tte.baseline_tte import _departure_timestamp, _travel_time_from_timestamps, extract_trip_features


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    err = y_pred - y_true
    abs_err = np.abs(err)
    denom = np.maximum(np.abs(y_true), 1e-6)

    return {
        "count": int(y_true.size),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(abs_err / denom) * 100.0),
        "bias": float(np.mean(err)),
    }


def _sample_weight(y_sec: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones_like(y_sec, dtype=np.float64)

    if mode == "short-boost":
        w = np.ones_like(y_sec, dtype=np.float64)
        w[y_sec <= 600.0] = 1.8
        w[(y_sec > 600.0) & (y_sec <= 1800.0)] = 1.2
        return w

    raise ValueError(f"Unknown weight mode: {mode}")


def _forward_target(y_sec: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y_sec
    if mode == "log1p":
        return np.log1p(y_sec)
    raise ValueError(f"Unknown transform: {mode}")


def _inverse_target(y_model: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y_model
    if mode == "log1p":
        return np.expm1(y_model)
    raise ValueError(f"Unknown transform: {mode}")


def build_supervised_dataset(
    records: Sequence[Dict],
    min_travel_time: float,
    max_travel_time: float,
    min_speed_kmh: float,
    max_speed_kmh: float,
    limit: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    xs: List[np.ndarray] = []
    ys: List[float] = []
    stats = {
        "total": 0,
        "used": 0,
        "drop_invalid_tt": 0,
        "drop_tt_range": 0,
        "drop_speed_range": 0,
        "drop_short_coords": 0,
    }

    n_take = len(records) if limit <= 0 else min(limit, len(records))
    for rec in records[:n_take]:
        stats["total"] += 1
        coords = np.asarray(rec["coords"], dtype=np.float64)
        if coords.shape[0] < 2:
            stats["drop_short_coords"] += 1
            continue

        tt = _travel_time_from_timestamps(rec)
        if tt is None or tt <= 0.0:
            stats["drop_invalid_tt"] += 1
            continue

        if tt < min_travel_time or tt > max_travel_time:
            stats["drop_tt_range"] += 1
            continue

        dep = _departure_timestamp(rec)
        feat = extract_trip_features(coords=coords, departure_ts=dep, map_runtime=None)

        path_len = float(feat[1])
        speed_kmh = path_len / tt * 3.6
        if speed_kmh < min_speed_kmh or speed_kmh > max_speed_kmh:
            stats["drop_speed_range"] += 1
            continue

        xs.append(feat)
        ys.append(float(tt))
        stats["used"] += 1

    if not xs:
        raise ValueError("No supervised samples after filtering.")

    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.float64), stats


def build_inference_dataset(records: Sequence[Dict], limit: int = 0) -> Tuple[np.ndarray, List[int]]:
    xs: List[np.ndarray] = []
    ids: List[int] = []

    n_take = len(records) if limit <= 0 else min(limit, len(records))
    for rec in records[:n_take]:
        traj_id = int(rec["traj_id"])
        coords = np.asarray(rec["coords"], dtype=np.float64)
        dep = _departure_timestamp(rec)
        feat = extract_trip_features(coords=coords, departure_ts=dep, map_runtime=None)
        xs.append(feat)
        ids.append(traj_id)

    if not xs:
        return np.zeros((0, 1), dtype=np.float32), []

    return np.vstack(xs).astype(np.float32), ids


def build_base1_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=450,
        min_samples_leaf=40,
        l2_regularization=0.05,
        random_state=seed,
    )


def build_base2_model(seed: int, max_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.035,
        max_depth=10,
        max_iter=max_iter,
        min_samples_leaf=30,
        l2_regularization=0.02,
        random_state=seed,
    )


def build_residual_model(seed: int, max_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=6,
        max_iter=max_iter,
        min_samples_leaf=55,
        l2_regularization=0.08,
        random_state=seed,
    )


def _predict_base(model, x: np.ndarray, transform: str) -> np.ndarray:
    raw = model.predict(x)
    y = _inverse_target(np.asarray(raw, dtype=np.float64), mode=transform)
    return np.clip(y, 1.0, None)


def _choose_blend_weight(
    y_true_val: np.ndarray,
    pred_base1_val: np.ndarray,
    pred_base2_val: np.ndarray,
    pred_resid_val: np.ndarray,
    baseline_ref: Optional[Dict[str, float]],
) -> Dict:
    candidates = []
    for w in np.linspace(0.0, 1.0, 81):
        pred = w * pred_resid_val + (1.0 - w) * pred_base2_val
        m = evaluate_regression(y_true_val, pred)
        candidates.append({"w_resid": float(w), "metrics": m})

    best = min(
        candidates,
        key=lambda c: (
            c["metrics"]["mape"],
            c["metrics"]["mae"],
            c["metrics"]["rmse"],
        ),
    )

    fallback_metrics = evaluate_regression(y_true_val, pred_base1_val)
    use_fallback = False

    if baseline_ref is not None:
        b_mae = float(baseline_ref.get("mae", math.inf))
        b_rmse = float(baseline_ref.get("rmse", math.inf))
        b_mape = float(baseline_ref.get("mape", math.inf))

        m = best["metrics"]
        dominates = (
            m["mae"] <= b_mae
            and m["rmse"] <= b_rmse
            and m["mape"] <= b_mape
            and (m["mae"] < b_mae or m["rmse"] < b_rmse or m["mape"] < b_mape)
        )
        if not dominates:
            use_fallback = True

    if use_fallback:
        return {
            "mode": "fallback_base1",
            "w_resid": 1.0,
            "selected_metrics": fallback_metrics,
            "best_candidate": best,
        }

    return {
        "mode": "blend",
        "w_resid": float(best["w_resid"]),
        "selected_metrics": best["metrics"],
        "best_candidate": best,
    }


def _load_baseline_reference(path: Optional[Path]) -> Optional[Dict[str, float]]:
    if path is None or (not path.exists()):
        return None

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if "val" in obj:
        g = obj["val"]
    elif "global" in obj:
        g = obj["global"]
    else:
        g = obj

    return {
        "mae": float(g.get("mae", math.inf)),
        "rmse": float(g.get("rmse", math.inf)),
        "mape": float(g.get("mape", math.inf)),
    }


def train_command(args: argparse.Namespace) -> None:
    train_records = load_pickle(args.train)
    x_train, y_train, train_stats = build_supervised_dataset(
        records=train_records,
        min_travel_time=args.min_travel_time,
        max_travel_time=args.max_travel_time,
        min_speed_kmh=args.min_speed_kmh,
        max_speed_kmh=args.max_speed_kmh,
        limit=args.train_limit,
    )

    val_input = load_pickle(args.val_input)
    x_val, val_ids = build_inference_dataset(val_input, limit=args.val_limit)
    val_gt = load_pickle(args.val_gt)
    y_val = np.asarray(
        [float(item["travel_time"]) for item in val_gt[: (len(val_gt) if args.val_limit <= 0 else args.val_limit)]],
        dtype=np.float64,
    )

    if y_val.size != x_val.shape[0]:
        raise ValueError(f"val mismatch: x={x_val.shape[0]} y={y_val.size}")

    base1 = build_base1_model(seed=args.seed)
    y_train_t = _forward_target(y_train, mode="log1p")
    w_train = _sample_weight(y_train, mode="short-boost")
    base1.fit(x_train, y_train_t, sample_weight=w_train)

    base2 = build_base2_model(seed=args.seed + 11, max_iter=args.base2_max_iter)
    base2.fit(x_train, y_train_t)

    pred_base1_train = _predict_base(base1, x_train, transform="log1p")
    pred_base2_train = _predict_base(base2, x_train, transform="log1p")

    residual_target_train = y_train - pred_base1_train
    resid_model = build_residual_model(seed=args.seed + 23, max_iter=args.residual_max_iter)
    resid_model.fit(x_train, residual_target_train)

    residual_train_pred = resid_model.predict(x_train)
    pred_resid_train = np.clip(pred_base1_train + residual_train_pred, 1.0, None)

    pred_base1_val = _predict_base(base1, x_val, transform="log1p")
    pred_base2_val = _predict_base(base2, x_val, transform="log1p")
    residual_val_pred = resid_model.predict(x_val)
    pred_resid_val = np.clip(pred_base1_val + residual_val_pred, 1.0, None)

    baseline_ref = _load_baseline_reference(args.baseline_metrics)
    choose = _choose_blend_weight(
        y_true_val=y_val,
        pred_base1_val=pred_base1_val,
        pred_base2_val=pred_base2_val,
        pred_resid_val=pred_resid_val,
        baseline_ref=baseline_ref,
    )

    if choose["mode"] == "fallback_base1":
        pred_final_train = pred_base1_train
        pred_final_val = pred_base1_val
    else:
        w_resid = float(choose["w_resid"])
        pred_final_train = np.clip(w_resid * pred_resid_train + (1.0 - w_resid) * pred_base2_train, 1.0, None)
        pred_final_val = np.clip(w_resid * pred_resid_val + (1.0 - w_resid) * pred_base2_val, 1.0, None)

    train_metrics = {
        "base1": evaluate_regression(y_train, pred_base1_train),
        "base2": evaluate_regression(y_train, pred_base2_train),
        "resid_path": evaluate_regression(y_train, pred_resid_train),
        "final": evaluate_regression(y_train, pred_final_train),
    }
    val_metrics = {
        "base1": evaluate_regression(y_val, pred_base1_val),
        "base2": evaluate_regression(y_val, pred_base2_val),
        "resid_path": evaluate_regression(y_val, pred_resid_val),
        "final": evaluate_regression(y_val, pred_final_val),
    }

    pred_rows = [{"traj_id": int(tid), "travel_time": float(pred)} for tid, pred in zip(val_ids, pred_final_val)]
    save_pickle(args.val_pred_out, pred_rows)

    artifact = {
        "kind": "phase4_residual_ensemble_v1",
        "base1": base1,
        "base2": base2,
        "residual": resid_model,
        "choose": choose,
        "feature_mode": "no-map",
        "config": {
            "seed": args.seed,
            "base2_max_iter": args.base2_max_iter,
            "residual_max_iter": args.residual_max_iter,
            "cleaning": {
                "min_travel_time": args.min_travel_time,
                "max_travel_time": args.max_travel_time,
                "min_speed_kmh": args.min_speed_kmh,
                "max_speed_kmh": args.max_speed_kmh,
            },
        },
        "summary": {
            "train_dataset": train_stats,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "baseline_reference": baseline_ref,
        },
    }

    save_pickle(args.model_out, artifact)

    metrics_json = {
        "train": train_metrics,
        "val": val_metrics,
        "selected": choose,
        "train_dataset": train_stats,
        "baseline_reference": baseline_ref,
    }
    save_json(args.metrics_out, metrics_json)

    print(f"Saved model: {args.model_out}")
    print(f"Saved val predictions: {args.val_pred_out} (n={len(pred_rows)})")
    print(f"Saved metrics: {args.metrics_out}")

    print("\n=== Phase4 Selected Mode ===")
    print(json.dumps(choose, ensure_ascii=False, indent=2))
    print("\n=== Val Final Metrics ===")
    print(json.dumps(val_metrics["final"], ensure_ascii=False, indent=2))


def predict_command(args: argparse.Namespace) -> None:
    artifact = load_pickle(args.model_in)
    base1 = artifact["base1"]
    base2 = artifact["base2"]
    resid_model = artifact["residual"]
    choose = artifact["choose"]

    recs = load_pickle(args.input)
    x, ids = build_inference_dataset(recs, limit=args.limit)

    pred_base1 = _predict_base(base1, x, transform="log1p")
    if choose["mode"] == "fallback_base1":
        pred_final = pred_base1
    else:
        pred_base2 = _predict_base(base2, x, transform="log1p")
        pred_resid = np.clip(pred_base1 + resid_model.predict(x), 1.0, None)
        w_resid = float(choose["w_resid"])
        pred_final = np.clip(w_resid * pred_resid + (1.0 - w_resid) * pred_base2, 1.0, None)

    out_rows = [{"traj_id": int(tid), "travel_time": float(tt)} for tid, tt in zip(ids, pred_final)]
    save_pickle(args.output, out_rows)
    print(f"Saved predictions: {args.output} (n={len(out_rows)})")


def evaluate_command(args: argparse.Namespace) -> None:
    pred = load_pickle(args.pred)
    gt = load_pickle(args.gt)

    pred_by_id = {int(item["traj_id"]): float(item["travel_time"]) for item in pred}
    y_true, y_pred = [], []

    for item in gt:
        tid = int(item["traj_id"])
        if tid not in pred_by_id:
            continue
        y_true.append(float(item["travel_time"]))
        y_pred.append(float(pred_by_id[tid]))

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    metrics = evaluate_regression(y_true_arr, y_pred_arr)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.metrics_out is not None:
        save_json(args.metrics_out, {"global": metrics})
        print(f"Saved metrics: {args.metrics_out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase4 residual + ensemble model (no-map)")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--train", type=Path, default=Path("data_ds15/train.pkl"))
    train.add_argument("--val-input", type=Path, default=Path("task_B_tte/val_input.pkl"))
    train.add_argument("--val-gt", type=Path, default=Path("task_B_tte/val_gt.pkl"))

    train.add_argument("--model-out", type=Path, default=Path("task_B_tte/model_phase4_residual_ensemble.pkl"))
    train.add_argument("--metrics-out", type=Path, default=Path("task_B_tte/metrics_phase4_residual_ensemble.json"))
    train.add_argument("--val-pred-out", type=Path, default=Path("task_B_tte/pred_val_phase4_residual_ensemble.pkl"))

    train.add_argument("--baseline-metrics", type=Path, default=Path("task_B_tte/analysis_outputs_baseline_hgb/global_metrics.json"))

    train.add_argument("--train-limit", type=int, default=0)
    train.add_argument("--val-limit", type=int, default=0)

    train.add_argument("--min-travel-time", type=float, default=60.0)
    train.add_argument("--max-travel-time", type=float, default=7200.0)
    train.add_argument("--min-speed-kmh", type=float, default=3.0)
    train.add_argument("--max-speed-kmh", type=float, default=120.0)

    train.add_argument("--seed", type=int, default=20260420)
    train.add_argument("--base2-max-iter", type=int, default=620)
    train.add_argument("--residual-max-iter", type=int, default=260)

    pred = sub.add_parser("predict")
    pred.add_argument("--model-in", type=Path, required=True)
    pred.add_argument("--input", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    pred.add_argument("--limit", type=int, default=0)

    eva = sub.add_parser("evaluate")
    eva.add_argument("--pred", type=Path, required=True)
    eva.add_argument("--gt", type=Path, required=True)
    eva.add_argument("--metrics-out", type=Path, default=None)

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
