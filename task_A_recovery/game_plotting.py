from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analyze_recovery import draw_road_overlay
from game_core import CaseData


def _finite_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    ok = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return arr[ok]


def compute_case_bbox(
    case: CaseData,
    player_pred_coords: Optional[np.ndarray] = None,
    pad_ratio: float = 0.08,
) -> Tuple[float, float, float, float]:
    chunks = [
        _finite_rows(case.gt_coords),
        _finite_rows(case.pred23_coords),
        _finite_rows(case.pred28_coords),
        _finite_rows(case.input_coords[case.mask]),
    ]
    if player_pred_coords is not None:
        chunks.append(_finite_rows(player_pred_coords))

    non_empty = [x for x in chunks if x.size > 0]
    if not non_empty:
        return 0.0, 1.0, 0.0, 1.0

    merged = np.vstack(non_empty)
    x_min = float(np.min(merged[:, 0]))
    x_max = float(np.max(merged[:, 0]))
    y_min = float(np.min(merged[:, 1]))
    y_max = float(np.max(merged[:, 1]))

    dx = max(1e-6, x_max - x_min)
    dy = max(1e-6, y_max - y_min)
    x_pad = dx * pad_ratio
    y_pad = dy * pad_ratio

    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def filter_road_segments_bbox(
    road_segments: Optional[Dict[str, np.ndarray]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_segments: int,
    seed: int,
) -> Optional[np.ndarray]:
    if road_segments is None:
        return None
    if road_segments.get("lon1") is None or road_segments["lon1"].size == 0:
        return None

    mask = (
        (road_segments["max_lon"] >= x_min)
        & (road_segments["min_lon"] <= x_max)
        & (road_segments["max_lat"] >= y_min)
        & (road_segments["min_lat"] <= y_max)
    )
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None

    if max_segments > 0 and idx.size > max_segments:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_segments, replace=False)

    lon1 = road_segments["lon1"][idx]
    lat1 = road_segments["lat1"][idx]
    lon2 = road_segments["lon2"][idx]
    lat2 = road_segments["lat2"][idx]
    return np.stack([lon1, lat1, lon2, lat2], axis=1).astype(np.float64)


def save_case_overlay_png(
    case: CaseData,
    player_pred_coords: np.ndarray,
    out_path: Path,
    road_segments: Optional[Dict[str, np.ndarray]] = None,
    map_max_segments: int = 5000,
    map_color: str = "#9fa6ad",
    map_alpha: float = 0.25,
    map_linewidth: float = 0.55,
    metrics: Optional[Dict] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_min, x_max, y_min, y_max = compute_case_bbox(case=case, player_pred_coords=player_pred_coords)

    fig, ax = plt.subplots(figsize=(8.2, 8.0))

    if road_segments is not None:
        draw_road_overlay(
            ax=ax,
            road_segments=road_segments,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            color=map_color,
            alpha=map_alpha,
            linewidth=map_linewidth,
            max_segments=map_max_segments,
            label="Road",
        )

    ax.plot(case.gt_coords[:, 0], case.gt_coords[:, 1], color="#1f77b4", linewidth=1.8, label="GT", zorder=3)
    ax.plot(case.pred23_coords[:, 0], case.pred23_coords[:, 1], color="#ff7f0e", linewidth=1.4, linestyle="--", label="Baseline23_e5", zorder=4)
    ax.plot(case.pred28_coords[:, 0], case.pred28_coords[:, 1], color="#2ca02c", linewidth=1.4, linestyle="-.", label="Baseline28", zorder=4)
    ax.plot(player_pred_coords[:, 0], player_pred_coords[:, 1], color="#d81b60", linewidth=1.8, label="Player", zorder=5)
    ax.scatter(
        case.input_coords[case.mask, 0],
        case.input_coords[case.mask, 1],
        s=12,
        color="#111111",
        label="Known",
        zorder=6,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)

    title = f"traj {case.traj_id} | B23 MAE={case.mae_b23:.1f}m | B28 MAE={case.mae_b28:.1f}m"
    if metrics is not None:
        p_mae = metrics.get("player", {}).get("mae", math.nan)
        if np.isfinite(p_mae):
            title = f"traj {case.traj_id} | Player MAE={p_mae:.1f}m | B23={case.mae_b23:.1f}m | B28={case.mae_b28:.1f}m"

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
