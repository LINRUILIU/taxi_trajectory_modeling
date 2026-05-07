from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .geo import polyline_lengths_m, project_points_to_polyline_monotonic, sample_polyline_at_s


def route_projection_fill(base_seg: np.ndarray, route_polyline: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    projected, mono_s, stats = project_points_to_polyline_monotonic(base_seg, route_polyline)
    if projected.shape != base_seg.shape or mono_s.size != base_seg.shape[0]:
        raise ValueError("Route projection output shape mismatch")
    stats["mode"] = "route_projection"
    return projected.astype(np.float32), stats


def route_s_fill(base_seg: np.ndarray, route_polyline: np.ndarray, beta: float) -> Tuple[np.ndarray, Dict[str, float]]:
    _, base_s, stats = project_points_to_polyline_monotonic(base_seg, route_polyline)
    cum = polyline_lengths_m(route_polyline)
    total = float(cum[-1]) if cum.size else 0.0
    if total <= 1e-9:
        raise ValueError("Route polyline length is too small")
    if base_seg.shape[0] == 1:
        uniform_s = np.array([0.5 * total], dtype=np.float64)
    else:
        uniform_s = np.linspace(total / (base_seg.shape[0] + 1), total * base_seg.shape[0] / (base_seg.shape[0] + 1), base_seg.shape[0])
    beta_val = float(np.clip(beta, 0.0, 1.0))
    final_s = beta_val * base_s + (1.0 - beta_val) * uniform_s
    final_s = np.maximum.accumulate(np.clip(final_s, 0.0, total))
    points = sample_polyline_at_s(route_polyline, final_s)
    stats["mode"] = "route_s"
    stats["route_s_beta"] = beta_val
    return points.astype(np.float32), stats
