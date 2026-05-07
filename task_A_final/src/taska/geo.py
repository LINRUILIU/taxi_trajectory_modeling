from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def haversine_meters(lon1, lat1, lon2, lat2):
    r = 6371000.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def point_distance_m(a: np.ndarray, b: np.ndarray) -> float:
    return float(haversine_meters(float(a[0]), float(a[1]), float(b[0]), float(b[1])))


def polyline_lengths_m(polyline: np.ndarray) -> np.ndarray:
    if polyline.shape[0] < 2:
        return np.zeros(1, dtype=np.float64)
    seg = haversine_meters(
        polyline[:-1, 0],
        polyline[:-1, 1],
        polyline[1:, 0],
        polyline[1:, 1],
    ).astype(np.float64)
    return np.concatenate([[0.0], np.cumsum(seg)])


def sample_polyline_at_s(polyline: np.ndarray, s: np.ndarray) -> np.ndarray:
    if polyline.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if polyline.shape[0] == 1:
        return np.repeat(polyline[:1], repeats=len(s), axis=0)
    cum = polyline_lengths_m(polyline)
    total = float(cum[-1])
    if total <= 1e-9:
        return np.repeat(polyline[:1], repeats=len(s), axis=0)
    clipped = np.clip(np.asarray(s, dtype=np.float64), 0.0, total)
    lon = np.interp(clipped, cum, polyline[:, 0])
    lat = np.interp(clipped, cum, polyline[:, 1])
    return np.stack([lon, lat], axis=1)


@dataclass
class ProjectionResult:
    point: np.ndarray
    s: float
    dist_m: float


def _segment_projection(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    lat_ref = float(point[1])
    cos_lat = max(0.2, math.cos(math.radians(lat_ref)))
    meter_per_lon = 111320.0 * cos_lat
    meter_per_lat = 111320.0

    ax = (float(a[0]) - float(point[0])) * meter_per_lon
    ay = (float(a[1]) - float(point[1])) * meter_per_lat
    bx = (float(b[0]) - float(point[0])) * meter_per_lon
    by = (float(b[1]) - float(point[1])) * meter_per_lat

    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        t = 0.0
    else:
        t = -(ax * dx + ay * dy) / denom
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * (b - a)
    return proj.astype(np.float64), t


def project_point_to_polyline(point: np.ndarray, polyline: np.ndarray) -> ProjectionResult:
    if polyline.shape[0] == 0:
        raise ValueError("Cannot project onto empty polyline")
    if polyline.shape[0] == 1:
        return ProjectionResult(point=polyline[0].astype(np.float64), s=0.0, dist_m=point_distance_m(point, polyline[0]))

    cum = polyline_lengths_m(polyline)
    best_dist = float("inf")
    best_proj = polyline[0].astype(np.float64)
    best_s = 0.0
    for i in range(polyline.shape[0] - 1):
        proj, t = _segment_projection(point, polyline[i], polyline[i + 1])
        dist = point_distance_m(point, proj)
        s = float(cum[i] + t * max(0.0, cum[i + 1] - cum[i]))
        if dist < best_dist:
            best_dist = dist
            best_proj = proj
            best_s = s
    return ProjectionResult(point=best_proj, s=best_s, dist_m=best_dist)


def project_points_to_polyline_monotonic(points: np.ndarray, polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    raw_s = []
    raw_dist = []
    for pt in points:
        res = project_point_to_polyline(pt, polyline)
        raw_s.append(res.s)
        raw_dist.append(res.dist_m)
    s_arr = np.asarray(raw_s, dtype=np.float64)
    mono_s = np.maximum.accumulate(s_arr)
    proj_pts = sample_polyline_at_s(polyline, mono_s)
    clamp_delta = np.maximum(0.0, mono_s - s_arr)
    stats = {
        "projection_raw_backtracks": int(np.sum(np.diff(s_arr) < -1e-6)),
        "projection_mean_dist_m": float(np.mean(raw_dist)) if raw_dist else 0.0,
        "projection_max_dist_m": float(np.max(raw_dist)) if raw_dist else 0.0,
        "projection_p95_dist_m": float(np.percentile(raw_dist, 95)) if raw_dist else 0.0,
        "projection_clamped_mean_m": float(np.mean(clamp_delta)) if clamp_delta.size else 0.0,
        "projection_clamped_max_m": float(np.max(clamp_delta)) if clamp_delta.size else 0.0,
    }
    return proj_pts, mono_s, stats


def point_to_polyline_distance_m(point: np.ndarray, polyline: np.ndarray) -> float:
    return float(project_point_to_polyline(point, polyline).dist_m)
