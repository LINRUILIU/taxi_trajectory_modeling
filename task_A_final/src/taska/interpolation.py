from __future__ import annotations

import numpy as np


def _safe_interpolate_1d(x_all: np.ndarray, x_known: np.ndarray, y_known: np.ndarray) -> np.ndarray:
    if len(x_known) == 0:
        return np.full_like(x_all, np.nan, dtype=np.float64)
    if len(x_known) == 1:
        return np.full_like(x_all, y_known[0], dtype=np.float64)
    return np.interp(x_all, x_known, y_known)


def _pchip_endpoint_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    m = ((2.0 * h0 + h1) * d0 - h0 * d1) / max(1e-12, (h0 + h1))
    if m * d0 <= 0.0:
        return 0.0
    if d0 * d1 < 0.0 and abs(m) > abs(3.0 * d0):
        return 3.0 * d0
    return m


def _safe_pchip_interpolate_1d(x_all: np.ndarray, x_known: np.ndarray, y_known: np.ndarray) -> np.ndarray:
    n = len(x_known)
    if n == 0:
        return np.full_like(x_all, np.nan, dtype=np.float64)
    if n == 1:
        return np.full_like(x_all, y_known[0], dtype=np.float64)
    if n == 2:
        return np.interp(x_all, x_known, y_known)

    h = np.diff(x_known).astype(np.float64)
    if np.any(h <= 0.0):
        return np.interp(x_all, x_known, y_known)

    delta = np.diff(y_known).astype(np.float64) / h
    m = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        d_im1 = delta[i - 1]
        d_i = delta[i]
        if d_im1 == 0.0 or d_i == 0.0 or d_im1 * d_i < 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            den = (w1 / d_im1 + w2 / d_i)
            m[i] = 0.0 if abs(den) < 1e-12 else (w1 + w2) / den
    m[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
    m[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])

    xq = np.asarray(x_all, dtype=np.float64)
    yq = np.empty_like(xq, dtype=np.float64)
    left_mask = xq <= x_known[0]
    right_mask = xq >= x_known[-1]
    yq[left_mask] = y_known[0]
    yq[right_mask] = y_known[-1]
    mid_mask = ~(left_mask | right_mask)
    if np.any(mid_mask):
        xm = xq[mid_mask]
        idx = np.searchsorted(x_known, xm, side="right") - 1
        idx = np.clip(idx, 0, n - 2)
        x0 = x_known[idx]
        x1 = x_known[idx + 1]
        y0 = y_known[idx]
        y1 = y_known[idx + 1]
        m0 = m[idx]
        m1 = m[idx + 1]
        hi = x1 - x0
        t = (xm - x0) / hi
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        yq[mid_mask] = h00 * y0 + h10 * hi * m0 + h01 * y1 + h11 * hi * m1
    return yq


def interpolate_traj(timestamps: np.ndarray, coords: np.ndarray, mask: np.ndarray, mode: str = "pchip") -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    known_idx = np.where(mask)[0]
    pred = coords.copy()
    if known_idx.size == 0:
        return pred.astype(np.float32)

    t_known = timestamps[known_idx]
    x_all = timestamps
    if mode == "linear":
        lon = _safe_interpolate_1d(x_all, t_known, coords[known_idx, 0])
        lat = _safe_interpolate_1d(x_all, t_known, coords[known_idx, 1])
    elif mode == "pchip":
        lon = _safe_pchip_interpolate_1d(x_all, t_known, coords[known_idx, 0])
        lat = _safe_pchip_interpolate_1d(x_all, t_known, coords[known_idx, 1])
    else:
        raise ValueError(f"Unsupported interpolate mode: {mode}")
    pred[:, 0] = lon
    pred[:, 1] = lat
    pred[mask] = coords[mask]
    return pred.astype(np.float32)
