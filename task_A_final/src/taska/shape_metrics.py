from __future__ import annotations

import numpy as np

from .geo import haversine_meters


def _point_set_to_vertex_set_mean_distance(points: np.ndarray, polyline_vertices: np.ndarray) -> float:
    if points.size == 0 or polyline_vertices.size == 0:
        return 0.0
    lon1 = points[:, 0:1]
    lat1 = points[:, 1:2]
    lon2 = polyline_vertices[None, :, 0]
    lat2 = polyline_vertices[None, :, 1]
    dmat = np.asarray(haversine_meters(lon1, lat1, lon2, lat2), dtype=np.float64)
    return float(np.mean(np.min(dmat, axis=1)))


def shape_symmetric_m(gt_missing_points: np.ndarray, pred_missing_points: np.ndarray, gt_polyline: np.ndarray, pred_polyline: np.ndarray) -> float:
    if gt_missing_points.size == 0 or pred_missing_points.size == 0:
        return 0.0
    d_gt_to_pred = _point_set_to_vertex_set_mean_distance(gt_missing_points, pred_polyline)
    d_pred_to_gt = _point_set_to_vertex_set_mean_distance(pred_missing_points, gt_polyline)
    return float(0.5 * d_gt_to_pred + 0.5 * d_pred_to_gt)
