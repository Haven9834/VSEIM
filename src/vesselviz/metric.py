"""Water-plane projection, vertical freeboard recovery and passage aggregation."""
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar
from .grounding import setting


def matrix(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all() or np.linalg.matrix_rank(result) < min(shape):
        raise ValueError(f"Supply a finite full-rank {shape[0]} by {shape[1]} camera matrix")
    return result


def water_to_metric(points, homography):
    H = matrix(homography, (3, 3))
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Water-plane points must be finite N by 2 coordinates")
    projected = np.column_stack([points, np.ones(len(points))]) @ H.T
    if np.any(abs(projected[:, 2]) <= np.finfo(float).eps*np.linalg.norm(H)):
        raise ValueError("Water-plane point projects to infinity")
    return projected[:, :2]/projected[:, 2:3]


def recover_vertical(point_xy, upper_uv, projection, depth_m):
    """Minimize image reprojection error along a known world vertical line."""
    P = matrix(projection, (3, 4))
    xy, uv = np.asarray(point_xy, float), np.asarray(upper_uv, float)
    if xy.shape != (2,) or uv.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(uv).all():
        raise ValueError("Invalid vertical-line correspondence")
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("Positive moulded depth is required")
    base = P @ np.array([*xy, 0., 1.])
    vertical = P[:, 2]
    if base[2]*(base[2]+depth_m*vertical[2]) <= 0:
        raise ValueError("Vertical search interval crosses the camera projection plane")
    if np.linalg.norm(vertical[:2]*base[2]-base[:2]*vertical[2]) <= 1e-12*np.linalg.norm(P)**2:
        raise ValueError("Camera geometry cannot resolve height at this location")
    def squared_error(height):
        q = base+height*vertical
        return float(np.sum((q[:2]/q[2]-uv)**2))
    result = minimize_scalar(squared_error, bounds=(0., float(depth_m)), method="bounded")
    if not result.success:
        raise ValueError("Freeboard reprojection fit did not converge")
    return float(result.x), float(np.sqrt(result.fun))


def frame_freeboard(upper, waterline, homography, projection, depth_m, config):
    upper, waterline = np.asarray(upper, float), np.asarray(waterline, float)
    if upper.shape != waterline.shape or upper.ndim != 2 or upper.shape[1] != 2:
        raise ValueError("Upper/waterline correspondence arrays must be aligned N by 2")
    tolerance = setting(config, "reprojection_tolerance_px", minimum=np.finfo(float).eps)
    minimum = setting(config, "min_samples", integer=True, minimum=1)
    multiplier = setting(config, "mad_multiplier", minimum=np.finfo(float).eps)
    floor = setting(config, "freeboard_floor_m", minimum=np.finfo(float).eps)
    xy = water_to_metric(waterline, homography)
    measurements = []
    for point, target in zip(xy, upper):
        try:
            height, residual = recover_vertical(point, target, projection, depth_m)
        except ValueError:
            continue
        if 0 < height < depth_m and residual <= tolerance:
            measurements.append((height, residual))
    if len(measurements) < minimum:
        raise ValueError("Insufficient reprojection-consistent freeboard samples")
    values = np.asarray(measurements)
    center = np.median(values[:, 0])
    scale = max(floor, 1.4826*np.median(abs(values[:, 0]-center)))
    values = values[abs(values[:, 0]-center) <= multiplier*scale]
    if len(values) < minimum:
        raise ValueError("Insufficient geometrically consistent freeboard samples")
    return dict(freeboard_m=float(np.median(values[:, 0])), n_valid_samples=len(values),
                median_reprojection_px=float(np.median(values[:, 1])))


def aggregate_drafts(records, min_valid_frames):
    if type(min_valid_frames) is not int or min_valid_frames < 1:
        raise ValueError("Set a positive min_valid_frames")
    groups = defaultdict(list)
    for row in records:
        if row.get("valid") is True:
            key = tuple(str(row[k]) for k in ("area", "camera", "clip", "passage_id", "track_id"))
            groups[key].append(row)
    passages, issues = [], []
    for key, group in sorted(groups.items()):
        identity = dict(zip(("area", "camera", "clip", "passage_id", "track_id"), key))
        if len(group) < min_valid_frames:
            issues.append(dict(identity, code="insufficient_valid_frames"))
            continue
        if len({str(g["vessel_id"]) for g in group}) != 1:
            issues.append(dict(identity, code="conflicting_vessel_identity"))
            continue
        depths = np.array([g["moulded_depth_m"] for g in group], float)
        if not np.isfinite(depths).all() or np.any(depths <= 0) or not np.allclose(depths, depths[0], rtol=0, atol=1e-9):
            issues.append(dict(identity, code="inconsistent_moulded_depth"))
            continue
        freeboards = np.array([g["freeboard_m"] for g in group], float)
        if not np.isfinite(freeboards).all():
            raise ValueError("Nonfinite valid-frame freeboard")
        freeboard = float(np.median(freeboards))
        draft = float(depths[0]-freeboard)
        if not 0 < draft < depths[0]:
            issues.append(dict(identity, code="nonphysical_draft"))
            continue
        passages.append(dict(identity, vessel_id=str(group[0]["vessel_id"]),
            moulded_depth_m=float(depths[0]), freeboard_m=freeboard, visual_draft_m=draft,
            n_valid_frames=len(group), first_timestamp=min(g["timestamp"] for g in group),
            last_timestamp=max(g["timestamp"] for g in group)))
    return passages, issues
