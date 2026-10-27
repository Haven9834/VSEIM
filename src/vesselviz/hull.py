"""Five full-frame views from saved, exact-frame segmentation outputs."""
import numpy as np
from PIL import Image, ImageDraw
from .io import configured_path, load_json, safe_key, sha256
from .geometry import crop_rectangle

PANEL_NAMES = ("01_original", "02_vessel_mask", "03_clipseg_heatmap", "04_boundaries", "05_final_hull")


def polygon_mask(points, size):
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
        raise ValueError("Polygon requires at least three finite xy points")
    if (xy < 0).any() or (xy[:, 0] > size[0]).any() or (xy[:, 1] > size[1]).any():
        raise ValueError("Polygon is outside the original-image coordinate system")
    area2 = abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))
    if area2 <= 1: raise ValueError("Degenerate hull polygon")
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in xy], fill=255)
    return np.asarray(mask) > 0


def dashed(draw, points, color, width, dash, gap):
    progress = 0.0
    for start, end in zip(points[:-1], points[1:]):
        p, q = np.array(start, float), np.array(end, float)
        length = float(np.linalg.norm(q - p)); pos = 0.0
        while pos < length:
            phase = progress % (dash + gap); on = phase < dash
            step = min(length - pos, (dash if on else dash + gap) - phase)
            if step < 1e-7: step = min(length - pos, 1e-5)
            if on: draw.line([tuple(p + (q - p) * pos / length), tuple(p + (q - p) * (pos + step) / length)], fill=color, width=width)
            pos += step; progress += step


def hull_panels(frame, base):
    for key in ("camera", "clip", "frame_id"): safe_key(frame[key])
    image_path = configured_path(frame.get("image"), base)
    mask_path = configured_path(frame.get("vessel_mask"), base)
    probability_path = configured_path(frame.get("probability"), base)
    hull_path = configured_path(frame.get("hulls"), base)
    with Image.open(image_path) as source: original = source.convert("RGB")
    if list(original.size) != frame.get("original_size"):
        raise ValueError("Source image dimensions differ from original_size")
    expected_hash = frame.get("image_sha256", "")
    if expected_hash and sha256(image_path) != expected_hash:
        raise ValueError("Source image hash differs from the manifest")
    with Image.open(mask_path) as mask:
        if mask.size != original.size or mask.mode not in ("1", "L"):
            raise ValueError("Vessel mask must be a same-size, single-channel binary image")
        mask_array = np.asarray(mask.convert("L"))
    if not set(np.unique(mask_array)).issubset({0, 255}):
        raise ValueError("Vessel mask is not binary 0/255")
    probability = np.load(probability_path, allow_pickle=False)
    if probability.shape != (original.height, original.width) or not np.issubdtype(probability.dtype, np.number):
        raise ValueError("Probability must be a numeric H x W array in original coordinates")
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("Probability values must be finite and within [0,1]")
    data = load_json(hull_path)
    if not isinstance(data, dict) or not isinstance(data.get("hull_instances"), list):
        raise ValueError("Hull JSON must contain hull_instances; consult the data schema")
    for key in ("camera", "clip", "frame_id"):
        if key in data and data[key] != frame[key]: raise ValueError("Hull file frame identity mismatch")
    if "original_size" in data and data["original_size"] != list(original.size):
        raise ValueError("Hull file original_size mismatch")
    from matplotlib import colormaps
    lo, hi = float(probability.min()), float(probability.max())
    display = (probability - lo) / max(hi - lo, 1e-9)
    heat = colormaps["jet"](display)[:, :, :3] * 255
    rgb = np.array(original)
    thermal = Image.fromarray((.78 * heat + .22 * rgb).clip(0, 255).astype("uint8"))
    boundaries = original.copy(); draw = ImageDraw.Draw(boundaries)
    width, dash = max(2, round(original.width / 800)), max(8, round(original.width / 160))
    union = np.zeros((original.height, original.width), bool); curves = []
    ids = set()
    for item in data["hull_instances"]:
        hull_id = str(item["id"])
        if hull_id in ids: raise ValueError("Duplicate hull instance ID")
        ids.add(hull_id)
        mask = polygon_mask(item["polygon"], original.size); union |= mask
        upper, lower = [], []
        for x in np.flatnonzero(mask.any(0)):
            ys = np.flatnonzero(mask[:, x]); upper.append([int(x), int(ys[0])]); lower.append([int(x), int(ys[-1])])
        dashed(draw, upper, (0, 230, 110), width, dash, dash * .6)
        dashed(draw, lower, (255, 35, 45), width, dash, dash * .6)
        curves.append(dict(hull_id=hull_id, upper=upper, lower=lower))
    overlay = rgb.copy()
    overlay[union] = (.22 * overlay[union] + .78 * np.array([66, 151, 255])).astype("uint8")
    assert np.array_equal(overlay[~union], rgb[~union])
    panels = [original, Image.fromarray(mask_array).convert("RGB"), thermal, boundaries, Image.fromarray(overlay)]
    rectangle = crop_rectangle(original.size, frame.get("display_crop"))
    panels = [im.crop(rectangle) for im in panels]
    warnings = [] if ids else ["No hull polygons; boundary and hull overlay panels remain unmarked"]
    if not expected_hash:
        warnings.append("No expected image hash supplied; exact-frame provenance requires external verification")
    missing_identity = [k for k in ("camera", "clip", "frame_id", "original_size") if k not in data]
    if missing_identity:
        warnings.append("Hull identity metadata missing: " + ", ".join(missing_identity))
    report = dict(camera=frame["camera"], clip=frame["clip"], frame_id=frame["frame_id"],
                  source_sha256=sha256(image_path), hull_count=len(ids), probability_range=[lo, hi],
                  crop_xyxy=list(rectangle), curves=curves,
                  warnings=warnings)
    return panels, report
