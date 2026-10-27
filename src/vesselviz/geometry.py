"""Explicit pixel-coordinate transforms; no camera calibration is inferred."""
import math


def crop_values(values, size):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("Crop must be [top, bottom, left, right]")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError("Crop values must be nonnegative integer pixels, not null")
    t, b, l, r = values
    if l + r >= size[0] or t + b >= size[1]:
        raise ValueError("Crop removes the entire image")
    return t, b, l, r


def crop_rectangle(size, crop):
    t, b, l, r = crop_values(crop, size)
    return l, t, size[0] - r, size[1] - b


def valid_box(box):
    return (len(box) == 4 and all(isinstance(v, (float, int)) and math.isfinite(v) for v in box)
            and box[2] > box[0] and box[3] > box[1])


def restore_box(box, original_size, transform):
    if not valid_box(box):
        raise ValueError("Invalid xyxy bounding box")
    space = transform.get("space")
    if space == "original":
        return list(box)
    if space != "crop_resize":
        raise ValueError("Coordinate space must explicitly be original or crop_resize")
    t, b, l, r = crop_values(transform.get("crop"), original_size)
    resized = transform.get("resized_size")
    if not isinstance(resized, list) or len(resized) != 2 or any(not isinstance(v, int) or v <= 0 for v in resized):
        raise ValueError("resized_size must be [width, height] in source detection coordinates")
    sx, sy = (original_size[0] - l - r) / resized[0], (original_size[1] - t - b) / resized[1]
    return [box[0] * sx + l, box[1] * sy + t, box[2] * sx + l, box[3] * sy + t]


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def clip_box(box, rectangle):
    clipped = [max(box[0], rectangle[0]), max(box[1], rectangle[1]),
               min(box[2], rectangle[2]), min(box[3], rectangle[3])]
    return clipped if valid_box(clipped) else None
