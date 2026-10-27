"""Text grounding, prompted segmentation and robust hull-boundary extraction."""
import numpy as np
from scipy import ndimage


def setting(config, name, *, integer=False, minimum=0, maximum=None):
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("Configure numeric parameter: " + name)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError("Parameter outside supported range: " + name)
    if integer and type(value) is not int:
        raise ValueError("Parameter must be an integer: " + name)
    return value


def response_map(positive, negative):
    pos, neg = np.asarray(positive, float), np.asarray(negative, float)
    if pos.ndim != 3 or neg.ndim != 3 or pos.shape[0] < 1 or neg.shape[0] < 1 or pos.shape[1:] != neg.shape[1:]:
        raise ValueError("Grounding arrays must be prompt-by-height-by-width")
    if any(not np.isfinite(a).all() or np.any((a < 0) | (a > 1)) for a in (pos, neg)):
        raise ValueError("Grounding responses must lie in [0,1]")
    exclusion = neg.mean(axis=0)
    return pos.mean(axis=0) * (1-exclusion), exclusion


def sample_prompts(hull, exclusion, config):
    positive_threshold = setting(config, "positive_threshold", maximum=1)
    negative_threshold = setting(config, "negative_threshold", maximum=1)
    counts = [setting(config, "positive_points", integer=True, minimum=1),
              setting(config, "negative_points", integer=True, minimum=1)]
    coordinates, labels = [], []
    for array, threshold, count, label in (
        (hull, positive_threshold, counts[0], 1),
        (exclusion, negative_threshold, counts[1], 0),
    ):
        accepted = array > threshold
        if label == 0:
            accepted &= ~(hull > positive_threshold)
        yy, xx = np.where(accepted)
        if not len(xx):
            raise ValueError("No eligible segmentation prompt pixels")
        selected = np.linspace(0, len(xx)-1, min(count, len(xx)), dtype=int)
        coordinates.extend(zip(xx[selected].tolist(), yy[selected].tolist()))
        labels.extend([label]*len(selected))
    return np.asarray(coordinates, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def select_candidate(masks, scores, hull, config):
    weights = [setting(config, key) for key in ("semantic_weight", "continuity_weight")]
    if sum(weights) == 0:
        raise ValueError("At least one candidate-selection weight must be positive")
    min_pixels = setting(config, "min_mask_pixels", integer=True, minimum=1)
    min_score = setting(config, "min_sam_score", maximum=1)
    if len(masks) != len(scores):
        raise ValueError("Candidate masks and scores differ in length")
    candidates = []
    for index, (mask, score) in enumerate(zip(masks, scores)):
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != hull.shape:
            raise ValueError("Segmentation candidates must be aligned binary masks")
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid segmentation confidence")
        size = int(mask.sum())
        if size < min_pixels or score < min_score:
            continue
        labels, _ = ndimage.label(mask)
        component_sizes = np.bincount(labels.ravel())[1:]
        continuity = float(component_sizes.max()/size)
        semantic = float(hull[mask].mean())
        objective = weights[0]*semantic + weights[1]*continuity
        candidates.append((objective, index, dict(semantic=semantic, continuity=continuity, model_score=float(score))))
    if not candidates:
        raise ValueError("No candidate satisfies segmentation quality settings")
    _, index, diagnostics = max(candidates, key=lambda item: (item[0], -item[1]))
    return np.asarray(masks[index], bool), diagnostics


def boundary_pairs(mask, config):
    min_columns = setting(config, "min_columns", integer=True, minimum=2)
    window = setting(config, "median_window", integer=True, minimum=3)
    if window % 2 != 1:
        raise ValueError("median_window must be odd")
    multiplier = setting(config, "mad_multiplier", minimum=np.finfo(float).eps)
    floor = setting(config, "residual_floor_px", minimum=np.finfo(float).eps)
    degree = setting(config, "polynomial_degree", integer=True, maximum=3)
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Expected a two-dimensional binary hull mask")
    x = np.flatnonzero(mask.any(axis=0))
    if len(x) < max(min_columns, degree+2):
        raise ValueError("Insufficient hull-column support")
    upper = np.array([np.flatnonzero(mask[:, column])[0] for column in x], float)
    lower = np.array([np.flatnonzero(mask[:, column])[-1] for column in x], float)
    valid = lower > upper
    # Local median residuals screen spikes before the smooth-boundary fit.
    for boundary in (upper, lower):
        residual = boundary-ndimage.median_filter(boundary, size=window, mode="nearest")
        center = np.median(residual)
        scale = max(floor, 1.4826*np.median(abs(residual-center)))
        valid &= abs(residual-center) <= multiplier*scale
    if valid.sum() < max(min_columns, degree+2):
        raise ValueError("Boundary filtering left insufficient support")
    for boundary in (upper, lower):
        curve = np.polyval(np.polyfit(x[valid], boundary[valid], degree), x)
        residual = boundary-curve
        center = np.median(residual[valid])
        scale = max(floor, 1.4826*np.median(abs(residual[valid]-center)))
        valid &= abs(residual-center) <= multiplier*scale
    if valid.sum() < max(min_columns, degree+2):
        raise ValueError("Boundary consistency criterion failed")
    x, upper, lower = x[valid], upper[valid], lower[valid]
    return np.column_stack([x, upper]), np.column_stack([x, lower])


class CLIPSeg:
    def __init__(self, config):
        if not config.get("model"):
            raise ValueError("Configure a CLIPSeg model identifier or local directory")
        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        self.torch, self.device = torch, config.get("device", "cpu")
        kwargs = {"local_files_only": config.get("local_files_only", True)}
        self.processor = CLIPSegProcessor.from_pretrained(config["model"], **kwargs)
        self.model = CLIPSegForImageSegmentation.from_pretrained(config["model"], **kwargs).to(self.device).eval()

    def responses(self, image, prompts):
        if not prompts:
            raise ValueError("No grounding descriptions supplied")
        inputs = self.processor(text=prompts, images=[image]*len(prompts), padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            if logits.ndim == 2:
                logits = logits.unsqueeze(0)
            logits = self.torch.nn.functional.interpolate(
                logits.unsqueeze(1), size=(image.height, image.width), mode="bilinear", align_corners=False)
            return self.torch.sigmoid(logits[:, 0]).cpu().numpy()


class SAM2:
    def __init__(self, config, checkpoint):
        if not config.get("model_config"):
            raise ValueError("Configure the SAM2 architecture configuration")
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        self.predictor = SAM2ImagePredictor(build_sam2(
            config["model_config"], str(checkpoint), device=config.get("device", "cpu")))

    def candidates(self, image, points, labels):
        with self.torch.inference_mode():
            self.predictor.set_image(np.asarray(image.convert("RGB")))
            masks, scores, _ = self.predictor.predict(
                point_coords=points, point_labels=labels,
                box=np.array([0, 0, image.width-1, image.height-1], dtype=np.float32),
                multimask_output=True)
        return np.asarray(masks, bool), np.asarray(scores, float)
