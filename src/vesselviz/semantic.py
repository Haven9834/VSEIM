"""Measurement-oriented semantics obtained from a configured multimodal API."""
import base64
from io import BytesIO
import json
import math
import os
from urllib import error, parse, request


SCHEMA = {
    "vessel_type": "string", "viewpoint": "side|oblique|bow|stern|unknown",
    "hull_configuration": "string", "hull_visible": "boolean",
    "upper_visible": "boolean", "waterline_visible": "boolean",
    "occlusion": "none|partial|severe|unknown", "eligible": "boolean",
    "positive_prompts": ["string"], "negative_prompts": ["string"],
}


def parse_semantics(value):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if len(lines) < 3 or lines[-1].strip() != "```":
                raise ValueError("Incomplete semantic JSON response")
            value = "\n".join(lines[1:-1])
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Semantic output must be a JSON object")
    result = {}
    for field in ("vessel_type", "hull_configuration"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError("Missing semantic text: " + field)
        result[field] = value[field].strip()
    for field in ("hull_visible", "upper_visible", "waterline_visible", "eligible"):
        if type(value.get(field)) is not bool:
            raise ValueError("Semantic field must be a JSON boolean: " + field)
        result[field] = value[field]
    for field, choices in (
        ("viewpoint", {"side", "oblique", "bow", "stern", "unknown"}),
        ("occlusion", {"none", "partial", "severe", "unknown"}),
    ):
        if value.get(field) not in choices:
            raise ValueError("Unrecognized semantic category: " + field)
        result[field] = value[field]
    for field in ("positive_prompts", "negative_prompts"):
        prompts = value.get(field)
        if not isinstance(prompts, list) or not prompts or any(
            not isinstance(p, str) or not p.strip() for p in prompts
        ):
            raise ValueError("Supply nonempty semantic descriptions: " + field)
        result[field] = [p.strip() for p in prompts]
    return result


def eligible(semantics):
    s = parse_semantics(semantics)
    return (s["eligible"] and s["hull_visible"] and s["upper_visible"]
            and s["waterline_visible"] and s["viewpoint"] in {"side", "oblique"}
            and s["occlusion"] in {"none", "partial"})


def vessel_crops(image, bbox, expansion):
    if not isinstance(expansion, (int, float)) or not math.isfinite(expansion) or expansion < 1:
        raise ValueError("Set context_expansion >= 1")
    if len(bbox) != 4 or any(not math.isfinite(float(x)) for x in bbox):
        raise ValueError("Invalid vessel bounding box")
    x1, y1, x2, y2 = map(float, bbox)
    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
        raise ValueError("Bounding box must lie inside the original image")
    rectangle = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
    cx, cy = (x1+x2)/2, (y1+y2)/2
    half_w, half_h = (x2-x1)*expansion/2, (y2-y1)*expansion/2
    context = (max(0, math.floor(cx-half_w)), max(0, math.floor(cy-half_h)),
               min(image.width, math.ceil(cx+half_w)), min(image.height, math.ceil(cy+half_h)))
    return image.crop(rectangle), image.crop(context), rectangle


def _image_part(image):
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}


class QwenAPI:
    """Chat-completions-compatible transport; endpoint and model are user supplied."""

    def __init__(self, settings, prompt):
        self.settings = settings
        endpoint = settings.get("endpoint", "")
        parsed = parse.urlsplit(endpoint)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.username:
            raise ValueError("Configure the complete multimodal API endpoint")
        if not settings.get("model") or not prompt.strip():
            raise ValueError("Configure the model identifier and measurement prompt")
        for field in ("temperature", "max_tokens", "timeout_s"):
            v = settings.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError("Configure API parameter: " + field)
        if settings["temperature"] < 0 or settings["max_tokens"] <= 0 or settings["timeout_s"] <= 0:
            raise ValueError("Invalid API generation or timeout setting")
        if type(settings["max_tokens"]) is not int:
            raise ValueError("max_tokens must be an integer")
        env_name = settings.get("api_key_env", "")
        self.key = os.environ.get(env_name, "") if env_name else ""
        if env_name and not self.key:
            raise ValueError("The configured API credential environment variable is empty")
        self.prompt = prompt

    def analyze(self, local_image, context_image):
        content = [{"type": "text", "text": "Local vessel crop, followed by contextual crop."},
                   _image_part(local_image), _image_part(context_image)]
        payload = {
            "model": self.settings["model"],
            "messages": [
                {"role": "system", "content": self.prompt + "\nReturn one JSON object with this schema: " + json.dumps(SCHEMA)},
                {"role": "user", "content": content},
            ],
            "temperature": self.settings["temperature"],
            "max_tokens": self.settings["max_tokens"],
        }
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = request.Request(self.settings["endpoint"], data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.settings["timeout_s"]) as response:
                result = json.load(response)
        except error.HTTPError as exc:
            raise ValueError(f"Multimodal API returned HTTP {exc.code}") from None
        except error.URLError:
            raise ValueError("Multimodal API connection failed") from None
        try:
            choice = result["choices"][0]
            if choice.get("finish_reason") not in (None, "stop"):
                raise ValueError("Multimodal response was incomplete")
            return parse_semantics(choice["message"]["content"])
        except (KeyError, IndexError, TypeError):
            raise ValueError("Unexpected multimodal API response format") from None
