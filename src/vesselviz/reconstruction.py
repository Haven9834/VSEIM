"""Observation-wise semantic grounding and metric draft reconstruction."""
from pathlib import Path
import numpy as np
from PIL import Image
from .grounding import CLIPSeg, SAM2, response_map, sample_prompts, select_candidate, boundary_pairs, setting
from .semantic import QwenAPI, eligible, vessel_crops
from .metric import aggregate_drafts, frame_freeboard, matrix
from .io import configured_path, create_output, load_json, save_json, safe_key


def run_reconstruction(config_file, output, *, vlm=None, grounder=None, segmenter=None):
    from .cli import write_csv
    config = load_json(config_file)
    base = Path(config_file).resolve().parent
    source = configured_path(config.get("observations_json"), base)
    observations = load_json(source).get("observations", [])
    source_base = source.parent
    if not observations:
        raise ValueError("No vessel observations supplied")
    for camera in config["cameras"].values():
        matrix(camera.get("homography"), (3, 3))
        matrix(camera.get("projection_matrix"), (3, 4))
    minimum = setting(config["geometry"], "min_valid_frames", integer=True, minimum=1)
    if vlm is None:
        prompt = configured_path(config["vlm"].get("prompt_file"), base).read_text(encoding="utf-8")
        vlm = QwenAPI(config["vlm"], prompt)
    if grounder is None:
        grounder = CLIPSeg(config["clipseg"])
    if segmenter is None:
        checkpoint = configured_path(config["sam2"].get("checkpoint"), base)
        segmenter = SAM2(config["sam2"], checkpoint)
    # Resolve all source files before creating outputs or sending API requests.
    paths = [configured_path(row.get("image"), source_base) for row in observations]
    seen = set()
    for row in observations:
        key = tuple(str(row.get(k, "")) for k in ("area", "camera", "clip", "passage_id", "track_id", "frame_id"))
        if not all(key) or key in seen:
            raise ValueError("Missing or duplicate observation identity")
        seen.add(key)
        for value in key[1:]:
            safe_key(value)
        if row.get("upper_reference_matches_moulded_depth") is not True:
            raise ValueError("Confirm compatible upper-boundary and moulded-depth references")
        if not row.get("vessel_id") or row.get("association_status") != "accepted":
            raise ValueError("Metric draft reconstruction requires an accepted vessel association")
    out = create_output(output)
    assets = out / "observations"
    assets.mkdir()
    records, issues = [], []
    for index, (row, path) in enumerate(zip(observations, paths)):
        identity = {k: row[k] for k in ("area", "camera", "clip", "passage_id", "track_id", "frame_id", "vessel_id", "timestamp")}
        sub = assets / f"{index:08d}"
        sub.mkdir()
        try:
            depth = float(row["moulded_depth_m"])
            if not np.isfinite(depth) or depth <= 0:
                raise ValueError("Positive moulded depth is required")
            with Image.open(path) as original:
                image = original.convert("RGB")
            local, context, rectangle = vessel_crops(image, row["bbox_xyxy"], config["context_expansion"])
            semantics = vlm.analyze(local, context)
            save_json(sub / "semantics.json", semantics)
            if not eligible(semantics):
                raise ValueError("Observation is not eligible for metric measurement")
            positive = grounder.responses(local, semantics["positive_prompts"])
            negative = grounder.responses(local, semantics["negative_prompts"])
            hull, exclusion = response_map(positive, negative)
            coordinates, labels = sample_prompts(hull, exclusion, config["grounding"])
            masks, scores = segmenter.candidates(local, coordinates, labels)
            mask, diagnostics = select_candidate(masks, scores, hull, config["grounding"])
            upper, water = boundary_pairs(mask, config["boundary"])
            offset = np.array(rectangle[:2])
            camera = config["cameras"][row["camera"]]
            result = frame_freeboard(upper+offset, water+offset, camera["homography"],
                                     camera["projection_matrix"], depth, config["geometry"])
            full_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            x1, y1, x2, y2 = rectangle
            full_mask[y1:y2, x1:x2] = mask.astype(np.uint8)*255
            full_response = np.zeros((image.height, image.width), dtype=np.float32)
            full_response[y1:y2, x1:x2] = hull
            Image.fromarray(full_mask).save(sub / "hull_mask.png")
            np.save(sub / "hull_response.npy", full_response, allow_pickle=False)
            save_json(sub / "boundaries.json", dict(upper=(upper+offset).tolist(), waterline=(water+offset).tolist(),
                diagnostics=diagnostics, coordinate_space="original_image", **identity))
            records.append(dict(identity, valid=True, moulded_depth_m=depth, **result))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            # Request/geometry failures remain visible; no replacement measurement.
            issues.append(dict(identity, code="excluded_observation", reason=str(exc)))
    passages, aggregation_issues = aggregate_drafts(records, minimum)
    issues.extend(aggregation_issues)
    write_csv(out / "frame_freeboard.csv", records)
    write_csv(out / "passage_drafts.csv", passages)
    save_json(out / "issues.json", issues)
    save_json(out / "run.json", dict(n_observations=len(observations), n_valid_frames=len(records),
        n_valid_passages=len(passages), n_issues=len(issues)))
    return out
