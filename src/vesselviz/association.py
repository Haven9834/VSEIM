"""Read existing track-to-AIS matches without performing new association."""
from collections import defaultdict
from PIL import Image
from .geometry import restore_box, crop_rectangle, clip_box, intersection
from .io import csv_rows, identifier, number, configured_path, confined_path, safe_key
from .privacy import mask_mmsi, short_name

FIELDS = {
    "image": "image_name", "track": "track_id", "mmsi": "ais_mmsi",
    "x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2", "speed": "ais_speed",
    "name": "ais_english_name", "status": "match_status", "confidence": "track_match_confidence",
    "within_gate": "within_adaptive_gate",
}


def collect_camera(config, base):
    camera, clip = safe_key(config["camera"]), safe_key(config["clip"])
    table = configured_path(config.get("matches_csv"), base)
    mapping = configured_path(config.get("mapping_csv"), base)
    images = configured_path(config.get("images_dir"), base, kind="dir")
    fields = dict(FIELDS, **config.get("fields", {}))
    mapping_fields = {"track": "track_id", "mmsi": "mmsi", **config.get("mapping_fields", {})}
    accepted = config.get("accepted_statuses")
    if not isinstance(accepted, list) or not accepted or any(not isinstance(s, str) for s in accepted):
        raise ValueError("accepted_statuses must explicitly list the statuses to display")
    if config.get("speed_unit") != "kn":
        raise ValueError("This adapter expects speed_unit='kn'; convert other source units explicitly")
    tracks = defaultdict(set)
    for _, row in csv_rows(mapping, mapping_fields.values()):
        tracks[identifier(row[mapping_fields["track"]])].add(identifier(row[mapping_fields["mmsi"]]))
    issues, grouped = [], defaultdict(list)

    def issue(code, frame, track, line, detail=""):
        issues.append(dict(camera=camera, clip=clip, frame=frame, track_id=track,
                           source_row=line, code=code, detail=detail))

    for line, row in csv_rows(table, [fields[k] for k in ("image", "track", "mmsi", "x1", "y1", "x2", "y2", "status")]):
        frame, track, mmsi = row[fields["image"]], identifier(row[fields["track"]]), identifier(row[fields["mmsi"]])
        status = row[fields["status"]]
        if status not in accepted:
            issue("unaccepted_status", frame, track, line); continue
        if not track:
            issue("missing_track", frame, track, line); continue
        try:
            masked = mask_mmsi(mmsi)
        except ValueError:
            issue("invalid_mmsi", frame, track, line); continue
        if tracks.get(track) != {mmsi}:
            issue("mapping_conflict", frame, track, line); continue
        speed = number(row.get(fields["speed"]))
        if status == "TRACK_MATCHED_AIS_MISSING_AT_FRAME":
            speed = None
        if speed is None or not 0 <= speed < 102.3:
            speed = None; issue("missing_speed", frame, track, line)
        name = short_name(row.get(fields["name"]), config.get("name_prefix", 2))
        if name == "N/A": issue("missing_name", frame, track, line)
        confidence = row.get(fields["confidence"], "")
        if confidence.lower() != "high": issue("non_high_confidence_kept", frame, track, line)
        if row.get(fields["within_gate"], "").lower() not in ("true", "1"):
            issue("outside_or_unknown_gate_kept", frame, track, line)
        grouped[frame].append(dict(track_id=track, mmsi_masked=masked, name_masked=name,
            speed_text=f"{speed:.1f} kn" if speed is not None else "N/A",
            bbox=[number(row[fields[k]]) for k in ("x1", "y1", "x2", "y2")],
            _mmsi=mmsi, source_row=line, status=status))

    output = []
    for frame, ships in sorted(grouped.items()):
        try:
            image_path = confined_path(images, frame)
            with Image.open(image_path) as image:
                size = image.size
        except (OSError, ValueError):
            issue("missing_or_invalid_image", frame, "", ""); continue
        if list(size) != config.get("original_size"):
            raise ValueError(f"{camera}/{clip}: image dimensions do not match original_size")
        rectangle = crop_rectangle(size, config.get("display_crop"))
        restore_box([0, 0, 1, 1], size, config.get("coordinates", {}))
        seen = defaultdict(list)
        for ship in ships: seen[ship["track_id"]].append(ship)
        kept = []
        for track, repeated in seen.items():
            # Ambiguous records are excluded as a group, not resolved by CSV order.
            if len(repeated) != 1:
                for ship in repeated: issue("duplicate_track_excluded", frame, track, ship["source_row"])
                continue
            ship = repeated[0]
            try:
                box = restore_box(ship["bbox"], size, config.get("coordinates", {}))
                clipped = clip_box(box, rectangle)
            except (ValueError, TypeError):
                issue("invalid_box_or_transform", frame, track, ship["source_row"]); continue
            if clipped is None:
                issue("box_outside_crop", frame, track, ship["source_row"]); continue
            if clipped != box: issue("box_clipped", frame, track, ship["source_row"])
            ship["bbox"] = clipped
            kept.append(ship)
        for i, a in enumerate(kept):
            for b in kept[i + 1:]:
                if a["_mmsi"] == b["_mmsi"]:
                    issue("same_mmsi_multiple_tracks_kept", frame, a["track_id"], a["source_row"], "Other track: " + b["track_id"])
                area = lambda q: (q[2] - q[0]) * (q[3] - q[1])
                if intersection(a["bbox"], b["bbox"]) / min(area(a["bbox"]), area(b["bbox"])) > .65:
                    issue("overlapping_boxes_kept", frame, a["track_id"], a["source_row"])
        for ship in kept: del ship["_mmsi"]
        kept.sort(key=lambda s: (s["bbox"][0] + s["bbox"][2]) / 2)
        if kept:
            output.append(dict(camera=camera, clip=clip, frame=frame, image_path=str(image_path),
                               size=list(size), display_crop=config["display_crop"], ships=kept))
    return output, issues
