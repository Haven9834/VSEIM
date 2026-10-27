"""Explicit joins between tracked observations, specifications and draft results."""
from collections import defaultdict
import math


def unique_index(rows, keys):
    result = {}
    for row in rows:
        key = tuple(str(row.get(k, "")).strip() for k in keys)
        if not all(key) or key in result:
            raise ValueError("Missing or duplicate join identity: " + ",".join(keys))
        result[key] = row
    return result


def attach_identities(observations, matches, specifications):
    keys = ("area", "camera", "clip", "passage_id", "track_id")
    matches = unique_index(matches, keys)
    specifications = unique_index(specifications, ("MMSI",))
    output, issues = [], []
    for row in observations:
        key = tuple(str(row.get(k, "")).strip() for k in keys)
        match = matches.get(key, {})
        vessel = str(match.get("predicted_MMSI", ""))
        spec = specifications.get((vessel,), {})
        if match.get("status") != "accepted":
            issues.append(dict(zip(keys, key), code="no_accepted_track_identity"))
            continue
        try:
            depth = float(spec["moulded_depth_m"])
            if not math.isfinite(depth) or depth <= 0:
                raise ValueError("Invalid moulded depth")
        except (KeyError, ValueError, TypeError):
            issues.append(dict(zip(keys, key), code="missing_valid_moulded_depth"))
            continue
        if row.get("vessel_id") and str(row["vessel_id"]) != vessel:
            raise ValueError("Observation identity conflicts with accepted association")
        output.append(dict(row, vessel_id=vessel, association_status="accepted", moulded_depth_m=depth))
    return output, issues


def attach_drafts(activity, drafts):
    keys = ("area", "clip", "passage_id", "vessel_id")
    drafts = unique_index(drafts, keys)
    output, issues = [], []
    for line, row in enumerate(activity, 2):
        key = tuple(str(row.get(k, "")).strip() for k in keys)
        draft = drafts.get(key)
        if draft is None:
            issues.append(dict(source_row=line, code="missing_passage_draft"))
            continue
        value = float(draft["visual_draft_m"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid passage-level visual draft")
        if str(row.get("visual_draft_m", "")).strip() and not math.isclose(float(row["visual_draft_m"]), value, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Activity input already contains a conflicting visual draft")
        output.append(dict(row, visual_draft_m=value))
    return output, issues


def inventory_tables(passages):
    totals, by_type = defaultdict(float), defaultdict(float)
    for row in passages:
        value = float(row["emission_kg"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid passage emission")
        for area in (row["area"], "Overall"):
            if row["area"] == "Overall":
                raise ValueError("Input area must identify an actual monitoring area")
            totals[(area, row["pollutant"], row["scenario"])] += value
            by_type[(area, row["pollutant"], row["scenario"], row["vessel_type"])] += value
    comparison, contribution = [], []
    for area, p in sorted({(a, p) for a, p, _ in totals}):
        if (area, p, "SEIM") not in totals or (area, p, "VSEIM") not in totals:
            raise ValueError("Both emission scenarios are required")
        first, second = totals[(area, p, "SEIM")], totals[(area, p, "VSEIM")]
        comparison.append(dict(area=area, pollutant=p, SEIM_kg=first, VSEIM_kg=second,
            difference_kg=first-second, relative_difference_pct=(first-second)/first*100 if first else ""))
    for (area, p, scenario, kind), value in sorted(by_type.items()):
        total = totals[(area, p, scenario)]
        contribution.append(dict(area=area, pollutant=p, scenario=scenario, vessel_type=kind,
            emission_kg=value, contribution_pct=value/total*100 if total else ""))
    return comparison, contribution
