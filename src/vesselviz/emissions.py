"""Main-engine interval emissions with explicit, externally supplied parameters.

No emission factors, draft measurements or low-load tables are inferred here.
All scenarios use the same interval, power, speed and emission factors.
"""
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
from .io import number

POLLUTANTS = ("CO2", "CO", "VOCs", "NOx", "PM10", "SO2")


def validate_settings(settings):
    for key in ("draft_exponent", "speed_exponent", "load_cap", "low_load_threshold"):
        v = number(settings.get(key))
        if isinstance(settings.get(key), (str, bool)) or v is None or v <= 0:
            raise ValueError(f"Set a positive numeric model parameter: {key}")
    if settings["load_cap"] > 1 or settings["low_load_threshold"] > settings["load_cap"]:
        raise ValueError("Load threshold/cap must satisfy 0 < threshold <= cap <= 1")
    points = np.asarray(settings.get("load_points"), dtype=float)
    if points.ndim != 1 or len(points) < 2 or not np.isfinite(points).all() or points[0] <= 0 or np.any(np.diff(points) <= 0):
        raise ValueError("load_points must be a positive strictly increasing lookup grid")
    if not np.isclose(points[-1], settings["low_load_threshold"]):
        raise ValueError("Last lookup load must equal the low-load threshold")
    for p in POLLUTANTS:
        factors = np.asarray(settings.get("low_load_factors", {}).get(p), dtype=float)
        if factors.shape != points.shape or not np.isfinite(factors).all() or np.any(factors <= 0):
            raise ValueError(f"Supply finite positive low-load factors for {p}")
    if not str(settings.get("parameter_source", "")).strip():
        raise ValueError("Document parameter_source before calculating emissions")


def low_load_factor(load, pollutant, settings):
    load = np.asarray(load, dtype=float)
    out = np.ones_like(load)
    low = (load > 0) & (load < settings["low_load_threshold"])
    points = np.asarray(settings["load_points"], float)
    out[low] = np.interp(np.clip(load[low], points[0], points[-1]), points,
                         settings["low_load_factors"][pollutant])
    out[~np.isfinite(load)] = np.nan
    return out


def calculate(power_kw, speed_kn, design_speed_kn, draft_m, design_draft_m,
              dt_s, factors_g_kwh, settings):
    """Vectorized equations on common-valid intervals.

    design_draft_m is the legacy argument name for the SEIM draft denominator;
    the canonical CSV input is seim_draft_m, which may be AIS-based.
    """
    validate_settings(settings)
    values = [np.asarray(v, dtype=float) for v in
              (power_kw, speed_kn, design_speed_kn, draft_m, design_draft_m, dt_s)]
    values = np.broadcast_arrays(*values)
    power, speed, design_speed, draft, design_draft, dt = values
    if any(not np.isfinite(v).all() for v in values): raise ValueError("Nonfinite model input")
    if np.any(speed < 0) or any(np.any(v <= 0) for v in (power, design_speed, draft, design_draft, dt)):
        raise ValueError("Positive power, design speed, drafts and duration are required; speed may be zero")
    raw = (speed / design_speed) ** settings["speed_exponent"] * (draft / design_draft) ** settings["draft_exponent"]
    if not np.isfinite(raw).all(): raise ValueError("Load calculation overflow")
    load = np.clip(raw, 0, settings["load_cap"])
    grams = {}
    for p in POLLUTANTS:
        ef = np.broadcast_to(np.asarray(factors_g_kwh[p], float), load.shape)
        if not np.isfinite(ef).all() or np.any(ef < 0): raise ValueError(f"Invalid emission factor: {p}")
        grams[p] = power * load * ef * low_load_factor(load, p, settings) * (dt / 3600)
    return load, grams


def process_intervals(rows, settings, include_reference=False):
    """Reject invalid/overlapping intervals and retain their source row numbers."""
    validate_settings(settings)
    required = ("power_kw", "speed_kn", "design_speed_kn", "seim_draft_m", "visual_draft_m", "dt_s")
    if include_reference: required += ("reference_draft_m",)
    required += tuple("ef_" + p + "_g_kwh" for p in POLLUTANTS)
    prepared, issues = [], []
    for line, row in enumerate(rows, 2):
        try:
            row = dict(row)
            canonical, legacy = number(row.get("seim_draft_m")), number(row.get("design_draft_m"))
            if canonical is not None and legacy is not None and not np.isclose(canonical, legacy, rtol=0, atol=1e-9):
                raise ValueError("Conflicting canonical and legacy SEIM draft inputs")
            if not str(row.get("seim_draft_m", "")).strip():
                row["seim_draft_m"] = row.get("design_draft_m", "")
            key = tuple(str(row.get(k, "")).strip() for k in ("area", "clip", "passage_id", "vessel_id"))
            if not all(key): raise ValueError("missing passage identity")
            time = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            # ISO timestamps may be local naive or timezone-aware; never mix them.
            values = {k: number(row.get(k)) for k in required}
            if any(v is None for v in values.values()): raise ValueError("missing or nonfinite numeric input")
            if any(v < 0 for v in values.values()): raise ValueError("negative model input")
            if any(values[k] <= 0 for k in required if not k.startswith("ef_") and k != "speed_kn"):
                raise ValueError("nonpositive model input")
            if include_reference and (not row.get("reference_source", "").strip() or not row.get("reference_kind", "").strip()):
                raise ValueError("reference provenance missing")
            prepared.append(dict(key=key, time=time, values=values, source_row=line,
                vessel_type=str(row.get("vessel_type", "") or "").strip() or "unknown",
                reference_source=row.get("reference_source", ""), reference_kind=row.get("reference_kind", "")))
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(dict(source_row=line, code="excluded_interval", reason=str(exc)))
    groups = defaultdict(list)
    for item in prepared: groups[item["key"][:3]].append(item)
    accepted = []
    for key, group in groups.items():
        if len({g["key"][3] for g in group}) != 1:
            issues.extend(dict(source_row=g["source_row"], code="passage_vessel_conflict", reason="one passage maps to multiple vessels") for g in group)
            continue
        if len({g["vessel_type"] for g in group}) != 1:
            issues.extend(dict(source_row=g["source_row"], code="passage_type_conflict", reason="one passage has multiple vessel types") for g in group)
            continue
        if len({g["time"].tzinfo is not None for g in group}) != 1:
            raise ValueError("Do not mix local and timezone-aware timestamps within a passage")
        end, seen = None, set()
        for item in sorted(group, key=lambda g: g["time"]):
            if item["time"] in seen:
                issues.append(dict(source_row=item["source_row"], code="excluded_interval", reason="duplicate passage timestamp"))
                continue
            seen.add(item["time"])
            if end is not None and item["time"] < end:
                issues.append(dict(source_row=item["source_row"], code="overlapping_interval", reason="interval not counted twice")); continue
            end = item["time"] + timedelta(seconds=item["values"]["dt_s"])
            accepted.append(item)
    if not accepted: raise ValueError("No valid emission intervals remain")
    # An explicit passage boundary does not justify duplicate inventory coverage.
    vessel_intervals = defaultdict(list)
    for item in accepted: vessel_intervals[(item["key"][0], item["key"][3])].append(item)
    for group in vessel_intervals.values():
        if len({g["time"].tzinfo is not None for g in group}) != 1:
            raise ValueError("Mixed timestamp conventions for the same area/vessel")
        end = None
        for item in sorted(group, key=lambda g: g["time"]):
            if end is not None and item["time"] < end:
                raise ValueError("Overlapping passages for the same area/vessel; resolve input coverage first")
            end = item["time"] + timedelta(seconds=item["values"]["dt_s"])
    array = lambda k: np.array([g["values"][k] for g in accepted])
    common = dict(power_kw=array("power_kw"), speed_kn=array("speed_kn"), design_speed_kn=array("design_speed_kn"),
                  design_draft_m=array("seim_draft_m"), dt_s=array("dt_s"),
                  factors_g_kwh={p: array("ef_" + p + "_g_kwh") for p in POLLUTANTS}, settings=settings)
    scenarios = {"SEIM": "seim_draft_m", "VSEIM": "visual_draft_m"}
    if include_reference: scenarios["REFERENCE"] = "reference_draft_m"
    detail, totals = [], defaultdict(float)
    for scenario, column in scenarios.items():
        loads, grams = calculate(draft_m=array(column), **common)
        for i, item in enumerate(accepted):
            area, clip, passage, vessel = item["key"]
            for p in POLLUTANTS:
                value = float(grams[p][i])
                detail.append(dict(area=area, clip=clip, passage_id=passage, vessel_id=vessel,
                    vessel_type=item["vessel_type"], timestamp=item["time"].isoformat(), source_row=item["source_row"],
                    dt_s=item["values"]["dt_s"], scenario=scenario, pollutant=p, load_factor=float(loads[i]), emission_g=value,
                    reference_kind=item["reference_kind"] if scenario == "REFERENCE" else "",
                    reference_source=item["reference_source"] if scenario == "REFERENCE" else ""))
                totals[(area, clip, passage, vessel, item["vessel_type"], scenario, p)] += value / 1000
    summary = [dict(zip(("area", "clip", "passage_id", "vessel_id", "vessel_type", "scenario", "pollutant"), key), emission_kg=value)
               for key, value in sorted(totals.items())]
    return detail, summary, issues
