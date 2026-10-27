"""Passage-based validation and explicitly referenced track-level accuracy."""
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
from .emissions import POLLUTANTS


KEYS = ("area", "clip", "passage_id", "vessel_id")


def error_metrics(reference, estimate):
    reference, estimate = np.asarray(reference, float), np.asarray(estimate, float)
    if reference.ndim != 1 or reference.shape != estimate.shape or not len(reference):
        raise ValueError("Supply aligned nonempty one-dimensional validation arrays")
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all() or np.any(reference <= 0):
        raise ValueError("References must be finite and positive; estimates must be finite")
    error = estimate-reference
    mse = float(np.mean(error**2))
    centered = float(np.sum((reference-reference.mean())**2))
    return dict(N=len(reference), MAE=float(np.mean(abs(error))), RMSE=float(np.sqrt(mse)),
        MAPE_pct=float(np.mean(abs(error)/reference)*100),
        NRMSE_pct=float(np.sqrt(mse)/reference.mean()*100),
        R2=float(1-np.sum(error**2)/centered) if centered > 0 else None,
        aggregate_bias_pct=float(error.sum()/reference.sum()*100))


def holm_adjust(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid probability values")
    order = np.argsort(values)
    adjusted = np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values))))
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted, 1)
    return result.tolist()


def paired_p(reference, first, second):
    differences = abs(np.asarray(first)-reference)-abs(np.asarray(second)-reference)
    return 1.0 if not np.any(differences) else float(wilcoxon(differences, alternative="two-sided").pvalue)


def emission_validation(emissions, draft_references):
    metadata = {}
    for row in draft_references:
        key = tuple(str(row.get(k, "")).strip() for k in KEYS)
        if not all(key) or key in metadata:
            raise ValueError("Missing or duplicate reference passage identity")
        if not row.get("reference_source", "").strip() or row.get("reference_kind") not in {
            "manual_verification", "authority_record", "operator_record", "survey"
        }:
            raise ValueError("Independent-validation references require a declared source and verification kind")
        values = [float(row[k]) for k in ("seim_draft_m", "visual_draft_m", "verified_draft_m")]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Validation drafts must be finite and positive")
        metadata[key] = dict(row, draft_values=values)
    grouped = defaultdict(dict)
    for row in emissions:
        key = tuple(str(row[k]) for k in KEYS)
        if key not in metadata:
            continue
        entry = (row["scenario"], row["pollutant"])
        if entry[0] not in {"SEIM", "VSEIM", "REFERENCE"} or entry[1] not in POLLUTANTS:
            raise ValueError("Unrecognized validation scenario or pollutant")
        if entry in grouped[key]:
            raise ValueError("Repeated passage/scenario/pollutant record")
        value = float(row["emission_kg"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Invalid validation emission total")
        grouped[key][entry] = value
    required = {(s, p) for s in ("SEIM", "VSEIM", "REFERENCE") for p in POLLUTANTS}
    accepted, issues = [], []
    for key in sorted(metadata):
        if set(grouped[key]) != required:
            issues.append(dict(zip(KEYS, key), code="incomplete_common_validation_passage"))
        elif any(grouped[key][("REFERENCE", p)] <= 0 for p in POLLUTANTS):
            issues.append(dict(zip(KEYS, key), code="zero_reference_emission"))
        else:
            accepted.append(key)
    if not accepted:
        raise ValueError("No complete common validation passages")
    summaries, pvalues, raw = [], [], []
    for p in POLLUTANTS:
        ref = np.array([grouped[k][("REFERENCE", p)] for k in accepted])
        first = np.array([grouped[k][("SEIM", p)] for k in accepted])
        second = np.array([grouped[k][("VSEIM", p)] for k in accepted])
        a, b = error_metrics(ref, first), error_metrics(ref, second)
        for scenario, statistics in (("SEIM", a), ("VSEIM", b)):
            summaries.append(dict(pollutant=p, scenario=scenario, **statistics))
        pvalues.append(paired_p(ref, first, second))
        for key, benchmark, seim, vseim in zip(accepted, ref, first, second):
            raw.append(dict(zip(KEYS, key), pollutant=p, E_SEIM_kg=float(seim), E_VSEIM_kg=float(vseim),
                E_verified_kg=float(benchmark), APE_SEIM_pct=float(abs(seim/benchmark-1)*100),
                APE_VSEIM_pct=float(abs(vseim/benchmark-1)*100),
                reference_source=metadata[key]["reference_source"], reference_kind=metadata[key]["reference_kind"]))
    adjusted = holm_adjust(pvalues)
    for i, p in enumerate(POLLUTANTS):
        ref = np.array([grouped[k][("REFERENCE", p)] for k in accepted])
        first = np.array([grouped[k][("SEIM", p)] for k in accepted])
        second = np.array([grouped[k][("VSEIM", p)] for k in accepted])
        for row in summaries[2*i:2*i+2]:
            row.update(lower_error_cases=int(np.sum(abs(second-ref) < abs(first-ref))),
                       paired_p=pvalues[i], holm_p=adjusted[i])
    drafts = np.array([metadata[key]["draft_values"] for key in accepted])
    draft_summary = [dict(scenario=name, **error_metrics(drafts[:, 2], drafts[:, column]))
                     for column, name in enumerate(("SEIM", "VSEIM"))]
    for row in draft_summary:
        row["paired_p"] = paired_p(drafts[:, 2], drafts[:, 0], drafts[:, 1])
    return raw, summaries, draft_summary, issues


def visual_metric_validation(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(k, "")).strip() for k in ("area", "clip", "track_id", "MMSI"))
        if not all(key):
            raise ValueError("Metric observations require area, clip, track_id and MMSI")
        groups[key].append(row)
    raw, issues = [], []
    for key, frames in sorted(groups.items()):
        observation = dict(zip(("area", "clip", "track_id", "MMSI"), key))
        for dimension in ("L", "B"):
            reference_values, visual_values = [], []
            for row in frames:
                if str(row.get("valid_"+dimension, "")).lower() not in {"true", "1"}:
                    continue
                try:
                    ref, vis = float(row[dimension+"_ref_m"]), float(row[dimension+"_vis_m"])
                    if not np.isfinite([ref, vis]).all() or min(ref, vis) <= 0:
                        raise ValueError("Invalid dimension")
                except (ValueError, KeyError, TypeError):
                    issues.append(dict(observation, dimension=dimension, code="invalid_valid_frame_dimension"))
                    continue
                reference_values.append(ref)
                visual_values.append(vis)
            if reference_values and not np.allclose(reference_values, reference_values[0], rtol=0, atol=1e-9):
                issues.append(dict(observation, dimension=dimension, code="conflicting_static_reference"))
                visual_values = []
            observation[dimension+"_ref_m"] = reference_values[0] if visual_values else ""
            observation[dimension+"_vis_m"] = float(np.median(visual_values)) if visual_values else ""
            observation["n_valid_"+dimension+"_frames"] = len(visual_values)
        raw.append(observation)
    summaries = []
    for area in sorted({r["area"] for r in raw})+["Overall"]:
        for dimension in ("L", "B"):
            selected = [r for r in raw if (area == "Overall" or r["area"] == area) and r[dimension+"_vis_m"] != ""]
            statistics = error_metrics([r[dimension+"_ref_m"] for r in selected], [r[dimension+"_vis_m"] for r in selected]) if selected else {"N": 0}
            summaries.append(dict(area=area, dimension=dimension, **statistics))
    return raw, summaries, issues


def association_validation(predictions, references):
    keys = ("area", "clip", "passage_id", "track_id")
    pred, ref = {}, {}
    for rows, target in ((predictions, pred), (references, ref)):
        for row in rows:
            key = tuple(str(row[k]).strip() for k in keys)
            if not all(key) or key in target:
                raise ValueError("Missing or duplicate track-level validation identity")
            target[key] = row
    raw = []
    for key in sorted(set(pred)|set(ref)):
        r, p = ref.get(key, {}), pred.get(key, {})
        label = str(r.get("reference_MMSI", "")).strip()
        if label and not r.get("reference_source", "").strip():
            raise ValueError("Reference association requires source provenance")
        proposed = str(p.get("predicted_MMSI", "")).strip()
        evaluated = bool(label)
        correct = bool(proposed == label and p.get("status") == "accepted") if evaluated else None
        raw.append(dict(zip(keys, key), predicted_MMSI=proposed, reference_MMSI=label,
            correct=correct, status=p.get("status", "missing_prediction"), reference_source=r.get("reference_source", "")))
    summary = []
    for area in sorted({r["area"] for r in raw})+["Overall"]:
        subset = [r for r in raw if (area == "Overall" or r["area"] == area) and r["correct"] is not None]
        n, correct = len(subset), sum(r["correct"] for r in subset)
        summary.append(dict(area=area, Reference_matches=n, Correct_matches=correct,
            Incorrect_matches=n-correct, Accuracy_pct=correct/n*100 if n else ""))
    return raw, summary
