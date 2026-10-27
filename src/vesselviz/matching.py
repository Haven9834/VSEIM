"""Synchronized local-metric AIS assignment and track-level consistency checks."""
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
from scipy.optimize import linear_sum_assignment
from .grounding import setting
from .metric import water_to_metric


def timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Matching timestamps must include a timezone")
    return parsed.astimezone(timezone.utc).timestamp()


def index_ais(rows):
    groups = defaultdict(list)
    for row in rows:
        mmsi = str(row.get("MMSI", "")).strip()
        if len(mmsi) != 9 or not mmsi.isdigit():
            raise ValueError("AIS identity must be a nine-digit MMSI string")
        t = timestamp(row["timestamp"])
        point = np.array([float(row["x_m"]), float(row["y_m"])])
        if not np.isfinite(point).all():
            raise ValueError("AIS local-metric position is nonfinite")
        speed = float(row["speed_kn"]) if str(row.get("speed_kn", "")).strip() else None
        if speed is not None and (not np.isfinite(speed) or speed < 0):
            raise ValueError("Invalid AIS speed")
        groups[mmsi].append((t, point, speed))
    output = {}
    for mmsi, values in groups.items():
        values.sort(key=lambda x: x[0])
        if len({v[0] for v in values}) != len(values):
            raise ValueError("Duplicate AIS vessel timestamps must be resolved before matching")
        output[mmsi] = ([v[0] for v in values], values)
    return output


def interpolate_targets(index, time, max_gap):
    targets = []
    for mmsi, (times, rows) in sorted(index.items()):
        j = bisect_left(times, time)
        if j < len(times) and times[j] == time:
            _, point, speed = rows[j]
        elif j == 0 or j == len(times) or times[j]-times[j-1] > max_gap:
            continue
        else:
            ratio = (time-times[j-1])/(times[j]-times[j-1])
            point = rows[j-1][1]*(1-ratio)+rows[j][1]*ratio
            a, b = rows[j-1][2], rows[j][2]
            speed = a*(1-ratio)+b*ratio if a is not None and b is not None else None
        targets.append(dict(MMSI=mmsi, point=point, speed_kn=speed))
    return targets


def associate(visual_rows, ais_rows, config):
    threshold = setting(config, "distance_threshold_m", minimum=np.finfo(float).eps)
    gap = setting(config, "max_interpolation_gap_s", minimum=np.finfo(float).eps)
    minimum = setting(config, "min_track_matches", integer=True, minimum=1)
    fraction = setting(config, "min_track_fraction", minimum=np.finfo(float).eps, maximum=1)
    margin = setting(config, "median_margin_m")
    ais_index = index_ais(ais_rows)
    frames, tracks, identities = defaultdict(list), defaultdict(list), {}
    for row in visual_rows:
        identity = {k: str(row.get(k, "")).strip() for k in ("area", "camera", "clip", "passage_id", "track_id")}
        if not all(identity.values()):
            raise ValueError("Visual observations require area, camera, clip, passage_id and track_id")
        key = tuple(identity.values())
        time = timestamp(row["timestamp"])
        xy = water_to_metric([[float(row["water_u"]), float(row["water_v"])]],
                             config["cameras"][identity["camera"]]["homography"])[0]
        item = dict(identity, key=key, time=time, timestamp=row["timestamp"], point=xy)
        frames[(identity["camera"], identity["clip"], time)].append(item)
        tracks[key].append(item)
        identities[key] = identity
    predictions, issues = [], []
    for (_, _, time), observations in sorted(frames.items()):
        if len({o["track_id"] for o in observations}) != len(observations):
            raise ValueError("Duplicate visual track at one camera/clip timestamp")
        targets = interpolate_targets(ais_index, time, gap)
        n, m = len(observations), len(targets)
        cost = np.full((n, m+n), threshold+1, dtype=float)
        for i, item in enumerate(observations):
            for j, target in enumerate(targets):
                distance = float(np.linalg.norm(item["point"]-target["point"]))
                cost[i, j] = distance if distance <= threshold else (threshold+1)*(n+1)
        ii, jj = linear_sum_assignment(cost)
        for i, j in zip(ii, jj):
            item = observations[i]
            if j >= m or cost[i, j] > threshold:
                issues.append(dict(identities[item["key"]], timestamp=item["timestamp"], code="unmatched_frame"))
                continue
            target = targets[j]
            predictions.append(dict(identities[item["key"]], timestamp=item["timestamp"],
                predicted_MMSI=target["MMSI"], distance_m=float(cost[i, j]),
                speed_kn=target["speed_kn"] if target["speed_kn"] is not None else ""))
    matched = defaultdict(list)
    for row in predictions:
        key = tuple(row[k] for k in ("area", "camera", "clip", "passage_id", "track_id"))
        matched[key].append(row)
    final = []
    for key, observations in sorted(tracks.items()):
        by_mmsi = defaultdict(list)
        for row in matched[key]:
            by_mmsi[row["predicted_MMSI"]].append(row["distance_m"])
        if len(by_mmsi) > 1:
            issues.append(dict(identities[key], code="multiple_candidate_mmsi", candidates=sorted(by_mmsi)))
        candidates = []
        for mmsi, distances in by_mmsi.items():
            support = len(distances)/len(observations)
            if len(distances) >= minimum and support >= fraction:
                candidates.append((float(np.median(distances)), mmsi, len(distances), support))
        candidates.sort()
        status, winner = "insufficient_support", None
        if candidates:
            winner = candidates[0]
            status = "accepted"
            if len(candidates) > 1 and candidates[1][0]-winner[0] <= margin:
                status = "ambiguous_identity"
        if status != "accepted":
            issues.append(dict(identities[key], code=status))
        final.append(dict(identities[key], predicted_MMSI=winner[1] if winner else "", status=status,
            n_frames=len(observations), n_matches=winner[2] if winner else 0,
            match_fraction=winner[3] if winner else 0., median_distance_m=winner[0] if winner else ""))
    # Frame assignments are one-to-one. Track summaries must also flag overlap.
    conflicts = set()
    for i, a in enumerate(final):
        if a["status"] != "accepted":
            continue
        for j, b in enumerate(final[i+1:], i+1):
            if b["status"] != "accepted" or any(a[k] != b[k] for k in ("camera", "clip", "predicted_MMSI")):
                continue
            keys = ("area", "camera", "clip", "passage_id", "track_id")
            ka, kb = tuple(a[k] for k in keys), tuple(b[k] for k in keys)
            at = [o["time"] for o in tracks[ka]]
            bt = [o["time"] for o in tracks[kb]]
            if max(min(at), min(bt)) <= min(max(at), max(bt)):
                conflicts.update((i, j))
                issues.append(dict(camera=a["camera"], clip=a["clip"], MMSI=a["predicted_MMSI"],
                                   tracks=[a["track_id"], b["track_id"]], code="overlapping_track_identity"))
    for i in conflicts:
        final[i]["status"] = "overlapping_track_identity"
    return predictions, final, issues
