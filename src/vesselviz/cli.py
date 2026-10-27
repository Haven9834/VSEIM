"""Command-line workflows. Inputs stay read-only; each run uses a new directory."""
import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
import sys
from PIL import Image
from .io import configured_path, create_output, load_json, save_json, safe_key


def write_csv(path, rows):
    if not rows: return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run_association(config_file, output, selection=None):
    from .association import collect_camera
    from .render import association_panel
    from .gallery import write_gallery
    config = load_json(config_file); base = Path(config_file).resolve().parent
    cameras = config.get("cameras", [])
    if not cameras: raise ValueError("Configure at least one camera/clip")
    if len({(c["camera"], c["clip"]) for c in cameras}) != len(cameras):
        raise ValueError("Duplicate camera/clip configuration")
    frames, issues = [], []
    for camera in cameras:
        found, problems = collect_camera(camera, base); frames.extend(found); issues.extend(problems)
    if not frames: raise ValueError("No matched frames; check paths, field mapping, accepted statuses and labels")
    if selection:
        selected = Path(selection).resolve()
        frames = [f for f in frames if Path(f["image_path"]).resolve() == selected]
        if len(frames) != 1: raise ValueError("Selected image has no unique camera/clip match")
    out = create_output(output); (out / "images").mkdir()
    style = config.get("render", {}); font_path = ""
    if style.get("font_path"): font_path = str(configured_path(style["font_path"], base))
    records = []
    for frame in frames:
        identity = frame["camera"] + "/" + frame["clip"] + "/" + frame["frame"]
        key = hashlib.sha256(identity.encode()).hexdigest()[:20]
        panel, layout = association_panel(frame, style.get("panel_width", 1600), font_path)
        image_name = "images/" + key + ".jpg"; panel.save(out / image_name, quality=95, subsampling=0)
        records.append(dict(id=key, camera=frame["camera"], clip=frame["clip"], frame=frame["frame"],
                            image=image_name, tracks=[s["track_id"] for s in frame["ships"]], layout=layout))
    save_json(out / "index.json", records); save_json(out / "issues.json", issues)
    write_gallery(out / "gallery.html", records)
    return out


def run_hull(manifest, output, selection=None):
    from .hull import hull_panels, PANEL_NAMES
    from .render import assemble_rows
    base = Path(manifest).resolve().parent; frames = load_json(manifest).get("frames", [])
    if selection:
        frames = [f for f in frames if configured_path(f["image"], base) == Path(selection).resolve()]
        if len(frames) != 1: raise ValueError("Selected image has no unique exact-frame label entry")
    if not frames: raise ValueError("Manifest contains no frames")
    keys = [tuple(f[k] for k in ("camera", "clip", "frame_id")) for f in frames]
    if len(set(keys)) != len(keys): raise ValueError("Duplicate hull frame identity")
    # Validate required files before creating an output directory.
    for frame in frames:
        for field in ("image", "vessel_mask", "probability", "hulls"): configured_path(frame.get(field), base)
    out = create_output(output); rows, reports = [], []
    for frame in frames:
        panels, report = hull_panels(frame, base)
        sub = out.joinpath(*(safe_key(frame[k]) for k in ("camera", "clip", "frame_id")))
        sub.mkdir(parents=True)
        for name, panel in zip(PANEL_NAMES, panels): panel.save(sub / (name + ".png"))
        row = assemble_rows([panels]); row.save(sub / "five_columns.png", dpi=(300, 300))
        rows.append(panels); reports.append(report)
    assemble_rows(rows).save(out / "all_frames.png", dpi=(300, 300)); save_json(out / "audit.json", reports)
    return out


def run_emissions(config_file, output):
    from .emissions import process_intervals
    config = load_json(config_file); base = Path(config_file).resolve().parent
    source = configured_path(config.get("input_csv"), base)
    with source.open(encoding="utf-8-sig", newline="") as stream: rows = list(csv.DictReader(stream))
    detail, passages, issues = process_intervals(rows, config["model"], config.get("include_reference", False))
    out = create_output(output)
    write_csv(out / "interval_emissions.csv", detail); write_csv(out / "passage_emissions.csv", passages)
    save_json(out / "issues.json", issues)
    for filename, keys in (("area_totals.csv", ("area", "scenario", "pollutant")),
                           ("vessel_type_totals.csv", ("area", "vessel_type", "scenario", "pollutant"))):
        grouped = defaultdict(float)
        for row in passages: grouped[tuple(row[k] for k in keys)] += row["emission_kg"]
        write_csv(out / filename, [dict(zip(keys, key), emission_kg=value) for key, value in sorted(grouped.items())])
    comparisons = defaultdict(dict)
    for row in passages:
        key = tuple(row[k] for k in ("area", "clip", "passage_id", "vessel_id", "pollutant"))
        comparisons[key][row["scenario"]] = row["emission_kg"]
    comparison_rows = []
    for key, values in sorted(comparisons.items()):
        seim, vseim = values["SEIM"], values["VSEIM"]
        comparison_rows.append(dict(zip(("area", "clip", "passage_id", "vessel_id", "pollutant"), key),
            SEIM_kg=seim, VSEIM_kg=vseim, signed_difference_kg=vseim-seim,
            absolute_difference_kg=abs(vseim-seim), relative_difference_pct=(vseim/seim-1)*100 if seim else "",
            reduction_pct=(seim-vseim)/seim*100 if seim else ""))
    write_csv(out / "passage_comparison.csv", comparison_rows)
    ranked = []
    groups = defaultdict(list)
    for row in passages: groups[(row["area"], row["scenario"], row["pollutant"])].append(row)
    for group in groups.values():
        previous = None; rank = 0
        for index, row in enumerate(sorted(group, key=lambda r: (-r["emission_kg"], r["clip"], r["passage_id"])), 1):
            if row["emission_kg"] != previous: rank = index
            previous = row["emission_kg"]; ranked.append(dict(row, rank=rank))
    write_csv(out / "passage_rankings.csv", ranked)
    from .inventory import inventory_tables
    comparisons, contributions = inventory_tables(passages)
    write_csv(out / "inventory_comparison.csv", comparisons)
    write_csv(out / "vessel_type_contributions.csv", contributions)
    save_json(out / "run.json", dict(parameter_source=config["model"]["parameter_source"],
        accepted_input_intervals=len(detail)//(6*(3 if config.get("include_reference") else 2)),
        excluded_intervals=len(issues), reference_is_independent_ground_truth=False,
        note="REFERENCE denotes a supplied draft scenario, not proof of independent verification."))
    return out


def run_assemble(index_path, selection_path, output):
    from .render import assemble_rows
    base = Path(index_path).resolve().parent
    records = {r["id"]: r for r in load_json(index_path)}
    ids = load_json(selection_path)["ids"]
    if not ids or len(ids) != len(set(ids)): raise ValueError("Selection must contain unique frame IDs")
    rows = defaultdict(list)
    from .io import confined_path
    for key in ids:
        r = records[key]
        with Image.open(confined_path(base, r["image"])) as image: rows[r["camera"]].append(image.copy())
    out = create_output(output)
    assemble_rows([rows[k] for k in sorted(rows)]).save(out / "selected_grid.png", dpi=(300, 300))
    save_json(out / "selection.json", {"ids": ids})
    return out


def run_match(config_file, output):
    from .matching import associate
    config = load_json(config_file)
    base = Path(config_file).resolve().parent
    def read(field):
        with configured_path(config.get(field), base).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    predictions, tracks, issues = associate(read("visual_csv"), read("ais_csv"), config)
    out = create_output(output)
    write_csv(out / "frame_associations.csv", predictions)
    write_csv(out / "track_mmsi_matches.csv", tracks)
    save_json(out / "issues.json", issues)
    return out


def run_validate(config_file, output):
    from .evaluation import emission_validation, visual_metric_validation, association_validation
    config = load_json(config_file)
    base = Path(config_file).resolve().parent
    def read(field):
        with configured_path(config.get(field), base).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    outputs, issues = {}, []
    if config.get("passage_emissions_csv") or config.get("verified_drafts_csv"):
        raw, summary, drafts, problems = emission_validation(read("passage_emissions_csv"), read("verified_drafts_csv"))
        outputs.update(emission_validation_raw=raw, emission_validation_summary=summary, draft_validation_summary=drafts)
        issues.extend(problems)
    if config.get("visual_metrics_csv"):
        raw, summary, problems = visual_metric_validation(read("visual_metrics_csv"))
        outputs.update(visual_metric_raw=raw, visual_metric_summary=summary)
        issues.extend(problems)
    if config.get("association_predictions_csv") or config.get("association_references_csv"):
        raw, summary = association_validation(read("association_predictions_csv"), read("association_references_csv"))
        outputs.update(association_validation_raw=raw, association_validation_summary=summary)
    if not outputs:
        raise ValueError("Configure at least one validation input set")
    out = create_output(output)
    for name, rows in outputs.items():
        write_csv(out / (name+".csv"), rows)
    save_json(out / "issues.json", issues)
    return out


def run_prepare(config_file, output):
    from .inventory import attach_identities, attach_drafts
    config = load_json(config_file)
    base = Path(config_file).resolve().parent
    def read(field):
        with configured_path(config.get(field), base).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if config.get("mode") == "observations":
        source = configured_path(config.get("observations_json"), base)
        rows = load_json(source).get("observations", [])
        rows = [dict(row, image=str(configured_path(row.get("image"), source.parent))) for row in rows]
        prepared, issues = attach_identities(rows, read("track_matches_csv"), read("technical_csv"))
    elif config.get("mode") == "inventory":
        prepared, issues = attach_drafts(read("activity_csv"), read("passage_drafts_csv"))
    else:
        raise ValueError("Preparation mode must be observations or inventory")
    out = create_output(output)
    if config["mode"] == "observations":
        save_json(out / "observations.json", {"observations": prepared})
    else:
        write_csv(out / "inventory_inputs.csv", prepared)
    save_json(out / "issues.json", issues)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vesselviz")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("association", "hull", "emissions", "reconstruct", "match", "prepare", "validate", "pick"):
        p = sub.add_parser(name); p.add_argument("--config", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
        if name == "pick": p.add_argument("--kind", choices=("association", "hull"), required=True)
    p = sub.add_parser("assemble"); p.add_argument("--index", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "pick":
            import tkinter as tk
            from tkinter import filedialog, messagebox
            root = tk.Tk(); root.withdraw()
            try:
                selected = filedialog.askopenfilename(title="Select an original frame", filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff")])
                if not selected: return 0
                (run_association if args.kind == "association" else run_hull)(args.config, args.output, selected)
            except (OSError, ValueError, KeyError) as exc:
                messagebox.showerror("Input or label error", str(exc)); return 2
            finally: root.destroy()
            return 0
        if args.command == "assemble": result = run_assemble(args.index, args.selection, args.output)
        else:
            from .reconstruction import run_reconstruction
            result = {"association": run_association, "hull": run_hull, "emissions": run_emissions,
                      "reconstruct": run_reconstruction, "match": run_match, "prepare": run_prepare,
                      "validate": run_validate}[args.command](args.config, args.output)
        print(result); return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
