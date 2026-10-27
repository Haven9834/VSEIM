"""Analytical geometry, association and workflow checks with generated fixtures."""
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from vesselviz.semantic import QwenAPI, parse_semantics, eligible, vessel_crops
from vesselviz.grounding import response_map, select_candidate, boundary_pairs
from vesselviz.metric import water_to_metric, recover_vertical, frame_freeboard, aggregate_drafts
from vesselviz.matching import associate, index_ais, interpolate_targets, timestamp
from vesselviz.evaluation import error_metrics, holm_adjust, emission_validation, association_validation, visual_metric_validation
from vesselviz.emissions import POLLUTANTS, process_intervals
from vesselviz.inventory import attach_drafts, attach_identities, inventory_tables
from vesselviz.specifications import fill_missing
from vesselviz.reconstruction import run_reconstruction


H = [[.1, 0, 0], [0, .1, -10], [0, 0, 1]]
P = [[10, 0, 0, 0], [0, 10, -10, 100], [0, 0, 0, 1]]
GEOMETRY = dict(reprojection_tolerance_px=.1, min_samples=3, mad_multiplier=3,
                freeboard_floor_m=.01, min_valid_frames=1)
BOUNDARY = dict(min_columns=3, median_window=3, mad_multiplier=3,
                residual_floor_px=1, polynomial_degree=1)
GROUNDING = dict(positive_threshold=.7, negative_threshold=.7, positive_points=3,
    negative_points=2, semantic_weight=1, continuity_weight=1,
    min_mask_pixels=5, min_sam_score=.5)


def semantics(**kwargs):
    return dict(vessel_type="test", viewpoint="side", hull_configuration="visible side shell",
        hull_visible=True, upper_visible=True, waterline_visible=True,
        occlusion="none", eligible=True, positive_prompts=["side shell"],
        negative_prompts=["water"], **kwargs)


def model_settings():
    return dict(draft_exponent=1., speed_exponent=3., load_cap=1., low_load_threshold=.2,
        load_points=[.01, .2], low_load_factors={p: [1., 1.] for p in POLLUTANTS},
        parameter_source="unit-test fixture")


def emission_row():
    return dict(area="Area I", clip="clip", passage_id="passage", vessel_id="fixture",
        timestamp="2000-01-01T00:00:00+00:00", power_kw=1000, speed_kn=5,
        design_speed_kn=10, seim_draft_m=4, visual_draft_m=2, dt_s=1,
        vessel_type="test", **{"ef_"+p+"_g_kwh": 1 for p in POLLUTANTS})


def match_settings():
    return dict(distance_threshold_m=10, max_interpolation_gap_s=10, min_track_matches=1,
        min_track_fraction=.5, median_margin_m=.1, cameras={"Camera_01": {"homography": np.eye(3).tolist()}})


def visual(track, second, x=0):
    return dict(area="Area I", camera="Camera_01", clip="clip", passage_id=track,
        track_id=track, timestamp=f"2000-01-01T00:00:{second:02d}+00:00", water_u=x, water_v=0)


def ais(mmsi="123450000", second=0, x=0):
    return dict(MMSI=mmsi, timestamp=f"2000-01-01T00:00:{second:02d}+00:00", x_m=x, y_m=0, speed_kn=2)


class SemanticsTests(unittest.TestCase):
    def test_boolean_strings_rejected(self):
        row = semantics(); row["eligible"] = "false"
        with self.assertRaises(ValueError):
            parse_semantics(row)

    def test_visibility_and_view_gate(self):
        self.assertTrue(eligible(semantics()))
        for field, value in (("viewpoint", "bow"), ("occlusion", "severe"), ("waterline_visible", False)):
            row = semantics(); row[field] = value
            self.assertFalse(eligible(row))

    def test_crops_keep_coordinates(self):
        local, context, box = vessel_crops(Image.new("RGB", (100, 80)), [20, 10, 60, 50], 2)
        self.assertEqual(local.size, (40, 40))
        self.assertEqual(context.size, (80, 70))
        self.assertEqual(box, (20, 10, 60, 50))

    def test_empty_api_configuration_rejected(self):
        with self.assertRaises(ValueError):
            QwenAPI({}, "")

    def test_api_payload_and_parse_without_network(self):
        config = dict(endpoint="https://service.invalid/chat/completions", model="configured-model",
                      temperature=0, max_tokens=256, timeout_s=1)
        response = BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(semantics())}}]}).encode())
        with patch("vesselviz.semantic.request.urlopen", return_value=response) as transport:
            value = QwenAPI(config, "Measurement task").analyze(Image.new("RGB", (5, 5)), Image.new("RGB", (7, 7)))
        self.assertTrue(value["eligible"])
        payload = json.loads(transport.call_args.args[0].data)
        self.assertEqual(payload["model"], "configured-model")
        self.assertEqual(len(payload["messages"][1]["content"]), 3)


class GroundingGeometryTests(unittest.TestCase):
    def test_grounding_suppression(self):
        hull, exclusion = response_map(np.full((2, 3, 4), .8), np.full((1, 3, 4), .25))
        np.testing.assert_allclose(hull, .6)
        np.testing.assert_allclose(exclusion, .25)

    def test_connected_candidate_selection(self):
        good = np.zeros((8, 8), bool); good[2:6, 2:6] = True
        bad = np.zeros((8, 8), bool); bad[::2, ::2] = True
        mask, diagnostic = select_candidate([bad, good], [.9, .9], np.ones((8, 8)), GROUNDING)
        np.testing.assert_array_equal(mask, good)
        self.assertEqual(diagnostic["continuity"], 1)

    def test_boundary_and_analytic_freeboard(self):
        mask = np.zeros((120, 60), bool); mask[70:101, 10:50] = True
        upper, water = boundary_pairs(mask, BOUNDARY)
        result = frame_freeboard(upper, water, H, P, 5, GEOMETRY)
        self.assertAlmostEqual(result["freeboard_m"], 3, places=4)

    def test_vertical_reprojection(self):
        point = water_to_metric([[20, 100]], H)[0]
        freeboard, residual = recover_vertical(point, [20, 70], P, 5)
        self.assertAlmostEqual(freeboard, 3, places=4)
        self.assertLess(residual, 1e-4)

    def test_height_unobservable(self):
        # At X=Y=0, the optical-axis vertical has no image displacement.
        degenerate = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]]
        with self.assertRaises(ValueError):
            recover_vertical([0, 0], [0, 0], degenerate, 4)

    def test_passage_median_and_conflicting_identity(self):
        base = dict(area="Area I", camera="Camera_01", clip="clip", passage_id="p", track_id="t",
                    vessel_id="fixture", timestamp="2000-01-01T00:00:00+00:00", valid=True, moulded_depth_m=5)
        rows = [dict(base, freeboard_m=v) for v in (1, 3, 2)]
        passages, issues = aggregate_drafts(rows, 2)
        self.assertEqual(passages[0]["visual_draft_m"], 3)
        self.assertFalse(issues)
        rows[0]["vessel_id"] = "other"
        self.assertFalse(aggregate_drafts(rows, 2)[0])


class MatchingTests(unittest.TestCase):
    def test_interpolation_and_no_extrapolation(self):
        index = index_ais([ais(second=0, x=0), ais(second=10, x=10)])
        middle = interpolate_targets(index, timestamp(ais(second=5)["timestamp"]), 10)
        self.assertEqual(middle[0]["point"][0], 5)
        self.assertFalse(interpolate_targets(index, timestamp(ais(second=11)["timestamp"]), 10))
        self.assertFalse(interpolate_targets(index, timestamp(ais(second=5)["timestamp"]), 9))

    def test_one_to_one_and_gating(self):
        predicted, tracks, issues = associate([visual("a", 0), visual("b", 0, 5), visual("c", 0, 100)],
            [ais(x=0), ais(mmsi="123450001", x=5)], match_settings())
        self.assertEqual(len(predicted), 2)
        self.assertEqual(len({r["predicted_MMSI"] for r in predicted}), 2)
        self.assertEqual(sum(r["status"] == "accepted" for r in tracks), 2)
        self.assertIn("unmatched_frame", [r["code"] for r in issues])

    def test_multiple_track_candidates_reported(self):
        rows = [visual("a", 0, 0), visual("a", 1, 5)]
        targets = [ais(second=s, x=0) for s in (0, 1)]+[ais(mmsi="123450001", second=s, x=5) for s in (0, 1)]
        _, tracks, issues = associate(rows, targets, match_settings())
        self.assertEqual(tracks[0]["status"], "ambiguous_identity")
        self.assertIn("multiple_candidate_mmsi", [r["code"] for r in issues])

    def test_all_overlapping_identity_conflicts_marked(self):
        rows = [visual(track, t) for track, times in (("a", (0, 4)), ("b", (1, 5)), ("c", (2, 6))) for t in times]
        _, tracks, _ = associate(rows, [ais(second=0), ais(second=6)], match_settings())
        self.assertTrue(all(r["status"] == "overlapping_track_identity" for r in tracks))

    def test_duplicate_ais_time_rejected(self):
        with self.assertRaises(ValueError):
            index_ais([ais(), ais()])


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_holm(self):
        statistics = error_metrics([100, 200], [110, 180])
        self.assertAlmostEqual(statistics["MAPE_pct"], 10)
        self.assertAlmostEqual(statistics["R2"], .9)
        np.testing.assert_allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])

    def test_passage_metric_medians_not_frame_weighted(self):
        rows = [dict(area="Area I", clip="clip", track_id="t", MMSI="fixture", valid_L="true",
                     L_ref_m=10, L_vis_m=value, valid_B="false") for value in (9, 10, 11)]
        raw, summary, _ = visual_metric_validation(rows)
        self.assertEqual(raw[0]["L_vis_m"], 10)
        self.assertEqual(summary[0]["N"], 1)
        self.assertEqual(summary[1]["N"], 0)

    def test_association_reference_denominator(self):
        key = dict(area="Area I", clip="clip", passage_id="p", track_id="t")
        _, summary = association_validation([dict(key, predicted_MMSI="fixture", status="accepted")], [])
        self.assertEqual(summary[0]["Reference_matches"], 0)
        _, summary = association_validation([], [dict(key, reference_MMSI="fixture", reference_source="fixture")])
        self.assertEqual(summary[0]["Incorrect_matches"], 1)

    def test_full_emission_validation(self):
        key = dict(area="Area I", clip="clip", passage_id="p", vessel_id="fixture")
        emissions = [dict(key, scenario=s, pollutant=p, emission_kg=value)
            for p in POLLUTANTS for s, value in (("SEIM", 12), ("VSEIM", 10.5), ("REFERENCE", 10))]
        references = [dict(key, seim_draft_m=4, visual_draft_m=3.1, verified_draft_m=3,
                           reference_source="fixture", reference_kind="manual_verification")]
        raw, summary, drafts, issues = emission_validation(emissions, references)
        self.assertEqual(len(raw), 6)
        self.assertAlmostEqual(summary[0]["MAPE_pct"], 20)
        self.assertAlmostEqual(summary[1]["MAPE_pct"], 5)
        self.assertEqual(drafts[0]["N"], 1)
        self.assertFalse(issues)

    def test_canonical_draft_and_legacy_conflict(self):
        detail, _, _ = process_intervals([emission_row()], model_settings())
        first = next(r for r in detail if r["scenario"] == "SEIM")
        second = next(r for r in detail if r["scenario"] == "VSEIM")
        self.assertAlmostEqual(second["emission_g"]/first["emission_g"], .5)
        row = dict(emission_row(), design_draft_m=5)
        with self.assertRaises(ValueError):
            process_intervals([row], model_settings())

    def test_parameter_regression_provenance(self):
        rows, issues = fill_missing([dict(vessel_type="test", L=10)],
            {"test": {"D": dict(predictors=["L"], coefficients=[.1], intercept=1, source="fixture")}}, ["D"])
        self.assertEqual(rows[0]["D"], 2)
        self.assertIn("regression", rows[0]["D_source"])
        self.assertFalse(issues)


class WorkflowTests(unittest.TestCase):
    def test_explicit_joins(self):
        row = dict(area="Area I", camera="Camera_01", clip="clip", passage_id="p", track_id="t")
        linked, issues = attach_identities([row], [dict(row, predicted_MMSI="fixture", status="accepted")],
                                          [dict(MMSI="fixture", moulded_depth_m=5)])
        self.assertEqual(linked[0]["moulded_depth_m"], 5)
        activity = [dict(row, vessel_id="fixture", timestamp="2000-01-01T00:00:00+00:00")]
        prepared, _ = attach_drafts(activity, [dict(activity[0], visual_draft_m=3)])
        self.assertEqual(prepared[0]["visual_draft_m"], 3)
        self.assertFalse(issues)

    def test_inventory_weights(self):
        rows = [dict(area="Area I", pollutant="CO2", scenario=s, vessel_type=t, emission_kg=v)
            for t, values in (("bulk", (60, 40)), ("container", (40, 35)))
            for s, v in zip(("SEIM", "VSEIM"), values)]
        summary, shares = inventory_tables(rows)
        self.assertEqual(summary[0]["relative_difference_pct"], 25)
        self.assertAlmostEqual(sum(r["contribution_pct"] for r in shares if r["area"] == "Overall" and r["scenario"] == "VSEIM"), 100)

    def test_reconstruction_with_injected_backends(self):
        class SemanticBackend:
            def analyze(self, local, context):
                return semantics()
        class Grounder:
            def responses(self, image, descriptions):
                a = np.full((1, image.height, image.width), .02)
                if descriptions == ["side shell"]:
                    a[:, 30:61, :] = .95
                else:
                    a[:, :15, :] = .95
                return a
        class Segmenter:
            def candidates(self, image, points, labels):
                mask = np.zeros((image.height, image.width), bool)
                mask[30:61, :] = True
                return np.asarray([mask]), np.array([.95])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (150, 120)).save(root / "frame.png")
            row = dict(area="Area I", camera="Camera_01", clip="clip", passage_id="p", track_id="t", frame_id="f",
                vessel_id="fixture", timestamp="2000-01-01T00:00:00+00:00", image="frame.png", bbox_xyxy=[10, 40, 40, 110],
                upper_reference_matches_moulded_depth=True, association_status="accepted", moulded_depth_m=5)
            (root / "observations.json").write_text(json.dumps({"observations": [row]}), encoding="utf-8")
            config = dict(observations_json="observations.json", context_expansion=1.5,
                cameras={"Camera_01": {"homography": H, "projection_matrix": P}},
                grounding=GROUNDING, boundary=BOUNDARY, geometry=GEOMETRY)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            output = run_reconstruction(root / "config.json", root / "out", vlm=SemanticBackend(), grounder=Grounder(), segmenter=Segmenter())
            with (output / "passage_drafts.csv").open(encoding="utf-8-sig", newline="") as stream:
                result = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(result["visual_draft_m"]), 2, places=4)
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["n_valid_passages"], 1)


if __name__ == "__main__":
    unittest.main()
