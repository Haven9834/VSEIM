# Input schemas

## Association input

One configuration entry covers one camera and one clip. All listed mappings must refer to that clip. Input CSV files use UTF-8, optionally with a BOM. Identifiers should be stored as strings.

Default per-frame columns:

| Column | Meaning |
| --- | --- |
| image_name | Relative image filename under images_dir |
| track_id | Actual visual track identifier |
| ais_mmsi | Nine-digit associated MMSI |
| x1, y1, x2, y2 | Whole-vessel bounding box in the declared coordinate system |
| ais_speed | Same-row speed over ground in kn |
| ais_english_name | Existing English name or already-transliterated name |
| match_status | Status evaluated against accepted_statuses |
| track_match_confidence | Optional stored confidence label |
| within_adaptive_gate | Optional stored gate flag |

Required mapping columns are `track_id,mmsi`. Use `mapping_fields` to rename them. `fields` keys are `image`, `track`, `mmsi`, `x1`, `y1`, `x2`, `y2`, `speed`, `name`, `status`, `confidence`, `within_gate`.

The software does not treat the mapping table as independent ground truth. It checks internal consistency between two existing results. A non-nine-digit MMSI, ambiguous mapping or repeated same-frame track is excluded and logged. All repeated same-frame track records are excluded together rather than retaining an arbitrary first row. Non-high confidence and outside/unknown gate flags remain visible with issue entries.

Names are not translated. Supply an English/transliterated name or the display reads N/A. A `TRACK_MATCHED_AIS_MISSING_AT_FRAME` record can retain its accepted track mapping, but never borrows speed from a neighboring timestamp.

## Hull input

The manifest has a `frames` array. Each entry contains:

| Field | Meaning |
| --- | --- |
| camera, clip, frame_id | Explicit identity, unique as a tuple |
| image | Original image file |
| image_sha256 | Optional source-image checksum |
| original_size | `[width, height]` in pixels |
| vessel_mask | Same-size single-channel binary PNG, values 0/255 |
| probability | Same-size H × W numeric `.npy` array, finite values in [0,1] |
| hulls | JSON containing original-coordinate hull polygons |
| display_crop | `[top,bottom,left,right]`, nonnegative integer pixels |

Hull JSON structure:

```json
{
  "camera": "",
  "clip": "",
  "frame_id": "",
  "original_size": [null, null],
  "hull_instances": []
}
```

Each nonempty hull instance has an `id` and a `polygon` containing at least three `[x,y]` points. Empty `hull_instances` is valid and explicitly reported; overlays remain unmarked. Existing `hull_polygon_original` or `polygon_original` records need to be mapped into this schema. ROI coordinates must be restored before use.

Identity metadata in a hull file is checked when present. For older files without metadata, the manifest is the user's explicit association; equal array dimensions alone cannot establish same-frame provenance. Missing metadata is reported. Supply identity metadata/checksums whenever possible. The software never searches for a nearby timestamp.

## Emission input

The CSV contains one row per non-overlapping integration interval, not one row per already-summed passage. Required fields:

| Field | Unit / definition |
| --- | --- |
| area, clip, passage_id | Scope and unique vessel-passage identifier |
| vessel_id | Consistent pseudonymous vessel identifier; raw MMSI is not required |
| timestamp | ISO 8601 interval-start time |
| dt_s | Positive interval duration, seconds |
| power_kw | Rated main-engine power, kW |
| speed_kn, design_speed_kn | Current and design speed, kn |
| seim_draft_m, visual_draft_m | Conventional SEIM baseline and visually reconstructed operating drafts, m |
| ef_CO2_g_kwh, ef_CO_g_kwh, ef_VOCs_g_kwh | Emission factors, g/kWh |
| ef_NOx_g_kwh, ef_PM10_g_kwh, ef_SO2_g_kwh | Emission factors, g/kWh |
| vessel_type | Optional category; missing becomes unknown |

With `include_reference=true`, also provide positive `reference_draft_m`, nonempty `reference_source` and `reference_kind`. Describe AIS-reported values as such, not as gauge measurements. `REFERENCE` is a scenario label, not a certification of independence.

No input values are imputed. Invalid rows are recorded in `issues.json` and excluded from every scenario. Duplicate passage timestamps and overlapping intervals are logged. Overlapping coverage for the same area/vessel across separate passages raises an error. A passage containing conflicting vessel identities is excluded. Zero-duration bookkeeping rows are excluded and do not add emissions.

The legacy design_draft_m alias is accepted when seim_draft_m is absent. If both are supplied, they must agree. The baseline field describes the actual conventional input used by the study, including its source; it does not imply a design-draft measurement.

The reader must assign valid passage boundaries upstream. Historical per-MMSI aggregates cannot be relabeled as individual passages without evidence. Separate clips/areas may retain the same vessel, but overlapping intervals must not count the same area/vessel twice.

## Matching inputs

The visual CSV requires area, camera, clip, passage_id, track_id, timestamp, water_u and water_v. The last two fields are a vessel reference point on the water plane in original-image pixels, not the bounding-box center. Each camera/clip/track timestamp is unique.

The AIS CSV requires MMSI, timestamp, x_m and y_m; speed_kn is optional. Timestamps include an explicit timezone. MMSI is stored as a nine-digit string. Positions are already projected into the camera's local metric frame. Duplicate MMSI timestamps must be resolved upstream.

Output frame_associations.csv contains frame assignments, spatial distances and interpolated same-time speeds. track_mmsi_matches.csv contains one row per area/camera/clip/passage/track, with predicted_MMSI, status, support count/fraction and median distance. issues.json lists rejected or conflicting records.

## Reconstruction observations

observations_json points to a JSON object with an observations array. Each member requires:

| Field | Definition |
| --- | --- |
| area, camera, clip, passage_id, track_id, frame_id | Explicit identity; area may contain spaces, other fields use filename-safe tokens |
| timestamp | Observation time, ISO 8601 |
| image | Original RGB frame path, relative to the observations JSON |
| bbox_xyxy | Vessel box in original-image pixels: x1,y1,x2,y2 |
| vessel_id | Accepted association identity |
| association_status | Must be accepted |
| moulded_depth_m | Positive vessel moulded depth, m |
| upper_reference_matches_moulded_depth | Boolean confirmation of the common vertical reference |

The camera configuration supplies a 3 by 3 image-to-water homography and a full-rank 3 by 4 projection matrix. Source images, coordinates and calibration must refer to the same pixel geometry.

The prepare command in observations mode attaches track identities and moulded depth. Its technical CSV requires MMSI and moulded_depth_m, one record per MMSI. It preserves the input boundary-reference confirmation.

Reconstruction outputs frame_freeboard.csv, passage_drafts.csv, per-observation semantics, hull masks, semantic response arrays and original-image boundary pairs. Rejected observations have explicit issue records. Passage drafts use the median of valid frame freeboards and carry their valid frame count.

The prepare command in inventory mode joins passage_drafts.csv to activity records by area, clip, passage_id and vessel_id. Activity records supply the emission fields except visual_draft_m. Conflicting existing drafts raise an error. Missing passage drafts are listed as issues.

## Evaluation inputs

The validation configuration accepts independent input groups:

- passage_emissions_csv: the emission command's passage_emissions.csv with SEIM, VSEIM and REFERENCE scenarios.
- verified_drafts_csv: one row per area/clip/passage_id/vessel_id, with seim_draft_m, visual_draft_m, verified_draft_m, reference_kind and reference_source. Accepted independent kinds are manual_verification, authority_record, operator_record and survey.
- visual_metrics_csv: frame rows with area, clip, track_id, MMSI, valid_L, valid_B, L_ref_m, L_vis_m, B_ref_m and B_vis_m. Flags accept true/1. Missing dimensions remain unevaluated.
- association_predictions_csv: one row per area/clip/passage_id/track_id, with predicted_MMSI and status.
- association_references_csv: the same track identity with reference_MMSI and reference_source.

Reference fields describe supplied evidence. They are not generated or filled from predictions. Preserve the source records separately.
