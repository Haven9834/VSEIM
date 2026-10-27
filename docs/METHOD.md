# Method implementation

## Scope and inputs

The input is a tracked vessel observation with its original image, bounding box, timestamp, camera, clip and passage identity. A passage is one vessel observed by one camera during a specified interval. Repeated MMSIs in other passages remain separate observations.

The implementation follows the manuscript formulations. Training samples, adapted model weights, measurement prompts, camera calibration and experiment settings are external inputs. AIS and camera coordinates must share a local metric reference frame. Calibration and images must use the same pixel convention, including distortion correction.

## Measurement-oriented semantic reasoning (MoVR)

The semantic module sends a vessel crop and an expanded contextual crop to a configured multimodal API. The deployment can serve a domain-adapted Qwen model. The endpoint, model identifier, prompt file and generation settings are supplied by the user.

The output schema covers vessel type, viewpoint, hull configuration, hull/upper-boundary/waterline visibility, occlusion, measurement eligibility and positive/negative grounding descriptions. A deterministic gate requires visible boundaries, a side or oblique view, and no severe or unknown occlusion. String values are not coerced into booleans.

Credentials come from the configured environment variable. Requests transmit the two crops to the specified endpoint. The package does not train the VLM or distribute its adapted weights.

## Reasoning-guided hull-side grounding (RHSG)

CLIPSeg evaluates each supplied description. Logits are resized to the vessel crop and passed through a sigmoid. Positive and negative descriptions are averaged separately:

    S_positive = mean(positive responses)
    S_negative = mean(negative responses)
    S_hull = S_positive * (1 - S_negative)

Positive SAM2 points come from high hull-response pixels. Negative points come from high exclusion-response pixels outside the positive region. Sampling is deterministic across eligible pixels in raster order. The vessel crop supplies the bounding prompt.

Candidate masks must pass the configured minimum area and model-score requirements. The selection objective is:

    semantic_weight * mean(S_hull within mask)
      + continuity_weight * largest_connected_component_area / mask_area

Continuity uses SciPy's default two-dimensional connected-component convention. This definition is explicit to permit controlled replacement.

Column-wise upper/lower mask envelopes are screened by local median residuals and robust median-absolute-deviation criteria. A configured polynomial fit supplies an additional consistency check. Retained boundary pairs are transformed back to original-image coordinates.

## CCTV–AIS association

AIS positions are interpolated only between bracketing records whose separation meets the configured time-gap limit. Extrapolation is excluded. The homography maps visual water-surface anchors to the AIS metric frame.

Each timestamp uses Euclidean spatial costs, a one-to-one assignment and unmatched dummy columns. Out-of-gate real assignments are forbidden. Track candidates must satisfy minimum match count and support fraction; the smallest median discrepancy identifies the leading candidate. A configured median-distance margin controls ambiguity.

Competing MMSIs, unmatched frames, unsupported tracks and overlapping tracks assigned to the same vessel are recorded. Downstream reconstruction requires accepted status.

## Freeboard and operating draft

A full 3 by 4 projection matrix supplies vertical geometry:

    image_point ∼ P * [X, Y, F, 1]^T

The water-plane homography supplies X and Y from the waterline. A bounded one-dimensional optimization minimizes upper-boundary reprojection error over 0 < F < moulded_depth. Reprojection residuals and within-frame median-absolute-deviation filtering define valid measurements.

The frame estimate is their median. The passage estimate is the median over valid frames, subject to a configured minimum count:

    visual_draft = moulded_depth - passage_freeboard

The observation must explicitly confirm that the upper-boundary reference is compatible with moulded depth. A planar homography alone cannot provide this vertical measurement.

## Technical specifications

The specifications.fill_missing function applies supplied ship-type regressions with predictor names, coefficients, intercept and provenance. Targets are evaluated in an explicit order to support dependencies among attributes. Existing positive measurements take priority. Missing models or predictors produce unresolved-field issues.

## Emissions and evaluation

Interval emissions precede passage aggregation. The three scenarios use shared time coverage and non-draft inputs, with the same low-load rule evaluated at each scenario's load.

Visual validation uses frame-quality flags and passage medians. Association validation uses declared reference correspondences. Independent-draft validation uses complete common passages with verification metadata. Numerical definitions are given in EMISSIONS.md and VALIDATION.md.
