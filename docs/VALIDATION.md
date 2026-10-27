# Verification scope

## Automated checks

The suite contains 66 tests covering numerical kernels, analytical camera geometry, semantic parsing, matching, data joins, visualization and command-line workflows.

Known projection examples recover a prescribed freeboard. Matching tests cover interpolation, time gaps, one-to-one assignment, ambiguous identities and overlapping tracks. Statistical tests cover passage medians, reference denominators, signed bias, R² and Holm adjustment. Emission tests cover energy units, interval duration, zero speed, shared intervals, low-load interpolation, draft aliases and conflicting inputs.

A reconstruction integration test injects deterministic semantic, grounding and segmentation backends. The API transport test uses a mocked response. Neither contacts a model service.

Fixtures are generated during testing and contain no study observations or benchmark records.

## External integration

Actual Qwen API, CLIPSeg and SAM2 execution requires a configured deployment and compatible weights. Live external model execution and GPU inference are not certified by these local software tests. API implementations may differ in their multimodal payload support.

Camera calibration, domain-adapted model accuracy and verification-record reliability require study-level assessment. The CI file declares a Windows/Linux Python matrix; local testing does not establish results of a future remote run.

## Evaluation conventions

Length and breadth use their own valid frame flags. Frame medians are grouped by area/clip/track/MMSI. Each dimension reports its own N.

Association accuracy uses correct referenced tracks divided by referenced tracks. Missing labels are unevaluated; an unavailable or rejected prediction with an available reference is incorrect. Duplicate track identities raise an error.

Independent emission validation uses common passages with all six pollutants and all three scenarios. Incomplete passages and zero benchmark denominators are reported. MAE, RMSE, MAPE and R² use passage-level samples. NRMSE normalizes RMSE by the mean benchmark emission. Aggregate bias compares emission sums. Two-sided paired Wilcoxon tests compare absolute errors, with Holm adjustment across pollutants.

Reference-source labels are provenance declarations. Supporting documents should be maintained by the study owner.
