# VSEIM implementation and usage

Research implementation of visual operating-draft reconstruction and its integration with AIS-based main-engine emission inventories.

The workflow combines measurement-oriented vessel semantics, text-conditioned hull grounding, structural segmentation, camera geometry, CCTV–AIS identity association and interval emission calculation. CCTV footage, AIS records, model weights and site-specific configurations are provided separately by the user.

## Workflow

    Tracked CCTV observations + synchronized AIS
                        |
             Water-plane track association
                        |
              Vessel technical records
                        |
          Qwen-compatible multimodal API
                        |
          CLIPSeg positive/negative grounding
                        |
          SAM2 segmentation and boundary filtering
                        |
          3D freeboard recovery → operating draft
                        |
          SEIM / VSEIM / supplied-reference emissions
                        |
          Passage / vessel-type / area summaries

Detection and tracking results are inputs. The package does not train a vessel detector or distribute a domain-adapted VLM checkpoint.

## Modules

| Component | Implementation |
| --- | --- |
| Measurement-oriented semantic reasoning (MoVR) | src/vesselviz/semantic.py |
| Hull-side grounding and boundary refinement (RHSG) | src/vesselviz/grounding.py |
| CCTV–AIS matching | src/vesselviz/matching.py |
| Camera projection, freeboard and draft | src/vesselviz/metric.py |
| Observation reconstruction workflow | src/vesselviz/reconstruction.py |
| Technical-attribute regression | src/vesselviz/specifications.py |
| Interval emission model | src/vesselviz/emissions.py |
| Explicit data joins and inventory summaries | src/vesselviz/inventory.py |
| Measurement, association and benchmark validation | src/vesselviz/evaluation.py |
| Association cards, hull panels and image selection | src/vesselviz/association.py, hull.py, render.py, gallery.py |

The reference implementation follows the manuscript formulations. Its numerical definitions and release coverage are described in [Method implementation](METHOD.md) and [Validation](VALIDATION.md).

## Installation

Python 3.10 or newer is required.

~~~bash
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install .
python -m vesselviz --help
~~~

The installed commands vesselviz and vseim are equivalent. For local CLIPSeg inference, install the optional dependencies:

~~~bash
python -m pip install ".[vision]"
~~~

Install SAM2 separately using its [official installation instructions](https://github.com/facebookresearch/sam2). Supply a compatible architecture configuration and checkpoint. The [CLIPSeg adapter](https://huggingface.co/docs/transformers/model_doc/clipseg) uses the Transformers processor and image-segmentation model classes.

The default CLIPSeg configuration only loads local files. Model downloading must be enabled explicitly by the user. No inference dependency is needed for existing-result visualization or emission calculation.

## Configuration

Copy each required example in configs/ to a corresponding .local.json file.

- The multimodal API endpoint, model identifier, prompt file and generation settings are empty.
- API credentials are read from the environment variable named by api_key_env.
- Camera matrices, matching gates, reconstruction thresholds, emission coefficients and correction tables require local configuration.
- Render sizes, colors, CPU device selection and request timeout are general software defaults.

Configuration examples describe the input contract; they are not the study's experiment settings. Blank scientific settings must be completed before the relevant calculation can run.

## Run the research workflow

All paths below are examples relative to the repository.

~~~bash
python -m vesselviz match --config configs/matching.local.json --output results/matching
python -m vesselviz prepare --config configs/prepare_observations.local.json --output results/linked_observations
python -m vesselviz reconstruct --config configs/reconstruction.local.json --output results/reconstruction
python -m vesselviz prepare --config configs/prepare_inventory.local.json --output results/inventory_inputs
python -m vesselviz emissions --config configs/emissions.local.json --output results/emissions
python -m vesselviz validate --config configs/validation.local.json --output results/validation
~~~

Each stage consumes the documented output of the preceding stage. Update the next configuration with the generated file path. Reconstruction can also consume already associated observations directly. Each run requires a new output directory.

Matching uses AIS positions already expressed in the same local metric reference frame as the camera homography. Metric reconstruction requires a separate 3D camera projection matrix. A water-plane homography alone cannot determine vessel freeboard.

## Figures and selection

~~~bash
python -m vesselviz association --config configs/association.local.json --output results/association_figures
python -m vesselviz hull --config configs/hull.local.json --output results/hull_figures
python -m vesselviz pick --kind association --config configs/association.local.json --output results/selected_frame
~~~

Open the generated gallery.html to select frames, export selection.json, then assemble a camera grid:

~~~bash
python -m vesselviz assemble --index results/association_figures/index.json --selection selection.json --output results/selected_grid
~~~

Association cards show masked MMSI, the actual Track ID, speed and an abbreviated English name. The hull renderer consumes exact-frame full-size masks and polygon results; the reconstruction workflow exports masks, semantic responses and fitted-boundary correspondences separately. See [Visualization](VISUALIZATION.md) for the distinction.

## Documentation

- [中文使用指南](QUICKSTART_ZH.md)
- [Input schemas](DATA_SCHEMA.md)
- [Method implementation](METHOD.md)
- [Emission equations and units](EMISSIONS.md)
- [Visualization conventions](VISUALIZATION.md)
- [Validation coverage](VALIDATION.md)

## Verification

~~~bash
python -m unittest discover -s tests -v
python scripts/check_release.py
~~~

Tests use generated software fixtures and analytical geometric examples. They do not contain the study dataset or independent benchmark records. External model and API execution requires a configured deployment.

Model weights and third-party libraries retain their own licenses. Authorship, publication metadata and repository licensing should follow the accompanying publication and institutional requirements.

