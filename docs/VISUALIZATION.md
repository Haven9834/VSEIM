# Visualization methods

## Coordinates

`coordinates.space` must be either `original` or `crop_resize`.

For a source image W × H, a detection-space crop `[t,b,l,r]`, and resized detection dimensions w × h:

```text
x_original = x_detection × (W − l − r) / w + l
y_original = y_detection × (H − t − b) / h + t
```

Cropping an original high-resolution image and then resizing is not interchangeable with resizing first. The configuration must describe the operation that actually produced the detections. No homography, intrinsic matrix or physical camera geometry is inferred by this module.

`display_crop` is a later display-only operation shared by images and overlays. It may remove top/bottom subtitles without changing the source pixels or model inputs. Resizing preserves aspect ratio; blank space is used when needed.

## Association cards

All detection boxes use the same blue. Badges show the actual track identifier, not sequential display numbering. Candidate badge locations minimize intersection with vessel boxes and already-placed badges; dense scenes may still require manual review.

Every vessel card contains two rows: MMSI with Track ID, followed by Speed with Name. The two rows use aligned columns. Names are shortened before fitting text. A minimum readable font check fails instead of silently drawing overflowing text. Rendering does not prove the underlying match is correct.

## Five columns

1. Original RGB image.
2. Supplied whole-vessel binary mask, not a filled detection rectangle or a hull-only substitute.
3. Saved CLIPSeg probability display: frame-level min-max normalization, jet colormap, 78% heat and 22% source RGB.
4. Upper/lower per-column envelopes of each supplied hull polygon, green/red dashed lines.
5. Union of supplied hull polygons, blue overlay at 78% opacity. Pixels outside the union remain unchanged before common display cropping.

The heatmap displays the supplied response array. Per-frame normalization means colors are not directly comparable as calibrated confidence across cameras. Existing-result rendering and model inference are separate commands with separate input contracts.

Mask envelopes are image-space contours. The lower envelope must not be called an independently measured physical waterline. Missing hulls remain unmarked even when the whole vessel is visible in column two.

## Portability choices

The rendering commands consume saved masks/probabilities. The reconstruction command runs the configured inference adapters and exports mask/response arrays plus boundary correspondences. Weights and runtime libraries are installed separately. See DATA_SCHEMA.md for each input contract.

Editable presentation builders tied to local presentation runtimes and hand-edited templates are not included. The generated PNG/JPEG panels can be inserted into a presentation, but the package does not claim those raster annotations are separately editable PowerPoint shapes.
