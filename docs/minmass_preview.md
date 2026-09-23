# Manual minmass preview

Open an input movie in **Preview & ROI**, then enable **Detection preview**.
This activates manual minmass (Auto minmass is turned off) and switches to a raw
frame. Adjust the slider in steps of 0.01 or type an exact value in the numeric
field. Scrub through multiple frames to check the threshold across the recording.

The preview calls the same adaptive preprocessing/localization entry point as
the worker, with the selected detector, background method/radius, diameter and
minmass. It no longer substitutes Trackpy for other detectors or runs detection
on maximum projections or companion images. CZI raw frames use the selected
channel, with the same channel clamping as the production loader.

Every displayed candidate has passed the detector's minmass test and its
internal duplicate suppression. Spots below minmass are not shown. Colours mean:

- **Green:** passes detection, raw contrast filtering and the evaluated ROI.
- **Orange:** rejected by raw contrast filtering, including unavailable contrast
  at an edge or where local noise cannot be measured.
- **Red:** passes raw contrast but is outside the evaluated ROI.
- **Blue:** passes detection/raw contrast, but ROI acceptance has not been evaluated.

Click near a spot to see its measured mass, raw CNR, ROI membership and decision.
Use **Ctrl-click** in polygon-drawing mode so ordinary clicks still draw vertices.
Contrast rejection takes precedence in the colour when both contrast and ROI
would reject a candidate; the inspector still shows ROI membership separately.

Changing the frame, threshold, analysis settings or polygon clears the old
spots/counts immediately and marks the result outdated until it refreshes.
A refresh button is also available. Zero detections is a valid fresh result;
a failed preview instead says it is unavailable.

## What the preview can promise

Polygon, ImageJ and sister-image ROIs use the same mask construction/filtering
rules as analysis. Changes made in the editor must be saved with **Save ROI** to
apply to a run. Multiple polygons are previewed as a union; separate ROI
replicate outputs apply each polygon individually.

Intensity-threshold ROIs require the production projection, which is built from
the processed movie. A sampled viewer projection is not equivalent. Detection
preview therefore shows these candidates in blue, labels ROI acceptance as
uncomputed and hides the potentially misleading intensity-mask overlay. Missing
or invalid requested ROIs also show blue candidates and explain what must be
fixed before running.

A single-frame result does **not** predict final track retention. Linking,
minimum track length and downstream filtering may still remove detections. This
update does not implement a tracking-window preview. Completed-run viewers do
not offer new detection predictions using unrelated current settings; open the
raw input instead.

Mass is measured after preprocessing and per-frame normalization. Equal minmass
values do not imply equal raw brightness or signal quality between frames or
conditions. Use raw contrast and several representative frames alongside minmass.

The exact-plane reader supports ordinary CZI planes and planar TIFF frames,
including memory-mappable contiguous TIFF blocks. It fails explicitly on
unsupported/ambiguous layouts instead of silently showing another plane. Frame
numbers in this viewer refer to the selected file, not an assembled split series.

## Verification

Regression tests compare preview and production coordinates/masses and
contrast/ROI decisions for Trackpy and Torch CPU, in fast and streaming memory
paths. Additional checks cover no detections, unknown ROIs, invalidation,
selected-channel CZI reads, TIFF plane selection and an actual rendered QML
preview with spot inspection. GPU parity is not separately certified by these
CPU tests; the preview still dispatches through the same backend selection code.
