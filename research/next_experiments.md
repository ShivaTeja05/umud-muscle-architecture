# UMUD Competitive Experiment Plan

> Historical early-stage plan. The verified baseline is now C18 at 0.33526.
> See [the 22 October review](review_2026-10-22.md) for completed experiments,
> limitations, new diagnostics and the next GPU pilot.

## Evidence So Far

- Public leaderboard: `0.92533`; lower is better. The current leaders are near
  `0.287`, so the submitted pipeline is not competitive.
- The old `0.71724` mask Dice was inflated by exact duplicate leakage. The
  fascicle set has 2,761 pairs but only 2,091 unique image-mask pairs.
- The difficult Telemed/video group is 140 of 309 test images. Correct scale and
  vertical MT reduce its raw means from 132.29 to 112.23 mm FL and from 29.08 to
  24.98 mm MT, but PA remains too low at 12.36 degrees.
- The 35-image static benchmark is too small and protocol-shifted to support a
  transferable linear calibration. Uncalibrated error is the primary metric.

## Ground-Truth-Matched Outputs

The competition labels average measurements from two raters. Each rater uses:

- MT: three global-vertical lines at left, middle, and right, then their mean.
- FL: three visible fascicles, extended to both aponeuroses with as little
  extrapolation as possible, then their mean.
- PA: three visible fascicle insertion angles, not necessarily the same
  fascicles selected for FL, then their mean.

The model should therefore predict anatomy, not one coupled PA/FL triangle.

## Experiment Order

### E1: Measurement And Scale Baseline

- Decode scale bars and crop the true ultrasound viewport for every scanner.
- Use three vertical MT samples.
- Select three distinct long fascicle fragments and reject high-extrapolation
  candidates.
- Keep benchmark-derived calibration disabled.
- Gate: all 309 rows valid, official bounds respected, stable five-frame traces.

### E2: Aponeurosis First

Train the prepared `kaggle/aponeurosis_unetpp_384` kernel before another
fascicle run. MT has the strongest metric sensitivity, and the current
aponeurosis model is the older 256 px custom U-Net.

- U-Net++/ResNet-34 with ImageNet initialization.
- ROI-normalized input, full aspect-aware geometry after inverse mapping.
- BCE plus Dice initially; compare boundary or distance-transform loss next.
- Validate curve position and MT error in pixels, not Dice alone.
- Gate: improve scanner-stratified MT error without degrading failure rate.

### E3: Leakage-Free Fascicle Model

- Train only the 2,091 unique pairs; store the exact train/validation manifest.
- Keep duplicate source images in one split.
- Normalize viewports and add scanner-style intensity, speckle, blur, and gain
  augmentation.
- Compare U-Net++, DeepLabV3+, and SegFormer only after the validation protocol
  is stable. Favor an ensemble only when fold errors are complementary.
- Add a second orientation/distance-field head so fragmented visible labels do
  not force the network to hallucinate full fascicle masks.

### E4: Public External Pretraining

Use DL_Track_US data for pretraining only after recording its exact archive,
checksum, license, and label convention. Fine-tune on deduplicated UMUD data.
Keep the UMUD static benchmark locked for final model comparison and the dynamic
benchmark for temporal tracking evaluation. Do not fit calibration on either
set and report the resulting score on the same records.

### E5: Sequence Model

- Detect contiguous five-frame clips from IDs and scanner metadata.
- Track aponeurosis curves and fascicle orientation with optical flow or a
  Kalman smoother.
- Smooth latent geometry rather than final clipped values.
- Reject a sequence model unless it improves held-out dynamic MAE and preserves
  real motion amplitude.

## Validation Gates

Every experiment must report:

- Duplicate-free mask Dice/IoU by scanner or source group.
- Aponeurosis vertical-position MAE and derived MT MAE.
- PA, FL, and MT MAE separately using tolerances 6 degrees, 12 mm, and 3 mm.
- Static and dynamic benchmark results with no fitted calibration.
- Prediction distributions and clipping counts by scanner family.
- Checkpoint SHA-256, package versions, seed, split manifest, and data sources.

Only submit when a change improves the predefined validation gates. Public
leaderboard blending, copying public prediction tables, and manual test labels
are excluded from the official pipeline.
