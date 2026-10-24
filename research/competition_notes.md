# UMUD Competition Notes

## What Kaggle Is Asking

For every image in the hidden test set, submit a CSV with:

- `image_id`
- `pa_deg`
- `fl_mm`
- `mt_mm`

The model should estimate:

- Pennation angle: angle between a fascicle and the deep aponeurosis.
- Fascicle length: distance between aponeuroses along the fascicle direction,
  sometimes requiring extrapolation outside the visible frame.
- Muscle thickness: mean of three global-vertical distances between superficial
  and deep aponeuroses at the left, middle, and right of the image.

The provided training data is not a normal tabular target file. It consists of
ultrasound images paired with aponeurosis masks and fascicle masks. That makes
this a segmentation plus geometry problem more than a pure image regression
problem.

## Why Lower Score Is Better

The competition score is based on normalized absolute prediction error. Errors
for PA, FL, and MT are normalized by tolerance values so that degrees and
millimetres contribute on comparable scales. Because it is an error score, lower
means closer to the hidden expert labels. A perfect score would be 0.

Community EDA logs suggest the tolerances used by the metric are approximately:

- PA: 6 degrees
- FL: 12 mm
- MT: 3 mm

That means MT is expensive: 1 mm MT error hurts about as much as 2 degrees PA or
4 mm FL error.

## Eligibility And Prize Notes

The competition is open to classic computer vision and deep learning methods.
Public pretrained models and public datasets are allowed when documented.

For top-3 official/prize eligibility, Kaggle notes an extra reproducibility
check after the private leaderboard:

- Public code repository.
- OSI-compatible open-source license.
- FAIR-style software metadata/documentation.
- Reproducible execution through a notebook, script, package, or installer.
- Clear dependency and inference instructions.

Code without a license is explicitly risky for prize eligibility.

## Research Direction

Most relevant references:

1. Ritsche et al., "Fully Automated Analysis of Muscle Architecture from B-Mode
   Ultrasound Images with DL_Track_US", Ultrasound in Medicine & Biology, 2024.
   They trained U-Net and VGG16-encoder U-Net models for fascicle/aponeurosis
   segmentation, then computed MT/PA/FL. Reported differences versus manual
   analysis were small enough to be a strong baseline reference.
2. DL_Track_US software repository. Useful for implementation details and for
   comparing our geometry extraction pipeline.
3. Katakis et al., "Automatic Extraction of Muscle Parameters with Attention
   UNet in Ultrasonography", Sensors, 2022. Attention U-Net for aponeurosis
   segmentation plus post-processing; fascicle angle from filtering/clustering.
4. TimTrack / SMA-style classical pipelines. Useful for robust line-orientation
   estimation using filtering, Hough/Radon-style transforms, and outlier
   rejection.
5. MSF-Net and residual-shrinkage U-Net papers. These suggest stronger
   segmentation backbones if a plain U-Net underfits the noisy fascicle masks.

## Modeling Plan

### Baseline 0: sample submission sanity

Submit only to verify formatting if needed. Not competitive.

### Baseline 1: classical geometry from image intensity

Detect bright aponeurosis bands with edge/line fitting and estimate fascicle
orientation using steerable/Gabor or Hough transforms. This is fast and gives us
interpretable failure cases.

### Baseline 2: train segmentation models

Train two models:

- Aponeurosis model: U-Net / U-Net++ / segmentation-models-pytorch encoder.
- Fascicle model: same family, but loss weighted for thin line structures.

Recommended losses:

- BCE + Dice/Tversky for aponeuroses.
- BCE + clDice/skeleton-aware or focal/Tversky for fascicles.

Recommended augmentation:

- Ultrasound intensity changes: gamma, CLAHE-like contrast, speckle/noise.
- Small rotations and affine transforms.
- Horizontal flips only when the downstream geometry handles orientation.
- Crops/resizes preserving aspect ratio.

### Baseline 3: geometry extraction

From predicted masks:

- Fit superficial and deep aponeurosis lines/curves.
- Estimate MT as the mean global-vertical distance at left, middle, and right
  positions, matching the competition ground-truth procedure.
- Fit multiple fascicle fragments and take a robust dominant orientation.
- PA is the angle between fascicle orientation and deep aponeurosis.
- FL can start as `MT / sin(PA)` and later use direct line intersections with
  the fitted aponeuroses.

### Competitive Iterations

- Test-time augmentation and ensemble multiple segmentation backbones.
- Sequence-aware smoothing for the 5-frame test groups.
- Predict uncertainty; reject outlier fascicles before computing medians.
- Calibrate output ranges using public training-mask-derived geometry and
  leaderboard feedback.
- Investigate DL_Track_US pretrained models as a benchmark, while documenting
  their public source for eligibility.

## Current Experiment Result

The CLI-managed Kaggle experiment `shivateja005/umud-u-net-fascicle-384-cli`
trained a U-Net++ model with a ResNet-34 encoder at 384 px on two Tesla T4 GPUs.
The final and best epoch reached mask Dice `0.71724`, compared with `0.68266`
for the earlier 256 px custom U-Net fascicle model.

On the 35-image static benchmark with six expert annotations:

- Best fascicle probability threshold: `0.95`.
- Raw normalized error: `0.5803` before linear calibration.
- Leave-one-out calibrated normalized error: `0.27290` (over-optimistic because
  calibration and evaluation reuse the same small domain).
- Normalized components: PA `0.21883`, FL `0.37017`, MT `0.22969`.
- Previous pipeline leave-one-out score: `0.30402`.

On the 167-frame dynamic benchmark, smoothed predictions correlated strongly
with the expert trajectories (PA `r=0.919`, FL `r=0.965`). The absolute FL
offset was fitted in pixel units and converted by per-image pixel spacing during
post-processing. Dynamic PA remains uncalibrated because one exercise sequence
does not establish a transferable offset across scanners and muscles.

The generated candidate
`artifacts/submission_unetpp384_t095_calibrated.csv` scored `0.92533` on the
public leaderboard. Since lower is better, this is substantially worse than the
roughly `0.287` leaders and must not be treated as a competitive model. Its main
failure is transfer to the 140 video frames, compounded by incorrect scale and
measurement-protocol assumptions. The old random mask split also leaked 670
exact duplicate fascicle image-mask pairs, inflating its Dice estimate.

### Submission format caveat

The competition-provided `sample_submission.csv` is semicolon-delimited, but
Kaggle's submission parser expects comma-delimited columns. A semicolon file is
accepted for upload and then fails with `ID column image_id not found in
submission`. A successfully scored public competition notebook confirms the
working header is `image_id,pa_deg,fl_mm,mt_mm`. The project writers and
validator therefore enforce comma-separated output with CRLF line endings.
