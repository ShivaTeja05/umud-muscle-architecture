# Curved-track GPU pilot — 22 October 2026

The paired fold-0 pilot completed on the supplied A100 80 GB. Adding expert VL
curve supervision improved held-out track and deep-strip direction errors, but
did **not** improve the PA proxy. This supports further diagnosis and conditional
second-fold confirmation, not replacing the submission model.

**Best verified Kaggle score remains C18: 0.33526. Target: 0.20.** No submission
was made. These angle diagnostics cannot establish a competition-score gain.

## Selected checkpoints

Both arms selected epoch 3 using the existing GT validation rule. Epoch 4 was
not substituted even where its VL results looked better.

| Metric | Control | Curves | Curves minus control |
|---|---:|---:|---:|
| GT angle MAE (°), 409 images | 1.42404 | 1.43270 | +0.00866 |
| GT Dice at 0.5 | 0.641169 | 0.641296 | +0.000127 |
| GT median-angle error (°) | 0.69342 | 0.72497 | +0.03156 |
| VL track angle MAE (°), 50 images | 1.43809 | 1.34943 | −0.08866 (6.17% lower) |
| VL deep-strip angle MAE (°) | 1.72070 | 1.55811 | −0.16260 (9.45% lower) |
| VL PA proxy MAE (°) | 1.65879 | 1.70810 | +0.04931 (2.97% higher) |
| VL PA signed bias (°) | +0.16092 | +0.36270 | +0.20178 |
| GT angle MAE with flip TTA (°) | 1.35696 | 1.35661 | −0.00034 |

TTA GT performance is essentially unchanged. Complete fascicle length and
independently measured muscle thickness were not evaluated by this pilot.

## Subject-level follow-up

Re-evaluated the saved checkpoints with the original batch-8, non-TTA evaluator
and extracted per-image results. Ordered raw image, crop, canvas and target
hashes match across arms. All 50 images have accepted track, strip and PA values.
There are 25 held-out subjects, with no training-subject overlap.

| Re-evaluated metric | Curves minus control (°) | Descriptive subject-resampling interval (°) | Subjects with lower error |
|---|---:|---:|---:|
| Track MAE | −0.088607 | [−0.121723, −0.057049] | 19/25 |
| Deep-strip MAE | −0.162479 | [−0.254902, −0.076168] | 20/25 |
| PA proxy MAE | +0.049330 | [−0.032359, +0.131174] | 11/25 |

These use 10,000 paired subject-cluster resamples, seed 20261022, preserving
image weighting. Equal-subject estimates give the same directions. Leaving out
any one subject retains the direction of each mean change. PA signed bias
increased for 23/25 subjects. These are exploratory descriptions of one fold,
not independent-fold confirmation or a statistical-significance claim. The
points and intervals in this table both come from the re-evaluation.

## Verification and numerical audit

- Original v1 fold-0 warm checkpoint SHA-256:
  `6d82cc133e4ef610b754720aa82d57a719bc53f06a6ec5c871131c79718ffdbe`.
  Both arms used this checkpoint, with no non-finite tensors or BN repair.
- GT fold manifests match exactly. Warm angle MAEs were 1.511663° and
  1.511655°, against the original reference 1.510992°. Both original warm guards
  passed. Both selected fp32 checkpoints passed save/reload checks without
  recalibration.
- The prepared source differs only in `VL_SUP` and `MODEL_NAME`. Both arms use
  no pseudo-labels and the same 309-image test-only style-bank policy. All
  245/245 embedded VL records matched; curves trained on 195 images from 99
  subjects and held out 50 images from 25 other subjects.
- The original strict comparison and the follow-up evaluator **failed** their
  0.00001° numerical agreement thresholds. Those failures are preserved.
  Three fresh-process warm VL evaluations showed a maximum aggregate range of
  0.000131964°; selected-checkpoint re-evaluation differed from saved values by
  at most 0.000058955°. Checkpoint, evaluation code, runtime flags and input
  hashes match. A separate, explicit **post-run engineering allowance of
  0.0002°** was therefore used for the audited reports. Dice tolerance stays
  0.00001. This changes no learning gate or selected epoch. GT was checked by
  source/split identity and its original guards, not freshly repeated.
- All 43 project tests pass locally and on the pod. CUDA forward/backward and
  fp32 reload smoke passed after fixing allocator initialization and allowing
  standard AMP gradient-scale backoff. Initial smoke failures remain saved.
- All 91 arm-output files, totaling 1,014,702,477 bytes, were retrieved and
  verified against remote SHA-256 checksums. This includes both checkpoints,
  prediction maps, metrics, configurations, logs and visualizations.

Each arm ran four epochs. Control used 204 optimizer steps per epoch; curves
used 272, adding 545 sampled supervised items per epoch. This is one stochastic
run per arm with different augmentation draws and more updates for curves, not
a compute-matched causal isolation of labels alone. VL is subject-held-out in a
supervised scanner domain, not an unseen-scanner benchmark.

Control took 7.35 minutes and curves 7.28 minutes, excluding setup, smoke and
follow-up diagnostics. Highest logged training allocation was approximately
6.1 GiB at batch 8; this is not total device memory. Although some inherited
configuration descriptions mention a five-fold ensemble, these outputs contain
**only verified fold 0**, with flip TTA where stated.

## Next work

1. Diagnose the PA change with a frozen 2×2 comparison: control/curves direction
   fields crossed with control/curves confidence weights, keeping expert strips,
   deep lines, thresholds and selected epochs fixed. The PA estimator effectively
   weights unit directions by segmentation probability × orientation magnitude²;
   improved unweighted track error need not improve that weighted estimate.
   Persisted maps are native-resolution, quantized TTA outputs, so this is a
   secondary mechanism diagnostic, not reproduction of the non-TTA table above.
   Do not fit an offset on these 25 subjects.
2. If the direction gains remain useful, confirm the same frozen paired setup
   on another subject fold before considering all-five-fold training. No
   additional folds were launched in this pilot.
3. Evaluate downstream PA and FL with fixed geometry and aponeurosis maps, then
   independent expert measurements for MT. Derived VL MT is not independent
   expert ground truth. Adoption requires measurement gains, not only angle loss.

The initial two-hour allowance was not expanded. All model execution occurred
on the pod. Kaggle inputs were transferred using existing local authentication
and scoped download links; no Kaggle account token was copied to the pod.

## Saved evidence

Local outputs: `artifacts/runpod_curve_v3_20261022/`.

- `comparison_audited/PILOT_REPORT.md` and `comparison.csv`: original saved metrics.
- `audited_analysis/SUBJECT_ANALYSIS.md` and `AUDITED_ANALYSIS.json`: numerical
  audit and re-evaluated subject analysis, including provenance and sensitivities.
- `vl_diagnostics/`: strict failure, per-image data and all three warm repeats.
- `ARM_OUTPUTS_SHA256.json` and `BACKUP_VERIFICATION.json`: output integrity.
- `control/`, `curves/`, environment records and process logs: original evidence.

Cloud compute finishing does not stop pod billing. After final retrieval and
verification, the pod can be stopped in Runpod; do not confuse a stopped
training process with a stopped pod.
