# Design: a PA network trained on the measurement itself (v33, 2026-11-10)

Status: design only. Nothing is built or pushed. Pushing the GPU kernel needs the user's OK.

## Why the earlier PA networks failed

Two earlier networks gave a better fascicle direction map and still no better pennation angle:

| Attempt | Direction map on VL (error at expert lines) | Measured PA |
|---|---|---|
| Production (fasc_orient_v1) | 1.69° | VL MAE 1.80° |
| v17 insertion field | 1.22° | no change |
| v21 ConvNeXt + FALLMUD | 1.43° | slightly worse |

A third attempt, the v30 direct regressor (image → PA with no geometry), was far too weak.

The pattern: the map improves, but the hand-written code that reads one angle from the map does not benefit. That reading code is where the VL noise comes from (v32). About half of the VL error is in the reference line (deep aponeurosis), not in the fascicle direction.

So the network was never trained on the quantity we are scored on. This design does that.

## The idea

Keep the geometry explicit, but make the reading step part of the network and put the loss on the final angle.

One network, one forward pass, four outputs:

1. Fascicle direction map (cos 2θ, sin 2θ), as today.
2. Fascicle fragment map, as today.
3. Deep aponeurosis edge map (new in this network; today it comes from a separate network).
4. A "where to read" weight map (new).

A small fixed readout, with no free parameters, turns these into one number:

- fascicle angle = weighted average of output 1, using output 4 as weights;
- deep-line angle = weighted straight-line fit through output 3 under the same region;
- PA = fascicle angle − deep-line angle.

The readout is differentiable, so the PA error trains all four outputs together. The network learns where to read and which line to use, instead of our hand rules.

## Training data (competition data only)

| Loss | Images | Label |
|---|---|---|
| Dense direction + fragments | 2,091 fascicle-mask images | as fasc_orient_v1 |
| Dense deep edge | 1,048 aponeurosis-mask images | lower edge of the mask band |
| PA measurement loss | 142 images with both masks, plus about 2,000 with a fascicle mask and our aponeurosis map | PA from mask geometry |
| Consistency | all training images | PA must change exactly as geometry predicts under rotation and stretch |

No external data, no test images, no expert benchmark images. VL, static and dynamic stay untouched as yardsticks.

Same encoder (ConvNeXt-small), same 5 folds and same near-duplicate grouping as fasc_orient_v1, so results compare one to one.

## Gates, fixed before the kernel runs

The new PA replaces the production PA head per image, with each scanner group's mean held at its Final A level. So the adopted LB constants stay valid, and only the per-image ordering changes.

| Gate | Set | Pass if |
|---|---|---|
| G0 | Training folds (out-of-fold) | PA MAE below production's on the 142 both-mask images |
| G1 | VL (subject bootstrap) | PA MAE change, upper 90% bound < 0 |
| G2 | VL | slope of predicted on expert angle rises by ≥ 0.10 |
| G3 | Static clean10 | no harm: upper 90% bound ≤ +0.10° |
| G4 | Dynamic | no harm: upper 90% bound ≤ +0.10° |
| G5 | Fascicle length through the new PA, VL and dynamic | no harm |

- All pass: build one candidate (Siemens and Lumify rows first), show the user, one submission.
- G1 fails but error correlation with production is below 0.9 at similar accuracy: test a fixed 50/50 blend under the same gates.
- Otherwise: closed, no rerun.

## Honest size of the prize

- Ceiling from our own logs: reading a good map exactly where the expert drew gives 1.43° on VL against 1.80° today. That is 20% less PA error on VL-like images at best.
- Translated to the score: about 0.002–0.006 if it transfers to the Siemens test rows. Worth 1–4 rank places from rank 29.
- Chance that all gates pass: about 10–15% [my estimate]. The last three PA networks all failed at G1.
- This alone does not reach top 3 (0.177). The bigger error pool is Telemed fascicle length (0.06), where we have no labelled yardstick.

## Cost and schedule

| Step | Where | Time |
|---|---|---|
| Write trainer from `kaggle/fasc_orient_v1/train.py` + readout + gates file | laptop, light | half a day |
| CPU smoke kernel | Kaggle CPU | 15 min |
| Full training, 5 folds | Kaggle 2×T4 | about 3.5 h |
| Gate evaluation | laptop, light | 1–2 h |

Results would be in on 11-11, leaving 11-12 for a submission and 11-13 for final selection.

## Risks

1. Mask-geometry PA is an annotator's mask, not an expert's angle. The level may differ; holding group means fixed covers that, but not a per-image difference in convention.
2. Only 142 images have both masks. The other 2,000 PA labels lean on our own aponeurosis map, so the network could partly learn to copy production.
3. Training images hold no Lumify and few Siemens VL images. The gain must transfer from VL (Esaote) evidence to Siemens test rows.
