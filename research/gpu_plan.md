# GPU plan: next training round (prepared 2026-10-21; nothing pushed to GPU)

> Historical plan. These v2/v3 runs subsequently completed on Runpod; see
> `artifacts/candidates_v6/README.md` for their results. The current plan is
> [review_2026-10-22.md](review_2026-10-22.md), with a prepared paired curvature pilot.

The Kaggle GPU quota (about 30 h a week, T4 x2, counted as session wall time) is used up until the weekly reset,
around Sat/Sun 00:00 UTC. Two kernels are ready to push once it resets. Both are private, and both have passed a
CPU-only Kaggle smoke test (see "Smoke tests" below).

| kernel | dir | what it produces | estimated GPU-h (wall) |
|---|---|---|---|
| `shivateja005/umud-fasc-orient-v2` | `kaggle/fasc_orient_v2/` | fascicle seg + orientation field, self-trained on the test-domain images, plus a multi-scale variant | **1.7 h** (cap 3.5 h) |
| `shivateja005/umud-apo-v3` | `kaggle/apo_v3/` | repaired apo_v2 with 5-fold flip TTA (`apo_v2r`), then a self-trained fp32 student (`apo_v3`) | **1.5 h** (teacher-only variant 0.3 h; cap 3.3 h) |

Both kernels follow the same rules:
- **Warm start, same folds.** Each kernel warm-starts fold k from the previous model's fold-k checkpoint and reuses
  that model's `folds.csv`. The train OOF maps stay out-of-fold, and the benchmark numbers stay comparable with v1/v2.
- **Safe checkpoints.** Checkpoints are saved in **fp32**. Each one is reloaded from disk, checked for inf/nan and
  re-validated against the metric it was selected on before any maskbank is written. A failed check triggers a
  BatchNorm recalibration on that fold's training split; a fold that still fails is dropped from the ensemble.
- **Self-training without hand labels.** Self-training uses only model pseudo-labels on unlabeled images: test 309,
  static 35, every 4th dynamic frame, and the 249 raw VL `.tiff` images. No test image is hand-labelled.
- **Output format.** Outputs are maskbank npz files (uint8, native resolution), with the same keys and encoding as
  the models they replace, plus `manifest.csv`.

## Priority order and budget for the reset week (~30 GPU-h)

| # | run | GPU-h | expected benefit (to be confirmed on the benchmarks only) | gate to continue |
|---|---|---|---|---|
| 1 | **fasc_orient_v2** (A: pseudo + domain aug + multi-scale; VL used only as validation) | 1.7 | PA is the noisiest test quantity; C2's shrinkage gain (0.467 → 0.390 LB) shows per-image PA/FL noise on the test scanners. The field PA already beat fusion PA on every benchmark (clean10 0.097 → 0.081, VL PA 0.334 → 0.294). v2 adapts the field to the test scanners and adds zoom views for Lumify. Plausible gain 0.005–0.02 on the score; no benchmark evidence yet | per-epoch VL track / strip / PA-proxy error (logged) must not get worse than epoch 0 (= v1); then the local benchmarks (below) |
| 2 | **apo_v3** (teacher + student) | 1.5 | **Teacher (`apo_v2r`):** removes the CPU-fix compromises (no TTA; 3 folds on VL/dynamic; VL raw only). Small, near-certain gain on MT and on the apo curves that feed PA/FL. **Student:** test-domain adaptation of the band/edge channels. MT is worth 0.111 per mm | teacher folds must reproduce the logged apo_v2 fold metrics (checked in-kernel); student kept only if the local MT benchmarks improve |
| 3 | fasc_orient_v2 **VL_SUP** variant (expert VL lines as targets) | 1.7 | Direct supervision of the insertion region, where v1 errs most on VL: 2.2° median error within 4 % of MT of the aponeuroses vs 0.9° mid-muscle. Caveat: VL becomes in-domain (subject-fold OOF), so it stops being an unseen-scanner proxy | run only after #1 is evaluated; judge on clean10 / dynamic / train142, with VL reported separately and flagged |
| 4 | round-2 self-training (`fasc_orient_v2r2`, teacher = v2 multi-scale maps) | 1.7 | Usual second-round noisy-student gain, smaller than round 1 | only if #1 beats v1 on the benchmarks |
| 5 | apo_v3 round 2 / longer schedule | 1.5 | small | only if #2's student beats its teacher |
| – | reserve | ~20 | re-runs after a failure; whole-fascicle / FL ideas; the Telemed FL lever | |

Runs #1 and #2 are independent and can go at the same time; Kaggle allows 2 GPU sessions. Total for #1–#2 is
about 3.2 GPU-h, and about 8 GPU-h if #1–#5 all run. Quota weeks left before the 2026-11-14 deadline: about 3
(resets around Oct 25, Nov 1, Nov 8).

## Exact commands (run from the project root)

```bash
cd /path/to/repo   # repository root
K=".venv/bin/kaggle"

# --- push (after the quota reset; retry later if Kaggle says too many GPU sessions are running)
$K kernels push -p kaggle/fasc_orient_v2 --accelerator NvidiaTeslaT4
$K kernels push -p kaggle/apo_v3 --accelerator NvidiaTeslaT4

# --- follow (plain `kernels logs` prints nothing while a kernel runs; use -f)
$K kernels status shivateja005/umud-fasc-orient-v2
$K kernels logs -f shivateja005/umud-fasc-orient-v2 | grep -E "EPOCH|epoch 0|reload check|TTA|FAILED|WARNING|DONE"
$K kernels status shivateja005/umud-apo-v3
$K kernels logs -f shivateja005/umud-apo-v3 | grep -E "teacher fold|EPOCH|reload check|pseudo-labels|ERROR|ALL DONE"

# --- download (skip the fp32 checkpoints: about 1.1 GB per kernel)
$K kernels output shivateja005/umud-fasc-orient-v2 -p artifacts/kaggle/fasc_orient_v2 --force \
    --file-pattern '.*\.(npz|csv|json|jpg|log|txt)$'
$K kernels output shivateja005/umud-apo-v3 -p artifacts/kaggle/apo_v3 --force \
    --file-pattern '.*\.(npz|csv|json|png|txt)$'
```

**Variants.** `kaggle/fasc_orient_v2/make_variant.py` writes a pushable copy with constants and kernel id changed.
The source directory is never modified.

```bash
# 3: expert VL lines as training targets
.venv/bin/python kaggle/fasc_orient_v2/make_variant.py kaggle/fasc_orient_v2 /tmp/umud_fo2_vlsup \
    --id shivateja005/umud-fasc-orient-v2-vlsup --set VL_SUP=True --set MODEL_NAME='"fasc_orient_v2vl"'
$K kernels push -p /tmp/umud_fo2_vlsup --accelerator NvidiaTeslaT4

# 4: round-2 self-training from the v2 output (teacher = v2 multi-scale maps, warm start = v2 checkpoints)
.venv/bin/python kaggle/fasc_orient_v2/make_variant.py kaggle/fasc_orient_v2 /tmp/umud_fo2_r2 \
    --id shivateja005/umud-fasc-orient-v2r2 --set MODEL_NAME='"fasc_orient_v2r2"' \
    --set WARM_MODEL='"fasc_orient_v2"' --set PSEUDO_MODEL='"fasc_orient_v2ms"' \
    --sources shivateja005/umud-fasc-orient-v2
$K kernels push -p /tmp/umud_fo2_r2 --accelerator NvidiaTeslaT4

# cheap apo teacher only (repaired apo_v2 + 5-fold flip TTA maps, ~0.3 GPU-h)
.venv/bin/python kaggle/fasc_orient_v2/make_variant.py kaggle/apo_v3 /tmp/umud_apo3_teacher \
    --id shivateja005/umud-apo-v3-teacher --set SELF_TRAIN=False
$K kernels push -p /tmp/umud_apo3_teacher --accelerator NvidiaTeslaT4

# GPU smoke (~0.15 GPU-h each; optional, since the CPU smoke already passed)
.venv/bin/python kaggle/fasc_orient_v2/make_variant.py kaggle/fasc_orient_v2 /tmp/umud_fo2_smoke \
    --id shivateja005/umud-fasc-orient-v2-smoke --set MODE='"smoke"'
# CPU smoke (no quota): --cpu --id shivateja005/umud-fasc-orient-v2-cpusmoke   (apo: umud-apo-v3-cpusmoke)
```

## What each kernel does

### fasc_orient_v2 (`kaggle/fasc_orient_v2/train.py`)
**Architecture.** Same as v1: smp 0.5.0 U-Net, ConvNeXt-S fb_in22k, 512 letterbox, 3 channels (fragment logit,
tanh cos2θ, tanh sin2θ). Fold k is initialised from v1 `checkpoints/fold{k}.pt`, attached through `kernel_sources`
and checked for inf/nan in the kernel. Training uses v1's `metrics/folds.csv`. Schedule: 10 epochs, AdamW
(encoder 1e-4, decoder 3e-4), 0.3-epoch warmup, cosine, EMA 0.998. Selection is by GT-fold angle error, as in v1;
epoch 0 is the v1 model itself.

**(a) Self-training.** One quarter of the samples in each epoch are pseudo-labelled images. The labels come from
the v1 5-fold x hflip test/static/dynamic/VL maps (the teacher maskbank of the attached v1 output).
- **Orientation target:** the teacher angle, mapped exactly through each sample's affine.
- **Orientation gate:** kept only where the 5-fold agreement |v| ≥ 0.97 **and** the 25 px field coherence
  ≥ 0.95, within 8 px of a teacher fragment (seg ≥ 0.3). Calibrated on the VL expert lines: kept pixels have
  1.27° mean error, rejected pixels 4.6° (`scratchpad/gpuprep/calib_vl.py`). On the test set the gate keeps
  96–100 % of near-fragment pixels.
- **Segmentation target:** hard, p ≥ 0.6 → 1 and p ≤ 0.15 → 0, ignored in between.
- **Weights:** seg 0.5, orientation 0.7 (GT = 1).

**(b) Multi-scale.**
- **Training:** zoom-crop augmentation on both axes (up to 2.2x magnification) plus the existing narrow-FOV crops.
- **Inference:** the full viewport plus a 2x2 grid of overlapping tiles, each 0.6 of a side (1.67x zoom). Each
  view is 5 folds x hflip, and the views are blended with feathered weights.
- **Written as two models:** `fasc_orient_v2` (full view only, comparable to v1) and `fasc_orient_v2ms`
  (multi-scale).

**(c) Domain augmentation.** Added on top of v1:
- Fourier amplitude swap (β 0.005–0.03) or histogram matching to a random unlabeled test/VL/static canvas;
- random tone curves, log/exp compression;
- acoustic-shadow stripes, lateral blur, CLAHE;
- rotation ±15°, aspect ±0.18, scale 0.7–1.3.

**(d) VL supervision (`VL_SUP`, default off).** The 245 expert-drawn VL fascicle lines (OSF xbawc, CC-BY-4.0) are
embedded in `train.py` as polylines in raw-image coordinates. They were parsed by
`scratchpad/gpuprep/export_vl_tracks.py`, which reproduces `vl_labels.csv`'s PA exactly. With `VL_SUP` on, they
become orientation targets inside the drawn muscle compartment, and the VL subjects are split into the 5 folds.
The VL maps are then subject-fold OOF.

**Always logged per epoch.** An out-of-domain check on the VL lines:
- `vl_track_err`: field angle error along the drawn tracks;
- `vl_strip_err`: the same, restricted to the deep strip (4–30 % of MT, the strip that geometry v2's
  `pa_fd_strip_g` uses);
- `vl_pa_mae` / `vl_pa_bias`: an image-level deep-strip PA proxy against the expert PA. v1 maps score 1.62° MAE on
  this proxy, against 2.42° for a constant.

**Outputs** (under `maskbank/`):
- `{test,static,dynamic,vl,train_oof}__fasc_orient_v2{,ms}_{seg,cos2,sin2}.npz`, encoded as in v1. VL covers the
  249 raw `.tiff` images only, with keys as in v1; map them by `source_filename` in `manifest.csv`.
- `metrics/` holds history (epoch 0 = v1), `fold*.json` (with epoch-0 vs best, the reload check and TTA),
  `cv_summary.json` and `pseudo_stats.csv`.
- `checkpoints/fold*.pt` (fp32).

**Time:**
- 5 min setup;
- per fold about 27 min (10 epochs x 140 s plus evaluation, the reload check and OOF);
- 3 folds on GPU0 ≈ 81 min;
- ensemble inference on 2 GPUs ≈ 8 min, merge ≈ 4 min;
- total ≈ **1.65 h**.

The budget guard stops new epochs at 2.7 h and hard-stops training at 3.0 h.

### apo_v3 (`kaggle/apo_v3/train.py`)
**Phase 1, teacher `apo_v2r` (always runs):**
- Loads the 5 apo_v2 fp16 checkpoints from `kernel_sources` `shivateja005/umud-apo-v2`.
- Recalibrates **only** the non-finite BatchNorm layers (`base.decoder.blocks.0.conv1.1`, which is inf in all 5) on
  160 training images of that fold.
- Verifies each fold on its **whole** held-out fold against the logged apo_v2 fold dice and score (one-sided
  tolerance) and saves fp32.
- Writes 5-fold x hflip maps for test, static, dynamic (all 168) and VL raw (249), and fold-wise OOF for all 1048
  training images.
- Takes about 15 min.

**Phase 2, student `apo_v3` (`SELF_TRAIN`):**
- **Init:** fold k starts from its repaired teacher.
- **Pseudo-labels:** from the teacher maps. Band: p ≥ 0.75 → 1, p ≤ 0.25 → 0, ignored in between. Edges are
  re-derived from the hard band exactly as the GT edges are. Images with no superficial pair, or more than 50 %
  uncertain band pixels, are dropped: about 96–100 % of test images and 84 % of VL survive on the apo_v2fix maps.
- **Training:** 25 % pseudo samples, weight 0.5; FDA / histogram-match augmentation; 15 epochs; EMA 0.997.
- **Selection:** the apo_v2 validation score (inner-edge MT error).
- **Per-epoch VL diagnostic:** predicted inner MT against the gap between the drawn VL aponeurosis lines, as a
  median ratio and its spread. On the apo_v2fix maps this gives 0.971 and 7.4 %, with 0 failures. Only the spread
  and the failure count are meaningful.
- **Outputs:** `maskbank/{...}__{apo_v2r,apo_v3}_{band,upper,lower}.npz`, keys as in apo_v2.
- **Time:** setup 5 min, phase 1 15 min, pseudo-label prep 2 min, student 3 folds x 23 min on GPU0 ≈ 70 min,
  inference 5 min. Total ≈ **1.5 h**.

## Evaluation after download (local, one image at a time; decisions from benchmarks, never the LB)

**fasc.** `scripts/eval_orient.py` reads `ORIENT_BANK`, `ORIENT_FOLDS` and `ORIENT_CACHE` from the environment,
but the model name `OMODEL = "fasc_orient_v1"` is hard-coded (line 53).
- The main session needs a one-line change: `OMODEL = os.environ.get("ORIENT_MODEL", "fasc_orient_v1")`.
- Then run with `ORIENT_BANK=artifacts/kaggle/fasc_orient_v2/maskbank`,
  `ORIENT_FOLDS=artifacts/kaggle/fasc_orient_v2/metrics/folds.csv`, `ORIENT_MODEL=fasc_orient_v2` (then
  `fasc_orient_v2ms`) and a fresh `ORIENT_CACHE`.
- Compare F1/F2-style recipes on clean10 two-rater, VL lodo3, dynamic and train142 (train_oof) against v1.
- Look at the `vl_*` columns in `metrics/history_fold*.csv` first: epoch 0 is v1, so it shows v1 → v2 on VL per fold.

**apo.**
```bash
.venv/bin/python scripts/eval_apo_v2.py measure --dataset <ds> --source v2 \
    --v2-root artifacts/kaggle/apo_v3/maskbank --v2-model apo_v2r   # then apo_v3
```
The VL keys map through `manifest.csv`, whose `source_filename` uses the same `vl_young/...` / `vl_old/...` form as
apo_v2. Delete the old v2 caches first, as noted in the apo2 report.

**Then.** Rebuild C7 with the new maps through `scripts/build_candidates_v4.py`, and compare S_sel against the
current 0.2708.

## Smoke tests (CPU-only Kaggle kernels, no GPU quota)

### `shivateja005/umud-fasc-orient-v2-cpusmoke`: COMPLETE on version 4, 6.5 min
Settings: fold 0, 8 train / 4 val / 6 pseudo / 2 VL-val images, 1 epoch of 3 steps; inference on 2 test, 1 static,
1 dynamic and 3 VL images.

Every stage ran through the real fork path:
- warm start from v1;
- epoch-0 baseline and the VL metric;
- the training step;
- the fp32 save, reload check and TTA;
- OOF writing (scale-1 and multi-scale);
- forked ensemble inference and the merge into `maskbank/`.

Format checks:
- **OOF:** the maps are identical to v1's (corr 1.000, 0.00° angle difference). This is expected: epoch 0 = v1
  fold 0 was the selected checkpoint.
- **Test / static / dynamic / VL:** maps correlate 0.94–0.96 with the v1 5-fold ensemble, at a 0.3–0.6° median
  angle difference (one fold here vs five).
- **Viewport encoding:** outside the viewport, seg = 0 and cos2 = sin2 = 128.
- **VL keys:** equal v1's.
- **Multi-scale vs scale-1:** 0.3–0.7° apart.

Version 5 adds only the OOM micro-batch fallback. It is COMPLETE in 5.2 min with identical results (epoch 0 = v1: dice 0.714, angle error 1.28°; reload check OK).

**Untested on CPU:** CUDA AMP / GradScaler / GPU memory. These code paths are unchanged from v1 and apo_v2, which ran on GPU. Watch the first ~10 min of the full run's log (`kernels logs -f`) and cancel it from the Kaggle web UI if the first `EPOCH` line or the epoch-0 `reproduces=True` line is missing. A GPU smoke (~0.15 GPU-h, `--set MODE='"smoke"'`) is optional.

### `shivateja005/umud-apo-v3-cpusmoke`: COMPLETE, 5.8 min
- Phase 1 found `base.decoder.blocks.0.conv1.1.running_var = inf` in fold 0, recalibrated that one layer, checked
  the fp32 reload (identical dice) and wrote `apo_v2r` maps.
- Pseudo-labels were usable on 8 of 9 images.
- The student epoch ran, the reload check passed and the `apo_v3` maps were written.
- The VL lines diagnostic worked (ratio 0.98).

### Bugs and findings from the smoke tests (all fixed before the GPU push)
1. **v1 fasc checkpoints overflow too.** `fasc_orient_v1` checkpoints for **folds 2 and 4** also have
   `decoder.blocks.0.conv1.1.running_var = inf`, the same fp16 overflow as apo_v2. Fold 3's largest running_var is
   63552, just under the fp16 maximum of 65504.
   - **Consequence for v1:** the v1 maps from those folds were made with the affected channels zeroed. v1's TTA
     metrics still looked fine, so the effect is small but real.
   - **Fix in v2:** the warm start resets only the non-finite layers and re-estimates them on 160 training crops of
     that fold.
   - **Change of plan:** v2 uses the v1 *maskbank* as the teacher, not the checkpoints, so pseudo-labels are as v1
     produced them.
2. **Fork deadlock.** fasc_orient_v2's forked worker hung right after fork, in cpusmoke versions 1 and 2.
   - **Cause:** the parent ran OpenCV ops with OpenCV's thread pool (pseudo-label prep, letterboxing), and the child
     then called `cv2.setNumThreads(1)`.
   - **Fix:** `cv2.setNumThreads(1)` at import, as apo_v2/apo_v3 already do. All parent torch work (checkpoint
     inspection, HF-cache warm-up) now also runs in an isolated subprocess.
   - **This would have hung the GPU run too**, wasting its session wall time.
3. **Possible out-of-memory.** v1 peaked at 13.7 GB of the T4's 15 GB at batch 8, and v2 adds an EMA copy
   (+0.2 GB). The training step now retries a batch as 2, then 4, accumulated micro-batches on CUDA OOM, instead of
   failing the fold.

**Local numpy-only checks** (`scratchpad/gpuprep/test_fo2_numpy.py`, `test_apo3_numpy.py`; no torch, < 0.4 GB RSS):

| check | result |
|---|---|
| Pseudo-orientation transform under random flips / rotations / anisotropic scaling | max error 0.007° |
| GT path vs pseudo path with the GT as the teacher | median 0.04–0.08° |
| VL track deep-strip PA vs expert PA (no augmentation) | MAE 0.54° |
| Multi-scale stitching of a perfect field | max error 0.002 |
| apo pseudo vs GT band targets outside the ignore band | 100 % agreement |
| 3-channel letterbox round trip | exact |
| Per-sample CPU cost | 14–17 ms (no DataLoader bottleneck) |

## Risks / notes

- **Self-training can confirm the teacher's own errors.** This is the main reason the gates are strict, and why
  VL is logged every epoch with epoch 0 = v1. If `vl_strip_err` or `vl_pa_mae` rises across folds, keep v1.
- **Static benchmark leakage.** 25 of the 35 static images were in v1's training data, so their pseudo-labels are
  in-sample. static35 stays optimistic; clean10, dynamic and VL are the honest tests, as before. Using unlabeled
  benchmark images for self-training mirrors exactly what is done on test, so it is honest transductive evaluation.
- **GPU1 idles about 25 min.** With 5 folds on 2 GPUs, GPU1 waits through GPU0's third fold; v1's folds must be
  kept for the warm start to stay honest.
- **Run-to-run differences.** If a fold fails its reload check it is dropped, and its train_oof keys will be
  missing (logged). VL under `VL_SUP` uses only the subject's held-out fold model.
- **Output size.** About 2 GB of npz per kernel, excluding checkpoints. Download with `--file-pattern`, and load
  one key at a time with `np.load` (lazy).
