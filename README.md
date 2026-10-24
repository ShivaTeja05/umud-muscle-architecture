# UMUD Challenge: muscle architecture from ultrasound images

Solution code for the Kaggle competition
[UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data).
For every test image it predicts the pennation angle `pa_deg` (degrees), the fascicle length `fl_mm` and the muscle
thickness `mt_mm` (millimetres). The metric is the mean over the three components of MAE / tau
(tau = 6 deg, 12 mm, 3 mm); lower is better.

License: GPL-3.0-only (`LICENSE`). Third-party code, models, data and tools: `THIRD_PARTY_NOTICES.md`.

## Final submissions

| | file | public LB | what it is |
|---|---|---|---|
| **Final A** | `configs/final_A.yaml` -> `final_A.csv` (sha256 `364e9cc3...afaa`) | **0.23297** | Final B + Telemed MT -0.5 mm (J4, a declared LB-chosen constant). Byte-identical to the scored file `02_K25b_B1J4.csv`. |
| **Final B** | `configs/final_B.yaml` -> `final_B.csv` (sha256 `30512ad2...9f22`) | not scored; predicted 0.2354 (see below) | the same pipeline without any LB-tuned continuous constant |
| base | `configs/C18.yaml` -> `C18.csv` (sha256 `3f9379f7...633d`) | 0.33526 | image-space base of both finals; byte-identical to the scored C18 file |

**Final B prediction.** The public score is exactly additive over disjoint (scanner group x component) cells. J4 changes
only the Telemed MT cells and K25b only the Telemed PA cells (checked by the tests), so B1 plus Final A's Telemed PA
cells would score exactly 0.23297 + (0.23931 - 0.23693) = **0.23535**. Final B differs from that file in the Telemed PA
cells only: its coupling pull sees the pipeline's own MT (no -0.5 mm), which raises the Telemed PA by a further 0.08 deg
on average (distance 0.0020). So LB(Final B) = 0.23535 + d, with |d| <= 0.0048 (the largest change any 102-row public
subset could see). The extra pull goes in the direction that won (K25b gained 0.0040 for a 0.0096 move), so d is more
likely negative (about -0.0008 at K25b's efficiency).

## Method summary

1. **Maps (GPU, private Kaggle T4 kernels; trained on the competition training data only).**
   - Aponeuroses: `apo_v2`, a ConvNeXt-S U-Net (timm `convnext_small.fb_in22k_ft_in1k`) with three outputs per pixel
     (band, upper edge, lower edge), 5 folds split by near-duplicate groups, letterbox 512. Its fp16 checkpoints
     overflowed BatchNorm variances, so `kaggle/apo_v2_fix` re-estimates the BatchNorm statistics on each fold's own
     training split and re-infers on CPU (`apo_v2fix`, used for every image).
   - Fascicle fragments: U-Net++ ResNet-34 at 384 px (`kaggle/fascicle_unetpp_384`, maps from
     `kaggle/infer_maskbank`).
   - Fascicle orientation field: `fasc_orient_v1`, ConvNeXt-S U-Net with a fragment head and a doubled-angle
     (cos 2theta, sin 2theta) orientation head.
2. **Layout from the image content** (`src/umud/layout.py`): scanner family, image size, scale (px/cm from the
   burned-in scale bar) and ultrasound viewport (`umud.geometry.ultrasound_metadata`, adapted from AmbrosM); the
   **video clips** (28 Telemed clips of 5 near-identical frames) are found by the normalised cross-correlation of
   consecutive viewport thumbnails (`src/umud/clips.py`, threshold 0.8; within-clip NCC >= 0.946, neighbours outside a
   clip <= 0.561). No file-name rules.
3. **Per-image measurement** (`src/umud/geometry_v2.py`, `scripts/build_candidates_v3.py`): superficial aponeurosis pair
   from the apo maps; MT = inner-edge distance at 25/50/75 % of the shared x-range; PA = mean of the fragment-fusion PA
   and the learned-field PA in the deep strip, against the deep aponeurosis; FL = mean of three estimates (straight
   fragments, the central three, and a straight fascicle constructed at the final PA between the fitted aponeuroses).
4. **Shrinkage** (`src/umud/shrinkage.py`): PA toward the scanner x muscle median with weight = per-image confidence;
   FL by empirical Bayes toward scanner x muscle centres, with a per-image error model fitted on the OSF benchmarks
   (leave-one-dataset-out). Muscle names come from OCR of the burned-in labels (Tesseract). MT is not shrunk.
   Telemed FL is not shrunk (raw per-clip median, C18). Every value is then the median over its clip's frames.
5. **Rater style** (`scripts/rater_calibration.py`): half-strength offsets estimated by a pre-registered rule from the
   OSF static benchmark xlsx (two-rater labels R7 + Rk): PA +0.3286 deg, MT +0.0388 mm (re-estimated at every build).
6. **Post rules** on the 4-decimal stage file (`src/umud/run.py`, each on its own cells):
   - **native geometry** for Philips Lumify (c = 0.844) and Siemens Juniper (c = 1.125): the labels of these scanners
     were drawn on their native square-pixel frames, so PA' = atan(c tan phi) + atan(c tan delta) and
     FL' = FL sqrt(cos^2 phi / c^2 + sin^2 phi) (phi = PA - deep-aponeurosis tilt delta; MT unchanged). The factors
     are fixed per scanner from the native geometry (16:9 screen -> 3:2; 1152x864 -> 1200x800) and are confirmed on
     every build by detectors that measure them from each image (`src/umud/native_geometry.py`: the aspect of the round
     'P' marker on Lumify, median 0.840; the slope of the 45-degree UI triangles on Siemens, median 1.123). The build
     stops if a detector does not confirm its factor. Telemed is left in image space;
   - **Final A only:** Telemed MT -0.5 mm;
   - **Telemed PA coupling pull:** PA += 0.25 (asin(1.067 MT / FL) - (PA - 0.3286)), with the expert coupling
     FL sin(PA) = 1.067 MT measured on the OSF benchmarks;
   - bounds PA [5, 45], FL [30, 200], MT [10, 50] (post-rule values kept 1e-3 inside).

## Reproduce Final A / B

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Tesseract 5 (only to re-run the muscle-label OCR).

```bash
uv sync --locked            # exact versions from uv.lock
# pip alternative: pip install -r requirements-lock.txt && pip install --no-deps -e .
```

**From the cached per-image tables (seconds, < 0.3 GB RAM).** With the per-image caches in `artifacts/` (the measure-stage
output, listed with their sha256 in `configs/C18.yaml` `inputs:`) and the competition images in `data/`:

```bash
uv run umud-run --config configs/final_A.yaml --stage build   # -> artifacts/final/final_A/final_A.csv (+ _report.json)
uv run umud-run --config configs/final_B.yaml --stage build   # -> artifacts/final/final_B/final_B.csv
uv run umud-run --config configs/C18.yaml     --stage build   # the base file
# equivalently: PYTHONPATH=src python scripts/umud_run.py --config configs/final_A.yaml --stage build
```

The build re-derives the layout table from the test images (scanner groups, clips), runs the factor detectors on the Lumify / Siemens
images, re-estimates the rater offsets from the OSF xlsx, validates the CSV (`scripts/validate_submission.py`) and
compares it with the reference file named in the config (Final A, C18: byte-identical; the exit code is 3 if not).

**End to end from the raw images.**

| step | command | where |
|---|---|---|
| data | `uv run python scripts/download_data.py` (competition data, after accepting the rules); OSF xbawc static / dynamic benchmarks and "Example Annotated Images" into `data/external_umud_benchmark/` and `artifacts/maskbank/vl_data/` | anywhere |
| layout + clips | `uv run umud-run --config configs/final_A.yaml --stage layout` | laptop, 4 s |
| maps | `uv run umud-run --config configs/final_A.yaml --stage maps` writes 7 ready-to-push kernel folders to `artifacts/final/<name>/kaggle/` with a `RUN.md` each (push / status / output commands, checkpoint sources): U-Net++ apo / fascicle training, `infer-maskbank`, `fasc-orient-v1`, `apo-v2`, `apo-v2-fix-test`, `apo-v2-fix-ext`. Push them by hand in that order; outputs go to fresh `artifacts/final/maps/` folders. | Kaggle T4 GPU (the fix kernels run on CPU) |
| VL expert labels | `uv run python scripts/vl_labels.py` | laptop, 20 s |
| muscle labels | `uv run python scripts/read_muscle_labels.py ocr` then `uv run python scripts/build_candidates_v4.py v5muscle` | laptop, ~5 min (Tesseract) |
| per-image measurement | `uv run umud-run --config configs/final_A.yaml --stage measure --full` (all datasets; `--datasets test` for the test set only), then build with `--set paths.v3_cache=<out>/cache --set paths.v4_cache=<out>/cache/v4extra` | laptop or Kaggle CPU, ~1.1 s / image |
| build | as above | laptop |

`--stage measure` without `--full` verifies the caches against their pins and re-measures one image per scanner group
from the maps (identical to the cache within 3e-14). The x86 (Kaggle / pod) and arm (laptop) per-image measurements
differ by <= 1.4e-5 mm; the build is deterministic on one platform.

### Configs

`configs/C18.yaml` holds every choice and constant of the base pipeline (map sources and thresholds, the geometry
configuration, PA / FL heads, shrinkage, error model, rater rule, layout / clip detection, output format) plus the
sha256 of every input and of every code file the build calls. `final_B.yaml` extends it; `final_A.yaml` extends
`final_B.yaml` and changes only the Telemed MT constant (tested). The runner refuses to build when a pinned file
changed, when a code default drifted from the config, when an `lb_constants` target does not hold its registered value
on its grid, or when a factor detector does not confirm its factor. Re-pin after an intended change:
`python tools/pin_config.py configs/C18.yaml`. One-off overrides: `--set key.sub=value`.

### Tests

```bash
uv run --with pytest pytest -q                          # all tests (about 1.5 min, < 0.5 GB)
uv run --with pytest pytest -q tests/test_final_package.py
```

`tests/test_final_package.py` rebuilds Final A, C18 and the intermediate scored files B1 and B1_J4 byte-for-byte from
the cached per-image tables, checks Final B's cells and the additivity argument above, reproduces seven ledger files
from single config changes, re-derives the rater rule, re-measures images from the maps, and unit-tests the clip
detection (synthetic frames and the 28 test clips) and the factor detectors (synthetic discs / triangles and the test
images). Data-dependent tests skip when `data/` or `artifacts/` is absent.

## External data and models

- **OSF xbawc** (CC BY 4.0): the UMUD static benchmark (35 images, 7 raters), the GM dynamic calf-raise sequence (167
  frames) and the "Example Annotated Images" (VL). Used for validation, the empirical-Bayes error model, the expert
  coupling k = 1.067 and the half-strength rater offset (as the organiser suggested, thread 739660).
- **Pretrained encoders**: ImageNet ResNet-34 (segmentation-models-pytorch 0.5.0 / torchvision) and timm
  `convnext_small.fb_in22k_ft_in1k` (Apache-2.0), downloaded at run time inside the kernels.
- **DL_Track_US v0.3 weights** (OSF 7mjsc, Apache-2.0): only a fallback aponeurosis map, never reached by the finals.
- **Not used by the finals**: models trained on test images (apo_v3, fasc_orient_v2), hand labels of test images (none
  exist), other participants' predictions.

## LB-guided choices and declarations

We used the public leaderboard only to choose among a few fixed, scanner-level alternatives. All of them are hard-coded
in the configs (`lb_constants`), so the pipeline runs on a new dataset without any leaderboard. No single images or
clips were probed, no submissions were blended by score, and no submission contains values from other participants.

| choice | config key | evidence | LB | in Final A | in Final B |
|---|---|---|---|---|---|
| C18: Telemed FL not shrunk (raw per-clip median) | `fl_rules.shrink_lambda.telemed_*: 0` | the Telemed dynamic benchmark predicted it (EB FL 0.171 -> 0.183) | C12 0.37670 -> 0.33526 | yes | yes |
| Half-strength rater offset (PA +0.3286 deg, MT +0.0388 mm) | `rater_style` | pre-registered rule on the OSF xlsx, re-estimated at every build | C17 (none) 0.38091 vs 0.37670 | yes | yes |
| J1: Lumify PA / FL in native geometry, c = 0.844 | `geometry_space.scanners.philips_lumify.space: physical` | factor from the 16:9 -> 3:2 screen geometry; 'P'-marker detector 0.840 | -0.04074 | yes | yes |
| J2: Siemens PA / FL in native geometry, c = 1.125 | `...siemens_juniper.space: physical` | 1152x864 native resolution of the training masks; UI-triangle detector 1.123 | -0.05521 | yes | yes |
| J3: Telemed 1200x800 stays in image space | `...telemed_artus_1200x800.space: image` | its native transform lost | +0.05413 (rejected) | yes | yes |
| K25b: Telemed PA coupling pull, lambda = 0.25 | `pa_head.coupling_pull.weight: 0.25` | expert FL sin(PA) / MT = 1.067 (static 1.07, VL 1.067) vs Telemed 0.95-1.02; grid {0, 0.25, 0.5} declared before scoring | -0.00396 | yes | yes |
| J4: Telemed MT -0.5 mm | `mt_edge.constant_mm.telemed_*: -0.5` | none offline (a hand-picked point of a pre-declared grid {0, -0.25, -0.5, -0.8}) | -0.00238 | **yes (declared LB-chosen constant)** | **no** |

Further declarations: the scanner layout decoder is adapted from AmbrosM's public notebook (Apache-2.0); the metric
re-implements the organiser's notebook; statistics are pooled across test images (clip medians; shrinkage toward
scanner x muscle medians of our own predictions); an AI coding assistant helped write the code.

## Hardware

- **Laptop** (layout, measurement, build, tests): Apple M3 Pro (11 cores, 36 GB RAM), macOS 26, Python 3.12.11. The
  build needs < 0.3 GB and about 5 s per final; the full test-set measurement about 6 min.
- **Kaggle kernels** (training and map inference): NVIDIA Tesla T4 (x2) GPU sessions, CPU sessions for the apo_v2 fix.
- **RunPod** (1x H100): experiments only (apo_v3, fasc_orient_v2); not part of either final.

## Repository layout

| path | content |
|---|---|
| `configs/` | `C18.yaml`, `final_A.yaml`, `final_B.yaml` |
| `src/umud/run.py` | the `umud-run` pipeline (stages layout / maps / measure / build) |
| `src/umud/layout.py`, `clips.py`, `native_geometry.py` | layout table, clip detection, native-geometry factors and transform |
| `src/umud/geometry.py`, `geometry_v2.py`, `shrinkage.py`, `muscle_label.py`, `metric.py` | measurement, shrinkage, OCR muscle labels, metric |
| `scripts/` | the per-image measurement and shrinkage layers (`build_candidates*.py`, `eval_*.py`), `rater_calibration.py`, `vl_labels.py`, `read_muscle_labels.py`, `validate_submission.py`, `umud_run.py`, research scripts |
| `kaggle/` | every Kaggle kernel (training and map inference); `apo_v2_fix/` holds the BatchNorm-fix kernels |
| `tests/`, `tools/` | tests; `tools/pin_config.py`; `tools/annotator/` (offline annotation tool, unused: no test image was hand-labelled) |
| `research/` | notes, reviews and the rules audit |
| `data/`, `artifacts/` | competition / benchmark data and all generated files (git-ignored) |

## History

Earlier stages (U-Net++ masks with line fits, public LB 0.925; geometry v1 0.467; shrinkage C2 0.390; C12 0.377; C18
0.335; native geometry B1 0.239; B1_J4 0.237; K25b 0.233) are documented in `research/` and in the research archive
(`artifacts/legacy_research`, not in git). The research-era builders (`scripts/build_candidates_v7.py`, `_v9.py`,
`aniso_phys_v9.py`) remain for reference; the finals are rebuilt only by `umud-run`.
