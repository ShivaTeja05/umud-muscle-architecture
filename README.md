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
| **Final A** | `configs/final_A.yaml` -> `final_A.csv` (sha256 `5d216fda...32bf`) | **0.23041** | KG0: K25b with the Telemed PA head G0 (below). Contains the declared LB-chosen Telemed MT -0.5 mm (J4). Byte-identical to the scored file `KG0_K25b.csv`. |
| **Final B** | `configs/final_B.yaml` -> `final_B.csv` (sha256 `3b95665c...66db`) | 0.23322 | B': Final A without J4, i.e. without any LB-tuned continuous constant. Byte-identical to the scored file `FinalBprime_KG0_noJ4.csv`. |
| previous best | `configs/K25b.yaml` -> `K25b.csv` (sha256 `364e9cc3...afaa`) | 0.23297 | Final A before the G0 head. Byte-identical to the scored file `02_K25b_B1J4.csv`. |
| base | `configs/C18.yaml` -> `C18.csv` (sha256 `3f9379f7...633d`) | 0.33526 | image-space base of every final; byte-identical to the scored C18 file |

**Final B.** Final B is Final A with the J4 cells removed: the Telemed MT goes back to the pipeline's own MT (no
-0.5 mm). The Telemed PA rule contains the coupling term 0.25 asin(1.067 MT / FL), which reads MT, so removing J4
also moves the Telemed PA: PA_B = PA_A + 0.25 (asin(1.067 MT_B / FL) - asin(1.067 MT_A / FL)), +0.08 deg on average.
Every other cell equals Final A. The build derives the file from Final A's file exactly as the scored file was made
(`derive` in `configs/final_B.yaml`). It also computes the same rules directly on the own MT and checks that the two
agree: all other cells identically, the Telemed PA cells within one unit of the 4th decimal (the rounding of the
intermediate file). The previous Final B (K25b without J4, sha256 `30512ad2...9f22`, never scored) is rebuilt by the
tests as a K25b variant.

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
   - **Telemed PA head G0** (KG0, both finals): on the Telemed clips the PA head becomes the learned-field angle in the
     deep strip (4-30 % of MT above the deep aponeurosis) measured against the *local* deep tangent (`pa_fd_strip`)
     wherever the insertion strip (2-12 % of MT) is steeper than the fused head (s = `pa_fd_ins` - fusF > 0, a sign
     test); elsewhere the head stays fusF. The Telemed PA is then rebuilt with the unchanged recipe (confidence rule
     toward the scanner-group median, clip median, + 0.3286 deg) and the unchanged coupling term (lambda 0.25 on the
     MT / FL of the rules above). The gate is on for 112 of 140 Telemed images, and the Telemed PA rises by
     0.83 deg on average. The Telemed FL is not rebuilt: it stays at the fusF-based PA. Every input is a column of
     the per-image cache. The build checks that the written-out recipe reproduces the pipeline's own fusF Telemed PA
     (within 1e-4 deg; the tolerance is 2e-4). Why: on the expert benchmarks, in the geometry regime of the test
     Telemed clips (fascicles steepening toward the insertion, curved deep aponeurosis), fusF reads 0.83 deg low and
     `pa_fd_strip` is unbiased. This also matches the protocol (insertion angle against the deep aponeurosis at the
     insertion). Details: `artifacts/candidates_v10/manifest.md` (not in git);
   - **Final B only:** derived from Final A's file with the J4 cells removed (see above);
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
uv run umud-run --config configs/K25b.yaml    --stage build   # the previous best file (Final A before the G0 head)
uv run umud-run --config configs/C18.yaml     --stage build   # the base file
# equivalently: PYTHONPATH=src python scripts/umud_run.py --config configs/final_A.yaml --stage build
```

The build re-derives the layout table from the test images (scanner groups, clips), runs the factor detectors on the Lumify / Siemens
images, re-estimates the rater offsets from the OSF xlsx, validates the CSV (`scripts/validate_submission.py`) and
compares it with the reference file named in the config (every config: byte-identical; the exit code is 3 if not).
The Final B build also builds Final A in the same process (it is derived from it).

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
sha256 of every input and of every code file the build calls. `K25b.yaml` extends it (native geometry, J4, coupling
pull); `final_A.yaml` extends `K25b.yaml` and only switches on the G0 Telemed PA head (`pa_head.telemed_head`);
`final_B.yaml` extends `final_A.yaml` and only removes the Telemed MT constant, with `derive` (all tested). The runner
refuses to build when a pinned file changed, when a code default drifted from the config, when an `lb_constants` target
does not hold its registered value on its grid, when a factor detector does not confirm its factor, when the G0 head's
written-out PA recipe does not reproduce the pipeline's Telemed PA, or when a `derive` source differs from the config
beyond the MT constants. Re-pin after an intended change:
`python tools/pin_config.py configs/C18.yaml`. One-off overrides: `--set key.sub=value`.

### Tests

```bash
uv run --with pytest pytest -q                          # all tests (about 1.5 min, < 0.5 GB)
uv run --with pytest pytest -q tests/test_final_package.py
```

`tests/test_final_package.py` rebuilds Final A (KG0), Final B (B'), K25b and C18 byte-for-byte from the cached
per-image tables. It also rebuilds, as config variants, the intermediate scored files B1 and B1_J4, the previous
Final B and the G0n contrast file. It checks that Final A is K25b plus the G0 head only (Telemed PA cells), and that
Final B is Final A without J4 (Telemed MT and PA cells, the coupling-on-own-MT rule). It reproduces seven ledger files
from single config changes, re-derives the rater rule and re-measures images from the maps. Unit tests cover the G0
gate (a sign test), the clip detection (synthetic frames and the 28 test clips) and the factor detectors (synthetic
discs / triangles and the test images). Data-dependent tests skip when `data/` or `artifacts/` is absent.

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
| KG0: Telemed PA head G0 (deep-strip field angle vs the local deep tangent where the insertion strip is steeper than fusF) | `pa_head.telemed_head.enabled: true` | an offline-justified per-image head. On the 78 expert-labelled benchmark images in the test-Telemed regime, fusF + offset reads 0.83 deg low and the G0 strip angle -0.08 deg; MAE vs the K25 rule -0.28 deg [-0.46, -0.11], over all 445 benchmark images -0.05 deg [-0.09, -0.01]. The gate is a sign test (s > 0); the strip is the v3 `FieldConfig` default and lambda is K25b's 0.25, so there is no LB-tuned constant. The adopt rule was pre-registered and one point was scored | -0.00256 (0.23297 -> 0.23041) | yes | yes |
| J4: Telemed MT -0.5 mm | `mt_edge.constant_mm.telemed_*: -0.5` | none offline (a hand-picked point of a pre-declared grid {0, -0.25, -0.5, -0.8}) | -0.00238 (B1 -> B1_J4); removing it from KG0: +0.00281 (B') | **yes (declared LB-chosen constant)** | **no** (its coupling-term effect on the Telemed PA is removed too) |

Further declarations: the scanner layout decoder is adapted from AmbrosM's public notebook (Apache-2.0); the metric
re-implements the organiser's notebook; statistics are pooled across test images (clip medians; shrinkage toward
scanner x muscle medians of our own predictions); an AI coding assistant helped write the code.

## Hardware

- **Laptop** (layout, measurement, build, tests): Apple M3 Pro (11 cores, 36 GB RAM), macOS 26, Python 3.12.11. The
  build needs < 0.3 GB and about 10 s per final (Final B about 15 s: it builds Final A too); the full test-set
  measurement about 6 min.
- **Kaggle kernels** (training and map inference): NVIDIA Tesla T4 (x2) GPU sessions, CPU sessions for the apo_v2 fix.
- **RunPod** (1x H100): experiments only (apo_v3, fasc_orient_v2); not part of either final.

## Repository layout

| path | content |
|---|---|
| `configs/` | `C18.yaml`, `K25b.yaml`, `final_A.yaml`, `final_B.yaml` |
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
0.335; native geometry B1 0.239; B1_J4 0.237; K25b 0.233; KG0 0.2304; B' 0.2332) are documented in `research/` and in
the research archive (`artifacts/legacy_research`, not in git). The research-era builders
(`scripts/build_candidates_v7.py`, `_v9.py`, `aniso_phys_v9.py`) remain for reference, as do the KG0 / B' builders in
`artifacts/candidates_v10`, `_v11` (not in git). The finals are rebuilt only by `umud-run`.
