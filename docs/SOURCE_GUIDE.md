# Source guide — the model we submit (2026-11-10)

This is a map of the code behind our two finals (Final A 0.21637, A-plus v2 0.21529), and where each known problem sits.

## 1. How a prediction is made

One image goes through these steps, in this order:

1. **Layout and scale** — find the scanner family, the ultrasound viewport and the pixel-to-mm scale.
2. **Maps** — two networks produce probability maps: aponeuroses, and a fascicle orientation field.
3. **Measure** — geometry code turns the maps into raw PA, FL and MT per image.
4. **Heads** — several PA and FL estimates are combined into one per image.
5. **Shrinkage** — noisy values are pulled toward a group value.
6. **Clip median** — frames of one video clip share a value.
7. **Rater offsets and post rules** — fixed corrections for how experts draw.
8. **LB constants** — five group constants found by leaderboard campaign. Final A only.

Steps 4–7 live in `Assembler` in [run.py](../src/umud/run.py). Step 8 is not in tracked code yet (see section 5).

## 2. Files

### Pipeline (laptop-safe once maps exist)

| File | What it does | Key places |
|---|---|---|
| [src/umud/run.py](../src/umud/run.py) | Entry point and assembly | `Assembler` [l.343](../src/umud/run.py:343), `pa_head` [l.400](../src/umud/run.py:400), `cons_at` [l.405](../src/umud/run.py:405), `fl_head` [l.421](../src/umud/run.py:421), `conf_rule` [l.452](../src/umud/run.py:452), `eb_fl` [l.474](../src/umud/run.py:474), `offsets` [l.673](../src/umud/run.py:673), `post_rules` [l.734](../src/umud/run.py:734), `telemed_head` [l.830](../src/umud/run.py:830), `stage_build` [l.1077](../src/umud/run.py:1077), `stage_measure` [l.1135](../src/umud/run.py:1135), `stage_maps` [l.1283](../src/umud/run.py:1283) |
| [src/umud/geometry_v2.py](../src/umud/geometry_v2.py) | Maps to raw PA / FL / MT | `select_pair` [l.615](../src/umud/geometry_v2.py:615), `extract_fragments` [l.682](../src/umud/geometry_v2.py:682), `directional_pa` [l.850](../src/umud/geometry_v2.py:850), `constructed_fl` [l.1034](../src/umud/geometry_v2.py:1034), `measure` [l.1097](../src/umud/geometry_v2.py:1097), `apo_mt_variants` [l.1537](../src/umud/geometry_v2.py:1537), `field_architecture` [l.1931](../src/umud/geometry_v2.py:1931), `measure_field` [l.2071](../src/umud/geometry_v2.py:2071) |
| [src/umud/shrinkage.py](../src/umud/shrinkage.py) | Shrinkage toward group values | `fit_group_prior` [l.310](../src/umud/shrinkage.py:310), `shrink` [l.487](../src/umud/shrinkage.py:487), `confidence_shrink` [l.598](../src/umud/shrinkage.py:598) |
| [src/umud/native_geometry.py](../src/umud/native_geometry.py) | Per-scanner scale marks (Lumify disc, Siemens triangles, Telemed ticks) | |
| [src/umud/geometry.py](../src/umud/geometry.py) | Scale decoder and viewport boxes (l.280–286 adapted from a public notebook; open decision) | |
| [src/umud/clips.py](../src/umud/clips.py), [layout.py](../src/umud/layout.py), [muscle_label.py](../src/umud/muscle_label.py) | Clip detection, scanner layout, muscle guess | |
| [src/umud/metric.py](../src/umud/metric.py) | The competition score | |

### Networks (GPU, Kaggle kernels)

| File | Network |
|---|---|
| [kaggle/apo_v2/train.py](../kaggle/apo_v2/train.py) | Aponeurosis segmentation, 5 folds. Inference fixes in `kaggle/apo_v2_fix/`. |
| [kaggle/fasc_orient_v1/train.py](../kaggle/fasc_orient_v1/train.py) | Fascicle orientation field. ConvNeXt-small encoder, 512 px, 5 folds, outputs seg / cos2 / sin2. |
| [kaggle/infer_maskbank/pipeline.py](../kaggle/infer_maskbank/pipeline.py) | Runs the networks over the test images and saves the maps. |
| [src/umud/torch_unet.py](../src/umud/torch_unet.py) | Model definition used at inference. |

### Configs

| File | Builds |
|---|---|
| [configs/C18.yaml](../configs/C18.yaml) | Base pipeline |
| [configs/K25b.yaml](../configs/K25b.yaml) | Base + PA coupling pull (λ 0.25) |
| [configs/final_A.yaml](../configs/final_A.yaml) | K25b + Telemed G0 PA head (this is KG0, 0.23041) |
| [configs/final_B.yaml](../configs/final_B.yaml) | Final B′. Do not change. |

## 3. How to run it

Building a submission from saved maps is light and safe on the laptop:

```bash
lockf -k /tmp/umud.lock env PYTHONPATH=src .venv/bin/python scripts/umud_run.py --config configs/final_A.yaml --stage build
```

- Stages: `layout`, `maps`, `measure`, `build`, `all`.
- **Do not run `maps` or `all` on the laptop.** They run the networks and need a GPU.
- `build` writes into `artifacts/final/final_A/`. To try a change, copy the config, and set `paths: out_dir:` to a new folder first. Otherwise the reference files get overwritten.
- Do not run `tests/test_final_package.py` as written. It rewrites `artifacts/final/test_layout.csv`.

To score a change offline, use the benchmark frame the `Assembler` builds (`bench_frame`, [l.544](../src/umud/run.py:544)). It holds our predictions next to expert labels for the public labelled sets.

## 4. Where the problems are

Error budget of the current score (0.215): FL 0.120, PA 0.066, MT 0.031.

### Problem 1 — Telemed fascicle length (0.06 of the score, the largest single item)

- Code: `fl_head` [run.py l.421](../src/umud/run.py:421), `cons_at` [l.405](../src/umud/run.py:405), `constructed_fl` [geometry_v2.py l.1034](../src/umud/geometry_v2.py:1034).
- FL is the mean of three estimates: straight, central3, and constructed-at-PA.
- On Telemed test clips the three disagree by 25–31 mm (median range). The straight one is long (×1.14–1.25), the constructed one is short (×0.89). The final level is a cancellation between them.
- The visible fascicle pieces cover only 6–7% of a fascicle, so FL is mostly extrapolated.
- We have no labelled images that look like Telemed test (long fascicles, FL about 1.5–2.2 × field of view). Only the Siemens VL set covers that range.

### Problem 2 — Pennation angle is compressed on hard images

- Code: `field_architecture` [geometry_v2.py l.1931](../src/umud/geometry_v2.py:1931), `directional_pa` [l.850](../src/umud/geometry_v2.py:850), `conf_rule` [run.py l.452](../src/umud/run.py:452).
- On the VL and dynamic sets, predicted angle moves only about 0.7–0.86 as much as the expert angle. On clean static images there is no compression (0.97–0.99).
- About half of this comes from the reference (aponeurosis) line, not the fascicle direction.
- Shrinkage in `conf_rule` adds a little more (ratio 0.86–0.94). Removing it makes scores worse, because the raw value is noisy.
- The real fix is a less noisy PA estimate, not a correction afterwards. Post-hoc corrections were tested and did not transfer (v31, v32).

### Problem 3 — T12 and T10 need opposite PA constants

- Code: `telemed_head` [run.py l.830](../src/umud/run.py:830), `g0_head` [l.326](../src/umud/run.py:326).
- LB constants: T12 −0.75°, T10 +1.10°. No code bug explains this (v31). The G0 head lifts both by +0.8°; T12's constant undoes that lift.

### Problem 4 — LB constants are outside the code

- Final A = KG0 output + 4 constants (T12.PA −0.75, T12.MT +0.325, T10.PA +1.1, SIE.PA +0.85). A-plus adds LUM.PA +0.5.
- They are applied to the CSV by `artifacts/candidates_v27/aplus/build_aplus.py` and the campaign code, not by `umud-run`.
- A drafted fix (a `cell_constants` block in the config) is in `artifacts/candidates_v31/package/draft/`. It reproduces both final files exactly. Not installed yet.

### Problem 5 — model weights

- 10 of the 13 checkpoints exist only as outputs of our private Kaggle kernels (`umud-apo-v2`, `umud-fasc-orient-v1`). They must be downloaded before anything can be rebuilt from scratch.

## 5. What was tried and did not work

Do not spend time on these again:

- Ensembling our own variants (errors correlate 0.93–1.00).
- A direct image → PA/FL/MT regressor trained on the ~500 public labelled images (v30 S5; far too weak).
- One global deep line for Telemed FL (v31 F1).
- A per-image PA term from aponeurosis tilt (v32).
- Tying FL to MT / sin(PA) per image (v32; all three rules made VL worse).
- More traced fascicle length, DeepLab-style head swaps, about 25 other head changes (see `artifacts/candidates_v29/map/LEDGER_tried.md`).

Full reports: `artifacts/candidates_v29/REPORT_v29_new_directions.md`, `artifacts/candidates_v31/REPORT_v31.md`, `artifacts/candidates_v32/REPORT_v32.md`. Running log: `artifacts/notes/FACTS.md`.
