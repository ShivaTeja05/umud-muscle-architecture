# UMUD offline annotator

A small browser tool for hand-measuring PA, FL and MT on the 309 UMUD test images. It follows
the organiser's labelling protocol. It runs fully offline: plain HTML, CSS and JS with no
network dependencies, served by a stdlib Python server bound to `127.0.0.1`.

> **Competition data must stay local.** `data/annotator/` is gitignored. Never upload, publish
> or share the PNGs, `manifest.json`, `autosave.json`, `labels.csv` or the `calib_*` files. Hand
> labels of test images count as **external data**. If they are used in any way (direct
> submission, blending, calibration, fine-tuning), they **must be declared** in the competition
> write-up and discussion. The organiser has said this is "permitted-if-declared" (threads
> 690868, 737782), but Foundational Rule 4.b reads the other way. Deciding whether to use them is
> the team's call. The calibration images are public OSF benchmark images (CC BY 4.0), not test
> data, but declare them too, because they decide how much weight the hand labels get.

## How to do it

1. Start the tool: `.venv/bin/python tools/annotator/serve.py`, then open http://127.0.0.1:8765/ in
   Chrome or Safari. Read the help (<kbd>?</kbd>) once. It has a schematic of where to click.
2. **Calibration first (12 images, about 15-25 min).** The queue starts with CAL_01 to CAL_12.
   These are public benchmark images whose expert answers are known but hidden from you. Measure
   them exactly as you would a test image. Do not rush them and do not try to "guess the
   answer". Their only purpose is to measure *your* error and bias per parameter.
   - CAL_03, CAL_06, CAL_09 and CAL_12 are frames of a calf-raise video. On these, **only FL and PA
     are needed** (MT is optional). Measure PA against the deep aponeurosis as on every other image.
   - The test images stay locked until all 12 are done (or flagged with <kbd>X</kbd> if one really
     cannot be measured). *Skip calibration...* exists, but then your labels can only get a
     low, generic weight.
3. **Then the 28 Telemed clips** (IMG_00056-195, first frame of each clip, `Telemed clips` in the
   top bar). These matter most: they are the images where our model is weakest.
4. The other 169 test images come after that, only if there is time.
5. When the calibration is done, run the scorer (below) and send the printout. It decides how
   your Telemed labels are blended with the model (and whether a bias correction is applied).

## Protocol tips (what experts do)

```
   superficial aponeurosis (bright band)
 ==================================================   <- click its LOWER edge (inner = muscle side)
   |  MT   \   \   FL: a central    \   \        |
   |        \   \    fascicle,       \   \       |     MT: 3 vertical lines, left / middle / right
   |         \   \   apo to apo       \   \      |         (the dashed guides), inner edge to inner edge
   |          \   \                    \  /\     |
 ==================================================   <- click its UPPER edge (inner = muscle side)
   first deep aponeurosis                PA: angle between the fascicle and the deep
                                             aponeurosis, measured right at the insertion
```

- **Inner edges.** MT runs between the *inner* edges of the two aponeuroses: the bottom of the
  bright superficial band and the top of the bright deep band. It never goes through a bright band.
  Zoom in (wheel) and use the loupe. A 0.5 mm error on MT costs as much as 1 deg on PA.
- **Superficial muscle only.** The deep aponeurosis is the *first* bright band below the top one.
  Some Telemed clips show a second muscle below it. Ignore that one.
- **FL: central fascicles, least extrapolation.** Choose fascicles in the middle of the image that
  you can follow from one aponeurosis to the other. If a fascicle leaves the image, keep clicking
  *outside* the image along its straight continuation until it meets the (extended) aponeurosis.
  Prefer long, clearly visible fascicles. Flick the clip frames (<kbd>,</kbd> <kbd>.</kbd>) to see them better.
- **PA at the insertion.** The first 2 clicks go along the fascicle *close to where it meets the deep
  aponeurosis* (not mid-muscle). The next 2 go along the deep aponeurosis *right there*. Use 3
  different fascicles, not the FL ones.
- Values of a parameter that disagree a lot with each other (the review card shows the spread)
  usually mean one measurement used the wrong structure.

## Calibration and scoring

```bash
# once (already done): 12 benchmark PNGs + calib_manifest.json (no answers) + calib_answers.json
PYTHONPATH=src .venv/bin/python tools/annotator/prepare.py --calib
# after the calibration (reads data/annotator/autosave.json, or --csv an exported calib CSV)
PYTHONPATH=src .venv/bin/python tools/annotator/score_calibration.py
```

- The set has 8 static OSF images and 4 dynamic frames. The static images are 6 Telemed Echo
  Blaster images at 77 px/cm, all from the "clean" set our models never trained on: im_02, 03, 14,
  17, 20 and 31. The other 2 static images are im_04 and im_15, at 126 px/cm, the same scale as the
  1088x644 Telemed test clips. The dynamic frames are 8, 48, 96 and 128 of the GM calf-raise video,
  which span its PA/FL range. The order is fixed and interleaved.
- Expert answers (`calib_answers.json`):
  - static: the mean of R1-R7, with im_19 R7_MT set to NaN. The per-rater values are kept.
  - dynamic: the mean of the 3 raters.
  - The scale is the expert one. Static uses the workbook `Scale_pixel_per_cm`. Dynamic uses 77.4
    px/cm (`DYN_PPC` in scripts/build_candidates.py); dynamic FL is compared in pixels anyway.
  - `serve.py` refuses to serve `calib_answers*`, `calib_score*`, `calib_pipeline*` and
    `backups/`, and the page never requests them.
- The dynamic benchmark measures PA against the image **horizontal**, not the deep aponeurosis.
  The tool records both:
  - `pa_deg` is the usual angle against the deep aponeurosis.
  - `pa_h_deg` is the angle of the fascicle line (first 2 PA clicks) against the horizontal.
  The scorer uses `pa_h_deg` on the dynamic frames and `pa_deg` everywhere else. This is not shown in the tool.
- `score_calibration.py` prints and writes `data/annotator/calib_score.json`. It reports:
  - per image: your value, the expert value, the error, and the error divided by the tolerance.
  - per component (PA 6 deg, FL 12 mm, MT 3 mm): MAE, bias with its t value, RMSE, MAE/tol, the
    leave-one-out bias-corrected MAE, and a split by image type.
  - you against single raters, next to the expert-vs-expert spread on the same images, and your
    rank among the raters.
  - pipeline C12, using its honest leave-one-dataset-out benchmark predictions, cached in
    `calib_pipeline.json`. They take about 2 s to compute and need no torch. The report also
    shows the SMA and DL_Track values from the workbook.
  - blend weights for your labels. The weight measured on the benchmark understates you, because
    the pipeline was built on these benchmarks and its leaderboard error is about 3x its benchmark
    error. The *test-domain* weight corrects for this, and is the one to use.

## Quick start (from the project root)

```bash
# 1. Convert the test images to PNG and write the manifest. This is light: about 15 s and 125 MB RSS, one image at a time.
PYTHONPATH=src .venv/bin/python tools/annotator/prepare.py \
    --hints artifacts/sub_v1_geom_clipmed.csv      # optional model hints, shown only AFTER you finish an image

# 1b. The calibration set (12 benchmark images; about 25 s)
PYTHONPATH=src .venv/bin/python tools/annotator/prepare.py --calib

# 2. Serve it (127.0.0.1 only) and open http://127.0.0.1:8765/
.venv/bin/python tools/annotator/serve.py           # Ctrl+C to stop

# 3. Optional: recompute or export in Python (identical to the browser's maths)
.venv/bin/python tools/annotator/labels.py          # autosave.json -> labels.csv + calib_labels.csv
PYTHONPATH=src .venv/bin/python tools/annotator/score_calibration.py   # calibration report
.venv/bin/python tools/annotator/labels.py --submission /tmp/hand_sub.csv \
    --fill artifacts/sub_v1_geom_clipmed.csv        # 309 rows: finished hand labels, model values for the rest
.venv/bin/python tools/annotator/test_labels.py     # unit checks
```

`prepare.py` writes `data/annotator/img/<stem>.png` at native resolution. It also writes
`data/annotator/manifest.json` with one entry per image: `image_id`, size, `px_per_cm` from
`umud.geometry.ultrasound_metadata` (the vertical scale bar), scanner `group`, ultrasound
`viewport`, `clip` and the optional `hint`. It also lists the **197 tasks**: 28 clip tasks plus 169 singles.

- Clips: IMG_00056 to IMG_00195 are 28 clips of 5 near-identical frames, with `clip = (n - 56) // 5`.
  Only the first frame of each clip is measured. On export its values are copied to all 5 frames.
  The keys <kbd>,</kbd> and <kbd>.</kbd> flick through the frames for viewing only, because the motion can make fascicles easier to see.
- Groups: siemens_juniper 91, telemed_artus 140, philips_lumify 58, legacy_853 12, legacy_512 8.

Task queue: 12 calibration tasks first, then the 28 Telemed clip tasks, then the other 169 (209 in
total). The top bar shows the phase and its position, for example `Calibration 3/12` or
`Telemed clips 5/28`. The progress grid has one section per phase.

## Protocol (organiser; what the tool enforces)

Measure the **superficial muscle only**. That means the top aponeurosis and the *first* deep
aponeurosis below it. Ignore any deeper muscle, which appears in some Telemed clips. Each
parameter is the **mean of 3** measurements.

| | Clicks | Value |
|---|---|---|
| **MT** x3 | Click the inner (lower) edge of the superficial aponeurosis, then the inner (upper) edge of the deep aponeurosis. **x is locked to the first click**, so the line is vertical in the global image. Use the dashed guides at 25/50/75 % of the ultrasound field (left, middle, right). A click within 8 px of a guide snaps to it. | \|Δy\| × mm/px |
| **FL** x3 | Trace a fascicle as a polyline from the deep to the superficial aponeurosis, following its curvature. Points **outside the image are allowed**, so you can extrapolate by hand. Pick fascicles that need the **least extrapolation**, near the centre. **Double-click** or <kbd>Enter</kbd> ends the line. | polyline length × mm/px |
| **PA** x3 | Click 2 points along a fascicle close to its **insertion**, then 2 points along the **deep aponeurosis** at that insertion. Use different fascicles from the FL ones. | acute angle between the two lines |

Scale: `px_per_cm` comes from the **vertical** scale bar and is applied to both axes. Pixels are
treated as square even on the stretched 1200x800 images, exactly as the organisers do. Never
correct for anisotropy. Press <kbd>K</kbd> to overlay a 1 cm grid and check that it lines up with the
on-image ruler. The top bar shows the scale. A `(heuristic!)` tag means the decode fell back to a
heuristic, which does not happen on the current test set.

The tool selects MT, then FL, then PA automatically as each one reaches 3. Once all 9
measurements are done, a **review card** appears. It shows the model hint and the difference,
flags large gaps (>4 deg, >15 mm, >2 mm), and gives a geometry check
`FL ~ 1.065 * MT / sin(PA)`. Only correct clear mistakes, such as using the wrong aponeurosis.
**Do not nudge values toward the model**, because the model is often wrong. On calibration images
the review card shows only your own values, their spread and the geometry check. It never shows a
model value or an answer. On the dynamic frames the tool starts with FL, and MT is marked `opt.`.

## Controls

| Input | Action |
|---|---|
| Click | Place a point. |
| Drag (any button) or Space+drag | Pan. |
| Wheel or pinch | Zoom at the cursor. Tick *trackpad mode* to make two-finger scroll pan. |
| Drag an existing point | Adjust it. An MT deep point keeps its x. |
| <kbd>M</kbd>/<kbd>1</kbd>, <kbd>F</kbd>/<kbd>2</kbd>, <kbd>P</kbd>/<kbd>3</kbd> | Choose the tool. |
| <kbd>Enter</kbd> or double-click | Finish the fascicle. |
| <kbd>Esc</kbd> | Cancel the measurement in progress. |
| <kbd>Backspace</kbd> or <kbd>Cmd/Ctrl+Z</kbd> | Undo the last point, then the last measurement. <kbd>Shift+Cmd/Ctrl+Z</kbd> redoes. |
| <kbd>N</kbd>/<kbd>→</kbd>, <kbd>B</kbd>/<kbd>←</kbd>, <kbd>Shift+N</kbd> | Next, previous, next unfinished. |
| <kbd>0</kbd>/<kbd>Z</kbd>, <kbd>+</kbd>, <kbd>-</kbd> | Fit, zoom in, zoom out. |
| <kbd>A</kbd> | Auto contrast (1st to 99.5th percentile of the ultrasound field). |
| <kbd>G</kbd>, <kbd>L</kbd>, <kbd>K</kbd>, <kbd>V</kbd> | Toggle guides, loupe, 1 cm grid, value labels. |
| <kbd>X</kbd> | Flag the image (unmeasurable or unsure). It counts as handled. |
| <kbd>?</kbd>/<kbd>H</kbd> | Help. |

The right panel has these parts:
- Brightness, contrast and gamma sliders, plus an *auto-contrast each image* option.
- A per-measurement table with delete buttons.
- Notes.
- A progress grid with one cell per task: grey to do, amber partial, green done, red flagged, outlined for clips. Clicking a cell jumps to that task.
- Time per image, total time and ETA.

## Saving and export

- **Autosave** runs on every change, to two places. One is the browser's `localStorage`. The other
  is `data/annotator/autosave.json`, written through `serve.py`. The server also writes
  `data/annotator/labels.csv` (test tasks only), `data/annotator/calib_labels.csv` (calibration
  tasks only) and an hourly copy in `data/annotator/backups/`.
- **Export calib CSV** downloads `umud_calib_labels_<stamp>.csv`. It has one row per calibration task,
  with the columns `image_id, kind, pa_deg, pa_h_deg, fl_mm, fl_px, mt_mm, mt_px, clicks_json, task_id, n_pa,
  n_fl, n_mt, need_mt, complete, flag, note, px_per_cm, active_s, updated`. It is byte-identical to
  the server's `calib_labels.csv`, and `score_calibration.py --csv` accepts it.
- Saves are merged per image, and the newest edit wins. A second browser tab or a stale browser
  copy therefore cannot wipe newer labels on disk.
- **Export CSV** downloads `umud_hand_labels_<stamp>.csv`, with one row per image and clip frames
  expanded. Its columns are `image_id, pa_deg, fl_mm, mt_mm, clicks_json, task_id, clip, n_pa, n_fl, n_mt,
  complete, flag, note, px_per_cm, active_s, updated`. `clicks_json` holds the raw native-pixel
  clicks: `{"task_id", "px_per_cm", "mt": [[p0,p1],...], "fl": [[p0,...,pk],...], "pa": [[f0,f1,a0,a1],...]}`.
  Values are the mean over the measurements present. Use rows with `complete == 1`.
- **Import CSV** restores labels from `clicks_json`, which works with any export. It asks before overwriting existing labels.
- The browser export and the server's `labels.csv` are byte-identical. Both were verified, and
  `labels.py` reproduces the JS maths. The submission format is `image_id,pa_deg,fl_mm,mt_mm`, and IDs keep their extension.

## Files

| File | Purpose |
|---|---|
| `prepare.py` | Converts the test images to PNG and writes the manifest. With `--calib` it builds the calibration set instead: the PNGs, `calib_manifest.json` and `calib_answers.json`. |
| `serve.py` | Local server on 127.0.0.1. It uses no external packages. It rejects non-loopback Host headers and cross-origin API calls, and never serves the calibration answers, scores or backups. |
| `index.html`, `app.js`, `style.css` | The viewer and annotator. |
| `labels.py` | Python version of the maths, the merge logic, CSV/submission writers and the CLI. |
| `score_calibration.py` | Scores the calibration labels against the hidden expert answers. It also compares them with the pipeline and suggests blend weights. |
| `test_labels.py` | Unit checks: maths, clip expansion, merge, calibration rows, scorer, and the answer files never being served. |

Data files (all in `data/annotator/`, gitignored): `img/`, `manifest.json` (test),
`calib/` and `calib_manifest.json` (calibration, no answers), `calib_answers.json` (expert answers,
never served), `calib_pipeline.json` (cached C12 benchmark predictions), `calib_score.json` (the
scorer's output), `autosave.json`, `labels.csv`, `calib_labels.csv` and `backups/`.

Coordinates are native image pixels with x to the right and y down. Each saved point is rounded to 0.01 px.
