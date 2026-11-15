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
.venv/bin/python tools/annotator/test_assist.py     # assist maths, JS/Python parity (needs node; skipped without)
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
| <kbd>E</kbd> | Geometry assist on / off (*Auto-extend to the bands*, on by default; see [Assists](#assists)). |
| <kbd>Y</kbd> | Use the suggested lines as a starting point (only on a data dir with `prefill_enabled`, while they are shown). |
| <kbd>?</kbd>/<kbd>H</kbd> | Help. |

The right panel has these parts:
- Brightness, contrast and gamma sliders, plus an *auto-contrast each image* option.
- A per-measurement table with delete buttons.
- Notes.
- A progress grid with one cell per task: grey to do, amber partial, green done, red flagged, outlined for clips. Clicking a cell jumps to that task.
- Time per image, total time and ETA.

## Assists

Two optional helps. Neither changes the stored measurement format: `mt`, `fl` and `pa` are the same point lists
as before. The maths, labels.py, the CSVs and the scorers are therefore unchanged. What each assist did is
recorded on the label (below), so assisted and unassisted labels can be compared later.

**Geometry assist** (every data dir; checkbox *Auto-extend to the bands (E)* in the right panel; setting
`state.settings.geomAssist`, default on). It uses the image's **own MT clicks**:
- Bands: the superficial band is a polyline through the upper point (smaller y) of each MT line. The deep band
  goes through the lower points. Both are sorted by x and continue straight past the outermost points with the
  slope of the nearest segment. With 2 MT lines a band is a straight line. Points under 0.5 px apart in x are
  averaged. With fewer than 2 MT lines the assist is inactive and the FL / PA hint says *Draw 2-3 MT lines first
  to enable auto-extend*. While the assist is active, the bands are drawn thin and dashed (green), past the image edges.
- FL: click 2 or more points along a visible fascicle, in any direction, and finish with <kbd>Enter</kbd> or a
  double-click. The end nearer the deep band (by relative depth between the bands) moves along its end segment
  onto the deep band, and the other end onto the superficial band. An end inside the muscle is extended. An end
  clicked beyond its band is trimmed where the line crosses the band. That includes a 2-click line from one
  bright band to the other. Stored: `[deep point, ...inner clicks..., superficial point]`, which is deep to
  superficial as the protocol asks. If an end segment never meets its band (parallel, or it points away), or meets it more
  than 3 image widths away, the raw line is kept and a warning is shown. A dashed preview and the live value
  (`..., to the bands NN mm`) show the result before you finish.
- PA: after the 2 fascicle clicks, the deep-aponeurosis arm is put on the deep band at x -/+ 5 mm
  (`0.5 * px_per_cm` px) around the point where the fascicle line meets it, and the PA is committed with 4 points
  as before. If the line does not meet the band, you click the 2 arm points yourself. With the assist off, PA
  takes 4 clicks.
- All points can be dragged afterwards. Moving an MT point later does not re-extend existing lines.

**Pre-drawn suggestions** (only when `manifest.prefill_enabled` is `true`, i.e. the work set; never on
calibration tasks; practice and check never show them, even if an entry had `prefill`):
- Manifest keys: top level `prefill_enabled`, `prefill_source`, `prefill_blind_rule`. Per main-frame image entry,
  `prefill` = `{"mt": [[p,p] x3], "fl": [[p,...,p] x3], "pa": [[f0,f1,a0,a1] x3], "apo_sup": [p,...],
  "apo_deep": [p,...], "confidence": 0..1}` in native image pixels, in the same formats as the labels. Malformed lines
  are dropped, at most 3 per tool are used, and `confidence` is not shown. Per task, `prefill_blind: true` marks a
  control task.
- A task with a suggestion shows it as dashed ghost lines in the tool colours, with a banner: *Suggested lines from
  our model: press Y to use them as a starting point, or just draw your own*. <kbd>Y</kbd> (or the banner button)
  copies mt / fl / pa into the label as normal editable measurements (MT deep x locked to the superficial x, as
  the tool does). Drawing your own first line instead hides the ghosts for that image. Undo brings them back.
- A blind control task, and an image without a suggestion, show only the neutral note *No suggestions on this image*.
- The work-set review card shows only your own values, their spread and the geometry check.

**Label fields** (added by app.js; labels.py, the CSV exports and the scorers ignore them; undo / redo and delete
keep them in step with the lines):

| Field | Content |
|---|---|
| `assist.mt`, `assist.fl`, `assist.pa` | One code per measurement, parallel to `mt` / `fl` / `pa`: `hand` (assist off, or MT), `ext` (FL auto-extended / trimmed), `arm` (PA deep arm auto-placed), `nomt` (assist on but fewer than 2 MT lines), `fail` (assist on, extension or arm impossible: drawn by hand), `prefill` (copied from the suggestion). `null` = made before this version. |
| `fl_raw` | The FL clicks as made, parallel to `fl` (`null` for a copied suggestion). |
| `assist.prefill_shown`, `assist.prefill_accepted`, `assist.prefill_blind`, `assist.prefill_t` | Work set only. Set at the first line or at <kbd>Y</kbd>: shown = the ghosts were on screen, accepted = <kbd>Y</kbd> was used (`prefill_t` its time in ms), blind = control task. An image without a suggestion gets all three `false`. |
| `prefill_start` | `{mt, fl, pa, t}`: a deep copy of what <kbd>Y</kbd> copied. The final lines minus this are the user's corrections. |

`first` (the first complete attempt, see below) works as before. A label that was only accepted with <kbd>Y</kbd>
gets `first` only when a line is committed while it is complete.

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
| `prepare_manual_kit.py` | Builds the manual labelling kit in `data/manual_labelling/` from the OSF static benchmark only (see below). |
| `score_check.py` | Grades the 10-image skills check (annotator autosave / CSV export, or a filled FIJI `answer_template.csv` plus its ROI files) against the hidden key and prints GOOD / PASS / FAIL, a copy check and the ROI comparison. `--practice` grades practice first attempts. |
| `check_work_overlap.py` | Read-only check that no work-set frame is a competition test image; blocks the work set if one is. |
| `test_score_check.py` | Unit checks for the check scorer, the kit files (no leaks, identical PNGs) and the deny list. |
| `assist_geom.py` | Python mirror of the geometry-assist maths (the `GEOM` block in app.js): bands from MT clicks, FL extension / trimming, the PA deep arm. |
| `test_assist.py` | Known-answer checks of that maths, a node run of the JS block against the Python mirror (930 cases), the page ids app.js needs, and labels.py reading labels that carry the assist fields. `--app-js PATH` checks another copy. |

Data files (all in `data/annotator/`, gitignored): `img/`, `manifest.json` (test),
`calib/` and `calib_manifest.json` (calibration, no answers), `calib_answers.json` (expert answers,
never served), `calib_pipeline.json` (cached C12 benchmark predictions), `calib_score.json` (the
scorer's output), `autosave.json`, `labels.csv`, `calib_labels.csv` and `backups/`.

Coordinates are native image pixels with x to the right and y down. Each saved point is rounded to 0.01 px.

## Manual labelling kit (practice + skills check)

`data/manual_labelling/` (git-ignored) is built from the 35 public OSF static benchmark images (CC BY 4.0) only.
No competition test image is used. One process, about 90 s the first time (2 s later), under 0.2 GB.

> **The labels are external data.** Labels made with the kit (practice, check or the work set) are new external
> data. If they change which model we submit, we must publish them (CC BY) and declare them in the competition
> write-up, with the image source and licence (URL, version, licence, preprocessing).

```bash
PYTHONPATH=src .venv/bin/python tools/annotator/prepare_manual_kit.py          # rebuild from private/selection.json
.venv/bin/python tools/annotator/serve.py --data data/manual_labelling/practice # 25 practice images (PRAC_01-25)
.venv/bin/python tools/annotator/serve.py --data data/manual_labelling/check    # 10 check images (CHK_01-10)
PYTHONPATH=src .venv/bin/python tools/annotator/score_check.py --state data/manual_labelling/check/autosave.json
PYTHONPATH=src .venv/bin/python tools/annotator/score_check.py --csv <filled answer_template.csv> [--rois <folder>]
PYTHONPATH=src .venv/bin/python tools/annotator/score_check.py --practice      # first attempts, last 5 practice images
.venv/bin/python tools/annotator/check_work_overlap.py                          # work set vs the test images (read only)
.venv/bin/python tools/annotator/test_score_check.py
```

- **practice/**: a normal data dir with `manifest.json` only (no `calib_manifest.json`, so there is no
  calibration phase and no gate). Each image has `clip: null`. Its `hint` is the expert mean and is shown only
  after the image is finished, with the experts' min-max range (`hint_range`) and the number of experts per value
  (`hint_n`).
- **check/**: the 10 images are calibration tasks (`calib_manifest.json`, no answers). `manifest.json` is an
  empty shell because serve.py needs one. The images are **mirrored left-right** (MT, FL and PA do not change; a
  file or pixel match with the public TIFF fails; listed in CREDITS.txt). There is **no answer file** in check/.
- **private/**: `check_key.json` (CHK -> source image, every rater's value), `calib_answers.json` (the same
  answers in the `data/annotator/calib_answers.json` format, `source` = CHK id), `pass_bar.json`,
  `practice_key.json`, `selection.json` and `selection.md`. `selection_v1_void.json` is the first draw, void
  because 7 of its check images were old calibration images. private/ sits outside every served dir. Never send it.
  The CHK order and the choice of check images exist only here: this script does not contain them.
- **Selection rules** (`--init-selection`, refused unless overridden): no check image from the old calibration
  set (`data/annotator/calib_answers.json`; its sources are listed above, so it is not blind), and no check image
  that is a near-duplicate of a practice image (32x32 viewport thumbnail NCC > 0.75; some benchmark images are near-duplicate groups), because
  practice shows its answers. Which images are in the check is written only in private/.
- **expert_package/**: `images/CHK_XX.png` (byte-identical to `check/calib/`), `scales.csv`,
  `answer_template.csv`, `CREDITS.txt` (CC BY 4.0 attribution and the changes made) and `INSTRUCTIONS.html/.pdf`.
  For an outside expert the check is honour-based: the source must be named (CC BY) and the public workbook holds
  every rater's values.
- **Verdict** (`score_check.py`): S = mean over PA / FL / MT of MAE / tau. The candidate is graded against the
  mean of all raters. Each expert rater is graded against the mean of the *other* raters (leave-one-out), on the
  same image x parameter cells. GOOD = S <= the median expert, PASS = S <= the worst expert, FAIL otherwise.
  The result is INCOMPLETE if fewer than 8 images have all three parameters. With only 10 images, the scorer
  also prints a bootstrap over images.
- **The check stays blind** (it can be scored only once): while INCOMPLETE the scorer prints no expert value, no
  error and no score; afterwards it prints expert values only for cells the candidate measured. `--reveal`
  overrides both. A missing or empty `--state` / `--csv` is an error. The saved report (`private/check_results/`)
  is redacted the same way.
- **Copy check**: an image matches when 2 or more parameter means are within 0.05 of one published value (a rater,
  the rater mean, or the workbook SMA / DL_Track value); 2 or more such images = SUSPECT. Single values are not
  used: on the workbook, honest raters match some published value on most images.
- **`--rois <folder>`** reads the returned FIJI `CHK_XX_rois.zip` files (ImageJ .roi: straight lines, segmented
  lines, angles; integer or sub-pixel), recomputes MT / FL / PA with the image scale, lists every mismatch with the
  CSV (> 0.1 mm MT, 0.5 mm FL, 0.5 deg PA) and prints S from the ROI files alone.
- **Unit slips** in a FIJI CSV are warned (and fixed with `--fix-units` only when clear): cm, radians, pixels,
  10x (Set Scale with px_per_mm and Known distance 10), and obtuse PA (180 - x). Pixels and 10x are often both
  plausible (7.7 px/mm vs 10): then the row is AMBIGUOUS and never changed; the ROI files settle it. The CSV may be
  UTF-8, cp1252 or MacRoman (Excel's plain "CSV").
- **`--practice`** grades the practice autosave on each image's **first** complete attempt (`first`, see below)
  for the last 5 finished images, against the experts' leave-one-out scores on the same images.
- **Work set**: `check_work_overlap.py` compares every WRK PNG with the competition test images (read only: file
  hash, pixel hash, HxW, 32x32 thumbnail NCC on the ultrasound field, both orientations). A hash hit or NCC > 0.95
  renames `work/manifest.json` to `manifest.BLOCKED.json`. Its report in private/ has no test file names.

Optional manifest keys used by the kit. They are all backward compatible, and `data/annotator/` uses none of them:
- `manifest.json`: `storage_key` (a separate browser storage per data dir, so practice and check served on the
  same port do not mix; `_v2` since the check was redrawn), `phase_name`, `progress_label`, `help_order`,
  `hint_is_answer`, `hint_label`, `hint_label_short`.
- per image: `hint_range`, `hint_n` (a number, or one per value when some expert values are missing).
- `calib_manifest.json`: `phase_name`, `progress_label`, `review_note`.

Label field added by app.js (every data dir; labels.py and the CSV exports ignore it): `first` = `{mt, fl, pa, t}`,
a frozen copy of the clicks the first time the image was complete. Moving points later does not change it.
The assist fields (`assist`, `fl_raw`, `prefill_start`) and the manifest keys `prefill_enabled`, `prefill_source`,
`prefill_blind_rule`, per-image `prefill` and per-task `prefill_blind` are described under [Assists](#assists).
serve.py compares denied folder names in lower case (the macOS disk ignores case) and also denies
`selection*`, `work_selection*` and `build_cache*`.
