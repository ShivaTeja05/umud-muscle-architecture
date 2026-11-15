"""Writes notebook/umud-solution.ipynb from the cells below. GPL-3.0-only."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# UMUD Challenge — geometry-first muscle architecture (public LB 0.215)

This notebook shares our complete solution for the
[UMUD Challenge](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data):
predicting pennation angle (PA), fascicle length (FL) and muscle thickness (MT) from B-mode ultrasound images.

**What you get**

- The full source code (GPL-3.0) and the per-image measurement tables, attached as a dataset.
- A cell that rebuilds our submission files in about 30 seconds on CPU and writes `submission.csv`.
- A step-by-step description of the method, with the leaderboard effect of every step.
- An honest list of what did **not** work, so you do not have to repeat it.

| File built here | Public LB | What it is |
|---|---|---|
| `A_plus_v2.csv` (= `submission.csv`) | **0.21529** | Pure model + 5 group constants tuned on the leaderboard |
| `final_A_lb.csv` | 0.21637 | Pure model + 4 group constants |
| `final_A.csv` | 0.23041 | Pure model, no leaderboard-tuned additive constants |

**Please read this before using the files.** The step from 0.230 to 0.215 comes from five additive constants,
one per scanner group and target, chosen with public-leaderboard feedback. They are declared openly in
section 6. They are not per-image or per-clip values. If you want a file without them, use `final_A.csv`.

If this is useful to you, an upvote helps other people find it.
""")

md(r"""
## Quick start

1. Click **Copy & Edit**, then **Run All** (CPU, about one minute, no GPU needed).
2. Submit the `submission.csv` it writes. It scores **0.21529** on the public leaderboard.
3. Read section 5 if you want to know why each step is there, and section 8 before trying something new.

## What we did differently — the five things that moved the score

A plain "segment the structures, then measure them" pipeline scored about 0.38 for us. These changes took it
to 0.215. None of them is a bigger network.

| # | Change | Public LB gain |
|---|---|---|
| 1 | **Undo the non-square pixels** of Lumify and Siemens images before measuring angle and length, and apply two Telemed-specific rules | **−0.102** |
| 2 | **Do not shrink Telemed fascicle length** toward the group value | −0.041 |
| 3 | **One additive constant per scanner group**, tuned with a small pre-registered leaderboard campaign | −0.015 |
| 4 | **Measure the angle at the fascicle insertion** on Telemed, where the raters measure it | −0.003 |
| 5 | Groundwork that everything else relies on: the true mm scale from each scanner's own marks, the **median over video clips** found from image content, thickness between predicted **inner edges** | part of the 0.38 baseline |

The short version: treat it as a **measurement** problem. Most of the gain came from working out how each
scanner's images were produced and how the raters draw their lines.
""")

md(r"""
## Contents

1. The task and the metric
2. Pipeline at a glance
3. Rebuild the submissions (runs here)
4. Look at the predictions, with figures and a check against expert labels
5. The method, step by step
6. Leaderboard-tuned constants (declared)
7. Score history: what each step was worth
8. What did not work
9. Where the remaining error is
10. How to reproduce from scratch, code map, credits and licences
""")

md(r"""
## 1. The task and the metric

For each of the 309 test images we predict three numbers:

| Target | Meaning | Tolerance τ |
|---|---|---|
| `pa_deg` | Angle between a fascicle and the deep aponeurosis | 6° |
| `fl_mm` | Fascicle length between the two aponeuroses, extended in a straight line if it leaves the image | 12 mm |
| `mt_mm` | Vertical distance between the inner edges of the two aponeuroses | 3 mm |

The score is the mean over the three targets of MAE / τ. Lower is better.

The labels were drawn by two raters, three fascicles and three thickness positions per image, then averaged
(see the host's answers in the discussion forum). There are no measured values for the training images, only
segmentation masks. So the problem has two halves: **find the structures**, then **measure them the way the
raters do**.

The test set mixes scanners, and they behave very differently:

| Group | Images | Notes |
|---|---|---|
| Siemens Juniper | 91 | Stretched to 1200×800 from a 1152×864 frame |
| Telemed ArtUS 1200×800 | 90 | 18 video clips of 5 near-identical frames |
| Telemed ArtUS 1088×644 | 50 | 10 video clips of 5 frames |
| Philips Lumify | 58 | Phone screenshots, non-square pixels |
| Older formats | 20 | 512 px and 853 px images |
""")

md(r"""
## 2. Pipeline at a glance

```
image
  |
  |- 1. layout     scanner, viewport, pixel-to-mm scale, clips
  |- 2. maps       two networks (GPU): aponeurosis edges,
  |                fascicle fragments + orientation field
  |- 3. measure    lines -> MT; angle vs deep line -> PA;
  |                straight extension -> FL (3 estimators)
  |- 4. heads      one PA and one FL per image
  |- 5. shrinkage  pull noisy values toward the group value
  |- 6. clips      frames of one video share the median
  |- 7. rater      small offsets from the OSF benchmark
  |- 8. native     undo non-square pixels (Lumify, Siemens)
  `- 9. constants  five group constants from the LB (declared)
```

Steps 1–2 need the images and a GPU. Their output is a table with one row of raw measurements per image.
The output tables of step 3 are attached to this notebook, and **steps 4–9 run below**.
""")

md(r"""
## 3. Rebuild the submissions

The attached dataset holds the repository (`src/umud`, `configs`, `scripts`, trainers under `kaggle/`) and the
per-image measurement tables. The cell below runs the real build stage (`umud.run`, config `final_A.yaml`),
recomputes the scanner layout and video clips from the test images, then applies the declared constants, and compares every file with the one we scored.
""")

code(r"""
import hashlib, json, shutil, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# locate the bundled repository inside the attached dataset
hits = sorted(Path("/kaggle/input").rglob("configs/final_A.yaml"))
assert hits, "attach the dataset that holds the repository (see the Input panel)"
SRC_REPO = hits[0].parents[1]
WORK = Path("/kaggle/working")
# the competition's test images (the build reads them to find scanner groups and video clips)
cands = [d for d in Path("/kaggle/input").rglob("*") if d.is_dir() and len(list(d.glob("IMG_*.tif"))) > 100]
assert cands, "attach the competition data"
TEST_IMAGES = cands[0]
print("repository:", SRC_REPO)
print("test images:", TEST_IMAGES)
""")

code((HERE / "notebook" / "rebuild.py").read_text().split('if __name__ == "__main__":')[0].split('"""', 2)[2].replace(
    "import hashlib, json, shutil, sys\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n", "") + r"""

BUILD = Path("/tmp/umud_build")          # scratch space; only the final files go to /kaggle/working
OUT, LAYOUT, REPORT = rebuild(SRC_REPO, BUILD, TEST_IMAGES)
display(pd.DataFrame(REPORT).T)
for f in ("final_A.csv", "final_A_lb.csv", "A_plus_v2.csv"):
    shutil.copy(OUT / f, WORK / f)
shutil.copy(OUT / "A_plus_v2.csv", WORK / "submission.csv")
print("wrote submission.csv (= A_plus_v2.csv) and the two other files to", WORK)
""")

md(r"""
The table compares each rebuilt file with the file we actually scored (bundled under `reference/`). On the
machine where the files were first built they are byte-identical. On Kaggle's CPU one value out of 927 differs
by 0.0001 mm (floating-point rounding on a different processor), which does not change the score. The build has
no randomness.
""")

md(r"""
## 4. Look at the predictions, with figures and a check against expert labels
""")

code(r"""
sub = pd.read_csv(WORK / "submission.csv").set_index("image_id")
kg0 = pd.read_csv(OUT / "final_A.csv").set_index("image_id")
grp = LAYOUT["group"].reindex(sub.index)

summary = sub.groupby(grp).agg(n=("pa_deg", "size"), pa_median=("pa_deg", "median"),
                               fl_median=("fl_mm", "median"), mt_median=("mt_mm", "median")).round(2)
display(summary)

print("Effect of the leaderboard constants (A_plus_v2 minus the pure model), mean per group:")
display((sub - kg0).groupby(grp).mean().round(3))
""")

code(r"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, c, t in zip(ax, ["pa_deg", "fl_mm", "mt_mm"], ["Pennation angle (deg)", "Fascicle length (mm)", "Muscle thickness (mm)"]):
    for g, d in sub.groupby(grp):
        a.hist(d[c], bins=20, alpha=0.55, label=f"{g} ({len(d)})")
    a.set_title(t)
ax[0].legend(fontsize=7)
plt.tight_layout()
plt.show()
""")

md(r"""
### What the geometry step sees

One example per scanner group. Orange: the fitted inner edge of the superficial aponeurosis. Cyan: the deep
aponeurosis. White: the three positions where thickness is measured. The title gives the final prediction.
""")

code(r"""
from PIL import Image

REPO = BUILD / "repo"
raw = pd.read_csv(REPO / "artifacts/candidates_v3/cache/test__v2.csv").set_index("image_id").reindex(sub.index)
show = ["siemens_juniper", "telemed_artus_1200x800", "telemed_artus_1088x644", "philips_lumify"]
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
for a, g in zip(axes.ravel(), show):
    iid = grp[grp == g].index[len(grp[grp == g]) // 2]
    r = raw.loc[iid]
    a.imshow(Image.open(TEST_IMAGES / iid).convert("L"), cmap="gray")
    for pre, col in (("sup", "orange"), ("deep", "cyan")):
        x = np.array([r[f"{pre}_x0"], r[f"{pre}_x1"]])
        a.plot(x, r[f"{pre}_p1_s"] * x + r[f"{pre}_p1_b"], color=col, lw=2)
    for q in (0.25, 0.5, 0.75):
        x = r["s_lo"] + q * (r["s_hi"] - r["s_lo"])
        a.plot([x, x], [r["sup_p1_s"] * x + r["sup_p1_b"], r["deep_p1_s"] * x + r["deep_p1_b"]], color="white", lw=1.2, ls="--")
    p = sub.loc[iid]
    a.set_title(f"{g}  {iid}\nPA {p.pa_deg:.1f} deg   FL {p.fl_mm:.1f} mm   MT {p.mt_mm:.1f} mm", fontsize=10)
    a.axis("off")
plt.tight_layout()
plt.show()
""")

md(r"""
### Video clips found from the image content

Telemed frames come in runs of five. The plot shows the similarity of each image to the previous one
(normalised cross-correlation of viewport thumbnails). Values near 1 are frames of the same clip.
""")

code(r"""
ncc = pd.to_numeric(LAYOUT["ncc_prev"], errors="coerce").reindex(sub.index)
in_clip = LAYOUT["clip"].reindex(sub.index).astype(int) >= 0
fig, a = plt.subplots(figsize=(14, 3))
a.scatter(np.arange(len(ncc))[~in_clip], ncc[~in_clip], s=8, color="tab:gray", label="single image")
a.scatter(np.arange(len(ncc))[in_clip], ncc[in_clip], s=8, color="tab:red", label="frame inside a clip")
a.axhline(0.8, color="k", lw=0.8, ls="--")
a.set_xlabel("test image (file order)")
a.set_ylabel("similarity to previous")
a.legend(fontsize=8)
plt.tight_layout()
plt.show()
print("clips found:", LAYOUT.loc[LAYOUT["clip"].astype(int) >= 0, "clip"].nunique())
""")

md(r"""
### The hardest part in one picture: fascicle length on Telemed

Only a small part of each fascicle is visible, so FL is mostly a straight-line extension. We compute three
estimators and average them. On Telemed images they disagree by tens of millimetres, which is the largest
single source of error in our score.
""")

code(r"""
heads = raw[["fl_straight", "fl_central3", "fl_constructed"]]
spread = (heads.max(axis=1) - heads.min(axis=1)).rename("range_mm")
display(pd.concat([heads, spread], axis=1).groupby(grp).median().round(1))

fig, a = plt.subplots(figsize=(7, 3.6))
for g, d in spread.groupby(grp):
    a.hist(d.dropna().clip(upper=80), bins=25, alpha=0.55, label=g)
a.set_xlabel("max − min of the three FL estimators (mm, clipped at 80)")
a.set_ylabel("images")
a.legend(fontsize=7)
plt.tight_layout()
plt.show()
""")

md(r"""
### Check against expert labels (public benchmarks, not the test set)

No test labels exist for us, so we validate on three public expert-labelled sets that are **not** part of the
competition data: the OSF static benchmark (35 images, mean of 7 raters), the vastus lateralis example images
(expert-drawn fascicles) and a dynamic calf-raise sequence. The numbers below are the per-image estimates
*before* shrinkage and rater offsets.
""")

code(r"""
from umud import run as ur

cfg = ur.load_config(REPO / "configs/final_A.yaml",
                     sets=[f"paths.out_dir={json.dumps(str(OUT / 'bench'))}",
                           f"paths.test_layout={json.dumps(str(OUT / 'test_layout.csv'))}",
                           f"paths.test_images={json.dumps(str(TEST_IMAGES))}"])
B = ur.Assembler(cfg, ur.bind(cfg)).bench_frame("lodo3")
TAU = {"pa": 6.0, "fl": 12.0, "mt": 3.0}
rows = []
for ds, d in B.groupby("dataset"):
    r = {"dataset": ds, "images": len(d)}
    for k, c in (("pa", "pa_deg"), ("fl", "fl_mm"), ("mt", "mt_mm")):
        e = (d[c] - d[f"{k}_true"]).abs()
        r[f"{k.upper()} MAE"] = round(float(e.mean()), 2) if e.notna().any() else np.nan
    rows.append(r)
display(pd.DataFrame(rows).set_index("dataset"))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for a, (k, c, u) in zip(ax, (("pa", "pa_deg", "deg"), ("fl", "fl_mm", "mm"), ("mt", "mt_mm", "mm"))):
    for ds, d in B.groupby("dataset"):
        a.scatter(d[f"{k}_true"], d[c], s=10, alpha=0.6, label=ds)
    lo, hi = np.nanmin(B[f"{k}_true"]), np.nanmax(B[f"{k}_true"])
    a.plot([lo, hi], [lo, hi], color="k", lw=0.8)
    a.set_xlabel(f"expert {k.upper()} ({u})")
    a.set_ylabel(f"predicted {k.upper()} ({u})")
ax[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
""")

md(r"""
Thickness sits on the diagonal. Angle and length are good on the static images and noisier on the long
fascicles of the vastus lateralis set — the same pattern that limits the test score.
""")

md(r"""
## 5. The method, step by step

### 5.1 Layout, scale and clips (`src/umud/layout.py`, `native_geometry.py`, `clips.py`)

- **Scanner family** from image size and content. No file-name rules.
- **Scale** from the marks burned into each image: tick marks on Telemed, triangles on Siemens, the depth
  disc on Lumify. The host confirmed that the raters used the vertical ruler for both axes.
- **Viewport**: the box that contains the actual ultrasound picture.
- **Video clips**: Telemed images come in runs of 5 nearly identical frames. We find them by the normalised
  cross-correlation of consecutive viewport thumbnails (threshold 0.8; inside a clip it is ≥ 0.946, outside
  ≤ 0.561). 28 clips are found.

### 5.2 Networks (`kaggle/apo_v2`, `kaggle/fasc_orient_v1`, `kaggle/infer_maskbank`)

Both are trained on the competition training masks only, with 5 folds split by near-duplicate groups
(670 of the fascicle pairs are exact duplicates, so a random split leaks).

| Network | Architecture | Outputs |
|---|---|---|
| Aponeurosis | U-Net, ConvNeXt-Small encoder, 512 px letterbox | band, upper edge, lower edge |
| Fascicle | U-Net, ConvNeXt-Small encoder, 512 px | fragment mask, orientation field as (cos 2θ, sin 2θ) |

Two details mattered:

- **Edges as separate outputs.** Thickness is measured between the *inner* edges. Predicting the edges
  directly was more accurate than taking the border of a predicted band.
- **Doubled-angle orientation.** A fascicle has no direction, so θ and θ + 180° are the same. Regressing
  (cos 2θ, sin 2θ) removes the wrap-around problem.

A warning from our own run: the fp16 checkpoints of the aponeurosis network overflowed some BatchNorm
variances. We re-estimated the BatchNorm statistics on each fold's training split (`kaggle/apo_v2_fix`).

### 5.3 Measurement (`src/umud/geometry_v2.py`)

- **Aponeuroses.** Pick the superficial/deep pair from the maps and fit a line (or a gentle curve) to each
  inner edge.
- **MT.** Vertical distance between the inner edges at 25 %, 50 % and 75 % of the shared width, averaged.
  This follows the host's description (column distance, three positions, inner borders).
- **PA.** Two estimates against the deep aponeurosis, averaged:
  1. fragment fusion: the dominant angle of the detected fascicle fragments;
  2. field angle: the orientation field averaged in a strip just above the deep aponeurosis.
- **FL.** Mean of three straight-line estimates:
  1. *straight*: extend each detected fragment to both aponeuroses;
  2. *central3*: the same, for the three most central fragments (the raters draw three);
  3. *constructed*: a straight fascicle built at the final PA between the two fitted aponeuroses.

### 5.4 Shrinkage (`src/umud/shrinkage.py`)

Per-image values are noisy on hard images. We pull them toward the median of their scanner × muscle group:

- PA: weight = a per-image confidence from the agreement of the estimators.
- FL: empirical Bayes with a per-image error model fitted on the public OSF benchmarks
  (leave-one-dataset-out).
- MT is not shrunk. **Telemed FL is not shrunk** — shrinking it cost 0.041 on the leaderboard.

Muscle names come from OCR of the labels burned into the images.

### 5.5 Clip median

Every frame of a clip gets the clip's median value. The frames are almost the same picture, so their labels
should be almost the same; the median removes single-frame failures.

### 5.6 Rater style

The raters have small systematic habits. On the public OSF benchmark (35 images, 7 raters) we estimated them
with a rule fixed in advance and applied half of the measured value: PA + 0.33°, MT + 0.04 mm.

### 5.7 Native geometry for Lumify and Siemens (`src/umud/run.py`, post rules)

These images were resized without keeping the aspect ratio, so angles and lengths measured on the file are
distorted. With aspect factor c (Lumify 0.844, Siemens 1.125), φ the fascicle angle and δ the tilt of the deep
aponeurosis:

- PA′ = atan(c·tan φ) + atan(c·tan δ)
- FL′ = FL · sqrt(cos²φ / c² + sin²φ)

MT is vertical and does not change. **This was the largest single gain** (about 0.10 together with the Telemed
rules of 5.8).

### 5.8 Telemed rules

- **MT − 0.5 mm.** A constant for both Telemed groups, chosen on the leaderboard (declared).
- **Coupling.** For a straight fascicle, FL·sin(PA) ≈ k·MT. On the expert benchmark k = 1.067. We pull the
  Telemed PA 25 % of the way toward asin(1.067·MT / FL).
- **Insertion-angle head (G0).** On Telemed test clips the fascicle is steeper near its insertion than in the
  mid-belly. Where that is the case we use the field angle in the deep strip against the *local* tangent of
  the deep aponeurosis. On 78 expert-labelled benchmark images of that kind this removed a 0.8° bias.
""")

md(r"""
## 6. Leaderboard-tuned constants (declared)

After the model was frozen (0.23041), we ran a small, pre-registered campaign: one additive constant per
scanner group and target, at most a few probes per day, adopted only when a statistical model gave a high
probability that it also helps on the private part.

| Group | Target | Constant | Public gain |
|---|---|---|---|
| Telemed 1200×800 | PA | −0.75° | 0.003 |
| Telemed 1200×800 | MT | +0.325 mm | — |
| Telemed 1088×644 | PA | +1.10° | 0.005 |
| Siemens Juniper | PA | +0.85° | 0.004 |
| Philips Lumify | PA | +0.50° | 0.001 |

Together they take 0.23041 to 0.21529. We never tuned per image or per clip. Whether these five numbers hold
on the private leaderboard is unknown at the time of writing; `final_A.csv` is the file without them.
""")

md(r"""
## 7. Score history

| Step | Public LB | Change |
|---|---|---|
| First complete geometry pipeline with shrinkage and clip median | 0.3767 | |
| Stop shrinking Telemed FL | 0.3353 | −0.041 |
| Native geometry (Lumify, Siemens) + Telemed MT −0.5 mm + PA coupling | 0.2330 | −0.102 |
| Telemed insertion-angle head | 0.2304 | −0.003 |
| Four group constants | 0.2164 | −0.014 |
| Fifth group constant (Lumify PA) | 0.2153 | −0.001 |

The lesson we would pass on: **the big steps came from understanding how the images were made and how the
labels were drawn, not from better networks.**
""")

code(r"""
steps = ["first full\npipeline", "no Telemed\nFL shrinkage", "native geometry\n+ Telemed rules", "insertion-angle\nhead",
         "4 group\nconstants", "5th group\nconstant"]
lb = [0.3767, 0.3353, 0.2330, 0.2304, 0.2164, 0.2153]
fig, a = plt.subplots(figsize=(11, 3.6))
bars = a.bar(steps, lb, color=["tab:blue"] * 4 + ["tab:orange"] * 2)
for b, v in zip(bars, lb):
    a.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
a.set_ylabel("public LB (lower is better)")
a.set_title("Blue: model changes.  Orange: leaderboard-tuned constants.")
plt.tight_layout()
plt.show()
""")

md(r"""
## 8. What did not work

We tested these offline against the public expert benchmarks under rules fixed before each test. None gave a
gain that survived.

| Idea | Outcome |
|---|---|
| Ensembling variants of our own pipeline | Their errors correlate 0.93–1.00; gain 0.000–0.002 |
| Direct image → (PA, FL, MT) regressor on ~500 labelled public images | Far too weak (PA MAE about 2.1° on held-out images, FL 6.9 mm) |
| Better fascicle orientation networks (two attempts) | Orientation error fell from 1.69° to 1.22° / 1.43°, measured PA did not improve |
| Replacing our deep-aponeurosis line with the expert's line | PA error did not fall (1.76° → 1.88°) |
| A per-image PA correction from aponeurosis tilt | Did not transfer between datasets |
| Tying FL to MT / sin(PA) per image | Worse on held-out images for all three rules tried |
| One global deep line for Telemed FL | Worse on both benchmarks |
| Using long-fascicle Siemens VL images as a stand-in for Telemed FL | Predicted the wrong sign for 2 of 3 known leaderboard results |
| Zero-shot SAM (ViT-B) for muscle thickness, with ideal point prompts | Median thickness error 112–144 % against 0.66 % for our network |
| External gastrocnemius tracking data as a Telemed benchmark | Wrong regime (pennation about 40° against about 13° in the test set) |
| Published tools as replacements (DL_Track_US, TimTrack-style field, SMA-style) | Benchmarked; none better than the pipeline above |
""")

md(r"""
## 9. Where the remaining error is

Our estimate of how the 0.215 splits:

| Target | Share of the score |
|---|---|
| Fascicle length | 0.120 (Telemed alone: 0.060) |
| Pennation angle | 0.066 |
| Muscle thickness | 0.031 |

- **Telemed FL** is the largest item. The detected fragments cover only 6–7 % of a fascicle, so one degree
  of angle changes the length by 6–7 mm. The three estimators typically disagree by 20–30 mm (table in section 4), and no public
  labelled set looks like these images.
- **PA** is compressed on hard images: predictions move only about 0.7–0.86 as much as the expert angles.
- **MT** is at the level of rater agreement on Siemens and about twice that on Telemed and Lumify.

If you want to improve on this, labelled images of long fascicles on Telemed-like frames would help more than
a new architecture.
""")

md(r"""
## 10. Reproduce from scratch, code map, credits

**From scratch**

1. Train the two networks: `kaggle/apo_v2/train.py`, `kaggle/fasc_orient_v1/train.py` (Kaggle 2×T4, a few hours each).
2. Run them over the images: `kaggle/infer_maskbank/pipeline.py`.
3. `python scripts/umud_run.py --config configs/final_A.yaml --stage measure` — maps to measurement tables.
4. `python scripts/umud_run.py --config configs/final_A.yaml --stage build` — tables to submission (this notebook).

**Code map**

| File | Role |
|---|---|
| `src/umud/run.py` | Entry point, heads, post rules, assembly |
| `src/umud/geometry_v2.py` | Maps → raw PA / FL / MT |
| `src/umud/shrinkage.py` | Shrinkage toward group values |
| `src/umud/native_geometry.py`, `layout.py`, `clips.py` | Scale marks, scanner layout, clip detection |
| `configs/*.yaml` | Every constant, with its evidence, and sha256 of every input |
| `README.md` | Longer description and the reproduction notes |

**Credits and licences**

- Code: GPL-3.0-only.
- The viewport and scale reader in `src/umud/geometry.py` is adapted from a public notebook by **AmbrosM**
  in this competition. Thank you.
- DL_Track_US (Ritsche et al., Apache-2.0) is used only as a fallback path that the final files never reach.
- Public benchmark labels: UMUD benchmark datasets on OSF (node `xbawc`, CC BY 4.0). They are used for
  validation, the FL error model and the rater offsets.
- Competition data: CC BY-NC-SA 4.0. No competition images are redistributed here; the attached tables hold
  our own measurements.

Thanks to Paul Ritsche and the organisers for the competition and for answering protocol questions in the forum.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
(HERE / "notebook" / "umud-solution.ipynb").write_text(json.dumps(nb, indent=1))
print("cells", len(cells))
