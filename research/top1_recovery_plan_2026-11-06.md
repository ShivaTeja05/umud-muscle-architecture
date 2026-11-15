# Top-1 recovery plan — 2026-11-06

## Decision in one sentence

Keep the verified `APLUS_v2_FinalA_LUMPA0.5.csv` (**0.21529 public**) as the
fallback, but stop treating output-constant searches as the path to first
place.  A credible top-1 attempt requires a new, independently validated
measurement model; the best available route is legal external expert-labelled
data plus subject-held-out, scanner-aware validation.

This plan deliberately does **not** use manually measured competition test
images, test-image pseudo-label adaptation, hidden-label inference, or a
leaderboard-fitted blend.

## Audited position

| Item | Value | Evidence |
|---|---:|---|
| Best scored submission | **0.21529** | Kaggle submission history, 2026-11-06 12:34 UTC |
| Best submission file | `APLUS_v2_FinalA_LUMPA0.5.csv` | A+ v2 ensemble plus Lumify PA +0.5 |
| Current public leader | **0.16485** | Live leaderboard, 2026-11-06 |
| User target | 0.17000 | Would presently be second, not first |
| Gap to 0.17 | **0.04529** | 21.0% relative reduction |
| Gap to current first | **0.05044** | 23.4% relative reduction |
| Deadline | 2026-11-14 23:00 | Competition metadata |

The score weights are approximately `(PA / 6 + FL / 12 + MT / 3) / 3`.
If a gain came from only one component, closing the first-place gap would mean
about **0.91 degrees PA**, **1.82 mm FL**, or **0.45 mm MT** of average error.
That is a model-quality improvement, not a calibration tweak.

## What the existing work proves

1. The A+ route works as a defensive ensemble: recent public scores improved
   from 0.23041 through 0.21637 to 0.21529.  It remains the protected fallback.
2. The pre-registered v19 constant campaign forecasts only roughly
   0.214–0.212.  Its own forecast has essentially no probability of 0.210,
   much less 0.170.  Do not spend the remaining submissions chasing it.
3. v21 rich-map variants were correctly gated out; their measured changes are
   at the thousandths scale and did not generalise across the held-out checks.
4. v26 L0 fails its dynamic FL gate (point change +0.122 mm; upper 90% bound
   +0.239 mm).  L1 has valid coverage but fails K0c fascicle recall
   (0.110, required >=0.50).  Neither is submission-eligible.
5. Public, licensed external data exists, but it is heterogeneous.  It must
   be used as a *subject-held-out validation and training* source, never as a
   source of test-image labels or leaderboard tuning.

## Operating rules

- Every candidate has an immutable input manifest, source URL/version/licence,
  subject split, seed, command line, metrics, and SHA-256 submission hash.
- Split by person/video before any crop/frame expansion.  Frames from a person
  can never be in both train and validation.
- No decision is made from the public leaderboard alone.  A submission is a
  confirmation of a candidate already promoted by validation.
- Do not request the Basel/Ritsche or Minghetti raw data: it is on request and
  may be the test cohort.  Do not use test-image manual labels.
- Data with unclear licence (notably FALLMUD) is validation-only until written
  reuse permission is obtained.  Record every external source in the final
  reproducibility package.
- Preserve at least two submission slots for final verified candidates and
  private-leaderboard hedging.  The competition limit is five/day, not a target.

## Eight-day execution plan

### Gate 0 — protect the current best (today, CPU; 1–2 hours)

1. **Completed initial integrity check:**
   `artifacts/candidates_v27/aplus/APLUS_v2_FinalA_LUMPA+0.5.csv` and
   `artifacts/ready/APLUS_v2_FinalA0.21637_LUMPA+0.5_LB0.21529.csv` are
   byte-identical SHA-256
   `8fd252872b4f03dfa8e5e8fccb76087668d33a33c20e29d6b3965ac0318a619d`.
   Both have the expected header, 309 unique IDs, and finite PA/FL/MT values.
   Rebuild the CSV from its pinned components before selecting it as a final.
2. Archive a manifest of every historical submission, score, exact components,
   and reproducibility status.  Mark only 0.21529 as the fallback.
3. Freeze the old constant-search route.  It may make one final *pre-registered*
   low-risk improvement only after a real model candidate exists; it must not
   consume exploratory slots now.

**Gate:** no modelling is allowed to replace the fallback unless it can be
re-created twice and differs from the fallback for a documented model reason.

### Gate 1 — make an honest external evaluation harness (today–Nov 7)

Acquire only the following public, licensed data, with original metadata kept:

| Priority | Dataset | Permitted role | Why it matters |
|---|---|---|---|
| P0 | TimTrack examples | PA/orientation external benchmark | Multi-rater labels; includes Telemed and Philips GM plus VL |
| P0 | GasFasTrack validation subset | held-out Telemed-family PA/FL benchmark | Expert PA/FL; protocol closest to needed geometry |
| P1 | UltraTimTrack | *unlabelled* domain pretraining/adaptation only | Telemed ArtUs scanner family |
| P1 | DL_Track_US | hash-dedupe and permitted pretraining only | Prevent accidental overlap with UMUD train |

Build one evaluator that reports the official normalized proxy and PA, FL and
MT separately, with bootstrap confidence intervals, per source, per subject,
and per scanner.  First run it unchanged on Final A.  Harmonise only explicit
measurement conventions (deep versus superficial aponeurosis, pixels versus
millimetres); if a source cannot be harmonised, it stays diagnostic-only.

**Gate:** the evaluator must reproduce published/hand-checked conventions on
at least ten examples and pass a leave-one-subject-out smoke test before any
GPU training begins.

### Lane A — model improvement (GPU; Nov 7–10)

The primary experiment is a **hybrid measurement model**, not another output
offset:

1. Start from the current apo/fascicle segmentation ensemble.
2. Train a compact residual measurement head from image features plus fixed
   geometry/map features to correct PA/FL/MT.  Train only on external labels
   whose convention is established; use augmentation and source/scanner
   indicators, but never a test-image target.
3. In parallel, train a segmentation/orientation specialist with external
   labelled masks only after the licence and hash-dedupe checks pass.  Its
   first purpose is to improve the maps in Telemed-like GM, not to replace all
   four scanner families.
4. Consider unlabelled UltraTimTrack only as source-side self-supervised
   adaptation (masked-image modelling / feature normalization), validated on
   *held-out people* from TimTrack and GasFasTrack.  Do not adapt on the
   competition test images unless an organiser rule explicitly permits it and
   we can document it reproducibly.

Run an ablation ladder on one A100 80 GB:

| Run | Question | Budget | Promotion condition |
|---|---|---:|---|
| A0 | Can Final A be measured fairly on external sources? | CPU | evaluator passes Gate 1 |
| A1 | Does a residual head improve unseen people? | 4–6 GPU-h | same direction on two sources |
| A2 | Does labelled source fine-tuning improve maps and measurements? | 8–12 GPU-h | better than A1 without a component regression |
| A3 | Does Telemed-only source adaptation transfer? | 6–10 GPU-h | improvement on held-out Telemed source, no loss elsewhere |
| A4 | Can a conservative specialist blend help the full ensemble? | 2 GPU-h | two-source lower confidence bound passes |

Use an A100 80 GB, 64 GB host RAM and 100 GB free disk.  Two 4090s are also
fine for A1–A4 if A100 is unavailable.  The first 24 hours should use only A0
and A1; there is no reason to reserve a large multi-GPU pod before an external
signal exists.

### Lane B — diagnosis and candidate selection (Nov 7–12)

For every A-run, produce a single blinded scorecard:

- leave-subject-out delta versus Final A in the normalized proxy;
- PA / FL / MT deltas and 90% bootstrap interval;
- source and scanner deltas; number of people, not just frames;
- correlation with Final A prediction changes and list of changed failure cases;
- exact data/weight/code provenance and inference time.

**Promotion to a Kaggle submission requires all of:**

1. Improvement in the same direction on at least two independent held-out
   sources, one being Telemed-family where applicable.
2. The pooled one-sided 90% lower-confidence improvement is at least 0.015
   normalized-score units versus Final A, with no individual component
   degradation over 0.005 on both sources.
3. The candidate changes a meaningful but controlled subset of predictions;
   it is not merely a hidden constant or duplicate submission.
4. A clean second inference exactly recreates the CSV and its hash.

The `0.015` gate is intentionally below the full 0.045–0.050 goal: external
benchmarks cannot certify the exact Kaggle gain.  A candidate without at least
this much robust external evidence is not worth a precious leaderboard slot.

### Submission policy and final days (Nov 10–14)

- At most one exploratory *model* candidate per day after promotion.
- Compare it against the exact fallback, not against a moving collection of
  unscored constants.
- If a candidate gains >=0.010 public and remains externally promoted, make
  one pre-registered conservative blend with Final A; do not search blend
  weights from the leaderboard.
- On Nov 13, freeze two fully reproducible finals: the best public candidate
  and the lowest-risk diverse fallback.  Use the official “select two finals”
  mechanism only after verifying the deadline/time zone.

## Kill criteria — how we avoid wasting days

- **By end of Nov 7:** no valid external harness => do not launch long runs;
  fix data/convention and use A+ as final fallback.
- **By end of Nov 9:** no A1/A2 signal meeting the two-source directional
  check => stop residual/head variations and investigate one documented
  geometry failure mode only.
- **By end of Nov 11:** no promoted model => no more ambitious training;
  conserve submissions, harden A+, and prepare reproducibility materials.
- A public gain smaller than 0.003 is noise-sized relative to the objective;
  log it, but do not call it a top-1 path.

## Honest outlook

Reaching 0.17 or overtaking 0.16485 is **not supported by the current stack**;
the current trajectory is roughly 0.21.  It is still worth making the attempt,
but only a genuinely better architecture/measurement model with transfer
evidence has enough headroom.  The plan maximizes that chance while preserving
the verified 0.21529 result and keeping the final entry eligible and
reproducible.
