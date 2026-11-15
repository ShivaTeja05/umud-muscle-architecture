# Handoff: where we are and what we plan to do (2026-11-10, 18:20 UTC)

Competition: Kaggle "UMUD Challenge: Muscle Architecture in Ultrasound Data".
Goal: top 3 on the private leaderboard with a genuine, reproducible model (GPL-3.0 code).
Deadline: 2026-11-14 23:00 UTC (treat as firm). Two final files are chosen by hand on 11-13.

## 1. Current standing

| File | Public LB | Notes |
|---|---|---|
| A-plus v2 | 0.21529 | Best. sha 8fd25287. Final A + LUM.PA +0.5 |
| Final A | 0.21637 | sha 1bf9fe4a. Pure model (KG0) + 4 group constants found on the LB |
| KG0 | 0.23041 | Pure model, no LB constants |
| Final B′ | 0.23322 | Do not touch |

Rank 29 of 637. Top 3 on the public LB: 0.161 / 0.174 / 0.177.

Metric: mean of MAE/τ over pennation angle (PA, τ 6°), fascicle length (FL, τ 12 mm), muscle thickness (MT, τ 3 mm).

Error budget at 0.215: FL 0.120 (Telemed FL alone 0.060), PA 0.066, MT 0.031.

Code map and known problems: `docs/SOURCE_GUIDE.md`. Running log of every result: `artifacts/notes/FACTS.md`. List of everything already tried: `artifacts/candidates_v29/map/LEDGER_tried.md`.

## 2. What is running right now

**v33 PA network, fold 0 only** — Kaggle GPU kernel `shivateja005/umud-pa-lines-v33-fold0`, pushed 13:37 UTC, still RUNNING at 18:16 UTC (4 h 39 min). Kaggle shows no log until it ends.

- Idea: one ConvNeXt-small network outputs fascicle direction, fragments, deep aponeurosis edge and a "where to read" weight map. A fixed differentiable readout turns these into PA, and the loss is on PA itself. Design: `research/pa_network_design_2026-11-10.md`.
- Trainer: `kaggle/pa_lines_v33/` (`build.sh` builds `gpu/train.py` and `cpusmoke/train.py` from `kaggle/fasc_orient_v1/train.py` + `src/part_*.py`).
- The version pushed has sha256 `6a378dc5…` and is the 13:24 file `kaggle/pa_lines_v33/gpu/train.py`.
- Decision rule, fixed before the run: `artifacts/candidates_v33/PREREG_fold0.txt` (sha `f7e116b9…`).
  - S1: on fold 0's 33 both-mask images, network PA MAE must be below production's (production = mean of `pa_fusion` and `pa_fd_strip_g` in `artifacts/candidates_v3/cache/train142__oof.csv`).
  - S2: on the VL expert set, the single fold-0 model must be no worse than production + 0.05° (`artifacts/candidates_v3/cache/vl__v2.csv`).
  - Both pass → ask the user before running the other 4 folds. Otherwise the lane closes (one fixed 50/50 blend test is allowed if error correlation with production is below 0.9).
- Outputs to read when it ends: `oof_readout.csv`, `readout.csv`, `metrics/early_fold0_ep{4,8}.csv`, `metrics/history_fold0.csv`. Download only small files:
  `kaggle kernels output shivateja005/umud-pa-lines-v33-fold0 -p <dir> --file-pattern '.*\.(json|csv|log|txt)$'`
- If it is still running at about 19:00 UTC, cancel it to protect the GPU quota. Kaggle keeps the files written so far.
- Expected value if every gate passes: about 0.002–0.006 (1–4 rank places). Chance of passing all gates: about 10–15% (my estimate). It does not reach top 3 alone.

**Note for Codex:** at 18:09–18:11 UTC something other than this session changed `kaggle/pa_lines_v33/build.sh` and `src/`, and created `artifacts/candidates_v33/kaggle_v1_source/`, `artifacts/candidates_v33/restart_build/` and `scripts/fit_pa_readout_gate.py`. I did not make or review those. The running kernel is unaffected.

## 3. What was tested today and closed

| Idea | Result | Where |
|---|---|---|
| Use VL images that match the Telemed head split as a yardstick for Telemed FL | Failed its pre-registered check: reproduced 1 of 3 known LB signs. VL says "shorten", the LB says the opposite | `artifacts/candidates_v34/vl_telsim/` |
| Fix the PA reference line (deep aponeurosis) | Killed at stage 0: swapping in the expert's line does not lower PA error (1.76° → 1.88°) | `artifacts/candidates_v35/refline/` |
| Literature search for an adoptable method | Nothing new; the published tools are already benchmarked here | `research/staged_experiments_2026-11-10.md` |

Consequence: Telemed FL, the largest error pool, has no trustworthy offline yardstick. Offline VL evidence must not be used to choose Telemed FL settings.

## 4. What we plan to do next

In order. Each step has a cheap first readout and a stop rule.

1. **Read the fold-0 result** under the fixed rule above. Report before any further GPU use.
2. **Daily LB campaign** (automated planner, group-level constants only, at most 2 per day). It runs 11-11, 11-12, 11-13. It submits only if the planner's value is ≤ −2e-4; on 11-10 nothing qualified (best −0.000187).
3. **Two decisions waiting on the user:**
   - Label 12 of the 96 non-test Telemed frames by hand, as a first real yardstick for Telemed FL. Stop if the labels cannot separate the FL heads.
   - Submit one group-level T12 FL probe (×1.005, planner value −0.000157, below the campaign's bar). Not approved.
4. **11-13: pick the two finals.** Default: A-plus v2 + Final A.
5. **Packaging for reproducibility** (needed for a top-3 claim):
   - Install the `cell_constants` fix so the two finals rebuild from tracked code (`artifacts/candidates_v31/package/draft/`).
   - Download and pin the 10 fold checkpoints that exist only in private Kaggle kernel outputs (`umud-apo-v2`, `umud-fasc-orient-v1`). Largest schedule risk.
   - Decide whether to keep or rewrite `src/umud/geometry.py` l.280–286 (adapted from a public notebook).

## 5. Rules that must be kept

Competition rules:
- No per-row, per-image or per-clip leaderboard probing. Group-level constants only, and only in Final A.
- No hand labels, pseudo-labels or self-training on test images. Test BatchNorm statistics are allowed.
- No other participants' code. External data must be public, licensed and declared.
- Never redistribute competition data.

User's rules:
- No heavy jobs on the laptop (it crashed before). No torch locally, under 1.5 GB RAM, one Python process at a time.
- Kaggle GPU: leave at least 3 hours of weekly quota unused. Ask before any GPU push. Used this week so far: about 5 hours by our log.
- RunPod: always ask first; terminate the pod straight after use.
- Submissions: at most 3 per day by the assistant; a non-probe submission needs clear offline evidence and the user's OK. Do not spend slots unless the new file is likely to beat the old one.
- Design every experiment in stages with an early readout and a kill rule.
- Never open or list `data/manual_labelling/` or `work_selection.json` (private).
- Never write into `artifacts/final/` or `artifacts/ready/`. Never change Final B′.
- Git: commit only when asked, on branch `final-package`; never push; leave `main` alone.

Traps in the code:
- `Assembler.offsets()` writes into `artifacts/final/final_A/rater_style/` by default. Redirect `paths.out_dir` first, or use the light loader `artifacts/candidates_v32/flpa/common.py`.
- Do not run `tests/test_final_package.py` as written (it rewrites `artifacts/final/test_layout.csv`).
- Do not run the `maps` or `all` stages of `scripts/umud_run.py` locally (they need a GPU).
- CPU smoke kernels: the data loader must use 0 workers on CPU, or it hangs.

## 6. Honest outlook

- No tested idea has produced a model gain since the LB constants.
- Gap to top 3 is about 0.038. The remaining levers in hand are each worth 0.0002–0.006.
- A real jump needs a yardstick for Telemed fascicle length. The only rule-compliant route we have found is hand-labelling non-test Telemed frames.
- A faster GPU would shorten waits but would not improve results; compute is not the limit.
