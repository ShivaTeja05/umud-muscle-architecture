# S5 direct regressor v30: build, push, status, download, score

Pre-registration: `artifacts/candidates_v30/s5/PREREG_v30_s5.json`. Its sha256 and UTC time are in `PREREG_v30_s5.sha256` (`72d2d5a6...d4f5`, 2026-11-06T23:57:43Z). The PREREG pins the sha256 of `s5_train.py`, both `run.py` files, the payload and the scorer. If any of them changes, write a hashed addendum first. An addendum may only fix a bug or tighten a gate.

Addendum (skeptic review, before any push): `artifacts/candidates_v30/s5/PREREG_v30_s5_addendum.json` (sha256 in `PREREG_v30_s5_addendum.sha256`). The kernel files are unchanged. The scorer is now r1: `score_s5.py`, with the registered r0 kept byte-identical as `score_s5_r0.py`. r1 adds three things:

- A1: the seed mean is re-derived from `pred_by_seed.csv` with NaN propagating. A non-finite seed is a FAIL.
- A2: the run must report the registered payload sha256.
- A3: the new G1b gate, on static29_minus_source_shared.

**Status at build time (2026-11-06 ~23:58Z): nothing has been pushed.** No S5 output exists. A push needs the user's OK in chat: first the cpusmoke, then the GPU kernel.

In every command below, `ROOT` is the project root and `$LOCK` stands for:

```
lockf -k /private/tmp/claude-501/-Users-gaddamshivateja-Documents-ChatGPT-UMUD-Challenge--Muscle-Architecture-in-Ultrasound-Data/1dd23780-c685-429e-98de-1210fac2eb6b/scratchpad/laptop.lock env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python
```

`K=.venv/bin/kaggle`

## 0. Files

| file | what |
|---|---|
| `s5_train.py` | The kernel script (GPL-3.0). It contains the ConvNeXt-T global-pool PA / FL heads, the exact-affine label geometry, the guards, training and inference. Its top level imports only numpy and PIL, so laptop tests can import it. |
| `build_payload.py` | Laptop step. It writes `payload/` (VL lines, TimTrack observer angles + file sha256, dynamic FL, inference geometry, ids / expected counts / expected guard drops). |
| `build_driver.py` | Laptop step. It writes `gpu/` and `cpusmoke/` (`run.py` with everything embedded and sha256-checked, plus `kernel-metadata.json`) and `build_info.json`. It never pushes. |
| `gpu/` | `shivateja005/umud-s5-direct-v30`: T4 x2, internet on, competition data attached. 2 seeds, one per GPU, 1 fold. |
| `cpusmoke/` | `shivateja005/umud-s5-direct-v30-cpusmoke`: CPU, the same embedded code with `UMUD_S5_MODE=cpusmoke`. |

## 1. Rebuild (only if code or payload changed; laptop-light, each step under 10 s)

```
cd ROOT
$LOCK artifacts/candidates_v30/s5/code/i05_guard_local.py      # expected guard drops -> out/i05_guard_local.json
$LOCK kaggle/s5_direct_v30/build_payload.py                    # payload/ (reads i05 output)
$LOCK kaggle/s5_direct_v30/build_driver.py                     # gpu/ cpusmoke/ build_info.json
$LOCK artifacts/candidates_v30/s5/code/t01_local_checks.py     # label geometry checks (must print "pass": true)
$LOCK artifacts/candidates_v30/s5/code/t05_runner_unpack.py    # run.py blobs == files, both arms identical code
$LOCK artifacts/candidates_v30/s5/code/score_s5.py --selftest  # scorer r1 formulas / plumbing -> out/score_selftest_r1.json (all_ok true)
$LOCK artifacts/candidates_v30/s5/code/t06_scorer_r1_regression.py  # r1 == r0 formulas, == v01_frontier on the v18 dense heads (pass true)
$LOCK artifacts/candidates_v30/s5/code/k01_skeptic_checks.py   # runner blobs / payload / ids cross-checks (all_ok true)
```

`t04_scorer_vs_v01.py` is the r0 check, and its output is pinned by the PREREG. Do not rerun it: it calls `run()` on a folder without `pred_by_seed.csv`, which r1 rejects as INVALID_OUTPUT. `t06` replaces it.

After any rebuild, compare the new `build_info.json` sha256 values with the PREREG. A mismatch needs a hashed addendum before you push.

## 2. Push the cpusmoke first (needs the user's OK; CPU, no GPU quota; about 15-25 min [speculative])

```
$K kernels push -p kaggle/s5_direct_v30/cpusmoke
$K kernels status shivateja005/umud-s5-direct-v30-cpusmoke
```

Download only the small outputs. There are no images, maps or checkpoints in `/kaggle/working`:

```
$K kernels output shivateja005/umud-s5-direct-v30-cpusmoke -p artifacts/candidates_v30/s5/kernel_out/cpusmoke_v1 \
    --file-pattern '(static_pred|test_pred|vl_val_pred|pred_by_seed)\.csv|metrics\.json|CREDITS\.txt|logs/.*\.log'
```

The cpusmoke passes only if `metrics.json` shows all of the following:

- `all_checks_pass: true`, `status: complete`, `n_seeds_used: 1`.
- `checks.smoke_3_steps: true`.
- `counts`: static 35, test 309, vl_payload 245, vl_matched 245, vl_used_ntracks_ge2 244, tt_images 130, tt_rows 290, dyn_frames_zip 168, dyn_fl_labels 167.
- `guard.dropped_ids` includes `tt__GM_P_6`, `tt__GM_T_1`, `tt__GM_T_4`, `tt__GM_T_8`. The dynamic window should be about frames 39-76; it is recorded in `guard.dropped_vs_local_symdiff`. A small boundary difference is information only.
- `guard.max_ncc_test_kept <= 0.90`, and the static checks hold by id, by flip-aware sha256 and by NCC.
- `label_aug.max_err < 1e-6`.
- `static_pred.csv` has 35 rows and `test_pred.csv` has 309 rows, with columns exactly `image_id,pa,fl` and all values finite.
- `logs/main.log` shows `timtrack: 130 images, all sha256 OK`.
- Addendum A4: no entry of `guard.dropped` has a `test_ncc` reason. If one does, stop and report it to the user before any GPU push: it would mean the external labelled data contains a near-copy of a test image.

Also run the scorer on the smoke output. It must say `NOT_A_GATE_RESULT`, because the mode is `cpusmoke`; this checks the plumbing only. Under r1 it must not say INVALID_OUTPUT. `output_check.invalid_reasons` must be absent, `payload_sha_registered` must be true and `seed_mean_consistent` must be true:

```
$LOCK artifacts/candidates_v30/s5/code/score_s5.py --kernel-dir artifacts/candidates_v30/s5/kernel_out/cpusmoke_v1
```

## 3. Push the GPU kernel (needs the user's OK, and only after the cpusmoke passes)

Before pushing, check the weekly GPU quota on the Kaggle web page; the CLI does not show it. Expect about 0.6-1.1 h of wall time on 2xT4 [speculative]. The hard caps are: 0.85 h of training per seed, the epoch loop stops at 1.25 h, and the parent kills workers at 2.0 h.

```
$K kernels push -p kaggle/s5_direct_v30/gpu --accelerator NvidiaTeslaT4
$K kernels status shivateja005/umud-s5-direct-v30
$K kernels output shivateja005/umud-s5-direct-v30 -p artifacts/candidates_v30/s5/kernel_out/gpu_v1 \
    --file-pattern '(static_pred|test_pred|vl_val_pred|pred_by_seed)\.csv|metrics\.json|CREDITS\.txt|logs/.*\.log'
```

The pattern is matched against the full output path. If a file arrives truncated (0 B), delete it and fetch it again. Never download anything else from the kernel.

## 4. Score (laptop, under 5 s)

```
$LOCK artifacts/candidates_v30/s5/code/score_s5.py --kernel-dir artifacts/candidates_v30/s5/kernel_out/gpu_v1
#   -> artifacts/candidates_v30/s5/out/score_gpu_v1.json  (verdict PASS / FAIL / INVALID_OUTPUT)
```

The gates come from the PREREG, are all required, and a NaN counts as a failure. The reference is `est_ref`, i.e. `bench_frame.csv`, with SD-matched deviations:

| gate | condition |
|---|---|
| G1 static35 | corr ≥ 0.86 AND k ≤ 1.5 AND r ≤ 0.6, for both PA and FL |
| G1b static29_minus_source_shared (addendum A3) | the same as G1 on the 29 images without im_01, im_04, im_10, im_15, im_18, im_35 |
| G2 clean10 | the same with k ≤ 1.8 |
| G3 test | PA SD ratio vs A-plus v2 ≥ 0.6 in at least 5 of the 6 groups |

Decision rules:

- **FAIL**: the direction is killed, with no rerun.
- **PASS**: ask the user about the remaining LOSO folds (+~4 GPU-h) and the blend gates. No candidate and no submission come from this run.
- **INVALID_OUTPUT**: a plumbing failure. The scorer lists the cause in `invalid_reasons`. If a file is only missing locally, download it again; that is not a rerun. A kernel-side failure allows at most one bug-fix rerun, and only after a hashed addendum.
- **A non-finite prediction from any seed** is a FAIL under the NaN rule (addendum A1), even when `static_pred.csv` looks finite.

Always report static35, clean10 and copies25 separately. `pkg_ref`, `static29_minus_source_shared`, the VL fold 0 results and the blend deltas are information only.

## Data hygiene note (found while building; see the PREREG `hold_out_and_exclusions`)

- 4 images in TimTrack dataset 4 are mirrored copies of static35 images: GM_T_1 = im_04, GM_T_4 = im_18, GM_T_8 = im_15, GM_P_6 = im_10.
- Dynamic frames 39-76 are mirrored near-copies of im_35.
- The guard drops all of these.
- clean10 stays far from every kept training image (max NCC 0.49).
- Sources: `artifacts/candidates_v30/s5/out/i03_overlap_flip.json`, `i05_guard_local.json`, `i06_guard_per_static.json`.
