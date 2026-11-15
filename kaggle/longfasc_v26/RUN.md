# LF-26 Kaggle kernels: build, push, download

Pre-registration: `artifacts/candidates_v26/PREREG_v26_longfasc.json` and `PREREG_v26_addendum_K.json` (training configuration). Each has a `.sha256` file with the UTC time. Nothing here has been pushed yet.

## Revision r1, 2026-11-05 08:04Z

- `PREREG_v26_r1.json` (b025d7a9...) and `PREREG_v26_addendum_K_r1.json` (2a7c3add..., template f0a83bca...) are hashed. `build_driver.py` refuses to build unless all four prereg files match their `.sha256` records and the template matches K_r1. It embeds the four sha256 values in `kernel.py`.
- The kernels now fail fast before training if a payload image is missing or the attached v1 maskbank differs from the laptop copy. They use per-draw augmentation seeds and assert Cunningham-only FALLMUD.
- They run `L0port` + `L0portexp` (VL expert geometry) on every payload image in every mode, and check them against the embedded laptop values at the r1 float-noise tolerance (`metrics/port_check.json`).
- `l0_ref` now needs the full laptop L0 runs: `runs/L0_prod_{test,static,dynamic,vl}.csv` + `runs/L0_expert_vl.csv`.
- The cpusmoke pass list and the GPU push conditions are in `PREREG_v26_addendum_K_r1.json` (`cpusmoke_pass_r1`, `gpu_push_r1`). The weekly GPU quota must be checked on the Kaggle web page, because the CLI does not show it.
- Local port check on the expert path: `local_port_check.py --set vl --geom vl_expert --laptop artifacts/candidates_v26/runs/L0_expert_vl.csv ...`

## Status, 2026-11-05 08:12Z

- The full laptop L0 runs exist. The kernels were rebuilt at 08:10Z: `kernel.py` is 915 KiB (slim fragments), with `l0_ref` for 760 production images and 248 VL expert images.
- The local port simulation of `gpu/kernel.py` matches the laptop: max |d| <= 1.4e-14 mm on test 40, VL expert 30, dynamic 20 and static 35 images.
- **cpusmoke v1 was pushed at 08:11Z** (`shivateja005/umud-longfasc-v26-cpusmoke`; kernel.py sha256 0e7abfea...).
- **GPU kernel NOT pushed.** It waits for every item in `cpusmoke_pass_r1` to hold and for a quota check on the Kaggle web page (`gpu_push_r1`).
- Check the cpusmoke outputs with:

  ```
  kaggle kernels output shivateja005/umud-longfasc-v26-cpusmoke -p artifacts/candidates_v26/kernel_out/cpusmoke_v1 \
      --file-pattern '(readout|metrics|logs)/.*|viz/.*\.jpg|.*config\.json|MANIFEST\.json|CREDITS\.txt|fallmud/(groups|overlap|stats).*'
  ```

## Status, 2026-11-05 07:08Z (before r1)

- The payload is exported and complete (07:05Z, 788 KB gz).
- `gpu/` and `cpusmoke/` are built (07:06Z; `kernel.py` 1.14 MB each, no `l0_ref`).
- Addendum K is hashed (07:08Z, template sha256 4781d61d...).
- If `lf_readout.py`, `geometry_v2.py` or `train_template.py` changes, redo steps 1-3. A changed template also needs a hashed addendum K revision.

In every command below, ROOT is the project root, and `$LOCK` stands for:

```
lockf -k /private/tmp/claude-501/-Users-gaddamshivateja-Documents-ChatGPT-UMUD-Challenge--Muscle-Architecture-in-Ultrasound-Data/1dd23780-c685-429e-98de-1210fac2eb6b/scratchpad/laptop.lock env PYTHONPATH=src .venv/bin/python
```

## 1. Payload: production geometry, laptop only, 4 chunks of about 1-2 minutes each

```
cd ROOT
$LOCK kaggle/longfasc_v26/export_payload.py --set test
$LOCK kaggle/longfasc_v26/export_payload.py --set static
$LOCK kaggle/longfasc_v26/export_payload.py --set dynamic
$LOCK kaggle/longfasc_v26/export_payload.py --set vl --summary   # VL also writes the expert geometry and K0d points
```

- The geometry goes to `payload/geom_{test,static,dynamic,vl,vl_expert}.json.gz`, the K0d points to `payload/k0d_points_vl.json.gz`, and the hashes and counts to `payload/payload_summary.json`. `complete` must be true.
- The K0d track angles stay on the laptop, in `artifacts/candidates_v26/runs/k0d_points_vl.csv`.
- No FL, PA or MT value is exported.

## 2. Optional: laptop L0 values for the in-kernel port check

`build_driver.py` embeds `payload/l0_ref.json` from `artifacts/candidates_v26/runs/L0_prod_<set>.csv` when that file exists. For the cpusmoke, the first 20 test images are enough:

```
$LOCK artifacts/candidates_v26/code/run_lf.py --set test --limit 20
```

Without these values the kernel still writes `readout/images_L0port.csv`, and `gates.py` runs the port check on the laptop after download.

## 3. Build the kernel folders

```
$LOCK kaggle/longfasc_v26/build_driver.py
```

- This writes `gpu/kernel.py` + `kernel-metadata.json` (shivateja005/umud-longfasc-v26: T4 x2, internet on, kernel source shivateja005/umud-fasc-orient-v1) and `cpusmoke/` (shivateja005/umud-longfasc-v26-cpusmoke: CPU only).
- It also writes `build_info.json` with every sha256 and size.
- It refuses to run on an incomplete payload.
- If `kernel.py` comes out larger than 1 MB, a warning is printed. See the open issue in the README.

Before pushing, run the local port simulation. It unpacks `kernel.py` exactly as the runner does and uses the embedded code, with no torch:

```
$LOCK kaggle/longfasc_v26/local_port_check.py --kernel kaggle/longfasc_v26/gpu/kernel.py \
     --laptop artifacts/candidates_v26/runs/L0_prod_test.csv --unpack <SP>/longfasc_v26/unpack --limit 20
```

## 4. Push (Push phase only; private; never from this build step)

1. Push the cpusmoke first:

   ```
   kaggle kernels push -p kaggle/longfasc_v26/cpusmoke
   ```

   About 25-40 minutes on CPU. Check that `logs/main.log` shows all of the following:
   - FALLMUD sha256 OK (3c1c0d4a...0cf9; the kernel raises on a mismatch);
   - the overlap guard keeps Cunningham only, and `fallmud/groups.csv` is present;
   - path-target viz in `viz/train_sample_fallmud_*.jpg`;
   - `metrics/cpusmoke_bn_repair.json` with `nonfinite_after_repair` empty (fold 2 or 4);
   - 3 training steps;
   - `readout/images_{L1prod,L1expvl,L0port}.csv` for 20 images;
   - `metrics/port_check.json` passed (when l0_ref is embedded).
2. Push the GPU kernel only if all of these hold:
   - the cpusmoke passed;
   - at most 1 other GPU session is running (AdaBN v25);
   - at least 6 h of weekly GPU quota remain;
   - `PREREG_v26_addendum_K.json` is hashed and its `template_sha256` equals the sha256 of `train_template.py`.

   ```
   kaggle kernels push -p kaggle/longfasc_v26/gpu
   ```

3. Fallback if K0b fails or quota is short: rebuild with `"UMUD_MAX_EPOCHS": "8"` in the gpu env (edit ARMS in `build_driver.py`) and record it in a dated addendum. Never reduce the number of folds.

## 5. Download (under 50 MB; maps and checkpoints stay on Kaggle)

```
kaggle kernels output shivateja005/umud-longfasc-v26 -p artifacts/candidates_v26/kernel_out/v1 \
    --file-pattern '(readout|metrics|logs)/.*|viz/.*\.jpg|.*config\.json|MANIFEST\.json|CREDITS\.txt'
```

The pattern is matched against the full output path. If a file arrives truncated (0 B), delete it and fetch it again.

## 6. Gates on the laptop

```
$LOCK artifacts/candidates_v26/code/gates.py --kernel-dir artifacts/candidates_v26/kernel_out/v1
```

## Runtime estimate (GPU kernel, T4 x2)

| Stage | Time |
|---|---|
| Data prep (pairs, OSF downloads, FALLMUD fetch + parse + overlap guard + groups) | about 5 min |
| Per fold (v1 ref eval, v1 init + BN repair, budget-fitted 6-12 epochs at about 2.2-2.6 min, reload + TTA eval) | about 30-35 min |
| Folds on 2 GPUs in parallel (3 + 2 folds) | about 1.6-1.8 h |
| Inference: 5 folds x hflip, 761 images | about 6 min |
| K0c on the FALLMUD hold-out | about 1 min |
| Readout: 3 runs, about 1770 readouts at about 0.5 s on 4 spawned CPU workers | about 5 min |
| VL field samples | under 1 min |
| **Total session** | **about 2.0-2.4 h** |

- This is about 2-2.5 h of the weekly GPU quota, or about 4-5 GPU-hours if the quota counts each T4 separately. Both are within the 8 GPU-h budget.
- Training proper is capped at 3.5 h (`TRAIN_CAP_S`). The hard stop for the epoch loop is 4.2 h after start.
