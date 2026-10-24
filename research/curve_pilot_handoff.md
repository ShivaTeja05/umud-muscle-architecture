# Curved-track pilot handoff

Status: fold-0 paired pilot completed on the supplied A100. See
`research/curve_pilot_results_2026-10-22.md` for measured outcomes, preserved strict
numerical failures and the explicit numerical audit. No further folds or Kaggle
submissions were launched. The instructions below describe reproduction; use a
fresh output directory and regenerate from the current source. All neural model
execution belongs on the GPU host, not the laptop.

Generate the two experiment directories:

```bash
.venv/bin/python scripts/prepare_curve_experiment.py
```

Use a fresh `--out` directory for another fold, for example:

```bash
.venv/bin/python scripts/prepare_curve_experiment.py \
  --fold 1 --out artifacts/gpu_experiments/curve_v3_fold1
```

Each arm contains `train.py` and private Kaggle kernel metadata. For a CUDA pod,
transfer both arm directories and the required data/checkpoints. Do not run the
training command on the laptop.

Expected remote input layout:

```text
/workspace/inputs/
  competition/     # recursive discovery of fasc_images_new_model_v1,
                   # fasc_masks_new_model_v1 and test_set_v2
  teacher_v1/
    fasc_orient_v1_config.json
    metrics/folds.csv
    metrics/fold0.json
    checkpoints/fold0.pt
```

The v1 checkpoint weights are not currently stored in the local workspace; the
original private Kaggle kernel output is the source. Download the required
checkpoint onto the GPU host, using the user's authenticated Kaggle access.
Do not silently substitute v2 weights or ImageNet: the pilot's strict warm-start
guard stops if the required v1 folds/checkpoint are absent or invalid.

On the GPU host, in a dedicated environment with CUDA PyTorch, numpy, OpenCV,
Pillow, scipy, scikit-image and segmentation-models-pytorch 0.5.0:

```bash
# Control first. Substitute the actual uploaded code path.
UMUD_INPUT_ROOT=/workspace/inputs \
UMUD_OUTPUT_DIR=/workspace/results/curve_v3/control \
UMUD_TEMP_DIR=/workspace/tmp/curve_v3/control \
python /workspace/curve_v3_pilot/control/train.py

# Run only after checking the control's epoch-0 reproduction and reload logs.
UMUD_INPUT_ROOT=/workspace/inputs \
UMUD_OUTPUT_DIR=/workspace/results/curve_v3/curves \
UMUD_TEMP_DIR=/workspace/tmp/curve_v3/curves \
python /workspace/curve_v3_pilot/curves/train.py
```

Internet is needed for the existing public OSF image downloads and dependency
resolution. Freeze the actual environment and save GPU/runtime information.
Run one arm at a time, keep outputs outside the temporary directory, and use an
external lifetime limit on the machine. The in-script training guards do not
limit inference time, downloads, or cloud billing.

Retrieve `metrics/`, `logs/`, config JSON, `manifest.csv`, `maskbank/`, and
`checkpoints/`. Check the following before expanding:

1. Warm source is the v1 fold-0 checkpoint; epoch 0 reproduces its validation.
2. Both arms report the same held-out VL count; the curves split manifest has no
   train/validation subject intersection.
3. Checkpoint fp32 reload and TTA verification pass. Epoch 0 remaining best is a
   valid negative result.
4. Compare the same GT fold and held-out VL subject IDs. The control predicts all
   VL images while the supervised arm writes only the available held-out fold;
   align IDs explicitly. Do not compare one arm's full VL to the other's subset.
5. Inspect PA, track and deep-strip errors separately. Before adopting a final
   model, evaluate downstream FL and MT plus independent expert benchmarks.

Do not submit the pilot outputs directly. A one-fold model is for deciding
whether to invest in a second fold and then full cross-validation.
