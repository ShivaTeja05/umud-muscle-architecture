# v16 insfield: run sheet (untracked; nothing here is pushed automatically)

Build (laptop, light):

    .venv/bin/python kaggle/v16_insfield/build_payload.py   # VL expert lines -> payload/ (18 s, 0.2 GB)
    .venv/bin/python kaggle/v16_insfield/build_driver.py    # train.py + cpusmoke/ A/ A0/ C/

Arms (same train.py, env switches in run.py):

| folder | kernel id | hardware | what |
|---|---|---|---|
| cpusmoke | shivateja005/umud-v16-insfield-cpusmoke | CPU | plumbing only: 160 comp pairs, 1 fold, 3 steps |
| A | shivateja005/umud-v16-insfield-a | T4 x2, ~4.0-4.5 h | ConvNeXt-S 512, VL expert lines ON (design A) |
| A0 | shivateja005/umud-v16-insfield-a0 | T4 x2, ~3.5 h | same code, VL OFF (control; optional: fasc_orient_v1 maps already exist) |
| C | shivateja005/umud-v16-insfield-c | T4 x2, ~7 h | ConvNeXt-B 640 + VL (only if A passes gate G1) |

CPU smoke (no GPU quota):

    K=.venv/bin/kaggle
    $K kernels push -p kaggle/v16_insfield/cpusmoke
    $K kernels status shivateja005/umud-v16-insfield-cpusmoke
    $K kernels output shivateja005/umud-v16-insfield-cpusmoke -p /private/tmp/claude-501/umud_v16/nextgen-models/cpusmoke_out \
        --file-pattern '(logs|metrics)/.*|.*config\.json|.*\.log'

GPU arm (judge decision only):

    $K kernels push -p kaggle/v16_insfield/A --accelerator NvidiaTeslaT4
    $K kernels output shivateja005/umud-v16-insfield-a -p artifacts/kaggle/v16_insfield_A \
        --file-pattern 'maskbank/.*|metrics/.*|logs/.*|.*config\.json|viz/.*'

After the GPU run (laptop, light, sequential): point the package's field bank at the new maps and re-measure
(see eval/README in artifacts/candidates_v16/nextgen-models/eval_plan.json):

    umud-run --config <overlay with maps.sources.fasc_field = {bank: artifacts/kaggle/v16_insfield_A/maskbank,
             model: fasc_ins_v16}> --stage measure --full --set paths.v3_cache=<empty> --set paths.v4_cache=<empty>
