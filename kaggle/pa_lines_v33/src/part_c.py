

# =============================================================================================
# v33 overlay, part C: self-checks, inference, main
# =============================================================================================
def self_checks(torch) -> dict:
    """Readout and angle-transport checks on synthetic maps (no data, no model)."""
    res = {}
    hh = ww = 256
    ys, xs = torch.meshgrid(torch.arange(hh, dtype=torch.float32), torch.arange(ww, dtype=torch.float32), indexing="ij")
    worst_f = worst_l = 0.0
    for th_deg, ln_deg in ((-20.0, 3.0), (-8.0, -6.0), (-35.0, 10.0), (15.0, 0.0)):
        th, ph = math.radians(th_deg), math.radians(ln_deg)
        cp = torch.full((1, hh, ww), math.cos(2 * th))
        sp = torch.full((1, hh, ww), math.sin(2 * th))
        dist = ((ys - 180.0) * math.cos(ph) - (xs - 128.0) * math.sin(ph)).abs()
        ed = (torch.exp(-0.5 * (dist / 1.5) ** 2) * 0.95 + 0.01).unsqueeze(0)
        f, l, _, _ = readout(cp, sp, torch.rand(1, hh, ww), ed, torch.ones(1, hh, ww), torch)
        worst_f = max(worst_f, _deg(float(f) - th))
        worst_l = max(worst_l, _deg(float(l) - ph))
    res["readout_fasc_err_deg"], res["readout_line_err_deg"] = worst_f, worst_l
    rng = np.random.default_rng(0)
    worst_t = 0.0
    for _ in range(50):
        la, lb = aug_matrix(rng)[:2, :2], aug_matrix(rng)[:2, :2]
        th = rng.uniform(-1.2, 1.2)
        d0 = np.linalg.inv(la) @ np.array([math.cos(th), math.sin(th)])
        d1 = lb @ d0
        got = float(map_angle(torch.tensor([th], dtype=torch.float32), torch.tensor(la[None], dtype=torch.float32),
                              torch.tensor(lb[None], dtype=torch.float32), torch))
        worst_t = max(worst_t, _deg(got - math.atan2(d1[1], d1[0])))
    res["transport_err_deg"] = worst_t
    # sample-level check: the label of view B equals the transported label of view A for one straight fragment
    it = {"kind": "fasc", "crop": np.full((300, 400), 90, np.uint8), "segs": np.array([[60, 220, 330, 120]], np.float32),
          "dpts": np.stack([np.arange(20, 380, 4.0), 250 + 0.05 * np.arange(20, 380, 4.0)], 1).astype(np.float32)}
    worst_s = worst_sl = 0.0
    for seed in range(30):
        r = np.random.default_rng(seed)
        a, b = make_sample(it, r), make_sample(it, r)
        la, lb = torch.tensor(a["lin"][None]), torch.tensor(b["lin"][None])
        if a["hf"] and b["hf"]:
            worst_s = max(worst_s, _deg(float(map_angle(torch.tensor([float(a["tf"])]), la, lb, torch)) - float(b["tf"])))
        if a["ha"] and b["ha"]:
            worst_sl = max(worst_sl, _deg(float(map_angle(torch.tensor([float(a["tl"])]), la, lb, torch)) - float(b["tl"])))
    res["sample_fasc_transport_err_deg"], res["sample_line_transport_err_deg"] = worst_s, worst_sl
    res["pass"] = bool(worst_f < 0.2 and worst_l < 0.3 and worst_t < 0.01 and worst_s < 0.05 and worst_sl < 0.3)
    return res


def predict_native(models, gray: np.ndarray, box, torch):
    """Fold-mean + hflip TTA maps at native resolution (uint8) and the readout of the mean maps."""
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    inp, m = make_input(crop)
    x = to_input(torch.from_numpy(inp[None]).to(DEV), torch)
    acc = None
    with torch.no_grad():
        for model in models:
            cur = torch.stack([t[0] for t in forward_maps(model, x, torch, True)], -1)
            acc = cur if acc is None else acc + cur
        acc = acc / len(models)
        valid = torch.from_numpy(_poly_mask(m, x2 - x1, y2 - y1)[None]).to(DEV).float()
        f, l, ws, qs = readout(acc[None, ..., 1], acc[None, ..., 2], acc[None, ..., 4], acc[None, ..., 3], valid, torch)
    maps = acc.cpu().numpy().astype(np.float32)
    back = unletterbox(maps, m, x2 - x1, y2 - y1)
    full = np.zeros((gray.shape[0], gray.shape[1], N_OUT), np.float32)
    full[y1:y2, x1:x2] = back
    enc = encode_channels(full[..., 0], full[..., 1], full[..., 2])
    enc["deep"] = np.clip(np.round(full[..., 3] * 255.0), 0, 255).astype(np.uint8)
    enc["wread"] = np.clip(np.round(full[..., 4] * 255.0), 0, 255).astype(np.uint8)
    fd, ld = math.degrees(float(f)), math.degrees(float(l))
    pa = math.degrees(float(wrap_half(l - f, torch)))
    return enc, [round(fd, 4), round(ld, 4), round(pa, 4), round(float(ws), 2), round(float(qs), 2)]


def main() -> None:
    global ITEMS, _LOG_FILE, DEV
    for d in (MASKBANK, CKPT_DIR, LOG_DIR, METRIC_DIR, VIZ_DIR, TMP / "ext"):
        d.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = LOG_DIR / "main.log"
    log(f"MODE={MODE} python={sys.version.split()[0]} cpu={os.cpu_count()}")
    log("nvidia-smi: " + sh("nvidia-smi -L").strip().replace("\n", " | "))
    import torch

    torch.set_num_threads(1)
    smp = ensure_smp()
    import timm

    log(f"torch={torch.__version__} smp={smp.__version__} timm={timm.__version__} cv2={cv2.__version__}")
    checks = self_checks(torch)
    (METRIC_DIR / "selfcheck.json").write_text(json.dumps(checks, indent=2))
    log("self checks: " + json.dumps(checks))
    if not checks["pass"]:
        raise RuntimeError("readout self checks failed")

    test_dir = find_dir("test_set_v2")
    pairs = find_pairs(find_dir("fasc_images_new_model_v1"), find_dir("fasc_masks_new_model_v1"))
    fasc_items, alias_rows, stats = prepare_items(pairs)  # folds identical to fasc_orient_v1
    for it in fasc_items:
        it["kind"] = "fasc"
    apo_pairs = find_apo_pairs()
    t = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(min(4, os.cpu_count() or 2)) as pool:
        apo_all = list(pool.imap(prep_apo, apo_pairs, chunksize=8))
    seen, apo_items = set(), []
    for r in sorted(apo_all, key=lambda r: r["stem"]):
        if r["empty"] or r["ih"] in seen:
            continue
        seen.add(r["ih"])
        apo_items.append(r)
    stats["apo_pairs"], stats["apo_items"] = len(apo_pairs), len(apo_items)
    log(f"aponeurosis: {len(apo_pairs)} pairs -> {len(apo_items)} items with a deep edge in {time.time() - t:.0f}s")
    stats["both_mask"] = attach_pairs(fasc_items, apo_items, {Path(p).stem: p for p, _ in pairs})
    stats["apo_folds"] = assign_apo_folds(fasc_items, apo_items)
    for it in apo_items:
        del it["dpts_full"]
    ITEMS = fasc_items + apo_items
    stats["fold_table"] = {f"{k}{'+deep' if d else ''}": [sum(1 for it in ITEMS if it["fold"] == f and it["kind"] == k
                                                              and ("dpts" in it) == d) for f in range(N_FOLDS)]
                           for k, d in (("fasc", True), ("fasc", False), ("apo", True))}
    log("data stats: " + json.dumps(stats))
    with open(METRIC_DIR / "folds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "representative", "group", "fold"])
        w.writerows(alias_rows)
        w.writerows([[it["stem"], it["stem"], -1, it["fold"]] for it in apo_items])
    (METRIC_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    _m = build_model(pretrained=True)  # warm the weight cache before any fork
    n_params = sum(p.numel() for p in _m.parameters())
    del _m
    log(f"pretrained {ENCODER} cached ({n_params / 1e6:.1f}M params)")

    n_gpu = sh("nvidia-smi -L").count("GPU ")
    datasets: dict[str, list[tuple[str, Path, str]]] = {}
    test_files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES], key=lambda p: p.name)
    datasets["test"] = [(p.stem, p, p.name) for p in test_files][:TEST_LIMIT]
    extra_names = ("static",) if CPU_SMOKE else ("static", "dynamic", "vl")

    def fetch_extras():
        for name in extra_names:
            try:
                datasets[name] = gather_extra(name)
            except Exception:  # noqa: BLE001
                log(f"extra {name} failed:\n{traceback.format_exc()}")

    if n_gpu == 0:
        if not SMOKE:
            raise RuntimeError("full mode needs a GPU")
        DEV = "cpu"
        torch.set_num_threads(os.cpu_count() or 2)
        state = {"plan_folds": len(TRAIN_FOLDS), "n_parallel": 1, "done": 0}
        for k in TRAIN_FOLDS:
            train_fold(k, state, torch)
            state["done"] += 1
        fetch_extras()
    elif len(TRAIN_FOLDS) == 1:
        # A single fold does not need a forked GPU worker. The first Kaggle
        # version stalled between process creation and the worker's first log.
        DEV = "cuda"
        torch.backends.cudnn.benchmark = True
        cv2.setNumThreads(1)
        log(f"training fold {TRAIN_FOLDS[0]} directly on {torch.cuda.get_device_name(0)}")
        fetch_extras()
        state = {"plan_folds": 1, "n_parallel": 1, "done": 0, "in_process": True}
        train_fold(TRAIN_FOLDS[0], state, torch)
    else:
        assign = [TRAIN_FOLDS[0::2], TRAIN_FOLDS[1::2]] if n_gpu >= 2 and len(TRAIN_FOLDS) > 1 else [TRAIN_FOLDS]
        plan = max(len(a) for a in assign)
        log(f"GPU fold assignment {assign} plan_folds={plan}")
        procs = []
        for g, folds in enumerate(assign):
            p = ctx.Process(target=gpu_worker, args=(g, folds, plan, len(assign)), daemon=False)
            p.start()
            procs.append(p)
            time.sleep(5)
        fetch_extras()
        while any(p.is_alive() for p in procs):
            time.sleep(30)
        for g, p in enumerate(procs):
            p.join()
            log(f"gpu worker {g} exit code {p.exitcode}")
        DEV = "cuda"

    # ---- summaries: per-fold metrics and the pooled out-of-fold readout table (gate G0 input)
    fold_res = [json.loads((METRIC_DIR / f"fold{f}.json").read_text()) for f in TRAIN_FOLDS
                if (METRIC_DIR / f"fold{f}.json").exists()]
    oof = []
    for f in TRAIN_FOLDS:
        fp = METRIC_DIR / f"oof_readout_fold{f}.csv"
        if fp.exists():
            with open(fp) as fh:
                oof += list(csv.DictReader(fh))
    with open(WORK / "oof_readout.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OOF_COLS)
        w.writeheader()
        w.writerows(oof)

    def pooled(sel, fn):
        v = [fn(r) for r in oof if sel(r)]
        return {"n": len(v), "mae": float(np.mean(v)) if v else None}

    g = lambda r, k: math.radians(float(r[k]))  # noqa: E731
    summary = {
        "folds": fold_res,
        "oof_fasc_angle": pooled(lambda r: r["has_fasc"] == "1", lambda r: _deg(g(r, "pred_fasc_deg") - g(r, "gt_fasc_deg"))),
        "oof_deep_line": pooled(lambda r: r["has_deep"] == "1", lambda r: _deg(g(r, "pred_line_deg") - g(r, "gt_line_deg"))),
        "oof_pa_both_mask": pooled(lambda r: r["has_fasc"] == "1" and r["has_deep"] == "1",
                                   lambda r: _deg((g(r, "pred_line_deg") - g(r, "pred_fasc_deg"))
                                                  - (g(r, "gt_line_deg") - g(r, "gt_fasc_deg")))),
    }
    (METRIC_DIR / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    log("CV summary: " + json.dumps({k: v for k, v in summary.items() if k != "folds"}))

    # ---- fold-mean + hflip TTA inference: maps and readout
    ckpts = sorted(CKPT_DIR.glob("fold*.pt"))
    if not ckpts:
        raise RuntimeError("no checkpoints were produced")
    models = []
    for cp in ckpts:
        m = build_model(pretrained=False)
        st = torch.load(cp, map_location="cpu", weights_only=False)
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st["state_dict"].items()})
        models.append(m.to(DEV).eval().to(memory_format=torch.channels_last))
    log(f"inference with {len(models)} fold models on {DEV}")
    manifest_rows, read_rows = [], []
    for name in ("test", *extra_names):
        entries = datasets.get(name) or []
        if not entries:
            continue
        t = time.time()
        writers = {ch: NpzZip(MASKBANK / f"{name}__{MODEL_NAME}_{ch}.npz") for ch in CHANNELS}
        n_ok = 0
        for j, (key, path, src) in enumerate(entries):
            try:
                gray = read_gray(path)
                if max(gray.shape) > 4000 or min(gray.shape) < 64:
                    continue
                box, kind = viewport(gray, path.suffix.lower())
                enc, rd = predict_native(models, gray, box, torch)
                for ch in CHANNELS:
                    writers[ch].add(key, enc[ch])
                manifest_rows.append([name, key, src, gray.shape[0], gray.shape[1]])
                read_rows.append([name, key, kind, *box, *rd])
                n_ok += 1
                if j < 3:
                    save_viz(VIZ_DIR / f"{name}_{key}.jpg", gray, enc)
            except Exception:  # noqa: BLE001
                log(f"inference failed {name}/{src}:\n{traceback.format_exc()}")
        for w in writers.values():
            w.close()
        log(f"inference {name}: {n_ok}/{len(entries)} images in {time.time() - t:.0f}s")
    with open(MASKBANK / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "source_filename", "height", "width"])
        w.writerows(manifest_rows)
    with open(WORK / "readout.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "layout", "x1", "y1", "x2", "y2", "fasc_deg", "line_deg", "pa_image_deg",
                    "w_sum", "q_sum"])
        w.writerows(read_rows)
    config = {
        "model": MODEL_NAME, "mode": MODE, "encoder": ENCODER, "arch": "smp.Unet", "smp": smp.__version__,
        "torch": torch.__version__, "size": SIZE, "channels": list(CHANNELS),
        "angle_convention": "angles in image pixel coords of the file (x right, y down), undirected; "
                            "pa_image_deg = line_deg - fasc_deg, NOT corrected for pixel aspect",
        "readout": "fascicle = weighted doubled-angle mean (weights wread*valid); deep line = total least squares "
                   "with weights deep^2*valid",
        "loss_weights": {"deep": W_DEEP, "fa": W_FA, "ln": W_LN, "pa": W_PA, "cons": W_CONS},
        "train": {"batch_items": BATCH_ITEMS, "views": 2, "max_epochs": MAX_EPOCHS, "folds": TRAIN_FOLDS, "seed": SEED},
        "data_stats": stats, "selfcheck": checks, "elapsed_min": (time.time() - T0) / 60,
    }
    (WORK / f"{MODEL_NAME}_config.json").write_text(json.dumps(config, indent=2, default=str))
    shutil.rmtree(TMP, ignore_errors=True)
    log(f"DONE elapsed={(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
