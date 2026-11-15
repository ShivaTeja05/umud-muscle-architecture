

# =============================================================================================
# v33 overlay, part B: losses, evaluation, fold training
# =============================================================================================
def _masked_mean(v, m):
    return (v * m).sum() / m.sum().clamp_min(1.0)


def compute_losses(out, b, torch) -> dict:
    """out (2B,5,H,W) float32; b: batch dict with views A = [:B], B = [B:]."""
    F = torch.nn.functional
    seg_t, skel, om = b["seg"].float(), b["skel"].float(), b["om"].float()
    ct, st, valid = b["cos2"].float(), b["sin2"].float(), b["valid"].float()
    deep_t, hfseg, hf, ha = b["deep"].float(), b["hfseg"], b["hf"], b["ha"]
    logit = out[:, 0]
    pw = torch.tensor([POS_WEIGHT], device=out.device)
    bce = _masked_mean(F.binary_cross_entropy_with_logits(logit, seg_t, pos_weight=pw, reduction="none").mean((1, 2)), hfseg)
    p = torch.sigmoid(logit)
    dice_l = _masked_mean(1 - (2 * (p * seg_t).sum((1, 2)) + 1) / (p.sum((1, 2)) + seg_t.sum((1, 2)) + 1), hfseg)
    sk_n = skel.sum((1, 2))
    skel_l = _masked_mean(1 - (p * skel).sum((1, 2)) / sk_n.clamp_min(1.0), (sk_n > 0).float() * hfseg)
    cp, sp = torch.tanh(out[:, 1]), torch.tanh(out[:, 2])
    ori_l = (((cp - ct).abs() + (sp - st).abs()) * om).sum() / (2 * om.sum().clamp_min(1))
    dl = out[:, 3]
    e = torch.sigmoid(dl)
    deep_bce = F.binary_cross_entropy_with_logits(dl, deep_t, pos_weight=pw, reduction="none").mean((1, 2))
    deep_dice = 1 - (2 * (e * deep_t).sum((1, 2)) + 1) / (e.sum((1, 2)) + deep_t.sum((1, 2)) + 1)
    deep_l = _masked_mean(deep_bce + deep_dice, ha)
    th_f, th_l, _, _ = readout(cp, sp, torch.sigmoid(out[:, 4]), e, valid, torch)
    fa_l = _masked_mean(wrap_half(th_f - b["tf"], torch).abs(), hf)
    ln_l = _masked_mean(wrap_half(th_l - b["tl"], torch).abs(), ha)
    pa_l = _masked_mean(wrap_half((th_l - th_f) - (b["tl"] - b["tf"]), torch).abs(), hf * ha)
    n = out.shape[0] // 2
    la, lb = b["lin"][:n].float(), b["lin"][n:].float()
    cons_l = (wrap_half(map_angle(th_f[:n], la, lb, torch) - th_f[n:], torch).abs().mean()
              + wrap_half(map_angle(th_l[:n], la, lb, torch) - th_l[n:], torch).abs().mean())
    loss = (W_BCE * bce + W_DICE * dice_l + W_SKEL * skel_l + W_ORI * ori_l + W_DEEP * deep_l
            + W_FA * fa_l + W_LN * ln_l + W_PA * pa_l + W_CONS * cons_l)
    return {"loss": loss, "bce": bce, "dice_l": dice_l, "skel_l": skel_l, "ori_l": ori_l, "deep_l": deep_l,
            "fa_l": fa_l, "ln_l": ln_l, "pa_l": pa_l, "cons_l": cons_l}


LOSS_KEYS = ("loss", "bce", "dice_l", "skel_l", "ori_l", "deep_l", "fa_l", "ln_l", "pa_l", "cons_l")


def to_dev(b: dict, torch) -> dict:
    """(B,2,...) loader batch -> (2B,...) on the device, view A first."""
    return {k: v.transpose(0, 1).reshape(-1, *v.shape[2:]).to(DEV, non_blocking=True) for k, v in b.items()}


def build_val_cache(items: list[dict], torch) -> list[dict]:
    samples = [make_sample(it, None) for it in items]
    cache = []
    for i in range(0, len(samples), BATCH):
        chunk = samples[i:i + BATCH]
        cache.append({k: torch.from_numpy(np.stack([np.asarray(s[k]) for s in chunk])) for k in chunk[0]})
    return cache


def _deg(d: float) -> float:
    d = abs(d) % math.pi
    return min(d, math.pi - d) * 180.0 / math.pi


def evaluate(model, cache, torch, tta: bool = False, rows_out: list | None = None, items: list | None = None) -> dict:
    """Dense metrics as fasc_orient_v1 plus readout errors (degrees, letterboxed frame = image frame)."""
    model.eval()
    n_img, dice_sum, ang_sum, ang_n = 0, 0.0, 0.0, 0
    fa, ln, pa = [], [], []
    k = 0
    with torch.no_grad():
        for b in cache:
            b = {kk: v.to(DEV) for kk, v in b.items()}
            x = to_input(b["img"], torch)
            p, c, s, e, w = forward_maps(model, x, torch, tta)
            valid = b["valid"].float()
            th_f, th_l, wsum, qsum = readout(c, s, w, e, valid, torch)
            seg_t, m = b["seg"] > 0, (b["skel"] > 0) & (b["om"] > 0)
            d = torch.remainder(0.5 * torch.atan2(s, c) - 0.5 * torch.atan2(b["sin2"].float(), b["cos2"].float()), math.pi)
            d = torch.minimum(d, math.pi - d) * (180.0 / math.pi)
            for i in range(x.shape[0]):
                hf, ha = bool(b["hf"][i] > 0), bool(b["ha"][i] > 0)
                if bool(b["hfseg"][i] > 0):
                    n_img += 1
                    pred = p[i] > 0.5
                    den = float(pred.sum() + seg_t[i].sum())
                    dice_sum += 2 * float((pred & seg_t[i]).sum()) / den if den > 0 else 1.0
                    if int(m[i].sum()) >= 10:
                        ang_sum += float(d[i][m[i]].mean())
                        ang_n += 1
                tf, tl, pf, pl = float(b["tf"][i]), float(b["tl"][i]), float(th_f[i]), float(th_l[i])
                if hf:
                    fa.append(_deg(pf - tf))
                if ha:
                    ln.append(_deg(pl - tl))
                if hf and ha:
                    pa.append(_deg((pl - pf) - (tl - tf)))
                if rows_out is not None:
                    it = items[k]
                    rows_out.append([it["stem"], it["kind"], it.get("twin", ""), it["fold"], int(hf), int(ha),
                                     round(math.degrees(tf), 4), round(math.degrees(tl), 4),
                                     round(math.degrees(pf), 4), round(math.degrees(pl), 4),
                                     round(float(wsum[i]), 2), round(float(qsum[i]), 2)])
                k += 1
    mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    out = {"dice50": dice_sum / max(1, n_img), "ang_mae": ang_sum / max(1, ang_n), "fa_mae": mean(fa),
           "ln_mae": mean(ln), "pa_mae": mean(pa), "n_fa": len(fa), "n_ln": len(ln), "n_pa": len(pa)}
    parts = [v for v in (out["fa_mae"], out["ln_mae"]) if v == v]
    out["score"] = float(np.mean(parts)) if parts else float("inf")  # model selection: readout errors only
    return out


OOF_COLS = ["stem", "kind", "twin", "fold", "has_fasc", "has_deep", "gt_fasc_deg", "gt_line_deg", "pred_fasc_deg",
            "pred_line_deg", "w_sum", "q_sum"]


def _smoke_pick(items: list[dict], n: int, rng) -> list[dict]:
    """Equal thirds of both-mask, fascicle-only and aponeurosis-only items."""
    pools = [[it for it in items if it["kind"] == "fasc" and "dpts" in it],
             [it for it in items if it["kind"] == "fasc" and "dpts" not in it],
             [it for it in items if it["kind"] == "apo"]]
    out = []
    for pool in pools:
        take = min(len(pool), n // 3)
        out += [pool[i] for i in rng.choice(len(pool), take, replace=False)]
    return out


def load_v1_init(model, fold: int, torch) -> str:
    """Optional warm start from the fasc_orient_v1 checkpoint of the SAME fold (same fascicle folds)."""
    hits = [p for p in Path("/kaggle/input").rglob(f"fold{fold}.pt") if "checkpoints" in p.parts]
    if not hits:
        return "imagenet"
    try:
        sd = torch.load(hits[0], map_location="cpu", weights_only=False)["state_dict"]
        own = model.state_dict()
        n_copy = 0
        for k, v in sd.items():
            v = v.float() if v.is_floating_point() else v
            if k in own and own[k].shape == v.shape:
                own[k] = v
                n_copy += 1
            elif k in own and v.shape[0] == 3 and own[k].shape[0] == N_OUT:  # 3 -> 5 output channels
                own[k][:3] = v
                n_copy += 1
        model.load_state_dict(own)
        return f"v1:{hits[0]} ({n_copy}/{len(own)} tensors)"
    except Exception:  # noqa: BLE001
        log(f"warm start failed, using ImageNet:\n{traceback.format_exc()}")
        return "imagenet"


def train_fold(fold: int, gpu_state: dict, torch) -> None:
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(SEED + fold)
    tr = [it for it in ITEMS if it["fold"] != fold]
    va = [it for it in ITEMS if it["fold"] == fold]
    if SMOKE:
        tr, va = _smoke_pick(tr, SMOKE_TRAIN, rng), _smoke_pick(va, 12, rng)
    torch.manual_seed(SEED + fold)
    n_cpu = os.cpu_count() or 4
    n_workers = 2 if DEV == "cpu" else max(2, min(4, n_cpu // gpu_state["n_parallel"] + 1))
    ds = PairSet(tr, augment=True, repeat=1)
    if DEV == "cpu" or gpu_state.get("in_process"):
        n_workers = 0  # Avoid forking after initializing the Torch thread pool or CUDA.
    wk = dict(persistent_workers=True, worker_init_fn=_worker_init, prefetch_factor=4) if n_workers else {}
    dl = DataLoader(ds, batch_size=BATCH_ITEMS, shuffle=True, num_workers=n_workers, pin_memory=DEV == "cuda",
                    drop_last=True, **wk)
    t_cache = time.time()
    cache = build_val_cache(va, torch)
    kinds = {k: sum(1 for it in tr if (it["kind"], "dpts" in it) == k) for k in (("fasc", True), ("fasc", False), ("apo", True))}
    log(f"fold {fold}: train={len(tr)} {kinds} val={len(va)} loader_workers={n_workers} steps/epoch={len(dl)} "
        f"val_cache={time.time() - t_cache:.1f}s device={DEV}")

    model = build_model(pretrained=True)
    init = load_v1_init(model, fold, torch)
    log(f"fold {fold}: init {init}")
    model = model.to(DEV).to(memory_format=torch.channels_last)
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENC}, {"params": dec_params, "lr": LR_DEC}],
                            weight_decay=WEIGHT_DECAY)
    steps = len(dl)
    warm = max(1, int(steps * WARMUP_EPOCHS))
    n_epochs = gpu_state.get("n_epochs", MAX_EPOCHS)
    if "ep_time" in gpu_state:
        folds_left = max(1, gpu_state["plan_folds"] - gpu_state["done"])
        n_fit = int((T0 + TRAIN_BUDGET_S - time.time()) / folds_left / (gpu_state["ep_time"] * 1.03))
        n_epochs = int(max(MIN_EPOCHS, min(n_epochs, n_fit)))
        log(f"fold {fold}: n_epochs={n_epochs} (budget fit {n_fit})")
    sched_state = {"total": n_epochs * steps}

    def lr_lambda(step: int) -> float:
        if step < warm:
            return 0.05 + 0.95 * step / warm
        total = max(sched_state["total"], warm + 1)
        prog = min(1.0, (step - warm) / (total - warm))
        return LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    hist_path = METRIC_DIR / f"history_fold{fold}.csv"
    mcols = ["dice50", "ang_mae", "fa_mae", "ln_mae", "pa_mae", "score"]
    with open(hist_path, "w", newline="") as fh:
        csv.writer(fh).writerow(["epoch", *LOSS_KEYS, *mcols, "lr", "sec"])
    best, best_epoch, nonfinite = None, 0, 0
    ckpt_path = CKPT_DIR / f"fold{fold}.pt"
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        model.train()
        t_ep = time.time()
        sums = np.zeros(len(LOSS_KEYS))
        nb = 0
        for step, b in enumerate(dl, 1):
            b = to_dev(b, torch)
            out = model_out(model, to_input(b["img"], torch), torch)
            ls = compute_losses(out, b, torch)
            if not bool(torch.isfinite(ls["loss"])):
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                log(f"fold {fold} ep {epoch} step {step}: non-finite loss, step skipped")
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(ls["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums += [float(ls[k]) for k in LOSS_KEYS]
            nb += 1
            if step % 50 == 0 or (SMOKE and step % 4 == 0):
                el = time.time() - t_ep
                log(f"fold {fold} ep {epoch} step {step}/{steps} " + " ".join(f"{k}={sums[i] / nb:.4f}" for i, k in enumerate(LOSS_KEYS))
                    + f" s/step={el / step:.2f}")
        ep_rows = [] if epoch in EARLY_EPOCHS else None  # staged readout: per-image val rows, no TTA
        met = evaluate(model, cache, torch, rows_out=ep_rows, items=va if ep_rows is not None else None)
        if ep_rows:
            with open(METRIC_DIR / f"early_fold{fold}_ep{epoch}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(OOF_COLS)
                w.writerows(ep_rows)
        sec = time.time() - t_ep
        lr_now = opt.param_groups[0]["lr"]
        sums /= max(1, nb)
        log(f"EPOCH fold={fold} ep={epoch}/{n_epochs} " + " ".join(f"{k}={sums[i]:.4f}" for i, k in enumerate(LOSS_KEYS))
            + " | val " + " ".join(f"{k}={met[k]:.3f}" for k in mcols) + f" | lr={lr_now:.2e} total={sec:.0f}s")
        with open(hist_path, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, *[round(float(v), 5) for v in sums], *[met[k] for k in mcols], lr_now, sec])
        if best is None or met["score"] < best["score"]:
            best, best_epoch = met, epoch
            sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
            bad = [k for k, v in model.state_dict().items() if v.is_floating_point() and not bool(torch.isfinite(v.half()).all())]
            if bad:
                log(f"fold {fold}: WARNING {len(bad)} tensors overflow in fp16, saving fp32 instead: {bad[:3]}")
                sd = dict(model.state_dict())
            torch.save({"state_dict": sd, "epoch": epoch, "metrics": met, "fold": fold, "encoder": ENCODER,
                        "size": SIZE, "classes": CHANNELS}, ckpt_path)
            log(f"fold {fold}: saved best epoch {epoch} score={met['score']:.3f}")
        if epoch == 1 and "n_epochs" not in gpu_state and not SMOKE:
            ep_time = sec
            left = T0 + TRAIN_BUDGET_S - time.time() + ep_time
            n_epochs = int(max(MIN_EPOCHS, min(MAX_EPOCHS, (left / gpu_state["plan_folds"]) // (ep_time * 1.03))))
            gpu_state["n_epochs"], gpu_state["ep_time"] = n_epochs, ep_time
            sched_state["total"] = n_epochs * steps
            log(f"fold {fold}: epoch time {ep_time:.0f}s -> n_epochs={n_epochs} for all folds on this device")
        if time.time() > T0 + HARD_TRAIN_END_S:
            log(f"fold {fold}: HARD time limit reached after epoch {epoch}; stopping this fold")
            break
    del dl
    state = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state["state_dict"].items()})
    rows: list = []
    met_tta = evaluate(model, cache, torch, tta=True, rows_out=rows, items=va)
    log(f"fold {fold}: best epoch {best_epoch}; TTA val " + json.dumps(met_tta))
    with open(METRIC_DIR / f"oof_readout_fold{fold}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(OOF_COLS)
        w.writerows(rows)
    res = {"fold": fold, "best_epoch": best_epoch, "n_epochs": n_epochs, "best": best, "tta": met_tta, "init": init,
           "n_train": len(tr), "n_val": len(va), "nonfinite_steps": nonfinite}
    (METRIC_DIR / f"fold{fold}.json").write_text(json.dumps(res, indent=2))
    del model, opt, cache
    if DEV == "cuda":
        torch.cuda.empty_cache()


def gpu_worker(gpu: int, folds: list[int], plan_folds: int, n_parallel: int) -> None:
    global _LOG_PREFIX, _LOG_FILE, DEV
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _LOG_PREFIX = f"[gpu{gpu}]"
    _LOG_FILE = LOG_DIR / f"gpu{gpu}.log"
    code = 0
    try:
        import torch

        DEV = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        log(f"worker start folds={folds} device={torch.cuda.get_device_name(0)}")
        state = {"plan_folds": plan_folds, "n_parallel": n_parallel, "done": 0}
        for k in folds:
            try:
                train_fold(k, state, torch)
            except Exception:  # noqa: BLE001
                code = 1
                log(f"fold {k} FAILED:\n{traceback.format_exc()}")
                torch.cuda.empty_cache()
            state["done"] += 1
    except BaseException:  # noqa: BLE001
        code = 2
        log(f"worker FAILED:\n{traceback.format_exc()}")
    os._exit(code)
