

# =============================================================================================
# v33 overlay, part A: data (aponeurosis items, both-mask pairs), samples, differentiable readout
# Everything below overrides the fasc_orient_v1 definitions above where names repeat.
# =============================================================================================
MODEL_NAME = "pa_lines_v33"
CHANNELS = ("seg", "cos2", "sin2", "deep", "wread")
N_OUT = 5
CPU_SMOKE = MODE == "cpusmoke"
EARLY_EPOCHS = (1, 2, 4, 8, 12, 16, 20, 24)  # epochs whose per-image validation readout is saved
BATCH_ITEMS = 2 if CPU_SMOKE else 4  # items per step; every item gives 2 views -> 2x forward images
W_DEEP, W_FA, W_LN, W_PA, W_CONS = 1.0, 4.0, 4.0, 4.0, 2.0  # angle losses are in radians
DEEP_STRIDE, DEEP_WIDTH = 4, 3
MIN_FASC_LEN, MIN_LINE_SPAN = 20.0, 60.0  # canvas px needed for a readout label
PAIR_MIN_CORR = 0.90
DEV = "cpu"  # set in the worker
DATA_ROOT = os.environ.get("PA_V33_DATA_ROOT")
if os.environ.get("PA_V33_WORK_DIR"):
    WORK = Path(os.environ["PA_V33_WORK_DIR"]).resolve()
    MASKBANK, CKPT_DIR = WORK / "maskbank", WORK / "checkpoints"
    LOG_DIR, METRIC_DIR, VIZ_DIR = WORK / "logs", WORK / "metrics", WORK / "viz"

_kaggle_find_dir = find_dir


def find_dir(name: str) -> Path:
    if DATA_ROOT is None:
        return _kaggle_find_dir(name)
    root = Path(DATA_ROOT).resolve()
    hits = [p for p in root.rglob(name) if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f"{name} under {root}")
    return min(hits, key=lambda p: (len(str(p)), str(p)))


def find_apo_pairs() -> list[tuple[str, str]]:
    return find_pairs(find_dir("apo_images_new_model_v1"), find_dir("apo_masks_new_model_v1"))


def deep_edge_points(mask: np.ndarray, ref_width: float):
    """Inner (upper) edge of the deep aponeurosis band as (K,2) x,y points, or None.

    Bands are sorted top to bottom; the deep band is the first one clearly below the
    superficial band (same rule as kaggle/apo_v2 superficial_pair).
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = []
    for k in range(1, n):
        x, y, ww, hh, area = stats[k]
        if ww < max(8, 0.15 * ref_width) or area < 30:
            continue
        sub = lab[y:y + hh, x:x + ww] == k
        present = sub.any(axis=0)
        top = sub.argmax(axis=0).astype(np.float32) + y
        bot = (hh - 1 - sub[::-1].argmax(axis=0)).astype(np.float32) + y
        comps.append({"x": int(x), "present": present, "top": top, "bot": bot,
                      "yc": float(np.median((top + bot)[present] / 2)), "width": int(present.sum())})
    comps.sort(key=lambda c: c["yc"])
    if len(comps) < 2:
        return None
    sup = comps[0]
    w = mask.shape[1]
    sbot = np.full(w, np.nan, np.float32)
    sbot[sup["x"]:sup["x"] + len(sup["bot"])][sup["present"]] = sup["bot"][sup["present"]]
    for c in comps[1:]:
        ctop = np.full(w, np.nan, np.float32)
        ctop[c["x"]:c["x"] + len(c["top"])][c["present"]] = c["top"][c["present"]]
        both = ~np.isnan(sbot) & ~np.isnan(ctop)
        if both.sum() >= 0.2 * min(sup["width"], c["width"]) and float(np.median(ctop[both] - sbot[both])) > 2:
            xs = np.flatnonzero(~np.isnan(ctop))[::DEEP_STRIDE]
            return np.stack([xs.astype(np.float32), ctop[xs] - 0.5], 1)
    return None


def prep_apo(pair: tuple[str, str]) -> dict:
    image_path, mask_path = Path(pair[0]), Path(pair[1])
    img = read_gray(image_path)
    m = read_gray(mask_path)
    if m.shape != img.shape:  # masks at original resolution: plain stretch is the alignment
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = m > 127
    out = {"stem": "apo_" + image_path.stem, "raw_stem": image_path.stem, "name": image_path.name, "kind": "apo",
           "path": str(image_path), "ih": sha1_array(img), "H": img.shape[0], "W": img.shape[1], "empty": True,
           "segs": np.zeros((0, 4), np.float32)}
    box, kind = viewport(img, image_path.suffix.lower())
    pts = deep_edge_points(mask, box[2] - box[0])
    if pts is None or len(pts) < 8:
        return out
    x1, y1, x2, y2 = box
    inside = (pts[:, 0] >= x1) & (pts[:, 0] < x2) & (pts[:, 1] >= y1) & (pts[:, 1] < y2)
    if inside.mean() < 0.97:
        box, kind = (0, 0, img.shape[1], img.shape[0]), kind + "_lowcov"
        x1, y1, x2, y2 = box
    crop = np.ascontiguousarray(img[y1:y2, x1:x2])
    out.update(box=box, layout=kind, crop=crop, dpts_full=pts, dpts=pts - np.array([x1, y1], np.float32),
               thumb=thumbnail(crop), empty=False)
    return out


def attach_pairs(fasc_items: list[dict], apo_items: list[dict], fasc_paths: dict) -> dict:
    """Give the fascicle item of every verified both-mask pair the deep edge of its aponeurosis twin."""
    fmap = {}
    for it in fasc_items:
        for a in it["aliases"]:
            fmap[a] = it
    amap = {it["raw_stem"]: it for it in apo_items}
    st = {"listed": len(BOTH_PAIRS), "missing": 0, "low_corr": 0, "attached": 0}
    for fs, as_ in BOTH_PAIRS:
        fi, ai = fmap.get(fs), amap.get(as_)
        if fi is None or ai is None or fs not in fasc_paths:
            st["missing"] += 1
            continue
        fimg = read_gray(Path(fasc_paths[fs]))
        aimg = cv2.resize(read_gray(Path(ai["path"])), (fimg.shape[1], fimg.shape[0]), interpolation=cv2.INTER_AREA)
        a = cv2.resize(fimg, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        b = cv2.resize(aimg, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        a -= a.mean()
        b -= b.mean()
        corr = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
        if corr < PAIR_MIN_CORR or "dpts" in fi:
            st["low_corr"] += 1
            continue
        sx, sy = fimg.shape[1] / ai["W"], fimg.shape[0] / ai["H"]
        p = ai["dpts_full"].astype(np.float64)
        p = np.stack([(p[:, 0] + 0.5) * sx - 0.5 - fi["box"][0], (p[:, 1] + 0.5) * sy - 0.5 - fi["box"][1]], 1)
        fi["dpts"] = p.astype(np.float32)
        fi["twin"] = ai["stem"]
        ai["twin"] = fi["stem"]
        st["attached"] += 1
    return st


def assign_apo_folds(fasc_items: list[dict], apo_items: list[dict]) -> dict:
    """Fascicle folds stay exactly those of fasc_orient_v1. An aponeurosis image that is a twin or a
    near duplicate of a fascicle image inherits that fold; the rest are grouped and split on their own."""
    ft = np.stack([thumbnail(it["crop"]) for it in fasc_items])
    at = np.stack([it["thumb"] for it in apo_items])
    fmap = {it["stem"]: it for it in fasc_items}
    corr = at @ ft.T
    ffold = np.array([it["fold"] for it in fasc_items])
    st = {"twin": 0, "near_dup": 0, "conflict": 0, "free": 0}
    free = []
    for i, it in enumerate(apo_items):
        hit = np.flatnonzero(corr[i] > DUP_CORR)
        if "twin" in it:
            it["fold"] = fmap[it["twin"]]["fold"]
            st["twin"] += 1
            st["conflict"] += int(len(hit) and len(set(ffold[hit]) - {it["fold"]}) > 0)
        elif len(hit):
            it["fold"] = int(ffold[hit[np.argmax(corr[i][hit])]])
            st["near_dup"] += 1
            st["conflict"] += int(len(set(ffold[hit])) > 1)
        else:
            free.append(i)
    if free:
        sub = at[free]
        ii, jj = np.nonzero(np.triu(sub @ sub.T > DUP_CORR, 1))
        groups = union_find_groups(len(free), np.stack([ii, jj], 1))
        folds = assign_folds(groups, SEED + 1)
        for k, i in enumerate(free):
            apo_items[i]["fold"] = int(folds[k])
    st["free"] = len(free)
    for it in apo_items:
        del it["thumb"]
    return st


# ----------------------------------------------------------------------------- samples
def _line_angle_np(p: np.ndarray) -> float:
    q = p - p.mean(0)
    sxx, syy, sxy = float((q[:, 0] ** 2).sum()), float((q[:, 1] ** 2).sum()), float((q[:, 0] * q[:, 1]).sum())
    return 0.5 * math.atan2(2 * sxy, sxx - syy)


def make_sample(it: dict, rng: np.random.Generator | None) -> dict:
    """One view. Like fasc_orient_v1.make_sample plus the deep-edge target, the two readout
    labels (fascicle angle tf, deep-line angle tl; radians, canvas frame) and the linear map lin."""
    crop, segs, dpts = it["crop"], it["segs"], it.get("dpts")
    h, w = crop.shape
    x0, y0, ww, hh = 0, 0, w, h
    if rng is not None:
        if rng.random() < 0.35:
            ww = max(32, int(round(rng.uniform(0.3, 1.0) * w)))
            x0 = int(rng.integers(0, w - ww + 1))
        if rng.random() < 0.15:
            hh = max(32, int(round(rng.uniform(0.65, 1.0) * h)))
            y0 = int(rng.integers(0, h - hh + 1))
    sub = crop[y0:y0 + hh, x0:x0 + ww]
    m = letterbox_matrix(ww, hh)
    if rng is not None:
        m = aug_matrix(rng) @ m
    img = warp_image(sub, m)
    valid = _poly_mask(m, ww, hh)

    def inside(p):
        xi, yi = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < SIZE) & (yi >= 0) & (yi < SIZE)
        ok[ok] = valid[yi[ok], xi[ok]] > 0
        return ok

    hf, tf = 0, 0.0
    if len(segs):
        s = segs - np.array([x0, y0, x0, y0], np.float32)
        p1 = (m[:2, :2] @ s[:, :2].T + m[:2, 2:3]).T
        p2 = (m[:2, :2] @ s[:, 2:].T + m[:2, 2:3]).T
        segs_out = np.concatenate([p1, p2], 1)
        d = p2 - p1
        ln = np.hypot(d[:, 0], d[:, 1])
        ok = inside((p1 + p2) / 2) & (ln >= 3)
        if ok.any() and ln[ok].sum() >= MIN_FASC_LEN:
            c2 = ((d[ok, 0] ** 2 - d[ok, 1] ** 2) / ln[ok]).sum()  # length-weighted doubled angle
            s2 = (2 * d[ok, 0] * d[ok, 1] / ln[ok]).sum()
            hf, tf = 1, 0.5 * math.atan2(s2, c2)
    else:
        segs_out = np.zeros((0, 4))
    tg = render_targets(segs_out, valid)
    if it["kind"] == "apo":
        hf = 0
    deep = np.zeros((SIZE, SIZE), np.uint8)
    ha, tl = 0, 0.0
    if dpts is not None and len(dpts):
        p = dpts.astype(np.float64) - np.array([x0, y0])
        p = (m[:2, :2] @ p.T + m[:2, 2:3]).T
        gap = np.abs(np.diff(dpts[:, 0])) <= 3 * DEEP_STRIDE
        for a, b, g in zip(p[:-1], p[1:], gap):
            if g and np.isfinite(a).all() and np.isfinite(b).all() and np.abs(np.r_[a, b]).max() < 50 * SIZE:
                cv2.line(deep, (int(round(a[0] * 16)), int(round(a[1] * 16))),
                         (int(round(b[0] * 16)), int(round(b[1] * 16))), 1, DEEP_WIDTH, cv2.LINE_8, 4)
        deep &= valid
        ok = inside(p)
        if ok.sum() >= 8 and np.ptp(p[ok], axis=0).max() >= MIN_LINE_SPAN:
            ha, tl = 1, _line_angle_np(p[ok])
    if rng is not None:
        img = intensity_aug(img, valid, rng)
    tg.update(img=img, valid=valid, deep=deep, hf=np.float32(hf), ha=np.float32(ha), tf=np.float32(tf),
              tl=np.float32(tl), lin=m[:2, :2].astype(np.float32), hfseg=np.float32(it["kind"] != "apo"))
    return tg


class PairSet:
    """Two independently augmented views of every item (stacked on a new axis 0)."""

    def __init__(self, items: list[dict], augment: bool, repeat: int = 1) -> None:
        self.items, self.augment, self.repeat = items, augment, repeat

    def __len__(self) -> int:
        return len(self.items) * self.repeat

    def __getitem__(self, idx: int):
        import torch

        it = self.items[idx % len(self.items)]
        seed = (torch.initial_seed() + idx * 7919) % (2**32)
        rng = np.random.default_rng(seed)
        a, b = make_sample(it, rng), make_sample(it, rng)
        return {k: torch.from_numpy(np.stack([np.asarray(a[k]), np.asarray(b[k])])) for k in a}


# ----------------------------------------------------------------------------- readout (no parameters)
def wrap_half(d, torch):
    """Difference of two undirected angles, wrapped to (-pi/2, pi/2]."""
    return 0.5 * torch.atan2(torch.sin(2 * d), torch.cos(2 * d))


def readout(cp, sp, wr, ed, valid, torch):
    """cp, sp: tanh direction maps; wr: read weight in [0,1]; ed: deep-edge probability; valid: 0/1.
    All (B,H,W) float32. Returns fascicle angle, deep-line angle (radians, undirected) and supports."""
    w = wr * valid
    c, s = (w * cp).sum((1, 2)), (w * sp).sum((1, 2))
    th_f = 0.5 * torch.atan2(s, c + 1e-6)
    q = ed * ed * valid  # squared so the diffuse background does not flatten the fit
    qs = q.sum((1, 2)).clamp_min(1e-6)
    hh, ww = cp.shape[1:]
    ys = torch.arange(hh, device=cp.device, dtype=cp.dtype).view(1, hh, 1)
    xs = torch.arange(ww, device=cp.device, dtype=cp.dtype).view(1, 1, ww)
    mx, my = (q * xs).sum((1, 2)) / qs, (q * ys).sum((1, 2)) / qs
    dx, dy = xs - mx.view(-1, 1, 1), ys - my.view(-1, 1, 1)
    sxx, syy, sxy = (q * dx * dx).sum((1, 2)) / qs, (q * dy * dy).sum((1, 2)) / qs, (q * dx * dy).sum((1, 2)) / qs
    th_l = 0.5 * torch.atan2(2 * sxy, sxx - syy + 1e-6)
    return th_f, th_l, w.sum((1, 2)), qs


def map_angle(theta, lin_from, lin_to, torch):
    """Undirected angle theta measured in view A -> the same line seen in view B (exact for affine views)."""
    L = lin_to @ torch.linalg.inv(lin_from)
    d = torch.stack([torch.cos(theta), torch.sin(theta)], 1).unsqueeze(2)
    v = (L @ d).squeeze(2)
    return torch.atan2(v[:, 1], v[:, 0])


def build_model(pretrained: bool):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=ENCODER, encoder_weights="imagenet" if pretrained else None, in_channels=3,
                    classes=N_OUT, decoder_channels=(256, 128, 64, 32, 16))


def model_out(model, x, torch):
    if DEV == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            return model(x).float()
    return model(x).float()


def split_out(out, torch):
    return (torch.sigmoid(out[:, 0]), torch.tanh(out[:, 1]), torch.tanh(out[:, 2]), torch.sigmoid(out[:, 3]),
            torch.sigmoid(out[:, 4]))


def forward_maps(model, x, torch, tta: bool):
    """prob, cos2, sin2, deep, wread (B,H,W) float32; hflip TTA flips the sin2 sign."""
    b = x.shape[0]
    xin = torch.cat([x, torch.flip(x, dims=[3])], 0) if tta else x
    out = model_out(model, xin, torch)
    p, c, s, e, w = split_out(out[:b], torch)
    if tta:
        pf, cf, sf, ef, wf = split_out(torch.flip(out[b:], dims=[3]), torch)
        p, c, s, e, w = 0.5 * (p + pf), 0.5 * (c + cf), 0.5 * (s - sf), 0.5 * (e + ef), 0.5 * (w + wf)
    return p, c, s, e, w
