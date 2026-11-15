# SPDX-License-Identifier: GPL-3.0-or-later
# UMUD v36 stage-0 probe: can zero-shot SAM (ViT-B, public Apache-2.0 weights) find the muscle belly between the
# aponeuroses precisely enough to measure muscle thickness? CPU only. Training aponeurosis images only (no test).
# Prompts are taken from the hand mask (oracle prompts), so the result is an upper bound for SAM.
import csv, json, subprocess, sys, time, urllib.request
from pathlib import Path

import cv2
import numpy as np

T0 = time.time()
N_IMAGES, SEED = 120, 20261111
WORK = Path("/kaggle/working")
ROOT = Path("/kaggle/input")


def log(m):
    print(f"[{(time.time() - T0) / 60:6.2f}m] {m}", flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segment-anything"], check=True)
import torch  # noqa: E402
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402

ckpt = WORK / "sam_vit_b_01ec64.pth"
urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", ckpt)
torch.set_num_threads(4)
sam = sam_model_registry["vit_b"](checkpoint=str(ckpt)).eval()
pred = SamPredictor(sam)
log("SAM ViT-B loaded")

EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
imgs = {p.stem: p for p in ROOT.rglob("*") if p.suffix.lower() in EXT and "apo_imgs" in str(p)}
masks = {p.stem: p for p in ROOT.rglob("*") if p.suffix.lower() in EXT and "apo_masks" in str(p)}
stems = sorted(set(imgs) & set(masks))
log(f"pairs {len(stems)}")
rng = np.random.default_rng(SEED)
stems = [stems[i] for i in rng.permutation(len(stems))]


def truth(mask):
    """Superficial band bottom edge and deep band top edge per column, from the two largest mask components."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n < 3:
        return None
    big = np.argsort(st[1:, cv2.CC_STAT_AREA])[::-1][:2] + 1
    a, b = sorted(big, key=lambda k: st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] / 2)
    H, W = mask.shape
    ys = np.arange(H)[:, None]
    sa, sb = lab == a, lab == b
    sbot = np.where(sa.any(0), (ys * sa).max(0), -1).astype(float)
    stop = np.where(sa.any(0), np.where(sa, ys, H).min(0), -1).astype(float)
    dtop = np.where(sb.any(0), np.where(sb, ys, H).min(0), -1).astype(float)
    dbot = np.where(sb.any(0), (ys * sb).max(0), -1).astype(float)
    ok = sa.any(0) & sb.any(0) & (dtop - sbot > 8)
    return sbot, stop, dtop, dbot, ok


rows, n_done = [], 0
for stem in stems:
    if n_done >= N_IMAGES:
        break
    try:
        img = cv2.imread(str(imgs[stem]), cv2.IMREAD_COLOR)
        m = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        t = truth(m)
        if t is None:
            continue
        sbot, stop, dtop, dbot, ok = t
        xs_ok = np.flatnonzero(ok)
        if len(xs_ok) < 0.3 * img.shape[1]:
            continue
        cols = [int(xs_ok[int(q * (len(xs_ok) - 1))]) for q in (0.25, 0.5, 0.75)]
        pos = [[x, 0.5 * (sbot[x] + dtop[x])] for x in cols]
        neg = [[cols[1], max(0.0, stop[cols[1]] - 12)], [cols[1], min(img.shape[0] - 1.0, dbot[cols[1]] + 12)]]
        pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        best = {}
        for name, pts, lbl in (("pos3", pos, [1, 1, 1]), ("pos3neg2", pos + neg, [1, 1, 1, 0, 0])):
            mk, sc, _ = pred.predict(point_coords=np.array(pts, float), point_labels=np.array(lbl), multimask_output=True)
            errs_all = []
            for k in range(len(mk)):
                e = []
                for x in cols:
                    col = mk[k][:, x]
                    y0 = int(round(0.5 * (sbot[x] + dtop[x])))
                    if not col[y0]:
                        e.append(np.nan)
                        continue
                    up = y0
                    while up > 0 and col[up - 1]:
                        up -= 1
                    dn = y0
                    while dn < len(col) - 1 and col[dn + 1]:
                        dn += 1
                    mt_s = dn - up + 1
                    mt_t = dtop[x] - sbot[x] - 1
                    e.append((mt_s - mt_t) / mt_t)
                errs_all.append(e)
            k_sam = int(np.argmax(sc))  # SAM's own choice
            mean_abs = [np.nanmean(np.abs(e)) if not np.all(np.isnan(e)) else np.inf for e in errs_all]
            k_orc = int(np.argmin(mean_abs))  # oracle choice among SAM's three masks
            best[name] = (errs_all[k_sam], errs_all[k_orc])
        mt_px = float(np.mean([dtop[x] - sbot[x] - 1 for x in cols]))
        r = [stem, img.shape[0], img.shape[1], round(mt_px, 1)]
        for name in ("pos3", "pos3neg2"):
            for e in best[name]:
                a = np.array(e, float)
                r += [float(np.nanmean(a)) if not np.all(np.isnan(a)) else np.nan,
                      float(np.nanmean(np.abs(a))) if not np.all(np.isnan(a)) else np.nan]
        rows.append(r)
        n_done += 1
        if n_done % 10 == 0:
            log(f"{n_done} images")
    except Exception as ex:  # noqa: BLE001
        log(f"fail {stem}: {ex}")

hdr = ["stem", "H", "W", "mt_px"] + [f"{n}_{c}_{s}" for n in ("pos3", "pos3neg2") for c in ("sam", "oracle") for s in ("rel", "absrel")]
with open(WORK / "sam_probe_rows.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(hdr)
    w.writerows(rows)
A = np.array([r[4:] for r in rows], float)
mt = np.array([r[3] for r in rows], float)
summ = {"n": len(rows), "mt_px_median": float(np.median(mt)), "production_oof_rel_median": 0.0066}
for j, h in enumerate(hdr[4:]):
    if h.endswith("absrel"):
        v = A[:, j]
        summ[h] = {"median": float(np.nanmedian(v)), "mean": float(np.nanmean(v)), "nan": int(np.isnan(v).sum()),
                   "median_px": float(np.nanmedian(v * mt)), "share_below_0.0066": float(np.nanmean(v < 0.0066))}
(WORK / "sam_probe_summary.json").write_text(json.dumps(summ, indent=2))
log("SUMMARY " + json.dumps(summ))
ckpt.unlink()
log("DONE")
