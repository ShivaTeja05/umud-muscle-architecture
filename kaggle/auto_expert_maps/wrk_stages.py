"""AUTO-EXPERT maps: the KG0 production map models on the 96 WRK work frames (Kaggle CPU kernel, no GPU).

Adapted from kaggle/tt_bench/tt_stages.py (the CPU map recipe used for TT-BENCH): the apo / ours / orient stages are
VERBATIM; only the data stage and the paths differ. Inputs as in kaggle/labelling_suggest: labelling_work_v1.zip from
the kernel output shivateja005/umud-labelling-frames (or an extracted copy), its work/manifest.json, the 96 main
frames (never the *_n1 view-only frames); the CORRECTED viewport / px_per_cm table of the LOCAL work manifest is
embedded by build_kernel.py (the zip manifest carries the uncorrected kernel scale) and every PNG sha256 is checked.
No competition test image is read (the apo stage reads the competition TRAINING images only, for the BatchNorm
re-estimation of fix_main, exactly as the test maps were made). No pseudo-truth / algorithm estimate is read here.

Stages (each in its own subprocess, launched by kernel.py):
  data     WRK zip -> viewport crops (/tmp/wrk/crops/wrk/<id>.png) + wrk_out/manifest.csv (box, size, crop sha1,
           px_per_cm, participant, PNG sha256 check). Map frame = the crop.
  apo      apo_v2fix band / upper / lower: kaggle/apo_v2_fix/ext/infer_fix_ext.py imported as a module (PART = "wrk"),
           its fix_main() unchanged (BatchNorm re-estimation on each fold's training split, reproduction check, all 5
           folds, CPU, no TTA = the test-map recipe); only the inference loop is swapped for the WRK crops.
  ours     ours_apo384 / ours_fasc384 (+ _noflip): the torch stage of kaggle/infer_maskbank/pipeline.py, same code.
  orient   fasc_orient_v1 seg / cos2 / sin2: kaggle/fasc_orient_v1/train.py imported as a module, its viewport(),
           letterbox, forward_maps (all 5 fold checkpoints as in the production v1 maskbank + hflip TTA);
           predict_native on the CPU device (float32).
Outputs: /kaggle/working/wrk_out/maskbank/wrk__<model>.npz (uint8, crop frame), manifest.csv, data_info.json, logs.
GPL-3.0-only.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

WORK = Path("/kaggle/working")
TMP = Path("/tmp/wrk")
OUT = WORK / "wrk_out"
BANK = OUT / "maskbank"
CROPS = TMP / "crops"
MANIFEST = OUT / "manifest.csv"
INPUT_ROOT = Path("/kaggle/input")
TABLE_PATH = TMP / "frames_table.json"
SMOKE = int(os.environ.get("TT_SMOKE", "0"))      # >0: images per subset
ZIP_NAME = "labelling_work_v1.zip"
FRAMES_SLUG = "umud-labelling-frames"
ORDER = ["wrk"]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[wrk {time.time() - T0:7.1f}s] {msg}", flush=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_manifest() -> list[dict]:
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return sorted(rows, key=lambda r: ORDER.index(r["subset"]))


def by_subset(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    return [(s, [r for r in rows if r["subset"] == s]) for s in ORDER if any(r["subset"] == s for r in rows)]


def crop_path(r: dict) -> Path:
    return CROPS / r["subset"] / f"{r['key']}.png"


# ----------------------------------------------------------------------------------------------- data
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_work_source() -> tuple[str, Path]:
    """('zip', path) or ('dir', folder holding manifest.json + img/); as kaggle/labelling_suggest."""
    zips = sorted(p for p in INPUT_ROOT.rglob(ZIP_NAME) if p.is_file())
    if not zips:
        zips = sorted(p for p in INPUT_ROOT.rglob("labelling_work*.zip") if p.is_file())
    log(f"zip candidates: {[str(p) for p in zips]}")
    if zips:
        pref = [p for p in zips if FRAMES_SLUG in str(p)]
        return "zip", (pref or zips)[0]
    hits = sorted(p for p in INPUT_ROOT.rglob("WRK_001.png") if p.is_file())
    log(f"no zip; WRK_001.png candidates: {[str(p) for p in hits]}")
    for p in hits:
        if (p.parent.parent / "manifest.json").exists():
            return "dir", p.parent.parent
    raise FileNotFoundError(f"neither {ZIP_NAME} nor an extracted work/ folder under {INPUT_ROOT}")


def extract_members(archive: Path, members: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for name in members:
            if name not in names:
                raise FileNotFoundError(f"{name} not in {archive}")
            target = (dest / name).resolve()
            if not str(target).startswith(str(root) + os.sep):
                raise ValueError(f"unsafe member {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def stage_data() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kind, src = find_work_source()
    log(f"work source: {kind} {src}")
    if kind == "zip":
        with zipfile.ZipFile(src) as zf:
            names = zf.namelist()
        man_names = [n for n in names if n.endswith("manifest.json") and "/img/" not in n and "private" not in n]
        if not man_names:
            raise FileNotFoundError(f"no manifest.json in {src}")
        man_name = sorted(man_names, key=len)[0]
        prefix = man_name[: -len("manifest.json")]
        extract_members(src, [man_name], TMP / "zip")
        manifest = json.loads((TMP / "zip" / man_name).read_text(encoding="utf-8"))
        main = [e for e in manifest["images"] if not e.get("view_only")]
        extract_members(src, [prefix + e["file"] for e in main], TMP / "zip")
        work_dir = TMP / "zip" / prefix
    else:
        work_dir = src
        manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
        main = [e for e in manifest["images"] if not e.get("view_only")]
    table = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    main = sorted(main, key=lambda e: e["image_id"])
    if SMOKE:
        main = main[:SMOKE]
    rows, problems = [], []
    for e in main:
        iid = e["image_id"]
        path = work_dir / e["file"]
        t = table.get(iid)
        if t is None:
            problems.append(f"{iid}: not in the embedded table (skipped)")
            continue
        with Image.open(path) as im:
            gray = np.asarray(im.convert("L"), dtype=np.uint8)
        H, W = gray.shape
        if (W, H) != (int(t["width"]), int(t["height"])):
            problems.append(f"{iid}: size {W}x{H} != table {t['width']}x{t['height']}")
        sha_ok = sha256_file(path) == t["sha256"]
        if not sha_ok:
            problems.append(f"{iid}: PNG sha256 differs from the local copy")
        vp = [int(v) for v in t["viewport"]]
        x1, y1, x2, y2 = max(0, vp[0]), max(0, vp[1]), min(W, vp[2]), min(H, vp[3])
        crop = np.ascontiguousarray(gray[y1:y2, x1:x2])
        p = CROPS / "wrk" / f"{iid}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(p)
        rows.append(dict(subset="wrk", key=iid, src=e["file"], H0=H, W0=W, x1=x1, y1=y1, x2=x2, y2=y2,
                         kind="table_viewport", Hc=crop.shape[0], Wc=crop.shape[1],
                         crop_sha1=hashlib.sha1(crop.tobytes()).hexdigest(), px_per_cm=float(t["px_per_cm"]),
                         participant=t.get("participant"), png_sha_ok=bool(sha_ok)))
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for pr in problems:
        log(f"WARNING {pr}")
    (OUT / "data_info.json").write_text(json.dumps({"source": f"{kind}:{src}", "n": len(rows), "smoke": SMOKE,
                                                     "sha_ok": sum(r["png_sha_ok"] for r in rows),
                                                     "problems": problems}, indent=1))
    log(f"data: {len(rows)} crops, sha ok {sum(r['png_sha_ok'] for r in rows)}/{len(rows)}, problems {len(problems)}")
    if problems:
        sys.exit(3)


# ----------------------------------------------------------------------------------------------- apo
def stage_apo() -> None:
    m = load_module("apo_fix_src", TMP / "apo_fix_src.py")
    m.PART = "wrk"
    if SMOKE:
        m.N_FOLDS = 1
    rows = read_manifest()

    def run_inference_tt(torch, models, items, fold_of, dev):
        BANK.mkdir(parents=True, exist_ok=True)
        chans = ["band", "upper", "lower"]
        flip = dev.type == "cuda"
        all_models = [models[k] for k in sorted(models)]
        with open(BANK / "apo_v2fix_viewports_wrk.csv", "w", newline="") as vpf:
            vw = csv.writer(vpf)
            vw.writerow(["dataset", "image_id", "x1", "y1", "x2", "y2", "sx", "sy", "ox", "oy", "net_size"])
            for sub, entries in by_subset(rows):
                writers = {c: m.NpzWriter(BANK / f"{sub}__{m.FIX_NAME}_{c}.npz") for c in chans}
                t0 = time.time()
                for r in entries:
                    gray = m.read_gray(crop_path(r))
                    vp = m.detect_viewport(gray)
                    canvas, tf, _ = m.letterbox(gray, vp, m.SIZE)
                    x = torch.from_numpy(canvas.astype(np.float32)[None, None] / 255.0).to(dev)
                    x = x.to(memory_format=torch.channels_last)
                    with torch.inference_mode():
                        pr = m._predict(torch, all_models, x, dev, flip)[0].cpu().numpy()
                    pn = m.to_native(pr, tf)
                    for c, ch in enumerate(chans):
                        writers[ch].add(r["key"], np.clip(np.rint(pn[c] * 255.0), 0, 255).astype(np.uint8))
                    vw.writerow([sub, r["key"], *vp, f"{tf['sx']:.6f}", f"{tf['sy']:.6f}", tf["ox"], tf["oy"], m.SIZE])
                for w in writers.values():
                    w.close()
                m.log(f"maskbank {sub}: {len(entries)} images in {time.time() - t0:.0f}s (flip={flip}, "
                      f"folds={sorted(models)})")

    m.run_inference_fixed = run_inference_tt
    m.fix_main()


# ----------------------------------------------------------------------------------------------- ours (384)
TORCH_MODELS = {"ours_apo384": "apo_unetpp_r34_384.pt", "ours_fasc384": "fasc_unetpp_r34_384.pt"}


def find_input_file(name: str) -> Path:
    """Verbatim from kaggle/infer_maskbank/pipeline.py."""
    patterns = [f"notebooks/*/*/{name}", f"*/{name}", f"*/*/{name}"]
    matches = sorted({path for pattern in patterns for path in INPUT_ROOT.glob(pattern) if path.is_file()})
    if not matches:
        matches = sorted(path for path in INPUT_ROOT.rglob(name) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"{name} not found under {INPUT_ROOT}")
    if len(matches) > 1:
        log(f"multiple matches for {name}: {matches}; using first")
    return matches[0]


def _largest_true_run(values: np.ndarray) -> tuple[int, int]:
    """Verbatim from kaggle/infer_maskbank/pipeline.py."""
    padded = np.pad(values.astype(np.int8), 1)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, len(values)
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def training_viewport(image: np.ndarray) -> tuple[int, int, int, int]:
    """Verbatim copy of kaggle/aponeurosis_unetpp_384/train.py:training_viewport (via pipeline.py)."""
    gray = image.astype(np.float32)
    vertical_texture = np.abs(np.diff(gray, axis=0)).mean(axis=0)
    left, right = _largest_true_run(vertical_texture > 1.0)
    if right - left < image.shape[1] * 0.35:
        left, right = 0, image.shape[1]
    horizontal_texture = np.abs(np.diff(gray[:, left:right], axis=1)).mean(axis=1)
    top, bottom = _largest_true_run(horizontal_texture > 0.5)
    if bottom - top < image.shape[0] * 0.45:
        top, bottom = 0, image.shape[0]
    return left, top, right, bottom


def to_uint8(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(probability * 255.0), 0, 255).astype(np.uint8)


def stage_ours() -> None:
    """The torch stage of kaggle/infer_maskbank/pipeline.py on the TT crops."""
    import cv2
    import torch

    try:
        import segmentation_models_pytorch as smp
        if smp.__version__ != "0.5.0":
            raise ImportError(smp.__version__)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "segmentation-models-pytorch==0.5.0"])
        import segmentation_models_pytorch as smp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rows = read_manifest()
    BANK.mkdir(parents=True, exist_ok=True)
    info = {}
    vrows = []
    for model_key, filename in TORCH_MODELS.items():
        path = find_input_file(filename)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        use_viewport = bool(checkpoint.get("viewport_preprocessing", False))
        size = int(checkpoint.get("size", 384))
        model = smp.UnetPlusPlus(encoder_name=checkpoint.get("encoder", "resnet34"), encoder_weights=None,
                                 in_channels=1, classes=1, activation=None)
        state = {k.removeprefix("module."): v for k, v in checkpoint["model"].items()}
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        info[model_key] = {"checkpoint": str(path), "viewport_crop": use_viewport, "size": size}
        log(f"{model_key}: viewport_crop={use_viewport} size={size} device={device}")
        for sub, entries in by_subset(rows):
            t0 = time.time()
            mean_bank, noflip_bank = {}, {}
            for r in entries:
                with Image.open(crop_path(r)) as image:
                    gray = np.asarray(image.convert("L"), dtype=np.uint8)
                height, width = gray.shape
                left, top, right, bottom = training_viewport(gray) if use_viewport else (0, 0, width, height)
                if model_key == "ours_apo384":
                    vrows.append([sub, r["key"], left, top, right, bottom])
                crop = gray[top:bottom, left:right]
                x = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                batch = np.ascontiguousarray(np.stack([x, x[:, ::-1]])[:, None])
                with torch.inference_mode():
                    probability = torch.sigmoid(model(torch.from_numpy(batch).to(device))).float().cpu().numpy()[:, 0]
                pass0, pass1 = probability[0], probability[1][:, ::-1]
                full_mean = np.zeros((height, width), dtype=np.float32)
                full_noflip = np.zeros((height, width), dtype=np.float32)
                crop_size = (right - left, bottom - top)
                full_mean[top:bottom, left:right] = cv2.resize(np.ascontiguousarray(0.5 * (pass0 + pass1)), crop_size,
                                                               interpolation=cv2.INTER_LINEAR)
                full_noflip[top:bottom, left:right] = cv2.resize(np.ascontiguousarray(pass0), crop_size,
                                                                 interpolation=cv2.INTER_LINEAR)
                mean_bank[r["key"]] = to_uint8(full_mean)
                noflip_bank[r["key"]] = to_uint8(full_noflip)
            np.savez_compressed(BANK / f"{sub}__{model_key}.npz", **mean_bank)
            np.savez_compressed(BANK / f"{sub}__{model_key}_noflip.npz", **noflip_bank)
            log(f"  {model_key} {sub}: {len(entries)} images {time.time() - t0:.0f}s")
            del mean_bank, noflip_bank
            gc.collect()
        del model
    with (BANK / "viewports_ours_apo384_wrk.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "left", "top", "right", "bottom"])
        w.writerows(vrows)
    (BANK / "run_info_ours_wrk.json").write_text(json.dumps(info, indent=1))


# ----------------------------------------------------------------------------------------------- orient / extra
def predict_native_dev(fo, models, gray, box, torch, dev) -> dict:
    """fasc_orient_v1 train.py predict_native with the device as a parameter (the original calls .cuda())."""
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    inp, m = fo.make_input(crop)
    x = fo.to_input(torch.from_numpy(inp[None]).to(dev), torch)
    acc = None
    with torch.no_grad():
        for model in models:
            p, c, s = fo.forward_maps(model, x, torch, True)
            cur = torch.stack([p[0], c[0], s[0]], -1)
            acc = cur if acc is None else acc + cur
    maps = (acc / len(models)).cpu().numpy().astype(np.float32)
    back = fo.unletterbox(maps, m, x2 - x1, y2 - y1)
    full = np.zeros((gray.shape[0], gray.shape[1], 3), np.float32)
    full[y1:y2, x1:x2] = back
    return fo.encode_channels(full[..., 0], full[..., 1], full[..., 2])


def stage_orient(tag: str = "fasc_orient_v1", path_contains: str = "umud-fasc-orient-v1", n_out: int = 3,
                 size: int | None = None, encoder: str | None = None) -> None:
    import warnings

    warnings.filterwarnings("ignore", message=".*autocast.*")
    fo = load_module("fo_src", TMP / "fo_src.py")
    if size:
        fo.SIZE = int(size)
    if encoder:
        fo.ENCODER = encoder
    smp = fo.ensure_smp()
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    ckpts = sorted(p for p in INPUT_ROOT.rglob("fold*.pt") if path_contains in str(p))
    if not ckpts:
        raise FileNotFoundError(f"no fold*.pt under {INPUT_ROOT} containing {path_contains!r}")
    models = []
    for cp in ckpts:
        st = torch.load(cp, map_location="cpu", weights_only=False)
        if n_out == 3:
            mdl = fo.build_model(pretrained=False)
        else:
            mdl = smp.Unet(encoder_name=fo.ENCODER, encoder_weights=None, in_channels=3, classes=int(n_out),
                           decoder_channels=(256, 128, 64, 32, 16))
        mdl.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st["state_dict"].items()})
        models.append(mdl.to(dev).eval().to(memory_format=torch.channels_last))
    log(f"{tag}: {len(models)} fold models {[str(c) for c in ckpts]} size={fo.SIZE} device={dev}")
    rows = read_manifest()
    vrows = []
    for sub, entries in by_subset(rows):
        t0 = time.time()
        writers = {ch: fo.NpzZip(BANK / f"{sub}__{tag}_{ch}.npz") for ch in fo.CHANNELS}
        for r in entries:
            try:
                gray = fo.read_gray(crop_path(r))
                box, kind = fo.viewport(gray, ".png")
                enc = predict_native_dev(fo, models, gray, box, torch, dev)
                for ch in fo.CHANNELS:
                    writers[ch].add(r["key"], enc[ch])
                vrows.append([sub, r["key"], kind, *box])
            except Exception:  # noqa: BLE001
                log(f"{tag} failed {sub}/{r['key']}:\n{traceback.format_exc()}")
        for w in writers.values():
            w.close()
        log(f"  {tag} {sub}: {len(entries)} images {time.time() - t0:.0f}s")
    with (BANK / f"{tag}_viewports_wrk.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "layout", "x1", "y1", "x2", "y2"])
        w.writerows(vrows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "apo", "ours", "orient", "extra"])
    a = ap.parse_args()
    if a.stage == "data":
        stage_data()
    elif a.stage == "apo":
        stage_apo()
    elif a.stage == "ours":
        stage_ours()
    elif a.stage == "orient":
        stage_orient()
    else:
        cfg = json.loads(os.environ["TT_EXTRA"])
        stage_orient(cfg["tag"], cfg["path_contains"], int(cfg.get("n_out", 3)), cfg.get("size"), cfg.get("encoder"))


if __name__ == "__main__":
    main()
