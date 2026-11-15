"""Checks for apply_prefill.py and anchoring_report.py (run: python tools/annotator/test_prefill.py).

Synthetic tests always run, in temporary folders. The test on the real work manifest uses a COPY of it (and of
private/work_selection.json) in a temporary folder and skips when the kit is absent; the real files are only read.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anchoring_report as ar  # noqa: E402
import apply_prefill as ap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "data" / "manual_labelling"
VP = [232, 43, 940, 596]
PPC = 110.4


def synthetic_manifest() -> dict:
    """96 tasks like the real work set: 8 participants x 12 ranks, queue WRK_{8 rank + p}, plus *_n1 view frames."""
    images, tasks = [], []
    for n in range(1, 97):
        iid, p = f"WRK_{n:03d}", f"p{(n - 1) % 8 + 1}"
        base = {"width": 1172, "height": 608, "px_per_cm": PPC, "viewport": VP, "kind": "work", "task_id": iid,
                "participant": p, "clip": None}
        images += [{"image_id": iid, "file": f"img/{iid}.png", "number": n, **base},
                   {"image_id": f"{iid}_n1", "file": f"img/{iid}_n1.png", "number": None, **base, "view_only": True}]
        tasks.append({"task_id": iid, "image_id": iid, "clip": None, "members": [iid], "frames": [iid, f"{iid}_n1"]})
    return {"version": 1, "kind": "work", "storage_key": "umud_manual_work_v1", "n_images": 192, "n_tasks": 96,
            "images": images, "tasks": tasks}


def synthetic_selection(manifest: dict) -> dict:
    return {"tasks": [{"task_id": t["task_id"], "rank": (int(t["task_id"][4:]) - 1) // 8} for t in manifest["tasks"]]}


def make_prefill(e: dict, k: int = 0) -> dict:
    x1, y1, x2, y2 = e["viewport"]
    W = x2 - x1
    sup = lambda x: y1 + 40.0 + 0.01 * (x - x1) + 0.3 * (k % 5)  # noqa: E731
    deep = lambda x: y1 + 300.0 - 0.02 * (x - x1) + 0.2 * (k % 7)  # noqa: E731
    mt = [[[x1 + f * W, sup(x1 + f * W)], [x1 + f * W, deep(x1 + f * W)]] for f in (0.25, 0.5, 0.75)]
    t = math.radians(-20.0 - 0.1 * (k % 9))
    fl = []
    for f in (0.35, 0.5, 0.65):
        xi = x1 + f * W - 300.0
        yd = deep(xi)
        s = (yd - sup(xi)) / max(1e-6, -math.sin(t))
        fl.append([[xi, yd], [xi + s * math.cos(t), yd + s * math.sin(t)]])
    pa = []
    for f in (0.3, 0.5, 0.7):
        xi = x1 + f * W
        yd = deep(xi)
        pa.append([[xi + 11 * math.cos(t), yd + 11 * math.sin(t)], [xi + 110 * math.cos(t), yd + 110 * math.sin(t)],
                   [xi - 55, yd + 1.1], [xi + 55, yd - 1.1]])
    xs = [x1 + W * i / 19 for i in range(20)]
    return ap.normalize_prefill({"mt": mt, "fl": fl, "pa": pa, "apo_sup": [[x, sup(x)] for x in xs],
                                 "apo_deep": [[x, deep(x)] for x in xs], "confidence": 0.5 + 0.004 * k})


def synthetic_suggestions(manifest: dict, broken=("WRK_005", "WRK_007")) -> dict:
    imgs = {}
    for k, e in enumerate(x for x in manifest["images"] if not x.get("view_only")):
        imgs[e["image_id"]] = {"ok": True, "prefill": make_prefill(e, k), "values": {}, "diag": {}}
    imgs[broken[0]] = {"ok": False, "reason": "geometry: no_aponeurosis_pair"}
    bad = copy.deepcopy(imgs[broken[1]])
    bad["prefill"]["mt"] = bad["prefill"]["mt"][:2]
    imgs[broken[1]] = bad
    return {"version": 1, "created": "2026-11-02T12:00:00Z", "kernel": "shivateja005/umud-labelling-suggest",
            "source_short": "synthetic test suggestions", "n_images": 96, "n_ok": 95, "images": imgs}


def setup(tmp: Path, manifest: dict | None = None, selection: dict | None = None) -> tuple[Path, Path, Path]:
    work, private = tmp / "work", tmp / "private"
    work.mkdir(parents=True)
    private.mkdir(parents=True)
    m = manifest or synthetic_manifest()
    (work / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    (private / "work_selection.json").write_text(json.dumps(selection or synthetic_selection(m)), encoding="utf-8")
    sugg = tmp / "suggestions_v1.json"
    sugg.write_text(json.dumps(synthetic_suggestions(m)), encoding="utf-8")
    return work, private, sugg


def run_cli(*args) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = ap.main([str(a) for a in args])
    return code, buf.getvalue()


# ------------------------------------------------------------------------------------------------ apply_prefill
def test_validate_prefill_rejects_bad_shapes() -> None:
    e = synthetic_manifest()["images"][0]
    good = make_prefill(e)
    assert ap.validate_prefill(good, 1172, 608) == []
    cases = {
        "3 measurements": lambda p: p["mt"].pop(),
        "not vertical": lambda p: p["mt"][0][1].__setitem__(0, p["mt"][0][1][0] + 2),
        "below": lambda p: p["mt"][1].reverse(),
        "bad point": lambda p: p["fl"][0][0].__setitem__(1, float("nan")),
        "deep (lower)": lambda p: p["fl"][2].reverse(),
        "exactly 4": lambda p: p["pa"][0].pop(),
        "x must increase": lambda p: p["apo_sup"].reverse(),
        "confidence": lambda p: p.__setitem__("confidence", 1.5),
        "unexpected keys": lambda p: p.__setitem__("note", "x"),
        "missing keys": lambda p: p.pop("apo_deep"),
        "far outside": lambda p: p["fl"][0][1].__setitem__(0, 1e7),
    }
    for want, edit in cases.items():
        p = copy.deepcopy(good)
        edit(p)
        errs = ap.validate_prefill(p, 1172, 608)
        assert any(want in x for x in errs), (want, errs)


def test_blind_selection_is_stratified_and_deterministic() -> None:
    m = synthetic_manifest()
    ranks, src = ap.task_ranks(m, None)
    assert src.startswith("queue order") and ranks["WRK_009"] == 1 and ranks["WRK_096"] == 11
    a = ap.select_blind(m, ranks)
    b = ap.select_blind(m, ranks)
    assert a == b and len(a) == 24 and ap.check_blind(a, m) == []
    for p in {x["participant"] for x in a}:
        mine = sorted(x["rank"] for x in a if x["participant"] == p)
        assert len(mine) == 3 and 0 <= mine[0] <= 3 and 4 <= mine[1] <= 7 and 8 <= mine[2] <= 11, (p, mine)
    per_rank = [sum(x["rank"] == r for x in a) for r in range(12)]
    assert per_rank == [2] * 12, per_rank      # every block of 8 queue tasks holds 2 blind tasks
    assert ap.select_blind(m, ranks, seed=1) != a


def test_apply_writes_backup_prefill_and_blind() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        work, private, sugg = setup(tmp)
        original = (work / "manifest.json").read_bytes()
        code, out = run_cli("--suggestions", sugg, "--work", work, "--private", private)
        assert code == 0, out
        assert (work / ap.BACKUP_NAME).read_bytes() == original
        m = json.loads((work / "manifest.json").read_text())
        blind = json.loads((private / ap.BLIND_NAME).read_text())
        bt = {b["task_id"] for b in blind["tasks"]}
        assert len(bt) == 24 and ap.check_blind(blind["tasks"], m) == []
        assert ap.validate_manifest(m, bt) == []
        keys = list(m)
        assert keys.index("prefill_enabled") < keys.index("images") and m["prefill_enabled"] is True
        assert "synthetic test suggestions" in m["prefill_source"] and "blind" in m["prefill_blind_rule"]
        imgs = {e["image_id"]: e for e in m["images"]}
        assert not any("prefill" in e for e in m["images"] if e.get("view_only"))
        for t in m["tasks"]:
            if t["task_id"] in bt:
                assert t["prefill_blind"] is True and "prefill" not in imgs[t["image_id"]]
            else:
                assert "prefill_blind" not in t
        pre = [e["image_id"] for e in m["images"] if "prefill" in e]
        nb_broken = len({"WRK_005", "WRK_007"} - bt)
        assert len(pre) == 96 - 24 - nb_broken, (len(pre), nb_broken)
        assert "WRK_005" not in pre and "WRK_007" not in pre
        # blind suggestions only in the private file (valid ones), never in the served folder
        assert set(blind["suggestions"]) <= {imgs[t]["image_id"] for t in bt}
        assert len(blind["suggestions"]) == 24 - len({"WRK_005", "WRK_007"} & bt)
        served = json.loads((work / "manifest.json").read_text())
        served_pf = {json.dumps(e["prefill"]) for e in served["images"] if "prefill" in e}
        assert served_pf and not any(json.dumps(p) in served_pf for p in blind["suggestions"].values())
        assert ap.validate_prefill(imgs[pre[0]]["prefill"], 1172, 608) == []
        assert "WRK_007: invalid" in out or "WRK_007" in bt


def test_apply_is_idempotent_and_dry_run_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        work, private, sugg = setup(tmp)
        assert run_cli("--suggestions", sugg, "--work", work, "--private", private, "--dry-run")[0] == 0
        assert not (work / ap.BACKUP_NAME).exists() and not (private / ap.BLIND_NAME).exists()
        assert "prefill" not in (work / "manifest.json").read_text()
        assert run_cli("--suggestions", sugg, "--work", work, "--private", private)[0] == 0
        files = [work / "manifest.json", work / ap.BACKUP_NAME, private / ap.BLIND_NAME]
        snap = [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]
        code, out = run_cli("--suggestions", sugg, "--work", work, "--private", private)
        assert code == 0 and "unchanged" in out, out
        assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in files] == snap
        # remove, then re-apply: the same manifest and the same blind set (reused)
        first = files[0].read_bytes()
        assert run_cli("--work", work, "--private", private, "--remove")[0] == 0
        stripped = json.loads(files[0].read_text())
        assert "prefill_enabled" not in stripped and not any("prefill" in e for e in stripped["images"])
        assert files[0].read_text() == (work / ap.BACKUP_NAME).read_text()
        assert run_cli("--suggestions", sugg, "--work", work, "--private", private)[0] == 0
        assert files[0].read_bytes() == first and files[1].read_bytes() == snap[1][0]


def test_reselect_refused_with_labels_and_practice_refused() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        work, private, sugg = setup(tmp)
        assert run_cli("--suggestions", sugg, "--work", work, "--private", private)[0] == 0
        (work / "autosave.json").write_text(json.dumps({"labels": {"WRK_001": {"mt": [[[1, 1], [1, 9]]]}}}))
        code, out = run_cli("--suggestions", sugg, "--work", work, "--private", private, "--reselect-blind")
        assert code == 2 and "refused" in out, out
        prac = synthetic_manifest()
        prac["kind"] = "practice"
        (work / "manifest.json").write_text(json.dumps(prac, indent=1))
        code, out = run_cli("--suggestions", sugg, "--work", work, "--private", private)
        assert code == 2 and "practice / check never get prefill" in out, out


def test_real_work_manifest_copy() -> None:
    real = KIT / "work" / "manifest.json"
    if not real.exists():
        print("  skip: no real work manifest")
        return
    before = real.read_bytes()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        m = json.loads(before)
        sel_path = KIT / "private" / "work_selection.json"
        sel = json.loads(sel_path.read_text()) if sel_path.exists() else synthetic_selection(m)
        work, private, sugg = setup(tmp, m, sel)
        code, out = run_cli("--suggestions", sugg, "--work", work, "--private", private)
        assert code == 0, out
        new = json.loads((work / "manifest.json").read_text())
        blind = json.loads((private / ap.BLIND_NAME).read_text())
        assert ap.validate_manifest(new, {b["task_id"] for b in blind["tasks"]}) == []
        assert "selection rank" in blind["selection"], blind["selection"]
        ranks = {t["task_id"]: t["rank"] for t in sel["tasks"]}
        for b in blind["tasks"]:
            assert b["rank"] == ranks[b["task_id"]] and b["stratum"] == b["rank"] // 4, b
        # nothing else of the manifest changed
        assert ap.strip_prefill(new) == m and (work / ap.BACKUP_NAME).read_bytes() == before
        print(f"  real manifest copy: {sum('prefill' in e for e in new['images'])} prefilled, blind "
              f"{sorted(b['task_id'] for b in blind['tasks'])[:6]}...")
    assert real.read_bytes() == before


# ------------------------------------------------------------------------------------------------ anchoring_report
def _perturb(meas: dict, rng: random.Random, sd: float) -> dict:
    """A user label near a suggestion: MT deep points moved vertically, FL ends and PA arms jittered."""
    out = copy.deepcopy({k: meas[k] for k in ("mt", "fl", "pa")})
    for m in out["mt"]:
        m[1][1] += rng.gauss(0, sd)
    for m in out["fl"]:
        m[-1][0] += rng.gauss(0, 4 * sd)
        m[-1][1] += rng.gauss(0, 2 * sd)
    for m in out["pa"]:
        m[1][1] += rng.gauss(0, 2 * sd)
    return out


def test_anchoring_report_detects_pull_and_kept_lines() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        work, private, sugg = setup(tmp)
        assert run_cli("--suggestions", sugg, "--work", work, "--private", private)[0] == 0
        m = json.loads((work / "manifest.json").read_text())
        blind = json.loads((private / ap.BLIND_NAME).read_text())
        applied = ar._iso_ms(blind["applied"])
        imgs = {e["image_id"]: e for e in m["images"]}
        rng = random.Random(1)
        labels = {}
        for k, t in enumerate(m["tasks"]):
            is_blind = t.get("prefill_blind") is True
            s = blind["suggestions"].get(t["image_id"]) if is_blind else imgs[t["image_id"]].get("prefill")
            if s is None:
                continue
            copied = not is_blind and k % 6 == 0
            lab = copy.deepcopy({x: s[x] for x in ("mt", "fl", "pa")}) if copied \
                else _perturb(s, rng, 8.0 if is_blind else 2.5)
            lab["first"] = {**copy.deepcopy(lab), "t": applied + 1000}
            lab["updated"] = applied + 2000
            if k % 2 == 0:   # app.js assist fields on half of the labels (older labels have none)
                lab["assist"] = {"prefill_blind": is_blind, "prefill_shown": not is_blind, "prefill_accepted": copied}
            labels[t["task_id"]] = lab
        shown_ids = [t["task_id"] for t in m["tasks"] if t["task_id"] in labels and not t.get("prefill_blind")]
        early = shown_ids[1]
        labels[early]["first"]["t"] = applied - 60_000       # done before the prefill went live
        flagged = shown_ids[2]
        labels[flagged]["flag"] = True
        hidden = next(i for i in shown_ids[3:] if "assist" in labels[i] and not labels[i]["assist"]["prefill_accepted"])
        labels[hidden]["assist"]["prefill_shown"] = False    # own lines drawn before the ghosts appeared
        state = tmp / "autosave.json"
        state.write_text(json.dumps({"version": 1, "labels": labels}))
        rows, skipped = ar.collect(m, json.loads(state.read_text()), blind, "final", 2.0)
        assert skipped.get("completed before the prefill was applied") == 1 and skipped.get("flagged") == 1, skipped
        assert skipped.get("suggestions not shown (own lines drawn first)") == 1, skipped
        res = ar.report(rows, n_boot=1000)
        for p in ("mt", "fl", "pa"):
            r = res[p]
            assert r["shown_copied"]["n"] >= 5 and r["shown_copied"]["mad"] < 1e-6, (p, r["shown_copied"])
            assert r["shown_own"]["n"] + r["shown_copied"]["n"] == r["shown"]["n"]
            assert r["shown"]["n"] > 50 and r["blind"]["n"] >= 20, (p, r["shown"]["n"], r["blind"]["n"])
            assert r["shown"]["mad"] < r["blind"]["mad"] and r["diff_ci95"][1] < 0, (p, r)
            assert r["verdict"].startswith("closer"), r["verdict"]
            assert r["shown"]["kept_rate"] > max(0.1, r["blind"]["kept_rate"]), (p, r["shown"], r["blind"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = ar.main(["--work", str(work), "--state", str(state), "--blind", str(private / ap.BLIND_NAME),
                            "--boot", "500",
                            "--json", str(tmp / "anch.json"), "--use", "first"])
        assert code == 0 and "shown - blind" in buf.getvalue(), buf.getvalue()
        out = json.loads((tmp / "anch.json").read_text())
        assert out["use"] == "first" and set(out["params"]) == {"mt", "fl", "pa"}
        # sanity: labels identical to the suggestions give d = 0 (same maths as the annotator)
        e = imgs[next(iter(i for i in imgs if "prefill" in imgs[i]))]
        s = ar.L.summarize(e["prefill"], e["px_per_cm"])
        assert s["n_mt"] == 3 and s["n_fl"] == 3 and s["n_pa"] == 3 and 5 < s["pa"] < 40 and s["mt"] > 5


def test_existing_annotator_files_untouched() -> None:
    """The converter only writes inside --work / --private; it never touches practice/ or check/."""
    for sub in ("practice", "check"):
        man = KIT / sub / "manifest.json"
        if man.exists():
            assert "prefill" not in man.read_text(encoding="utf-8"), sub


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
