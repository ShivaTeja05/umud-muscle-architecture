"""Python mirror of the annotator's geometry-assist maths (the GEOM block in app.js, between the
ASSIST-GEOM-BEGIN / ASSIST-GEOM-END markers).  Stdlib only; test_assist.py runs both on the same cases.

Bands: polylines through the image's own MT clicks.  The superficial band goes through the upper point
(smaller y) of each MT line, the deep band through the lower one, sorted by x.  Points less than 0.5 px apart
in x are averaged.  Past the outermost points a band continues straight with the slope of the nearest
segment.  Fewer than 2 distinct x positions give no band (the assist is then inactive).

FL: ``extend_fascicle`` orders the clicks so that the first one is the end nearer the deep band (by relative
depth between the bands), then moves each end along its end segment onto its band.  An end inside the muscle
is extended; an end beyond its band is trimmed where the line crosses the band (also when no click lies between
the bands but one segment crosses both, e.g. 2 clicks into the two bright bands).  The result is
[deep point, ...inner clicks..., superficial point].  It fails (the caller keeps the raw line) when no click is
between the bands and no segment crosses both, when an end segment never meets its band, or when it meets it
more than ``max_dist`` px (3 image widths in the tool) away.

PA: ``pa_arm`` intersects the fascicle line (2 clicks) with the deep band and returns 2 points on the band at
x -/+ half_px (5 mm in the tool) around the crossing.
"""

from __future__ import annotations

import math


def clean_band(pts: list) -> list | None:
    s = sorted(([float(p[0]), float(p[1])] for p in pts), key=lambda p: p[0])
    groups: list[dict] = []
    for p in s:
        if groups and p[0] - groups[-1]["x0"] < 0.5:
            g = groups[-1]
            g["n"] += 1
            g["sx"] += p[0]
            g["sy"] += p[1]
        else:
            groups.append({"x0": p[0], "n": 1, "sx": p[0], "sy": p[1]})
    band = [[g["sx"] / g["n"], g["sy"] / g["n"]] for g in groups]
    return band if len(band) >= 2 else None


def bands_from_mt(mt: list) -> dict | None:
    if not isinstance(mt, list) or len(mt) < 2:
        return None
    tops, bots = [], []
    for m in mt:
        if m[0][1] <= m[1][1]:
            tops.append(m[0])
            bots.append(m[1])
        else:
            tops.append(m[1])
            bots.append(m[0])
    sup, deep = clean_band(tops), clean_band(bots)
    return {"sup": sup, "deep": deep} if sup and deep else None


def band_y(band: list, x: float) -> float:
    k = len(band)
    j = 0
    while j < k - 2 and x > band[j + 1][0]:
        j += 1
    p, q = band[j], band[j + 1]
    return p[1] + (q[1] - p[1]) * (x - p[0]) / (q[0] - p[0])


def band_roots(band: list, a: list, d: list) -> list[float]:
    """Every t where the line a + t*d meets the band, sorted; parallel pieces give no root."""

    k = len(band)
    roots = []
    dn = math.hypot(d[0], d[1])
    if not dn > 0:
        return roots
    for j in range(k - 1):
        p, q = band[j], band[j + 1]
        m = (q[1] - p[1]) / (q[0] - p[0])
        den = d[1] - m * d[0]
        if abs(den) <= 1e-9 * dn * math.hypot(1, m):
            continue
        t = (p[1] + m * (a[0] - p[0]) - a[1]) / den
        x = a[0] + t * d[0]
        lo = -math.inf if j == 0 else p[0]
        hi = math.inf if j == k - 2 else q[0]
        if lo - 1e-7 <= x <= hi + 1e-7:
            roots.append(t)
    return sorted(roots)


def _end_point(band: list, pts: list, k: int, inward: int, max_dist: float) -> dict:
    end_idx = 0 if inward > 0 else len(pts) - 1
    a = pts[k]
    if k == end_idx:
        nb = pts[k + inward]
        d = [a[0] - nb[0], a[1] - nb[1]]
        tmax = math.inf
    else:
        o = pts[k - inward]
        d = [o[0] - a[0], o[1] - a[1]]
        tmax = 1 + 1e-9
    t = next((r for r in band_roots(band, a, d) if -1e-9 <= r <= tmax), None)
    if t is None:
        return {"why": "its end segment never meets the band (parallel, or it points away)"}
    x = [a[0] + t * d[0], a[1] + t * d[1]]
    if math.hypot(x[0] - a[0], x[1] - a[1]) > max_dist:
        return {"why": "the band is too far along its end segment (nearly parallel)"}
    return {"X": x, "mode": ("extend" if t > 1e-9 else "on") if k == end_idx else "trim"}


def extend_fascicle(raw: list, bands: dict, max_dist: float) -> dict:
    """{ok, pts, deep, sup} or {ok: False, why}; see the module docstring."""

    pts: list = []
    for p in raw:
        if not pts or math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-6:
            pts.append([float(p[0]), float(p[1])])
    if len(pts) < 2:
        return {"ok": False, "why": "the line has no length"}
    sup, deep = bands["sup"], bands["deep"]
    a, b = pts[0], pts[-1]
    ga = band_y(deep, a[0]) - band_y(sup, a[0])
    gb = band_y(deep, b[0]) - band_y(sup, b[0])
    norm = ga > 1e-9 and gb > 1e-9
    da = (a[1] - band_y(sup, a[0])) / ga if norm else a[1]
    db = (b[1] - band_y(sup, b[0])) / gb if norm else b[1]
    if db > da:
        pts.reverse()  # pts[0] = the end nearer the deep band
    n = len(pts)

    def beyond_deep(q):
        return q[1] - band_y(deep, q[0]) > 0

    def beyond_sup(q):
        return band_y(sup, q[0]) - q[1] > 0

    i = 0
    while i < n and beyond_deep(pts[i]):
        i += 1
    j = n - 1
    while j >= 0 and beyond_sup(pts[j]):
        j -= 1
    # i <= j: pts[i..j] are the clicks between the bands; i == j + 1: no click between them, but the segment
    # pts[j] -> pts[i] crosses the whole muscle (both ends clicked into the bright bands); both ends are trimmed
    crossing = i == j + 1 and i < n and j >= 0
    if not crossing and (i > j or beyond_sup(pts[i]) or beyond_deep(pts[j])):
        return {"ok": False, "why": "no clicked point lies between the two bands"}
    e0 = _end_point(deep, pts, i, 1, max_dist)
    if "X" not in e0:
        return {"ok": False, "why": "deep end: " + e0["why"]}
    e1 = _end_point(sup, pts, j, -1, max_dist)
    if "X" not in e1:
        return {"ok": False, "why": "superficial end: " + e1["why"]}
    if math.hypot(e1["X"][0] - e0["X"][0], e1["X"][1] - e0["X"][1]) < 1:
        return {"ok": False, "why": "the extended line has no length"}
    inner = pts[max(i, 1):min(j, n - 2) + 1]
    return {"ok": True, "pts": [e0["X"], *inner, e1["X"]], "deep": e0["mode"], "sup": e1["mode"]}


def pa_arm(f0: list, f1: list, deep: list, half_px: float, max_dist: float) -> dict | None:
    """{X, arm}: the crossing of line(f0, f1) with the deep band nearest the clicked segment, and 2 band
    points at x -/+ half_px; None when they do not meet within max_dist."""

    d = [f1[0] - f0[0], f1[1] - f0[1]]
    length = math.hypot(d[0], d[1])
    if not length > 1e-6:
        return None
    roots = band_roots(deep, f0, d)
    if not roots:
        return None

    def off(t):
        return -t if t < 0 else (t - 1 if t > 1 else 0)

    best = roots[0]
    for r in roots:
        if off(r) < off(best):
            best = r
    if off(best) * length > max_dist:
        return None
    x = [f0[0] + best * d[0], f0[1] + best * d[1]]
    return {"X": x, "arm": [[x[0] - half_px, band_y(deep, x[0] - half_px)],
                            [x[0] + half_px, band_y(deep, x[0] + half_px)]]}
