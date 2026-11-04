"""Scan OBJ assets into per-model fact cards.

The scanner records only what can be measured or read from the file itself:
bounding box, vertex/face counts, material names, group names. Everything that
requires an external source (real dimensions, RCS, thermal signature) is left
for a profile to declare, with its own citation.

A card is a claim about geometry, so the geometry section is never hand-written
and never guessed: re-running the scanner is the only way to change it.
"""
import json
import math
import os

SCHEMA = 1

# A model whose axis ratio is absurd is not a scale problem, it is a broken
# mesh (stray vertex, exploded part, wrong up-axis). Flagging it keeps a bad
# bounding box from silently becoming a wrong physical size downstream.
MAX_PLAUSIBLE_ASPECT = 12.0


def read_obj(path):
    """Return vertices, face count, usemtl names and group names of an OBJ."""
    verts = []
    faces = 0
    materials = []
    groups = []
    for line in open(path, "r", encoding="utf-8", errors="replace"):
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    continue
        elif line.startswith("f "):
            faces += 1
        elif line.startswith("usemtl "):
            name = line[7:].strip()
            if name and name not in materials:
                materials.append(name)
        elif line.startswith("g ") or line.startswith("o "):
            name = line[2:].strip()
            if name and name not in groups:
                groups.append(name)
    return verts, faces, materials, groups


def bbox(verts):
    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    return lo, hi


def convex_hull_area_2d(pts):
    """Area of the XY convex hull, used as a planform proxy for rotor/wing area."""
    pts = sorted(set(pts))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def min_area_rect(pts):
    """Minimum-area enclosing rectangle of a point set in the XY plane.

    Fabricated OBJ models are routinely stored rotated by an arbitrary yaw.
    The axis-aligned bbox of such a mesh overstates both span and length, so a
    scale derived from it is wrong by 20-30%. Returns (width, height, angle_deg).
    """
    hull = convex_hull_2d(pts)
    if len(hull) < 3:
        return None
    best = None
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        length = math.hypot(ex, ey)
        if length < 1e-12:
            continue
        ux, uy = ex / length, ey / length
        us = [px * ux + py * uy for px, py in hull]
        vs = [-px * uy + py * ux for px, py in hull]
        w = max(us) - min(us)
        h = max(vs) - min(vs)
        area = w * h
        if best is None or area < best[0]:
            best = (area, w, h, math.degrees(math.atan2(uy, ux)))
    if best is None:
        return None
    _area, w, h, ang = best
    return (round(w, 4), round(h, 4), round(ang % 180.0, 2))


def convex_hull_2d(pts):
    """Andrew monotone chain hull; returns hull vertices in CCW order."""
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def scan_model(path):
    verts, faces, materials, groups = read_obj(path)
    if not verts:
        return None
    lo, hi = bbox(verts)
    ext = [round(hi[i] - lo[i], 4) for i in range(3)]
    longest = max(ext) if max(ext) > 0 else 0.0
    shortest = min(e for e in ext if e > 0) if any(e > 0 for e in ext) else 0.0
    aspect = round(longest / shortest, 2) if shortest > 0 else None

    # planform area in the XY plane: for aircraft this tracks wing area, for
    # ground vehicles the footprint. Cheap, and a sanity anchor for later RCS.
    planform = convex_hull_area_2d([(v[0], v[1]) for v in verts])
    scale_ref = longest if longest > 0 else 1.0

    # Yaw-corrected footprint. A mesh stored at an arbitrary rotation makes the
    # axis-aligned bbox overstate both span and length, which would corrupt every
    # size derived from it downstream.
    rect = min_area_rect([(v[0], v[1]) for v in verts])
    yaw_deg = 0.0
    if rect is not None:
        rect_w, rect_h, ang = rect
        yaw_deg = round(ang, 2)
        if ang > 45.0:
            yaw_deg = round(ang - 90.0, 2)
    else:
        rect_w = rect_h = None

    quality = []
    if aspect is not None and aspect > MAX_PLAUSIBLE_ASPECT:
        quality.append("aspect_%.1f_exceeds_%.0f" % (aspect, MAX_PLAUSIBLE_ASPECT))
    if not materials:
        quality.append("no_usemtl")
    if len(groups) <= 1:
        quality.append("no_part_groups")
    if faces < 200:
        quality.append("very_low_poly")
    # A quarter-turn is harmless: the axis-aligned box still equals the true
    # footprint, only the axis labels swap. Oblique storage is the case that
    # inflates the box by up to 30%, so only that earns a flag.
    oblique = min(abs(yaw_deg), abs(abs(yaw_deg) - 90.0))
    if oblique > 5.0:
        quality.append("stored_oblique_%.0fdeg" % abs(yaw_deg))

    return dict(
        source=os.path.basename(path),
        geometry=dict(
            bbox_extent=ext,
            bbox_min=[round(v, 4) for v in lo],
            bbox_max=[round(v, 4) for v in hi],
            footprint_rect=[rect_w, rect_h],
            stored_yaw_deg=yaw_deg,
            true_longest=round(max(rect_w, rect_h), 4) if rect_w else round(longest, 4),
            verts=len(verts),
            faces=faces,
            aspect=aspect,
            planform_area=round(planform, 4),
            planform_ratio=round(planform / (scale_ref * scale_ref), 4) if scale_ref else None,
            materials=materials,
            groups=groups,
            quality_flags=quality,
        ),
    )


def walk_models(base):
    for cat in sorted(os.listdir(base)):
        d = os.path.join(base, cat)
        if not os.path.isdir(d):
            continue
        for dp, _dn, fn in os.walk(d):
            for f in sorted(fn):
                if f.lower().endswith(".obj"):
                    yield cat, os.path.join(dp, f)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Scan OBJ files into geometry fact cards")
    ap.add_argument("--models", default=os.path.join("assets", "models"))
    ap.add_argument("--out", default=os.path.join("assets", "asset_index.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cards = []
    for cat, path in walk_models(args.models):
        rec = scan_model(path)
        if rec is None:
            continue
        rec["category"] = cat
        rec["name"] = os.path.splitext(os.path.basename(path))[0]
        # Repo-relative: the resolver joins this against the project root, so the
        # assets/ prefix must stay. Relative to dirname(models) it would lose it.
        rec["path"] = path.replace("\\", "/")
        cards.append(rec)

    index = dict(schema=SCHEMA, count=len(cards), models=cards)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=1)
    if not args.quiet:
        print("scanned %d models -> %s" % (len(cards), args.out))


if __name__ == "__main__":
    main()