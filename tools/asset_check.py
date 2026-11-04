"""Check a model's facet-derived RCS against its profile's reference value.

The profile RCS is an external, literature-anchored order-of-magnitude figure.
The simulator never uses it as an input: it computes RCS from facets so that
aspect angle, shadowing and material all matter. This tool is the bridge -- it
recomputes the facet RCS the simulator would see, at a set of aspect angles,
and reports how far it lands from the published range.

A large disagreement is a signal, not a bug: it usually means the mesh scale,
the material assignment or the published figure refers to a different band or
a different aspect than the one the profile claims.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simkit import materials as matlib          # noqa: E402
from simkit import radar                        # noqa: E402
from simkit import scene                        # noqa: E402


def _face_centers(world):
    """Per-face centroids, matching the face ordering of _face_normals."""
    tris = world.verts[world.faces]
    return tris.mean(axis=1)


def rcs_at_aspect(world, az_deg, el_deg, band_ghz=15.0):
    """Facet RCS of every object, seen from one aspect, in m^2.

    Mirrors the geometry the radar kernel uses: one incident direction for the
    whole scene, facets weighted by projected area and material reflectivity.
    """
    lam = 3e8 / (band_ghz * 1e9)
    az, el = math.radians(az_deg), math.radians(el_deg)
    dvec = np.array([math.cos(el) * math.cos(az),
                     math.cos(el) * math.sin(az),
                     math.sin(el)])

    normals = radar._face_normals(world)
    areas = radar._face_areas(world)
    out = {}
    for o in world.objs:
        if not o.get("label", True):
            continue
        m = world.face_obj == o["id"]
        if not m.any():
            continue
        # Only front-facing facets are illuminated. Taking the absolute value
        # counts the far side of a closed body as well, which roughly doubles
        # the projected area and was one source of the inflated returns.
        cos_i = normals[m] @ (-dvec)
        lit = cos_i > 1e-6
        if not lit.any():
            out[o["name"]] = dict(rcs_m2=0.0, lit_faces=0)
            continue
        rho_tab = np.array([matlib.MATERIALS[n]["rf_rho"] for n in sorted(matlib.MATERIALS)])
        rough_tab = np.array([matlib.MATERIALS[n]["rough_m"] for n in sorted(matlib.MATERIALS)])
        face_mat_idx = world.face_mat[m]
        # Two numbers, because they answer different questions.
        # per_facet: what the simulator currently integrates. It is not
        #   tessellation invariant (the specular term goes as area squared, so a
        #   surface cut into n pieces loses a factor n), but it is the quantity
        #   the renderer actually uses, so it is the one to compare with a
        #   dataset's detection range.
        # coherent: correct physical optics, field summed before squaring.
        #   Verified mesh-invariant to 1e-4 dB and matching the closed form
        #   4*pi*A^2/lam^2. It is the right primitive, but on a proxy mesh it
        #   still cannot predict a complex target's absolute level.
        centers = _face_centers(world)[m][lit]
        per_facet = float(np.sum(radar.facet_sigma(
            areas[m][lit], cos_i[lit], rho_tab[face_mat_idx[lit]],
            rough_tab[face_mat_idx[lit]], lam,
            spec_scale=10.0 ** (-20.0 / 10.0))))
        coherent = radar.coherent_po_sigma(
            centers, normals[m][lit], areas[m][lit],
            rho_tab[face_mat_idx[lit]], -dvec, dvec, lam)
        sig = per_facet
        out[o["name"]] = dict(rcs_m2=float(sig), coherent_m2=float(coherent),
                              lit_faces=int(lit.sum()))
    return out


def db(x):
    return 10.0 * math.log10(max(float(x), 1e-12))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compare facet RCS with profile reference")
    ap.add_argument("asset", help="asset key, e.g. 军用飞机/单发战斗机")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--index", default=os.path.join("assets", "asset_index.json"))
    ap.add_argument("--band-ghz", type=float, default=15.0)
    ap.add_argument("--az", default="0,45,90,135,180")
    ap.add_argument("--el", type=float, default=5.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from simkit import asset as assetlib
    index = assetlib.load_index(args.index)
    spec = assetlib.expand(dict(asset=args.asset, profile=args.profile),
                           index=index, base_dir=".")
    prof = [p for p in assetlib._find(index, args.asset)["profiles"]
            if p.get("name") == spec["_asset"]["profile"]][0]

    cfg = dict(objects=[dict(spec, pos=[0, 0, 0], name="target")],
               ground=dict(enabled=False), _base_dir=os.path.abspath("."))
    world = scene.build_world(cfg)

    rows = []
    for az in [float(a) for a in args.az.split(",")]:
        r = rcs_at_aspect(world, az, args.el, args.band_ghz)
        val = r.get("target", {}).get("rcs_m2", 0.0)
        rows.append(dict(az_deg=az, rcs_m2=val, rcs_db=db(val)))

    ref = prof.get("rcs") or {}
    ref_val = ref.get("value_m2")
    result = dict(asset=args.asset, profile=prof.get("name"),
                  band_ghz=args.band_ghz, el_deg=args.el,
                  aspect=rows, reference=ref,
                  reference_db=db(ref_val) if ref_val else None)
    if ref_val:
        vals = [r["rcs_db"] for r in rows if r["rcs_m2"] > 0]
        if vals:
            result["computed_range_db"] = [min(vals), max(vals)]
            result["delta_db"] = [min(vals) - db(ref_val), max(vals) - db(ref_val)]

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        print("%s / %s   %.0f GHz, el %.1f deg" % (args.asset, prof.get("name"), args.band_ghz, args.el))
        print("  %-8s %12s %10s" % ("az(deg)", "rcs(m2)", "rcs(dBsm)"))
        for r in rows:
            print("  %-8.1f %12.4g %10.2f" % (r["az_deg"], r["rcs_m2"], r["rcs_db"]))
        if ref_val:
            print("  reference: %.4g m2 (%.1f dBsm)  [%s]" % (ref_val, db(ref_val), ref.get("source", "no source")))
            print("  delta: %.1f .. %.1f dB" % tuple(result["delta_db"]))


if __name__ == "__main__":
    main()