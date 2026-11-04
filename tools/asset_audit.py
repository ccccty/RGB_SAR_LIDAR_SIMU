"""Audit every profiled asset against its literature order of magnitude."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import asset_check as ac


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join("assets", "asset_index.json"))
    ap.add_argument("--band-ghz", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join("assets", "rcs_audit.json"))
    args = ap.parse_args()

    from simkit import asset as assetlib
    index = assetlib.load_index(args.index)

    rows = []
    for m in index["models"]:
        for p in (m.get("profiles") or []):
            key = "%s/%s" % (m["category"], m["name"])
            try:
                spec = assetlib.expand(dict(asset=key, profile=p["name"]), index=index,
                                       base_dir=".")
                g = m["geometry"]
                cfg = dict(objects=[dict(spec, pos=[0, 0, 0], name="target")],
                           ground=dict(enabled=False), _base_dir=os.path.abspath("."))
                from simkit import scene
                world = scene.build_world(cfg)
            except Exception as exc:
                rows.append(dict(asset=key, profile=p["name"], error=str(exc)[:160]))
                continue

            vals = []
            for az in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0):
                r = ac.rcs_at_aspect(world, az, 5.0, args.band_ghz)
                vals.append(r.get("target", {}).get("rcs_m2", 0.0))
            good = [v for v in vals if v > 0]
            ref = (p.get("rcs") or {}).get("value_m2")
            row = dict(asset=key, profile=p["name"], band_ghz=args.band_ghz,
                       scale=spec.get("scale"),
                       model_bbox=g["bbox_extent"], footprint=g.get("footprint_rect"),
                       computed_median_m2=sorted(good)[len(good)//2] if good else None,
                       computed_min_m2=min(good) if good else None,
                       computed_max_m2=max(good) if good else None,
                       reference_m2=ref)
            if good and ref:
                row["delta_db"] = [round(ac.db(v) - ac.db(ref), 1) for v in (min(good), max(good))]
            rows.append(row)

    json.dump(dict(band_ghz=args.band_ghz, rows=rows),
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("%-34s %-24s %10s %10s %s" % ("asset", "profile", "calc(m2)", "ref(m2)", "delta dB"))
    for r in rows:
        if "error" in r:
            print("%-34s %-24s  ERROR %s" % (r["asset"], r["profile"], r["error"]))
            continue
        d = r.get("delta_db")
        print("%-34s %-24s %10.4g %10.4g  %s" % (
            r["asset"], r["profile"], r["computed_median_m2"] or 0, r["reference_m2"] or 0,
            ("%+.1f..%+.1f" % (d[0], d[1])) if d else "-"))


if __name__ == "__main__":
    main()
