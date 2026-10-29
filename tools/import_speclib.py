"""Convert ECOSTRESS/ASTER spectral library .txt spectra into simkit CSV curves.

The library files are two-column ASCII (wavelength in um, reflectance or
emissivity).  Download them manually from https://speclib.jpl.nasa.gov/downloads
(the order form needs an e-mail address, so it cannot be scripted politely) and
point this tool at the extracted directory:

    python tools/import_speclib.py raw_speclib/ spectra/ [--match asphalt,metal]

Output: one CSV per spectrum, ready for a material entry

    "asphalt": dict(..., spectra="spectra/asphalt.csv")

and simkit.spectra.emissivity_for() then band-averages it for MWIR/LWIR.
"""
import os
import sys


def parse_txt(path):
    wl, val = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.replace(";", ",").split()
            if len(parts) < 2:
                parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if 0.2 <= a <= 200.0 and 0.0 <= b <= 1.5:
                wl.append(a)
                val.append(b)
    return wl, val


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    keep = None
    if "--match" in sys.argv:
        keep = sys.argv[sys.argv.index("--match") + 1].lower().split(",")
    os.makedirs(dst, exist_ok=True)
    files = []
    for root, _dirs, names in os.walk(src):
        for n in names:
            if n.lower().endswith((".txt", ".spectrum")):
                files.append(os.path.join(root, n))
    files.sort()
    kept = 0
    for p in files:
        rel = os.path.relpath(p, src).replace("\\", "/")
        if keep and not any(k in rel.lower() for k in keep):
            continue
        wl, val = parse_txt(p)
        if len(wl) < 20:
            continue
        out = os.path.join(dst, rel.replace("/", "__").rsplit(".", 1)[0] + ".csv")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("# source: %s\n# columns: wavelength_um,value\n" % rel)
            for a, b in zip(wl, val):
                fh.write("%.4f,%.5f\n" % (a, b))
        kept += 1
        print("%-52s %4d points  %.2f-%.2f um" % (rel[:52], len(wl), wl[0], wl[-1]))
    print("\nwrote %d curve(s) to %s" % (kept, dst))
    print("add to a material: spectra=\"%s/<name>.csv\"" % dst.replace("\\", "/"))
    return 0


sys.exit(main())
