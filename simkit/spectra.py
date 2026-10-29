"""Spectral data: band-averaged emissivity/reflectance from a CSV curve.

Format (CSV, as exported by tools/import_speclib.py from ECOSTRESS/ASTER .txt):

    wavelength_um,value
    0.40,0.0512
    ...
    14.01,0.9811

value is reflectance for 0.4-2.5 um and emissivity for 2.5-14 um in the source
libraries; the importer writes the kind into the header comment so this loader
does not have to guess.
"""
import os

import numpy as np

_CACHE = {}


def load_csv(path):
    """[(wavelength_um, value)] from a two-column CSV with '#' comments."""
    key = os.path.abspath(path)
    if key in _CACHE:
        return _CACHE[key]
    wl, val = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(";", ",").split(",")
            if len(parts) < 2:
                continue
            try:
                wl.append(float(parts[0]))
                val.append(float(parts[1]))
            except ValueError:
                continue
    if not wl:
        raise ValueError("no numeric rows in " + path)
    out = (np.asarray(wl, float), np.asarray(val, float))
    _CACHE[key] = out
    return out


def band_average(path, lo_um, hi_um):
    """Mean value over [lo, hi]; returns None when the curve does not cover it."""
    wl, val = load_csv(path)
    m = (wl >= lo_um) & (wl <= hi_um)
    if not m.any():
        return None
    return float(np.mean(val[m]))


def emissivity_for(spec, band_um):
    """Emissivity of a material spec at a band, preferring a measured curve."""
    p = spec.get("spectra")
    if p:
        v = band_average(p, band_um[0], band_um[1])
        if v is not None:
            return v
    name = spec.get("spectra_band")
    if band_um[0] >= 7.0:
        v = spec.get("emissivity_lwir")
    elif band_um[0] >= 2.5:
        v = spec.get("emissivity_mwir")
    else:
        v = None
    if v is None:
        v = spec.get("emissivity", 0.9)
    return float(v)
