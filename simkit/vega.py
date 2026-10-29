"""Vega Prime (Presagis) adapter -- discovery + integration contract.

simkit never links against Vega Prime.  A licensed VP installation is used
through a small host executable that YOU compile with the VP SDK; simkit talks
to that host through the external-renderer contract in simkit/external.py:

    scene.json:
      "optical":  {"backend": "vega"}
      "backends": {"vega": {"exe": "C:/Presagis/host/simkit_vega_host.exe"}}

The host must accept --config <json> --manifest <json> --out <dir> and write
<out>/images/optical/NNNNNN.png plus <out>/labels/optical/NNNNNN.json.
See docs/vega_prime_接入.md for the full checklist.
"""
import glob
import os

PATTERNS = [
    "C:/Presagis", "C:/Program Files/Presagis", "C:/Program Files (x86)/Presagis",
    "D:/Presagis", "E:/Presagis", "C:/Presagis/VegaPrime*",
]
DLL_HINTS = ["vp*.dll", "vs*.dll", "vp_sensor*.dll", "vsSensor*.dll"]


def find_install(cfg=None):
    cands = []
    if cfg:
        v = (cfg.get("backends", {}) or {}).get("vega", {}) or {}
        if v.get("install"):
            cands.append(v["install"])
    if os.environ.get("SIMKIT_VEGA_HOME"):
        cands.append(os.environ["SIMKIT_VEGA_HOME"])
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands.append(os.path.join(here, "third_party", "vega_prime"))
    for p in PATTERNS:
        cands.extend(glob.glob(p))
    for c in cands:
        if c and os.path.isdir(c):
            hits = []
            for pat in DLL_HINTS:
                hits.extend(glob.glob(os.path.join(c, "**", pat), recursive=True))
            if hits:
                return os.path.abspath(c), hits[:12]
    return None, []


def check(cfg=None):
    """Report what exists locally. Never tries to bypass licensing."""
    out = dict(install=None, dlls=[], host=None, ready=False, notes=[])
    inst, dlls = find_install(cfg)
    out["install"], out["dlls"] = inst, dlls
    host = ((cfg or {}).get("backends", {}) or {}).get("vega", {}).get("exe") if cfg else None
    host = host or os.environ.get("SIMKIT_VEGA_HOST")
    if host and os.path.exists(host):
        out["host"] = host
    if inst and not dlls:
        out["notes"].append("install directory found but no vp*/vs* DLLs -- is this the runtime?")
    if not inst:
        out["notes"].append("no Vega Prime installation found on this machine")
    if not out["host"]:
        out["notes"].append("host executable not built/configured (backends.vega.exe)")
    out["ready"] = bool(inst and out["host"])
    return out


def describe(cfg=None):
    r = check(cfg)
    print("Vega Prime adapter")
    print("  install : %s" % (r["install"] or "-"))
    if r["dlls"]:
        print("  dlls    : %s%s" % (", ".join(os.path.basename(d) for d in r["dlls"][:6]),
                                    " ..." if len(r["dlls"]) > 6 else ""))
    print("  host    : %s" % (r["host"] or "-"))
    print("  ready   : %s" % r["ready"])
    for n in r["notes"]:
        print("  note    : %s" % n)
    if not r["ready"]:
        print("\n  接入步骤见 docs/vega_prime_接入.md")
    return r
