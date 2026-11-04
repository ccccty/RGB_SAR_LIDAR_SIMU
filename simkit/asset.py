"""Resolve a scenario object that names an asset + profile into a plain spec.

The simulator has no notion of an asset: it consumes an object spec dict with
keys such as model/scale/mat/thermal_regions. This module is the single place
that turns a catalogued asset into that dict, so scenario files stop repeating
scale and RCS numbers that drift apart between scenes.

    {"asset": "旋翼无人机/四旋翼无人机", "profile": "DJI Mavic 3", "pos": [0,0,120]}

is expanded to

    {"model": "../assets/models/旋翼无人机/四旋翼无人机.obj", "scale": 1.0, ...}

Design rules:
  * scale is derived from the profile's real size divided by the mesh extent,
    never typed by hand.
  * rcs_db is deliberately NOT emitted from the profile. A scalar RCS override
    disables the facet model and with it the aspect-dependent fluctuation that
    makes a radar detection task non-trivial. The profile value is a check, not
    an input; see tools/asset_check.py.
"""
import json
import os

DEFAULT_INDEX = os.path.join("assets", "asset_index.json")


class AssetError(ValueError):
    """Raised when an asset reference cannot be resolved unambiguously."""


def load_index(path=DEFAULT_INDEX):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _find(index, key):
    """Look a model up by 'category/name' or by bare name."""
    models = index.get("models", [])
    if "/" in key:
        cat, name = key.split("/", 1)
        for m in models:
            if m["category"] == cat and m["name"] == name:
                return m
    hits = [m for m in models if m["name"] == key]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise AssetError("unknown asset %r" % key)
    raise AssetError("ambiguous asset %r: %s" % (key, ", ".join(m["category"] for m in hits)))


def _profile(asset, name):
    profiles = asset.get("profiles") or []
    if not profiles:
        raise AssetError("asset %s/%s has no profiles" % (asset["category"], asset["name"]))
    if name is None:
        return profiles[0]
    for p in profiles:
        if p.get("name") == name:
            return p
    raise AssetError("asset %s/%s has no profile %r (have: %s)" % (
        asset["category"], asset["name"], name,
        ", ".join(str(p.get("name")) for p in profiles)))


def _axis_scale(extent, real_size, footprint=None):
    """Uniform scale from a least-squares fit over the documented dimensions.

    No single axis is authoritative on a proxy mesh. Two distortions stack up:
    fabricated models are stored at an arbitrary yaw (so the axis-aligned box
    overstates the horizontal size by 20-30%), and their silhouette often does
    not match the real type at all (a fighter proxy can be nearly as wide as it
    is long). Anchoring on one dimension -- longest, or height -- then forces the
    other two to be wrong, and a model that comes out 40% too big produces a
    correspondingly wrong radar return.

    So fit the scale that minimises the squared log-error across all three
    documented dimensions, trying both horizontal assignments. The residual is
    returned alongside, because a proxy that cannot be made to match should be
    reported, not quietly averaged into a plausible-looking number.
    """
    height = real_size.get("height")
    dims = []
    if footprint and footprint[0] and footprint[1]:
        hw, hh = sorted((float(footprint[0]), float(footprint[1])), reverse=True)
        dims.append((hw, "horizontal-major"))
        dims.append((hh, "horizontal-minor"))
    else:
        dims.append((float(extent[0]), "x"))
        dims.append((float(extent[1]), "y"))
    if isinstance(height, (int, float)) and height > 0:
        dims.append((float(extent[2]), "z"))
    else:
        dims.append(None)

    span = real_size.get("span")
    length = real_size.get("length")
    pairs = []  # (mesh_value, real_value)
    horizontals = [d for d in dims[:2] if d]
    real_h = [v for v in (span, length) if isinstance(v, (int, float)) and v > 0]

    if len(horizontals) == 2 and len(real_h) == 2:
        # Try both ways of assigning the two real horizontal dimensions to the
        # two mesh horizontal extents and keep the better fit.
        best = None
        for flip in (False, True):
            rh = list(real_h)
            if flip:
                rh = rh[::-1]
            trial = [(horizontals[0][0], rh[0]), (horizontals[1][0], rh[1])]
            if dims[2]:
                trial.append((dims[2][0], height))
            s = _fit_scale(trial)
            if best is None or s[1] < best[1]:
                best = (s[0], s[1], trial)
        scale, resid, pairs = best
    else:
        trial = []
        if horizontals and real_h:
            trial.append((horizontals[0][0], real_h[0]))
        if dims[2]:
            trial.append((dims[2][0], height))
        scale, resid = _fit_scale(trial)
        pairs = trial

    if not pairs:
        raise AssetError("profile real_size_m has no usable positive dimension")
    return scale, resid, pairs


def _fit_scale(pairs):
    """Geometric-mean scale minimising squared log error; also returns RMS log error."""
    import math
    ratios = [r / m for m, r in pairs if m > 0 and r > 0]
    if not ratios:
        raise AssetError("no usable dimension pair for scale fit")
    logs = [math.log(v) for v in ratios]
    scale = math.exp(sum(logs) / len(logs))
    resid = math.sqrt(sum((v - sum(logs) / len(logs)) ** 2 for v in logs) / len(logs))
    return scale, resid


def expand(obj, index=None, base_dir="."):
    """Return a plain simulator object spec for an asset reference.

    Fields already present on the input win, so a scene can still override a
    single parameter without restating the whole profile.
    """
    if "asset" not in obj:
        return dict(obj)
    if index is None:
        index = load_index(os.path.join(base_dir, DEFAULT_INDEX))
    key = obj["asset"]
    asset = _find(index, key)
    prof = _profile(asset, obj.get("profile"))

    geom = asset["geometry"]
    out = dict(obj)
    out.pop("asset", None)
    out.pop("profile", None)

    out.setdefault("model", asset["path"])
    # load_object_mesh defaults shape to "box", so omitting this renders a
    # 12-face bounding box instead of the mesh -- silently, with a plausible
    # size and a completely wrong radar return. Every asset reference must
    # declare itself a mesh.
    out.setdefault("shape", "obj")
    scale = None
    resid = None
    if "scale" not in out:
        scale, resid, _pairs = _axis_scale(geom["bbox_extent"],
                                           prof.get("real_size_m") or {},
                                           geom.get("footprint_rect"))
        out["scale"] = round(scale, 6)
    if "mat" not in out and prof.get("materials", {}).get("default"):
        out["mat"] = prof["materials"]["default"]
    if "temp_c" not in out and prof.get("thermal", {}).get("skin_c") is not None:
        out["temp_c"] = prof["thermal"]["skin_c"]
    if "thermal_regions" not in out and prof.get("thermal_regions"):
        out["thermal_regions"] = [dict(r) for r in prof["thermal_regions"]]
    if "material_map" not in out and prof.get("material_map"):
        out["material_map"] = dict(prof["material_map"])
    # Record how far the scaled proxy deviates from the documented planform.
    # A proxy that is 40% too wide produces a correspondingly wrong RCS, and the
    # audit must be able to say so instead of blaming the radar model.
    rs = prof.get("real_size_m") or {}
    scaled = [extent * float(out["scale"]) for extent in geom["bbox_extent"]]
    out["_asset"] = dict(category=asset["category"], name=asset["name"],
                         profile=prof.get("name"), confidence=prof.get("confidence"),
                         scale_anchor="least-squares(3-axis)",
                         scale_log_rms=round(resid, 4) if resid is not None else None,
                         scaled_bbox_m=[round(v, 3) for v in scaled],
                         real_size_m=rs)
    return out


def expand_scene(cfg, base_dir="."):
    """Expand every asset reference in a scene config in place-safe fashion."""
    index = None
    objs = cfg.get("objects") or []
    if not any("asset" in o for o in objs):
        return cfg
    index = load_index(os.path.join(base_dir, DEFAULT_INDEX))
    cfg = dict(cfg)
    cfg["objects"] = [expand(o, index=index, base_dir=base_dir) for o in objs]
    return cfg