"""End-to-end registration acceptance for the three rendered modalities.

Why this exists
---------------
simkit.consistency judges the *state* -- one plan.ClipPlan feeding every label.
That is necessary and not sufficient: the shipped labels come from three
independent renderers.  A scene can pass every state-level rule and still ship
boxes that do not line up with each other or with the pixels.

This script therefore renders N real frames (optical + ir + radar), reads the
label files that were written to disk, and measures the agreement in the units a
consumer would use:

  azimuth/elevation  the centre of the optical box, back-projected through the
                     camera intrinsics, versus the radar label's az/el for the
                     same object: <= one beam width per axis.
  range              the radar's measured range against the target's own
                     labelled box: the echo must come from the body, i.e. land
                     between its leading face (minus one range cell) and its
                     centre (plus one cell).  A centroid comparison cannot be
                     used: a 20 m airframe seen nose-on returns from a face ten
                     metres in front of its centre of mass.
  thermal vs optical the IR box centre versus the optical box centre of the same
                     object, same 3D projection, same camera: <= 2 px.
  identity (G1)      the three channels name exactly the same objects, and every
                     labelled object actually appears in the radar detections.

The az/el comparison is made in ONE frame -- the radar's own aperture frame, the
frame its label reports in.  The optical direction is rotated into it through
the shared world state, so the number answers "do the two sensors agree about
where the target is", not "do two up-axis conventions print different numbers".
The boresight separation is printed separately: the radar follows the camera and
re-aims at the primary target only when it has to, which is a rendering choice,
not a registration error.

Usage
-----
    python tools/check_registration.py                     # both demo scenes
    python tools/check_registration.py scenes/x.json -n 6
    python tools/check_registration.py --reuse out/register/xxx  # no render
Exit code 0 when every row is inside tolerance, 1 otherwise.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simkit import dataset, scene  # noqa: E402
from simkit.mesh import rot_x, rot_y, rot_z  # noqa: E402

SCENES = ("scenes/demo_three_ships_plate.json", "scenes/radar_tri_aircraft.json")
PIXEL_TOL = 2.0          # ir box centre vs optical box centre, same projection
RENDER_PX_TOL = 2.0      # rendered silhouette vs its projected label box


def _json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _label_dir(out_dir, kind):
    return os.path.join(out_dir, "labels", kind)


def _labels(out_dir, kind):
    """frame -> label dict, for one modality directory."""
    out = {}
    for path in sorted(glob.glob(os.path.join(_label_dir(out_dir, kind), "*.json"))):
        data = _json(path)
        out[int(data.get("frame", len(out)))] = data
    return out


def _box_center(box):
    return (0.5 * (float(box["x0"]) + float(box["x1"])),
            0.5 * (float(box["y0"]) + float(box["y1"])))


def _camera_dir(cam, px):
    """Unit world direction through one pixel of one camera dict.

    Same intrinsics labels.project_points() uses -- one focal length from the
    horizontal field of view, centre at (W/2, H/2), +y down in pixels and up in
    the camera -- because this script has to reproduce what wrote the box.
    """
    pos = np.asarray(cam["pos"], float)
    fov = float(cam["fov_deg"])
    w, h = float(cam["width"]), float(cam["height"])
    _p, right, up, fwd = scene.sensor_basis(
        pos, cam["look_at"], cam.get("up", (0.0, 0.0, 1.0)),
        float(cam.get("roll_deg", 0.0) or 0.0))
    fx = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
    x = (float(px[0]) - w / 2.0) / fx
    y = -(float(px[1]) - h / 2.0) / fx
    d = fwd + right * x + up * y
    return d / max(float(np.linalg.norm(d)), 1e-12)


def _angles(direction, frame_pos, look_at, up):
    """(az, el) of a world direction in one sensor's aperture frame, in degrees."""
    _p, right, upv, fwd = scene.sensor_basis(frame_pos, look_at, up)
    z = float(np.dot(direction, fwd))
    if z <= 1e-9:
        return None
    x = float(np.dot(direction, right))
    y = float(np.dot(direction, upv))
    return (math.degrees(math.atan2(x, z)),
            math.degrees(math.atan2(y, math.hypot(x, z))))


def _obb_entry(box, origin, direction):
    """Distance along a ray from origin to the surface of one labelled box.

    A radar measures the echo that arrives first -- the range to the surface,
    not to the centre of mass.  For a 20 m fighter seen nose-on that is ten
    metres short of the centroid, more than one range cell, so a centroid truth
    would force a tolerance three times looser than the doc's and hide a real
    mistake inside it.

    scene.build_world makes this box from the mesh extents and rotates it by
    rot_z(yaw) @ rot_y(-pitch) @ rot_x(roll); labels.radar_labels publishes the
    mean of those corners as center and hi-lo as size, so center +- R @ size/2
    reconstructs the same eight corners and the slab test is exact.
    Returns None when the ray misses the box.
    """
    c = np.asarray(box["center"], float)
    size = np.asarray(box["size"], float)
    R = np.asarray(rot_z(float(box.get("yaw") or 0.0))
                  @ rot_y(-float(box.get("pitch") or 0.0))
                  @ rot_x(float(box.get("roll") or 0.0)), float)
    d = R.T @ np.asarray(direction, float)
    o = R.T @ (np.asarray(origin, float) - c)
    half = 0.5 * np.maximum(size, 1e-6)
    t0, t1 = -np.inf, np.inf
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(o[k]) > half[k]:
                return None
            continue
        a = (-half[k] - o[k]) / d[k]
        b = (half[k] - o[k]) / d[k]
        t0 = max(t0, min(a, b))
        t1 = min(t1, max(a, b))
    if t1 < max(t0, 0.0):
        return None
    return float(t0) if t0 > 0.0 else 0.0


# radar.beam_grid builds a beam as d = f + tan(az) r + tan(el) u, so the label
# az/el already is the projection angle this script measures; the two sides can
# be compared without re-deriving a convention (elevation differs by cos(az)^-1,
# 2e-5 deg at 1 deg).


def _instance_box(npz_path, oid):
    """Bounding box and centroid of one object in the RENDERED instance pass.

    The label box projects the 3D extent through the same camera dict, so it is
    only registered with the picture while both agree about the camera
    orientation.  A rolled basis (sensor_basis() swaps in a different up axis
    within 0.1 deg of vertical, and two modalities can land on opposite sides)
    moves the drawn aircraft tens of pixels with nothing else complaining.
    Returns None when the object has no pixels in this frame.
    """
    if not os.path.exists(npz_path):
        return None
    try:
        ids = np.load(npz_path)["instance_id"]
    except (KeyError, OSError, ValueError):
        return None
    m = np.asarray(ids) == int(oid)
    n = int(m.sum())
    if n <= 0:
        return None
    ys, xs = np.nonzero(m)
    return dict(x0=float(xs.min()), y0=float(ys.min()),
                x1=float(xs.max()) + 1.0, y1=float(ys.max()) + 1.0,
                cx=float(xs.mean()), cy=float(ys.mean()), pixels=n)

def _view_of(optical_frames, frame, name):
    """Per-frame camera dict of one optical view, as dataset._view_pose made it."""
    for entry in optical_frames.get("frames") or []:
        if int(entry.get("frame", -1)) != int(frame):
            continue
        v = (entry.get("views") or {}).get(name)
        if v is None:
            return None
        return dict(pos=v["pos"], look_at=v.get("look_at", [0.0, 0.0, 0.0]),
                    fov_deg=v.get("fov_deg"), width=v.get("width"),
                    height=v.get("height"), roll_deg=v.get("roll_deg", 0.0),
                    up=(0.0, 0.0, 1.0))
    return None


def render(scene_path, out_dir, frames, quiet=True):
    """Render N frames of optical + ir + radar through the real pipeline."""
    dataset.run(scene_path, out_dir, n_frames=int(frames),
                sensors=("optical", "ir", "radar"), quiet=quiet)


def check(out_dir, scene_path=None):
    """Read one rendered dataset back and return (rows, problems)."""
    optical = _labels(out_dir, "optical")
    radar = _labels(out_dir, "radar")
    ir_bands = [os.path.basename(d) for d in
                sorted(glob.glob(os.path.join(out_dir, "labels", "ir*")))
                if os.path.isdir(d)]
    ir = {b: _labels(out_dir, b) for b in ir_bands}
    if scene_path is None:
        scene_path = (_json(os.path.join(out_dir, "meta.json")).get("config") or "")
    scene_path = str(scene_path)
    if not os.path.exists(scene_path):
        # meta.json stores the absolute path the run used, and a capture moved
        # between machines (or written from a shell whose codepage mangled the
        # directory name) keeps only a readable basename.
        alt = os.path.join(ROOT, "scenes", os.path.basename(scene_path))
        scene_path = alt if os.path.exists(alt) else ""
    cfg = scene.load_config(scene_path) if scene_path else {}
    labelled = [o.get("name") for o in (cfg.get("objects") or [])
                if isinstance(o, dict) and o.get("label", True)]
    path = os.path.join(out_dir, "optical_frames.json")
    optical_frames = _json(path) if os.path.exists(path) else {}

    rows, problems, misses = [], [], []
    missed_names = set()
    seen_names = set()
    frames = sorted(set(optical) | set(radar))
    for f in frames:
        o_lab = optical.get(f)
        r_lab = radar.get(f)
        if o_lab is None or r_lab is None:
            problems.append("frame %d: missing %s label"
                            % (f, "optical" if o_lab is None else "radar"))
            continue
        view = _view_of(optical_frames, f, o_lab.get("view", "seeker"))
        if view is None:
            problems.append("frame %d: no optical view pose in optical_frames.json" % f)
            continue
        beam = r_lab.get("beam_raster") or {}
        bw_az = float(beam.get("beamwidth_az_deg") or 1.0)
        bw_el = float(beam.get("beamwidth_el_deg") or 6.0)
        cell = float((r_lab.get("waveform") or {}).get("range_res_m") or 0.0)
        pose = r_lab.get("sensor_pose") or {}
        r_pos = np.asarray(pose.get("pos") or [0, 0, 0], float)
        r_aim = np.asarray(pose.get("look_at") or [0, 0, 1], float)
        r_up = tuple(pose.get("up") or (0.0, 0.0, 1.0))

        npz = os.path.join(out_dir, "raw", "optical", "%06d.npz" % f)
        names_3d = {b.get("name"): b for b in (r_lab.get("boxes_3d") or [])}
        rcs_of = {t.get("name"): t.get("rcs_dbsm")
                  for t in (r_lab.get("targets") or [])}
        id_of = {b.get("id"): b.get("name") for b in (r_lab.get("boxes_3d") or [])}
        points = {}
        for det in r_lab.get("points") or []:
            name = id_of.get(det.get("obj"))
            if name is not None:
                points.setdefault(name, []).append(det)

        o_boxes = o_lab.get("boxes") or {}
        # G1: the three channels have to name the same objects
        sets = {"optical": set(o_boxes), "radar": set(names_3d)}
        for b, labs in ir.items():
            if f in labs:
                sets[b] = set(labs[f].get("boxes") or {})
        if len({frozenset(v) for v in sets.values()}) > 1:
            problems.append("frame %d: object names differ: %s"
                            % (f, {k: sorted(v) for k, v in sets.items()}))
        if labelled and sorted(o_boxes) != sorted(labelled):
            problems.append("frame %d: optical labels %s != scene's labelled %s"
                            % (f, sorted(o_boxes), sorted(labelled)))

        ir_px = []
        for name in sorted(o_boxes):
            box = o_boxes[name]
            u, v = _box_center(box)
            dir_opt = _camera_dir(view, (u, v))
            az_el_opt = _angles(dir_opt, r_pos, r_aim, r_up)
            dets = points.get(name) or []
            if not dets:
                # A missed detection is a legitimate radar outcome (Swerling
                # fluctuation, or a specular facet that vanishes at this
                # aspect), so it is reported as a MISS and not scored as a
                # geometry error.  A target missed in EVERY frame is a
                # different story: the aperture or the aim is wrong, and that
                # is caught below.
                misses.append(dict(frame=f, name=name,
                                   az=az_el_opt[0] if az_el_opt else None,
                                   el=az_el_opt[1] if az_el_opt else None,
                                   rcs=rcs_of.get(name)))
                missed_names.add(name)
                continue
            az = float(np.mean([d["az"] for d in dets]))
            el = float(np.mean([d["el"] for d in dets]))
            rng = float(np.mean([d["range"] for d in dets]))
            box3d = names_3d.get(name) or {}
            c = np.asarray(box3d.get("center") or [0, 0, 0], float)
            centre_rng = float(np.linalg.norm(c - r_pos))
            # Truth: where this frame's own line of sight meets the target's
            # box.  The ray is taken to the box centre, not along the reported
            # beam: the beam angle is quantised to the raster (and parked on the
            # nearest one when the target sits outside it), so it can be a tenth
            # of a degree off -- 7 m at 4 km, enough to step over a 5.8 m box.
            # The centroid is printed beside it because the gap between the two
            # is the half-depth along the beam, which is what makes a one-cell
            # tolerance mean something: the echo comes from the surface.
            to_centre = c - r_pos
            entry = _obb_entry(box3d, r_pos,
                               to_centre / max(float(np.linalg.norm(to_centre)), 1e-9))
            true_rng = float(entry) if entry is not None else centre_rng
            span = max(0.0, centre_rng - true_rng)
            d_az = az_el_opt[0] - az if az_el_opt else float("nan")
            d_el = az_el_opt[1] - el if az_el_opt else float("nan")
            d_rng = rng - true_rng
            # The measurement has to land ON the body: no nearer than the
            # leading face (minus a cell) and no farther than the centre (plus
            # a cell), because the echo comes from the surface, not the centre.
            ok_rng = -max(cell, 1e-6) <= d_rng <= span + max(cell, 1e-6)
            ok = abs(d_az) <= bw_az and abs(d_el) <= bw_el and ok_rng
            seen_names.add(name)
            rows.append(dict(frame=f, name=name, opt_az=az_el_opt[0] if az_el_opt else None,
                             opt_el=az_el_opt[1] if az_el_opt else None,
                             r_az=az, r_el=el, d_az=d_az, d_el=d_el,
                             bw_az=bw_az, bw_el=bw_el, r_range=rng,
                             c_range=centre_rng, true_range=true_rng,
                             span=span, d_range=d_rng, ok_range=bool(ok_rng),
                             cell=cell, dets=len(dets), ok=bool(ok)))
            if not ok:
                problems.append(
                    "frame %d %s: d_az %+.3f (limit %.2f) d_el %+.3f (limit "
                    "%.2f) d_range %+.2f m outside [%.2f, %.2f]"
                    % (f, name, d_az, bw_az, d_el, bw_el, d_rng,
                       -cell, span + cell))
            # what the optical pass actually DREW for this object, versus the
            # box the label projects: mask outside the box means the two disagree
            # about the camera, not about the object
            _name_of = {int(k): v for k, v in id_of.items() if k is not None}
            oid = next((i for i, n in _name_of.items() if n == name), None)
            mask = (_instance_box(npz, oid) if oid is not None else None)
            if mask is not None:
                # positive = the drawn pixels reach outside the label box,
                # in both senses: silhouette bbox and silhouette centroid
                edges = [(float(box["x0"]), mask["x0"]), (float(box["y0"]), mask["y0"]),
                         (mask["x1"], float(box["x1"])), (mask["y1"], float(box["y1"]))]
                centre = [(float(box["x0"]), mask["cx"]), (mask["cx"], float(box["x1"])),
                          (float(box["y0"]), mask["cy"]), (mask["cy"], float(box["y1"]))]
                out_px = max([x - y for x, y in edges] +
                             [x - y for x, y in centre])
                for row in rows:
                    if row["frame"] == f and row["name"] == name:
                        row["mask_out_px"] = out_px
                if out_px > RENDER_PX_TOL:
                    problems.append("frame %d %s: the rendered silhouette sits "
                                    "%.1f px outside its label box (limit %.0f)"
                                    % (f, name, out_px, RENDER_PX_TOL))
            for band, labs in ir.items():
                lab = labs.get(f) or {}
                ibox = (lab.get("boxes") or {}).get(name)
                if ibox is None:
                    continue
                iu, iv = _box_center(ibox)
                off = math.hypot(iu - u, iv - v)
                ir_px.append(off)
                if off > PIXEL_TOL:
                    problems.append("frame %d %s: %s box centre %.2f px from the "
                                    "optical box (limit %.1f)"
                                    % (f, name, band, off, PIXEL_TOL))
        # boresight separation: a pose difference, printed, not scored
        sep = _angles(_camera_dir(view, (0.5 * float(o_lab["width"]),
                                         0.5 * float(o_lab["height"]))),
                      r_pos, r_aim, r_up)
        for row in rows:
            if row["frame"] == f:
                row["boresight_deg"] = math.hypot(sep[0], sep[1]) if sep else None
                if ir_px:
                    row["ir_px_max"] = max(ir_px)
    # A target the radar never once detected is not a fluctuation: either the
    # aperture does not cover it or the aim is wrong, and reporting the other
    # rows as "green" would hide it.
    for name in sorted(missed_names - seen_names):
        problems.append("%s is labelled in the optical channel but the radar "
                        "never detected it in %d frame(s)" % (name, len(frames)))
    return rows, problems, sorted(ir), misses


def print_rows(rows, ir_bands):
    head = ("%-5s %-12s %8s %8s %8s %8s %7s %7s %5s %5s %9s %8s %6s %7s %4s "
            "%6s %6s"
            % ("frame", "target", "opt_az", "opt_el", "rd_az", "rd_el",
               "d_az", "d_el", "bw_az", "bw_el", "rd_range", "edge_rng",
               "depth", "d_edge", "det", "ir_px", "mask"))
    print(head)
    print("-" * len(head))
    for r in rows:
        print("%-5d %-12s %8.3f %8.3f %8.3f %8.3f %7.3f %7.3f %5.2f %5.2f "
              "%9.2f %8.2f %6.2f %+7.2f %4d %6s %6s"
              % (r["frame"], r["name"], r["opt_az"], r["opt_el"], r["r_az"],
                 r["r_el"], r["d_az"], r["d_el"], r["bw_az"], r["bw_el"],
                 r["r_range"], r["true_range"], r["span"], r["d_range"],
                 r["dets"],
                 "-" if r.get("ir_px_max") is None else "%.2f" % r["ir_px_max"],
                 "-" if r.get("mask_out_px") is None else "%+.2f" % r["mask_out_px"]))
    print("tolerance: |d_az| <= one az beam, |d_el| <= one el beam; "
          "range echo within [leading face - 1 cell, box centre + 1 cell]; "
          "ir/optical <= %.0f px (bands: %s)"
          % (PIXEL_TOL, ",".join(ir_bands) or "none"))
    print("mask = px the RENDERED silhouette sticks out of its label box; "
          "negative means it is inside.")
    print("edge_rng is where this beam meets the labelled box (the leading "
          "face a radar returns first), depth is how far the body extends "
          "behind it along the beam, d_edge is measured minus edge_rng.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="three-modality registration acceptance")
    ap.add_argument("scenes", nargs="*", default=list(SCENES))
    ap.add_argument("-n", "--frames", type=int, default=4)
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "out", "register"))
    ap.add_argument("--reuse", default=None,
                    help="score an existing capture instead of rendering")
    args = ap.parse_args(argv)

    jobs = []
    if args.reuse:
        jobs.append((args.reuse, None))
    else:
        for rel in (args.scenes or SCENES):
            path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            stem = os.path.splitext(os.path.basename(path))[0]
            out_dir = os.path.join(args.out, "%s_%s_%df" % (stem, stamp, args.frames))
            os.makedirs(out_dir, exist_ok=True)
            print("rendering %d frame(s) of optical+ir+radar: %s -> %s"
                  % (args.frames, rel, out_dir))
            t0 = time.time()
            render(path, out_dir, args.frames)
            print("  rendered in %.1f s" % (time.time() - t0))
            jobs.append((out_dir, path))

    bad = 0
    for out_dir, path in jobs:
        print("")
        print("=" * 96)
        print("capture %s" % out_dir)
        rows, problems, ir_bands, misses = check(out_dir, path)
        if not rows:
            print("no registration rows: %d problem(s)" % len(problems))
            for p in problems[:10]:
                print("  ! %s" % p)
            bad += 1
            continue
        print_rows(rows, ir_bands)
        worst_az = max(abs(r["d_az"]) for r in rows)
        worst_el = max(abs(r["d_el"]) for r in rows)
        worst_px = max([r.get("ir_px_max") or 0.0 for r in rows])
        cell = rows[0]["cell"]
        # how far each echo falls outside [edge - cell, centre + cell]
        worst_r = max(max(0.0, -r["d_range"] - cell,
                          r["d_range"] - r["span"] - cell) for r in rows)
        print("worst |d_az| %.3f deg (beam %.2f) | |d_el| %.3f (beam %.2f) | "
              "range excursion %.2f m (cell %.2f, window [edge-%.2f, centre+%.2f])"
              " | ir/opt %.2f px" % (worst_az, rows[0]["bw_az"], worst_el,
                                     rows[0]["bw_el"], worst_r, cell, cell, cell,
                                     worst_px))
        sep = [r.get("boresight_deg") for r in rows if r.get("boresight_deg")]
        if sep:
            print("boresight separation optical vs radar axis: %.2f..%.2f deg "
                  "(the re-aim pose difference, not a registration error)"
                  % (min(sep), max(sep)))
        if misses:
            print("%d frame(s) with no radar echo (legitimate miss, not scored):"
                  % len(misses))
            for m in misses[:10]:
                print("  - frame %d %s: optical box az %.2f el %.2f deg, rcs %s"
                      % (m["frame"], m["name"], m["az"], m["el"], m["rcs"]))
        if problems:
            bad += 1
            print("%d registration problem(s):" % len(problems))
            for p in problems[:20]:
                print("  ! %s" % p)
            if len(problems) > 20:
                print("  ... %d more" % (len(problems) - 20))
        else:
            print("OK: %d row(s) inside tolerance, names identical across "
                  "optical/ir/radar" % len(rows))
    print("")
    print("%d capture(s), %d with problems" % (len(jobs), bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
