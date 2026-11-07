"""Redraw RD / RA scope figures from an already generated dataset.

Why this exists: a paper figure or a dataset preview must be regenerated
without paying for the simulation again. raw/radar/NNNNNN.npz keeps the
range-Doppler power and both axes, so the RD figure is a pure redraw. The
range-angle cube is NOT stored (raw only keeps the RD map), so the RA figure is
rebuilt from the label detections on the scene's own azimuth raster with an
exponential noise floor scaled to the RD noise median; the README for the
figure says so in its title line and the run note below.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simkit import radar_scope  # noqa: E402
from simkit import weather  # noqa: E402

DEFAULT_RUN = os.path.join(ROOT, "out", "studio_runs", "preview_20261106_205649_recipe")


def _json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _radar_cfg(scene):
    return dict((scene.get("sensors") or {}).get("radar") or {})


def _targets(label):
    """One row per labelled object, keeping its strongest detection."""
    names = {b.get("id"): b.get("name") for b in label.get("boxes_3d") or []}
    best = {}
    for p in label.get("points") or []:
        oid = int(p.get("obj", -1))
        if oid < 0:
            continue                      # clutter or ECM, not a target row
        snr = float(p.get("snr_db", 0.0))
        if oid not in best or snr > float(best[oid].get("snr_db", 0.0)):
            best[oid] = dict(name=names.get(oid) or "obj%d" % oid,
                             range_m=p.get("range"), v_r_mps=p.get("v_r"),
                             az_deg=p.get("az"), snr_db=snr)
    return [best[k] for k in sorted(best)]


def _meta(scene, run, frame, radar):
    band = radar.get("band")
    fc = radar.get("fc_ghz")
    if fc is None and band in weather.RADAR_BANDS:
        fc = weather.RADAR_BANDS[band]["fc_ghz"]
    return dict(band=band, fc_ghz=fc, range_res_m=radar.get("range_res_m"),
                frame=int(frame), scene=os.path.basename(os.path.abspath(run)))


def _vel_res(axis):
    a = np.asarray(axis, float)
    return float(abs(a[1] - a[0])) if a.size > 1 else None


def _range_angle(power, ranges, label, radar, seed):
    """Rebuild a range-angle map from detections at the scene's azimuth raster.

    The RD map is the only map the dataset stores, so the RA picture is a
    reconstruction: an exponential noise floor whose median equals the RD
    noise median, plus one compact Gaussian per detection (sigma = one display
    cell in range, 0.6 of the scene's azimuth step in azimuth). Anything
    claiming finer angular detail than that would be invented.
    """
    n_az = max(int(radar.get("n_az") or 40), 2)
    span = float(radar.get("az_span_deg") or 2.0)
    n_disp = max(n_az, 160)               # display raster, not a new measurement
    angles = np.linspace(-span / 2.0, span / 2.0, n_disp)
    noise_med = max(float(np.median(power)), 1e-30)
    rng = np.random.default_rng((int(seed) * 7919 + 11) % (2 ** 63))
    floor = rng.exponential(noise_med / np.log(2.0), size=(n_disp, ranges.size))
    dr = float(ranges[1] - ranges[0]) if ranges.size > 1 else 1.0
    sig_az = 0.6 * span / n_az
    for t in _targets(label):
        if None in (t["range_m"], t["az_deg"], t["snr_db"]):
            continue
        amp = noise_med * 10.0 ** (float(t["snr_db"]) / 10.0)
        g_az = np.exp(-0.5 * ((angles - float(t["az_deg"])) / sig_az) ** 2)
        g_rg = np.exp(-0.5 * ((ranges - float(t["range_m"])) / (1.2 * dr)) ** 2)
        floor += amp * g_az[:, None] * g_rg[None, :]
    return floor, angles


def _write(rgb, path):
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), "RGB").save(path, optimize=True)
    w, h = Image.open(path).size
    return "%s  (%d×%d, %.0f KB)" % (path, w, h, os.path.getsize(path) / 1024.0)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="从已生成的 simkit 数据集重画雷达 RD / RA 示波器成图")
    ap.add_argument("--run", default=DEFAULT_RUN, help="数据集目录")
    ap.add_argument("--frame", type=int, default=0, help="帧号（默认 0）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <run>/figures/radar_scope）")
    ap.add_argument("--dyn-rd", type=float, default=30.0, help="RD 显示窗宽 dB")
    ap.add_argument("--dyn-ra", type=float, default=45.0, help="RA 显示窗宽 dB")
    ap.add_argument("--cmap", default="turbo", choices=("turbo", "inferno"))
    args = ap.parse_args(argv)

    run = os.path.abspath(args.run)
    npz_path = os.path.join(run, "raw", "radar", "%06d.npz" % args.frame)
    if not os.path.isfile(npz_path):
        print("[错误] 找不到 raw 雷达帧：%s（该数据集没有保存 research_baseline 原始帧）"
              % npz_path, file=sys.stderr)
        return 2
    out_dir = args.out or os.path.join(run, "figures", "radar_scope")
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npz_path)
    power = np.asarray(data["power"], float).T          # (n_range, n_doppler)
    ranges = np.asarray(data["range_m"], float)
    doppler = np.asarray(data["closing_speed_mps"], float)
    scene = _json(os.path.join(run, "scene.json"), {}) or {}
    radar = _radar_cfg(scene)
    label = _json(os.path.join(run, "labels", "radar", "%06d.json" % args.frame), {}) or {}
    targets = _targets(label)
    meta = _meta(scene, run, args.frame, radar)
    meta["vel_res_mps"] = _vel_res(doppler)

    rd = radar_scope.render_rd(power, ranges, doppler, targets=targets, meta=meta,
                               dyn_db=args.dyn_rd, cmap=args.cmap)
    rd_path = os.path.join(out_dir, "rd_%06d.png" % args.frame)
    print("RD 图：" + _write(rd["rgb"], rd_path))

    ra_power, angles = _range_angle(power, ranges, label, radar, args.frame)
    ra_meta = dict(meta)
    ra_meta["az_res_deg"] = float(radar.get("az_span_deg") or 2.0) / \
        max(int(radar.get("n_az") or 40), 2)
    ra = radar_scope.render_ra(ra_power, ranges, angles, targets=targets,
                               meta=ra_meta, dyn_db=args.dyn_ra, cmap=args.cmap)
    ra_path = os.path.join(out_dir, "ra_%06d.png" % args.frame)
    print("RA 图：" + _write(ra["rgb"], ra_path))

    print("目标 %d 个：%s" % (len(targets), "、".join(
        "%s %.0f m / %.1f m/s / %.1f dB" % (t["name"], float(t["range_m"]),
                                              float(t["v_r_mps"]), float(t["snr_db"]))
        for t in targets)))
    print("RD 显示窗 %.1f…%.1f dB，目标表%s" %
          (rd["db_window"][0], rd["db_window"][1],
           "压在热图右上角（未遮目标）" if rd["panel_mode"] == "inset" else "放在右侧预留区"))
    print("注：RA 图由检测点按场景方位栅格重建（原始 RA 立方体未入库），"
          "滤波器噪声按 RD 噪声中值标定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
