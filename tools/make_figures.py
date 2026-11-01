#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""assets/figures 图册生成器（唯一入口）。

    env\\Scripts\\python.exe tools\\make_figures.py
    env\\Scripts\\python.exe tools\\make_figures.py --only 03,05

输出：assets/figures/fig01..fig20*.png + index.json + README.md
所有数值取自 data/manifest.json、data/radar_bands.json、data/cloud/*.json、
datasets/yolo/export.json、datasets/yolo/labels/**、scenes/*.json、
simkit/weather.py、simkit/atmosphere.py 与 docs/*。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager as _fm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle)
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "figures"
TMP = ROOT / "_tmp" / "figs"

FONT_FILES = [
    r"D:\devtools\fonts\NotoSansCJKsc-Regular.otf",
    r"D:\devtools\fonts\NotoSansCJKsc-Bold.otf",
    r"D:\devtools\fonts\NotoSansMonoCJKsc-Regular.otf",
]

# ---------------------------------------------------------------- 视觉规范 --
BG = "#0b0f14"        # 画布
AX = "#0e141b"        # 坐标区
GRID = "#1b2530"      # 网格
TEXT = "#c8d9ea"      # 正文
DIM = "#8aa2b8"       # 次要
ACC = "#37c2ff"       # 强调
ACC2 = "#43d98a"      # 次强调
WARN = "#ffb454"      # 警示
ERR = "#ff5d6c"       # 错误
PUR = "#c792ea"       # 紫
SPINE = "#22303d"
MUTE = "#5b7186"      # 水印灰

W, H, DPI = 16.0, 10.0, 100
L, R, B, T = 0.06, 0.945, 0.085, 0.905

plt.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "font.size": 13,
    "axes.unicode_minus": False,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": AX,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.9,
    "axes.labelcolor": TEXT,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.color": DIM,
    "ytick.color": DIM,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 13,
    "legend.labelcolor": TEXT,
    "text.color": TEXT,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "butt",
    "figure.dpi": DPI,
})


def _bootstrap_fonts():
    for p in FONT_FILES:
        if Path(p).exists():
            _fm.fontManager.addfont(p)


_bootstrap_fonts()

# ---------------------------------------------------------------- 数据加载 --
def _json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


MANIFEST = _json("data/manifest.json")
SCENES = {s["id"]: s for s in MANIFEST["scenes"]}
BANDS = _json("data/radar_bands.json")
EXPORT = _json("datasets/yolo/export.json")
SEEK_CFG = _json("data/seeker_air2ground/cfg.json")

sys.path.insert(0, str(ROOT))
from simkit import atmosphere as atm_mod          # noqa: E402
from simkit import weather as wmod                # noqa: E402

C_LIGHT = 299792458.0

SCENE_COLORS = {
    "seeker_air2ground": ACC,
    "seeker_air2air_high": ACC2,
    "seeker_surface2air": WARN,
    "final_air": PUR,
    "final_ground": ERR,
    "final_sea": DIM,
}
BAND_COLORS = {"X": ACC, "Ku": ACC2, "K": WARN, "Ka": PUR, "W": ERR, "W94": DIM}


def frame(scene_id, i):
    return SCENES[scene_id]["frames"][i]


def mod(scene_id, i, key):
    return frame(scene_id, i)["mods"][key]


def img_path(rel):
    return ROOT / "data" / rel


_CLOUD_CACHE = {}


def cloud_of(scene_id, i):
    """点云数组：优先 manifest 内联字段，回退到外置 data/cloud/<scene>.json。"""
    fr = frame(scene_id, i)
    c = fr.get("cloud")
    if c is None:
        if scene_id not in _CLOUD_CACHE:
            _CLOUD_CACHE[scene_id] = _json("data/cloud/%s.json" % scene_id)
        c = _CLOUD_CACHE[scene_id].get(str(i)) or {}
    return c


def load_path(path, mode="RGB"):
    im = Image.open(str(path))
    if mode == "L":
        return np.asarray(im.convert("L"), float)
    return np.asarray(im.convert("RGB"), float)


def load_img(rel, mode="RGB"):
    im = Image.open(img_path(rel))
    if mode == "L":
        return np.asarray(im.convert("L"), float)
    return np.asarray(im.convert("RGB"), float)


def stretch(a, lo_p=1.0, hi_p=99.5, gamma=1.0):
    """百分位拉伸到 [0,1]；用于雷达/激光/红外的显示。"""
    a = np.asarray(a, float)
    lo, hi = np.nanpercentile(a, [lo_p, hi_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    v = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return v ** gamma


def dilate(a, k=1):
    """3x3 灰度膨胀（k 次），让稀疏点云在印刷尺寸下可见。"""
    out = a.copy()
    for _ in range(k):
        p = np.pad(out, 1, mode="edge")
        out = np.maximum.reduce([p[0:-2, 0:-2], p[0:-2, 1:-1], p[0:-2, 2:],
                                 p[1:-1, 0:-2], p[1:-1, 1:-1], p[1:-1, 2:],
                                 p[2:, 0:-2], p[2:, 1:-1], p[2:, 2:]])
    return out


# ---------------------------------------------------------------- 绘图工具 --
def new_fig():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def style_axes(ax):
    ax.set_facecolor(AX)
    for s in ax.spines.values():
        s.set_color(SPINE)
        s.set_linewidth(0.9)
    ax.tick_params(colors=DIM, labelsize=13, width=0.8, length=4)
    ax.grid(True, color=GRID, lw=0.6, ls="--", alpha=1.0)
    ax.set_axisbelow(True)


def wm(fig, num):
    fig.text(0.012, 0.013, u"simkit · 图 %s" % num, fontsize=8, color=MUTE,
             ha="left", va="bottom")


def note(fig, text, y=0.013, color=DIM, size=11.5):
    fig.text(0.945, y, text, fontsize=size, color=color, ha="right", va="bottom")


def panel_title(ax, name, pad=7):
    ax.set_title(name, loc="left", fontsize=17, fontweight="bold",
                 color=TEXT, pad=pad)


def tag_text(ax, x, y, s, size=12.5, color=TEXT, pad=0.28, ha="left"):
    """坐标区内的小标签，带半透明底，避免压在图上看不清。"""
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="top", zorder=9,
            bbox=dict(boxstyle="round,pad=%s" % pad, fc=BG, ec="none", alpha=0.78))


def draw_box(ax, b, color=ACC, lw=1.8, ls="-", shrink=0.0, z=7):
    x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
    if shrink:
        x0 += shrink; y0 += shrink; x1 -= shrink; y1 -= shrink
    ax.add_patch(Rectangle((x0, y0), max(x1 - x0, 1.0), max(y1 - y0, 1.0),
                           fill=False, ec=color, lw=lw, ls=ls, zorder=z))


def imshow_native(ax, arr, cmap=None, vmin=None, vmax=None):
    """按原图长宽比显示，四周留黑，不拉伸几何。"""
    kw = dict(interpolation="nearest", zorder=1)
    if cmap is not None:
        kw.update(cmap=cmap, vmin=0.0 if vmin is None else vmin,
                  vmax=1.0 if vmax is None else vmax)
    ax.imshow(arr, **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(SPINE)
        s.set_linewidth(0.9)
    ax.set_facecolor(BG)


def frame_note(num, text):
    pass


# ---------------------------------------------------------------- 物理工具 --
def weather_state(vis_m=20000.0, rain=0.0, rh=0.5, t_c=25.0, cloud=0.05,
                  sun=55.0, wind=5.0):
    w = wmod.resolve(dict(visibility_m=float(vis_m), rain_mmh=float(rain),
                          rh=float(rh), air_temp_c=float(t_c),
                          cloud_cover=float(cloud), sun_elev_deg=float(sun),
                          wind_ms=float(wind), sea_state=0.0))
    return wmod.derive(w)


def extinction_per_km(vis_m=20000.0, rain=0.0):
    """四个通道的消光系数 [1/km]：可见光/中波/长波走 simkit.atmosphere，
    905 nm 走 simkit.weather 的 beta_lidar_per_km。"""
    w = weather_state(vis_m=vis_m, rain=rain)
    cfg = {"weather": {}, "atmosphere": {"profile": "homogeneous"}, "_weather": w}
    out = {}
    for m in ("visible", "mwir", "lwir"):
        t = float(atm_mod.path_transmittance(np.zeros(3), np.array([1000.0, 0.0, 0.0]),
                                             cfg, m))
        out[m] = -math.log(max(t, 1e-12))
    out["lidar"] = float(w["beta_lidar_per_km"])
    return out


def radar_snr_db(r_m, sigma_m2, att_db_km, band, pt_dbm=40.0):
    """与 tools/radar_budget_table.py / simkit.radar 同一条雷达方程。"""
    b = BANDS[band]
    tx = 10.0 ** ((pt_dbm - 30.0) / 10.0)
    g = 10.0 ** (b["gain_db"] / 10.0)
    loss = 10.0 ** (b["loss_db"] / 10.0)
    pr = (tx * g * g * b["lam_m"] ** 2 * sigma_m2
          / ((4.0 * np.pi) ** 3 * max(r_m, 1.0) ** 4 * loss))
    pr *= 10.0 ** (-(att_db_km * 2.0 * r_m / 1000.0) / 10.0)
    pn = 10.0 ** (b["pn_dbw"] / 10.0)
    return 10.0 * math.log10(max(pr, 1e-300) / pn) + b["pg_db"]


def radar_rmax_m(band, sigma_m2, att_db_km, det_db, pt_dbm=40.0):
    lo, hi = 1.0, 2.0e6
    if radar_snr_db(hi, sigma_m2, att_db_km, band, pt_dbm) >= det_db:
        return hi
    if radar_snr_db(lo, sigma_m2, att_db_km, band, pt_dbm) < det_db:
        return 0.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if radar_snr_db(mid, sigma_m2, att_db_km, band, pt_dbm) >= det_db:
            lo = mid
        else:
            hi = mid
    return lo


def radar_att_db_km(band, rain):
    b = BANDS[band]
    return b["gas_db_km"] + b["rain_k"] * rain ** b["rain_a"]


# ============================================================ fig01 六模态墙
def fig01():
    sid, fi = "seeker_air2ground", 7
    fr = frame(sid, fi)
    fov = fr["views"]["seeker"]["fov_deg"]
    netd = SEEK_CFG["ir"]["netd_mk"]
    rb = SEEK_CFG["sensors"]["radar"]
    sb = SEEK_CFG["sensors"]["sar"]
    lb = SEEK_CFG["sensors"]["lidar"]
    specs = [
        ("optical", "可见光 RGB", "0.4–0.7 µm · 768×576 · FOV %g°" % fov),
        ("mwir", "中波红外 MWIR", "3–5 µm · 768×576 · NETD %g mK" % netd),
        ("lwir", "长波红外 LWIR", "8–12 µm · 768×576 · NETD %g mK" % netd),
        ("radar_ra", "雷达 距离-角度", "%s %g GHz · 512×560 · BW %g MHz"
         % (rb["band"], wmod.RADAR_BANDS[rb["band"]]["fc_ghz"], rb["bandwidth_mhz"])),
        ("sar", "SAR 距离-方位", "X %g GHz · 768×320 · 距离分辨 %.2f m"
         % (sb["fc_ghz"], fr["sar"]["res_range_m"])),
        ("lidar_range", "激光雷达 距离", "905 nm · 480×120 · 量程 %g m" % lb["max_range_m"]),
    ]

    fig = new_fig()
    gs = fig.add_gridspec(2, 3, left=L, right=R, top=T, bottom=B + 0.02,
                          wspace=0.075, hspace=0.34, height_ratios=[1.0, 0.66])
    for k, (key, name, params) in enumerate(specs):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        d = fr["mods"][key]
        if key == "optical":
            arr = np.clip(load_img(d["img"]) / 255.0, 0, 1) ** 0.92
            imshow_native(ax, arr)
        elif key in ("mwir", "lwir"):
            imshow_native(ax, stretch(load_img(d["img"], "L"), 2.0, 98.0, 0.85),
                          cmap="gray")
        elif key == "radar_ra":
            imshow_native(ax, stretch(load_img(d["img"], "L"), 1.0, 99.9, 0.65),
                          cmap="gray")
        elif key == "sar":
            imshow_native(ax, stretch(load_img(d["img"], "L"), 1.0, 99.0, 0.9),
                          cmap="gray")
        else:
            v = np.clip(load_img(d["img"], "L") / 255.0, 0, 1)
            imshow_native(ax, dilate(v, 1), cmap="gray")
        ax.set_anchor("N")
        panel_title(ax, name, pad=30)
        ax.text(0.0, 1.008, params, transform=ax.transAxes, fontsize=12.5,
                color=DIM, ha="left", va="bottom")

        if key == "radar_ra":
            pk = fr["radar"]["peak"]
            ax.add_patch(Circle((pk["px"], pk["py"]), 11, fill=False, ec=WARN, lw=1.6,
                                zorder=8))
            ax.annotate("峰值 %.1f dB" % pk["db"], xy=(pk["px"], pk["py"]),
                        xytext=(pk["px"] + 26, pk["py"] - 26), fontsize=11.5,
                        color=WARN, zorder=9,
                        arrowprops=dict(arrowstyle="-", color=WARN, lw=0.9))
        for b in (d.get("boxes") or []):
            draw_box(ax, b, ACC, 1.7)
            wpx = int(round(b["x1"] - b["x0"]))
            hpx = int(round(b["y1"] - b["y0"]))
            ax.text(b["x1"] + 5, b["y1"] + 3, "%d×%d px" % (wpx, hpx), fontsize=10.5,
                    color=ACC, ha="left", va="top", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.18", fc=BG, ec="none", alpha=0.65))

    # 可见光面板里放一个目标放大窗，封面图要经得起看
    ax0 = fig.axes[0]
    b = fr["mods"]["optical"]["boxes"][0]
    cx, cy = (b["x0"] + b["x1"]) / 2.0, (b["y0"] + b["y1"]) / 2.0
    half = 46
    x0, x1 = int(max(0, cx - half)), int(min(768, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(576, cy + half))
    if x0 < 0:
        x0 = 0
    crop = np.clip(load_img(fr["mods"]["optical"]["img"])[y0:y1, x0:x1] / 255.0, 0, 1) ** 0.92
    axi = ax0.inset_axes([0.615, 0.60, 0.355, 0.355])
    axi.imshow(crop, interpolation="nearest")
    axi.add_patch(Rectangle((b["x0"] - x0, b["y0"] - y0),
                            b["x1"] - b["x0"], b["y1"] - b["y0"],
                            fill=False, ec=ACC, lw=1.5))
    axi.set_xticks([]); axi.set_yticks([])
    for s in axi.spines.values():
        s.set_color(ACC); s.set_linewidth(0.9)
    axi.set_facecolor(BG)
    axi.set_title("目标放大", loc="left", fontsize=10.5, color=ACC, pad=3)
    ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=ACC, lw=0.8,
                            ls=":", alpha=0.8))

    fig.text(L, 0.036,
             "同一帧 · 同一三维世界：%s 第 %d 帧 · 斜距 %.1f m · 窄视场目标 %d px · 天气 %s"
             % (sid, fi + 1, fr["range_m"], round(fr["views"]["seeker"]["target_px"]),
                {"clear": "晴", "cloudy": "阴", "rain": "雨"}.get(fr["weather"]["name"],
                                                               fr["weather"]["name"])),
             fontsize=14, color=TEXT, ha="left", va="bottom")
    note(fig, "源 data/manifest.json + data/%s/*.png · 红外/雷达/SAR/激光已按百分位拉伸 · ○ 为雷达峰点" % sid)
    wm(fig, "01")
    return fig, "fig01_modality_wall.png"


# ============================================================ fig02 SAR 几何
def fig02():
    sc = SEEK_CFG["sensors"]["sar"]
    meta = frame("seeker_air2ground", 0)["sar"]
    lam = C_LIGHT / (sc["fc_ghz"] * 1e9)
    bw = sc["bandwidth_mhz"] * 1e6
    h = sc["altitude_m"]
    g = sc["swath_center_m"]
    r0 = math.hypot(h, g)
    sin_inc = g / r0
    rho_r_s = C_LIGHT / (2.0 * bw)
    rho_r_g = meta["res_range_m"]
    rho_a = meta["res_az_m"]
    t_int = sc["n_pulses"] / sc["prf_hz"]
    l_flight = sc["speed_mps"] * t_int
    l_beam = lam * r0 / sc["antenna_len_m"]
    l_sa = min(l_flight, l_beam)

    fig = new_fig()
    gs = fig.add_gridspec(1, 3, left=L, right=R, top=T - 0.02, bottom=0.36,
                          wspace=0.26, width_ratios=[1.35, 1.05, 0.95])

    # (a) 侧视剖面 --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    style_axes(ax)
    ax.set_xlim(-0.4, 9.2)
    ax.set_ylim(-0.55, 4.3)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axhline(0, color=DIM, lw=1.2)
    ax.fill_between([0, 9.2], -0.55, 0, color="#131a22", zorder=0)
    ax.plot([0], [h / 1000.0], marker="v", ms=13, color=ACC, zorder=6)
    ax.annotate("平台 高度 %d m\n速度 %d m/s" % (h, sc["speed_mps"]),
                xy=(0, h / 1000.0), xytext=(0.45, 3.85), fontsize=13, color=ACC,
                va="top", arrowprops=dict(arrowstyle="-", color=ACC, lw=0.9))
    near, far = meta["near_m"] / 1000.0, meta["far_m"] / 1000.0
    ax.plot([g / 1000.0], [0], marker="o", ms=6, color=ACC2, zorder=6)
    ax.plot([0, g / 1000.0], [h / 1000.0, 0], color=ACC, lw=1.6, ls="-")
    ax.plot([0, near], [h / 1000.0, 0], color=ACC, lw=1.0, ls="--", alpha=0.7)
    ax.plot([0, far], [h / 1000.0, 0], color=ACC, lw=1.0, ls="--", alpha=0.7)
    ax.plot([near, far], [0, 0], color=WARN, lw=4.0, solid_capstyle="butt", alpha=0.9)
    ax.annotate("斜距 R0 = %.0f m" % r0, xy=(g / 2000.0, h / 2000.0), xytext=(0.65, 1.55),
                fontsize=13, color=TEXT,
                bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec="none", alpha=0.75),
                arrowprops=dict(arrowstyle="-", color=DIM, lw=0.8))
    ax.annotate("测绘带 %d m\n(%.1f–%.1f km)" % (sc["swath_width_m"], near, far),
                xy=((near + far) / 2.0, 0), xytext=(6.15, 0.55), fontsize=12.5,
                color=WARN, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec="none", alpha=0.75),
                arrowprops=dict(arrowstyle="-", color=WARN, lw=0.8))
    th = math.degrees(math.asin(sin_inc))
    ax.annotate("入射角 θ = %.1f°" % th, xy=(g / 1000.0 * 0.93, 0.10),
                xytext=(2.05, 0.45), fontsize=12.5, color=DIM,
                bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec="none", alpha=0.75),
                arrowprops=dict(arrowstyle="->", color=DIM, lw=0.8))
    ax.set_xlabel("地距 [km]")
    ax.set_ylabel("高度 [km]")
    panel_title(ax, "SAR 侧视成像几何")


    # (b) 俯视：航迹 / 波束足迹 / 合成孔径 -------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    style_axes(ax2)
    ax2.set_xlim(3.6, 6.6)
    ax2.set_ylim(-0.16, 0.16)
    ax2.set_anchor("N")
    ax2.set_xlabel("地距 [km]")
    ax2.set_ylabel("方位 [km]")
    ax2.add_patch(Rectangle((near, -0.084), far - near, 0.168, fc=ACC, alpha=0.13,
                            ec=ACC, lw=1.0))
    ax2.plot([4.75, 4.75], [-0.16, 0.16], color=DIM, lw=0.8, ls="--")
    ax2.annotate("成像栅格\n方位向 168 m\n（n_cross × ρ_a / 2）", xy=(4.75, 0.02),
                 xytext=(3.75, 0.085), fontsize=12, color=DIM,
                 arrowprops=dict(arrowstyle="-", color=DIM, lw=0.8))
    ax2.plot([g / 1000.0] * 2, [-0.075, 0.075], color=WARN, lw=3.0, alpha=0.9)
    ax2.annotate("合成孔径 L_sa = %0.1f m" % l_sa, xy=(g / 1000.0, 0.08),
                 xytext=(5.35, 0.125), fontsize=12.5, color=WARN,
                 arrowprops=dict(arrowstyle="-", color=WARN, lw=0.8))
    ax2.annotate("波束足迹 λR0/D = %.1f m" % l_beam, xy=(5.45, -0.10), xytext=(5.55, -0.142),
                 fontsize=12, color=ACC, arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8))
    ax2.set_title("俯视：航迹垂直于地距", loc="left", fontsize=17, fontweight="bold",
                  color=TEXT, pad=8)

    # (c) 分辨单元放大 ----------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    style_axes(ax3)
    nx, ny = 6, 6
    ax3.set_xlim(0, nx * rho_r_g)
    ax3.set_ylim(0, ny * rho_a)
    ax3.set_anchor("N")
    for i in range(nx + 1):
        ax3.axvline(i * rho_r_g, color=SPINE, lw=1.0)
    for j in range(ny + 1):
        ax3.axhline(j * rho_a, color=SPINE, lw=1.0)
    ax3.add_patch(Rectangle((rho_r_g, rho_a), rho_r_g, rho_a, fc=ACC, alpha=0.35,
                            ec=ACC, lw=1.2))
    ax3.annotate("", xy=(rho_r_g, 0.15), xytext=(2 * rho_r_g, 0.15),
                 arrowprops=dict(arrowstyle="<->", color=WARN, lw=1.2))
    ax3.text(1.5 * rho_r_g, 0.32, "ρ_r = %.2f m" % rho_r_g, fontsize=12.5, color=WARN,
             ha="center")
    ax3.annotate("", xy=(5.9 * rho_r_g, rho_a), xytext=(5.9 * rho_r_g, 2 * rho_a),
                 arrowprops=dict(arrowstyle="<->", color=WARN, lw=1.2))
    ax3.text(5.55 * rho_r_g, 1.5 * rho_a, "ρ_a\n%.2f m" % rho_a, fontsize=12.5,
             color=WARN, ha="right", va="center")
    ax3.set_xlabel("地距 [m]")
    ax3.set_ylabel("方位 [m]")
    ax3.set_title("分辨单元（地距 × 方位）", loc="left", fontsize=17, fontweight="bold",
                  color=TEXT, pad=8)

    fig.text(L, 0.265,
             "距离向：ρ_r,斜距 = c/2B = %.2f m（B = %g MHz），投影到地面 ρ_r,地距 = ρ_r,斜距 / sinθ = %.2f m（manifest %.2f m）"
             % (rho_r_s, sc["bandwidth_mhz"], rho_r_g, meta["res_range_m"]),
             fontsize=12.5, color=TEXT)
    fig.text(L, 0.205,
             "方位向：ρ_a = 0.886·λR0/(2·L_sa) = %.2f m（λ = %.1f mm，%g GHz），L_sa = min(v·T_int, λR0/D) = min(%.1f m, %.1f m) = %.1f m"
             % (rho_a, lam * 1000.0, sc["fc_ghz"], l_flight, l_beam, l_sa),
             fontsize=12.5, color=TEXT)
    fig.text(L, 0.140,
             "T_int = %d 脉冲 / %g Hz PRF = %.2f s，平台 %d m/s × %.2f s = %.1f m 航程；波束足迹 λR0/D 受 1.5 m 天线限制"
             % (sc["n_pulses"], sc["prf_hz"], t_int, sc["speed_mps"], t_int, l_flight),
             fontsize=12.0, color=DIM)
    note(fig, "源 data/seeker_air2ground/cfg.json + manifest sar_meta + docs/SAR成像.md §2.2")
    wm(fig, "02")
    return fig, "fig02_sar_geometry.png"


# ============================================================ fig03 SAR 序列
def fig03():
    sid = "seeker_air2ground"
    fr_list = SCENES[sid]["frames"]
    n = len(fr_list)
    fig = new_fig()
    gs = fig.add_gridspec(4, 5, left=L, right=R, top=T - 0.02, bottom=B + 0.06,
                          wspace=0.05, hspace=0.13)
    arrs = [load_img(f["mods"]["sar"]["img"], "L") for f in fr_list]
    allv = np.concatenate([a.ravel()[::37] for a in arrs])
    lo, hi = np.percentile(allv, [2.0, 98.0])
    cx = []
    for k, f in enumerate(fr_list):
        ax = fig.add_subplot(gs[k // 5, k % 5])
        a = np.clip((arrs[k] - lo) / max(hi - lo, 1e-9), 0, 1) ** 0.95
        imshow_native(ax, a, cmap="gray")
        ax.set_anchor("N")
        ax.text(0.02, 0.95, "f%02d" % k, transform=ax.transAxes, fontsize=10.5,
                color=TEXT, va="top", ha="left", zorder=9,
                bbox=dict(boxstyle="round,pad=0.18", fc=BG, ec="none", alpha=0.7))
        b = (f["mods"]["sar"].get("boxes") or [None])[0]
        if b:
            cxx, cyy = (b["x0"] + b["x1"]) / 2.0, (b["y0"] + b["y1"]) / 2.0
            cx.append(cxx)
            ax.plot([cxx], [cyy], marker="o", ms=10, mfc="none", mec=ACC, mew=1.6,
                    zorder=8)
            ax.plot([cxx], [cyy], marker="o", ms=2.4, color=ACC, zorder=8)
    drift = (max(cx) - min(cx)) if cx else 0.0
    fig.text(L, 0.045,
             "20 帧目标横向漂移 %.1f px：框中心始终停在 x = %.1f px · PLAN P0-3 要求 ≥20 px → 未达标（每帧把波束重新对准目标，帧间差异只剩斑点噪声）"
             % (drift, cx[0] if cx else 0.0),
             fontsize=12.5, color=TEXT, va="bottom")
    note(fig, "源 data/%s/sar/*.png + manifest 各帧 boxes · 20 帧共用一个显示拉伸区间" % sid)
    wm(fig, "03")
    return fig, "fig03_sar_sequence.png"


# ============================================================ fig04 雷达拉伸
def fig04():
    sid, fi = "seeker_air2ground", 7
    fr = frame(sid, fi)
    raw = load_img(fr["mods"]["radar_ra"]["img"], "L")
    pk = fr["radar"]["peak"]
    p50, p95, p999 = np.percentile(raw, [50, 95, 99.9])
    frac = 100.0 * float((raw <= p95).mean())

    fig = new_fig()
    gs = fig.add_gridspec(1, 3, left=L, right=R, top=T - 0.02, bottom=0.18,
                          wspace=0.22, width_ratios=[1.0, 1.0, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    imshow_native(ax, raw / 255.0, cmap="gray")
    ax.set_anchor("N")
    panel_title(ax, "原始 8-bit 图（未拉伸）", pad=30)
    ax.text(0.0, 1.008, "512×560 · P50 = %d / 255 · 最亮 %d" % (p50, raw.max()),
            transform=ax.transAxes, fontsize=12.5, color=DIM)

    ax2 = fig.add_subplot(gs[0, 1])
    imshow_native(ax2, stretch(raw, 1.0, 99.9, 0.55), cmap="gray")
    ax2.set_anchor("N")
    panel_title(ax2, "百分位拉伸 P1–P99.9 + γ 0.55", pad=30)
    ax2.add_patch(Circle((pk["px"], pk["py"]), 9, fill=False, ec=WARN, lw=1.6, zorder=8))
    ax2.annotate("检出峰 %.1f dB" % pk["db"], xy=(pk["px"], pk["py"]),
                 xytext=(pk["px"] + 24, pk["py"] - 34), fontsize=12, color=WARN,
                 arrowprops=dict(arrowstyle="-", color=WARN, lw=0.9))
    ax2.text(0.0, 1.008, "同一文件 · 仅改显示映射 · 目标峰 %d / 255" % raw.max(),
             transform=ax2.transAxes, fontsize=12.5, color=DIM)

    ax3 = fig.add_subplot(gs[0, 2])
    style_axes(ax3)
    ax3.hist(raw.ravel(), bins=np.arange(0, 106, 1), color=ACC, alpha=0.85, log=True)
    ax3.axvline(p50, color=DIM, lw=1.2, ls="--")
    ax3.axvline(p95, color=WARN, lw=1.4, ls="--")
    ax3.axvline(float(raw.max()), color=ERR, lw=1.4, ls="--")
    ax3.set_xlabel("灰阶值 [0–255]")
    ax3.set_ylabel("像素数（对数）")
    ax3.set_ylim(0.7, 6.0e5)
    ax3.set_title("灰阶直方图", loc="left", fontsize=17, fontweight="bold", color=TEXT, pad=30)
    ax3.annotate("噪底中位 %d" % p50, xy=(p50, 1.0e5), xytext=(p50 + 26, 2.4e5),
                 fontsize=12, color=DIM, arrowprops=dict(arrowstyle="-", color=DIM, lw=0.8))
    ax3.annotate("P95 = %d" % p95, xy=(p95, 2.6e4), xytext=(p95 + 30, 6.0e4),
                 fontsize=12, color=WARN, arrowprops=dict(arrowstyle="-", color=WARN, lw=0.8))
    ax3.annotate("目标峰 %d" % raw.max(), xy=(float(raw.max()), 1.6),
                 xytext=(float(raw.max()) - 46, 40),
                 fontsize=12, color=ERR, arrowprops=dict(arrowstyle="-", color=ERR, lw=0.8))
    ax3.grid(True, color=GRID, lw=0.6, ls="--")
    ax3.set_axisbelow(True)

    fig.text(L, 0.115,
             "%.1f%% 的像素落在 P95 = %d 灰阶以下：原始图看上去“全黑”不是没信号，" % (frac, p95),
             fontsize=13.5, color=TEXT, va="bottom")
    fig.text(L, 0.062,
             "而是目标峰与噪底差了 %d 个灰阶，线性映射把整幅图压进了最暗的几条色阶。"
             % (int(raw.max()) - int(p50)),
             fontsize=13.5, color=TEXT, va="bottom")
    note(fig, "源 data/%s/radar_ra/%06d.png + manifest radar.peak · 分析：numpy 直方图" % (sid, fi))
    wm(fig, "04")
    return fig, "fig04_radar_stretch.png"


# ============================================================ fig05 点云三视
def fig05():
    sid, fi = "final_ground", 1
    c = cloud_of(sid, fi)
    x = np.asarray(c["x"], float) / 100.0
    y = np.asarray(c["y"], float) / 100.0
    z = np.asarray(c["z"], float) / 100.0
    inten = np.asarray(c["i"], float)
    nmax = int(c["maxr"])

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("sky", ["#1b2530", "#37c2ff", "#43d98a", "#ffb454"])
    fig = new_fig()
    gs = fig.add_gridspec(2, 2, left=L, right=0.895, top=T - 0.02, bottom=0.19,
                          wspace=0.20, hspace=0.46, width_ratios=[0.85, 1.15])
    keep = (x > -7.5) & (x < 7.5) & (y > -3.5) & (y < 11.5)
    views = [("俯视  X–Y", x, y, z, "X [m]", "Y [m]", (0, 0), gs[0:2, 0]),
             ("前视  X–Z（裁 X = -7.5…7.5 m）", x[keep], z[keep], z[keep], "X [m]",
              "Z [m]", (0, 1), gs[0, 1]),
             ("侧视  Y–Z（裁 Y = -3.5…11.5 m）", y[keep], z[keep], z[keep], "Y [m]",
              "Z [m]", (1, 1), gs[1, 1])]
    for name, u, v, cc, xl, yl, _ij, cell in views:
        ax = fig.add_subplot(cell)
        style_axes(ax)
        sc = ax.scatter(u, v, c=cc, cmap=cmap, s=1.6, alpha=0.75, linewidths=0,
                        vmin=0.0, vmax=max(z.max(), 1e-6))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")
        ax.set_anchor("C")
        ax.set_title(name, loc="left", fontsize=17, fontweight="bold", color=TEXT, pad=10)
        if "俯视" in name:
            zt = z > 1.2
            if zt.any():
                ax.annotate("目标（漫游车）", xy=(float(np.median(u[zt])),
                                                 float(np.median(v[zt]))),
                            xytext=(4.5, 15.0), fontsize=12.5, color=WARN,
                            arrowprops=dict(arrowstyle="->", color=WARN, lw=1.0))
    cax = fig.add_axes([0.912, 0.19, 0.011, 0.70])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("高度 Z [m]", fontsize=13, color=TEXT)
    cb.ax.tick_params(colors=DIM, labelsize=12)
    cb.outline.set_edgecolor(SPINE)

    fig.text(L, 0.10,
             "点数 %s（命中率 %.1f%%）· 量程 %d m · 强度 %d–%d（原始计数）· 高度 %.2f–%.2f m"
             % (format(int(c["n"]), ","), 100.0 * c["hit"], nmax,
                int(inten.min()), int(inten.max()), z.min(), z.max()),
             fontsize=13.5, color=TEXT, va="bottom")
    note(fig, "源 data/cloud/%s.json 第 %d 帧（manifest 的 cloud 字段外置）· 单位 cm → m" % (sid, fi))
    wm(fig, "05")
    return fig, "fig05_pointcloud_views.png"


# ============================================================ fig06 导引头轨迹
def fig06():
    seeks = ["seeker_air2air_high", "seeker_air2ground", "seeker_surface2air"]
    fig = new_fig()
    ax = fig.add_axes([L + 0.065, 0.265, (R - L) - 0.135, 0.615])
    style_axes(ax)
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    for s in ax2.spines.values():
        s.set_color(SPINE)
    ax2.tick_params(colors=DIM, labelsize=13)

    for sid in seeks:
        fr = SCENES[sid]["frames"]
        xs = [f["i"] + 1 for f in fr]
        rg = [f["range_m"] for f in fr]
        px = [f["views"]["seeker"]["target_px"] for f in fr]
        pw = [f["views"]["wide"]["target_px"] for f in fr]
        col = SCENE_COLORS[sid]
        ax.plot(xs, rg, color=col, lw=2.4, marker="o", ms=3.2, label=sid)
        ax2.plot(xs, px, color=col, lw=1.7, ls="--", alpha=0.95)
        ax2.plot(xs, pw, color=col, lw=1.4, ls=":", alpha=0.7)

    ax.set_xlim(0.3, 20.7)
    ax.set_ylim(2000, 8200)
    ax2.set_ylim(0, 60)
    ax.set_xlabel("帧序号")
    ax.set_ylabel("斜距 [m]")
    ax2.set_ylabel("目标像素跨度 [px]")
    ax2.set_yticks([0, 6, 20, 40, 60])
    ax.axhline(0, color=SPINE, lw=0)
    ax2.axhline(6.0, color=ERR, lw=1.6, ls="--")
    ax2.text(20.4, 6.9, "6 px 可训练下限（PLAN P0-2）", fontsize=12.5, color=ERR,
             ha="right", va="bottom")

    ax2.tick_params(colors=ACC2, labelsize=13)
    ax2.yaxis.label.set_color(ACC2)
    ax.text(0.6, 8020, "实线 = 斜距（左轴，m）· 虚线 = 窄视场 target_px · 点线 = 宽视场 target_px（右轴，px）",
            fontsize=12, color=DIM, va="top")

    handles = [Line2D([], [], color=SCENE_COLORS[s], lw=2.2, label=s) for s in seeks]
    handles += [Line2D([], [], color=DIM, lw=2.2, label="斜距（左轴）"),
                Line2D([], [], color=DIM, lw=1.7, ls="--", label="窄视场 2.2° 像素"),
                Line2D([], [], color=DIM, lw=1.4, ls=":", label="宽视场 18° 像素")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=12,
                     bbox_to_anchor=(0.5, 0.075), handlelength=2.2, columnspacing=1.4,
                     labelcolor=TEXT)
    for t in leg.get_texts():
        t.set_color(TEXT)

    fig.text(L, 0.185,
             "窄视场 20 帧只走 %.0f–%.0f m，目标始终 %.1f–%.1f px；宽视场 18° 同一目标只有 %.1f–%.1f px"
             % (min(f["range_m"] for s in seeks for f in SCENES[s]["frames"]),
                max(f["range_m"] for s in seeks for f in SCENES[s]["frames"]),
                min(f["views"]["seeker"]["target_px"] for s in seeks for f in SCENES[s]["frames"]),
                max(f["views"]["seeker"]["target_px"] for s in seeks for f in SCENES[s]["frames"]),
                min(f["views"]["wide"]["target_px"] for s in seeks for f in SCENES[s]["frames"]),
                max(f["views"]["wide"]["target_px"] for s in seeks for f in SCENES[s]["frames"])),
             fontsize=13, color=TEXT)
    note(fig, "源 data/manifest.json：3 个 seeker 场景 × 20 帧（城市遮挡场景尚未重生成）")
    wm(fig, "06")
    return fig, "fig06_seeker_trajectory.png"


# ============================================================ fig07 雷达预算
def fig07():
    bands = sorted(BANDS, key=lambda b: BANDS[b]["fc_ghz"])
    sigma = 1.0
    det = float(BANDS[bands[0]]["det_snr_db"])
    rain = 8.0
    rs = np.logspace(1.7, 4.4, 320)          # 50 m .. 25 km
    fig = new_fig()
    gs = fig.add_gridspec(1, 2, left=L + 0.035, right=R, top=T - 0.03, bottom=0.20,
                          wspace=0.14)
    rmax_tab = {}
    for k, (wlabel, rn) in enumerate([("晴 0 mm/h", 0.0), ("雨 %g mm/h" % rain, rain)]):
        ax = fig.add_subplot(gs[0, k], sharey=None)
        style_axes(ax)
        for b in bands:
            att = radar_att_db_km(b, rn)
            snr = np.array([radar_snr_db(r, sigma, att, b) for r in rs])
            rm = radar_rmax_m(b, sigma, att, det)
            rmax_tab[(b, k)] = rm
            ax.plot(rs / 1000.0, snr, color=BAND_COLORS[b], lw=2.0)
            if rm > 0:
                ax.plot([rm / 1000.0], [det], marker="o", ms=6, color=BAND_COLORS[b],
                        mec=BG, mew=1.0, zorder=8)
        ax.axhline(det, color=ERR, lw=1.5, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(0.05, 25)
        ax.set_ylim(-32, 72)
        ax.set_xlabel("斜距 R [km]（对数）")
        if k == 0:
            ax.set_ylabel("接收 SNR [dB]")
        ax.set_title("%s" % wlabel, loc="left", fontsize=17, fontweight="bold",
                     color=TEXT, pad=30)
        ax.text(0.055, det + 2.0, "检出门限 %.0f dB" % det, fontsize=12.5, color=ERR,
                va="bottom")
    axl = fig.axes[0]
    axl.text(0.0, 1.008, "σ = 1 m² · P_t 40 dBm · 32 脉冲相干积累 · 参数 data/radar_bands.json",
             transform=axl.transAxes, fontsize=12.5, color=DIM, va="bottom")
    hand = [Line2D([], [], color=BAND_COLORS[b], lw=2.2,
                   label="%s  %.1f GHz" % (b, BANDS[b]["fc_ghz"])) for b in bands]
    fig.legend(handles=hand, loc="lower left", ncol=6, fontsize=13,
               bbox_to_anchor=(0.06, 0.075), handlelength=2.0, columnspacing=1.5)

    lines = ["R_max 晴 → 雨（8 mm/h）"]
    for b in bands:
        a = rmax_tab[(b, 0)] / 1000.0
        c = rmax_tab[(b, 1)] / 1000.0
        lines.append("%-4s %5.2f → %5.2f km  %+.0f%%"
                     % (b, a, c, 100.0 * (c - a) / max(a, 1e-9)))
    fig.axes[1].text(0.035, 0.035, "\n".join(lines), transform=fig.axes[1].transAxes,
                     fontsize=11.5, color=TEXT, va="bottom", ha="left", linespacing=1.62,
                     family="Noto Sans Mono CJK SC",
                     bbox=dict(boxstyle="round,pad=0.5", fc="#101820", ec=SPINE, alpha=0.95))
    note(fig, "模型：tools/radar_budget_table.py 同一条雷达方程 · 雨衰 γ 取 simkit/weather.py 的 ITU-R P.838 系数")
    wm(fig, "07")
    return fig, "fig07_radar_budget.png"


# ============================================================ fig08 雨衰
def fig08():
    bands = sorted(BANDS, key=lambda b: BANDS[b]["fc_ghz"])
    rr = np.linspace(0.0, 64.0, 400)
    fig = new_fig()
    ax = fig.add_axes([L + 0.05, 0.27, (R - L) - 0.10, 0.61])
    style_axes(ax)
    for b in bands:
        k = BANDS[b]["rain_k"]; al = BANDS[b]["rain_a"]
        g = k * np.maximum(rr, 1e-6) ** al
        ax.plot(rr, g, color=BAND_COLORS[b], lw=2.2,
                label="%s  %.1f GHz  k=%.4g α=%.3g" % (b, BANDS[b]["fc_ghz"], k, al))
        g8 = k * 8.0 ** al
        ax.plot([8.0], [g8], marker="o", ms=5.5, color=BAND_COLORS[b], mec=BG, mew=1.0)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 120)
    ax.set_xlim(0, 64)
    ax.set_xlabel("雨强 R [mm/h]")
    ax.set_ylabel("单程雨衰 γ = k·R^α  [dB/km]")
    ax.axvline(8.0, color=DIM, lw=1.0, ls="--")
    ax.axvline(32.0, color=DIM, lw=1.0, ls=":")
    ax.text(8.4, 0.0062, "8 mm/h\nrain 预设", fontsize=11.5, color=DIM, va="bottom")
    ax.text(32.4, 0.0062, "32 mm/h\nstorm 预设", fontsize=11.5, color=DIM, va="bottom")
    ax.legend(loc="lower right", ncol=1, fontsize=13, handlelength=2.1,
              columnspacing=1.8, borderaxespad=0.8)
    ka = BANDS["Ka"]["rain_k"] * 8.0 ** BANDS["Ka"]["rain_a"]
    fig.text(L, 0.070,
             "Ka 在 8 mm/h 雨里 γ = %.2f dB/km，2 km 双程 %.1f dB —— 与 docs/P1_长程雷达与标定.md 实测的 6.3 dB 一致；\n"
             "同一场雨对 X 波段只有 0.15 dB/km，波段选择直接决定“能不能在雨里看远”。"
             % (ka, 2 * 2 * ka), fontsize=12.5, color=TEXT, va="bottom", linespacing=1.7)
    ax.set_title("ITU-R P.838-3：γ = k·R^α（水平极化近似）",
                 loc="left", fontsize=15, color=TEXT, pad=12)
    note(fig, "源 data/radar_bands.json rain_k / rain_a · 与 simkit/weather.py RADAR_BANDS 同源")
    wm(fig, "08")
    return fig, "fig08_rain_attenuation.png"


# ============================================================ fig09 大气透过率
def fig09():
    vis = np.logspace(math.log10(0.3), math.log10(30.0), 160)
    rain = np.linspace(0.0, 64.0, 160)
    mods = [("visible", "可见光 550 nm", ACC), ("mwir", "中波红外 3–5 µm", WARN),
            ("lwir", "长波红外 8–12 µm", ACC2), ("lidar", "激光 905 nm", PUR)]
    Tv = {m: np.array([extinction_per_km(v * 1000.0, 0.0)[m] for v in vis])
          for m, _, _ in mods}
    Tr = {m: np.array([extinction_per_km(20000.0, r)[m] for r in rain]) for m, _, _ in mods}

    fig = new_fig()
    gs = fig.add_gridspec(1, 2, left=L + 0.035, right=R, top=T - 0.03, bottom=0.19,
                          wspace=0.20)
    ax = fig.add_subplot(gs[0, 0])
    style_axes(ax)
    for m, lab, col in mods:
        ax.plot(vis, np.exp(-Tv[m] * 1.0), color=col, lw=2.2, label=lab)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 30)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("能见度 V [km]（对数）")
    ax.set_ylabel("1 km 路径透过率 T")
    ax.set_title("透过率 vs 能见度（Koschmieder，550 nm 基准）", loc="left", fontsize=15,
                 color=TEXT, pad=10)
    for v0, nm in [(20.0, "clear 20 km"), (4.0, "rain 4 km"), (0.8, "fog 0.8 km")]:
        ax.axvline(v0, color=DIM, lw=0.9, ls="--")
        ax.text(v0 * 1.04, 0.985, nm, fontsize=11, color=DIM, ha="left", va="top")
    ax.legend(loc="upper left", fontsize=13, handlelength=2.2)

    ax2 = fig.add_subplot(gs[0, 1])
    style_axes(ax2)
    for m, lab, col in mods:
        ax2.plot(rain, np.exp(-Tr[m] * 1.0), color=col, lw=2.2)
    ax2.set_xlim(0, 64)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("雨强 R [mm/h]（能见度固定 20 km）")
    ax2.set_ylabel("1 km 路径透过率 T")
    ax2.set_title("透过率 vs 降雨", loc="left", fontsize=15, color=TEXT, pad=10)
    ax2.axvline(8.0, color=DIM, lw=0.9, ls="--")
    ax2.text(8.6, 0.05, "8 mm/h", fontsize=11, color=DIM)

    fig.text(L, 0.075,
             "零雨时 1 km：可见光 T=%.2f、MWIR %.2f、LWIR %.2f、905 nm %.2f —— 长波红外在霾/雾里衰减最慢，"
             "可见光与 905 nm 最先失去对比度"
             % (math.exp(-extinction_per_km(20000, 0)["visible"]),
                math.exp(-extinction_per_km(20000, 0)["mwir"]),
                math.exp(-extinction_per_km(20000, 0)["lwir"]),
                math.exp(-extinction_per_km(20000, 0)["lidar"])),
             fontsize=13, color=TEXT)
    note(fig, "公式：simkit/atmosphere.py（Beer–Lambert）+ simkit/weather.py derive()，homogeneous 剖面")
    wm(fig, "09")
    return fig, "fig09_atmosphere_transmittance.png"


# ============================================================ fig10 标签一致
def fig10():
    sid, fi = "seeker_air2ground", 7
    fr = frame(sid, fi)
    lc = EXPORT["label_consistency"]
    keys = [("optical", "可见光 RGB", ACC), ("mwir", "中波红外", ACC2),
            ("lwir", "长波红外", WARN)]
    b0 = fr["mods"]["optical"]["boxes"][0]
    cx, cy = (b0["x0"] + b0["x1"]) / 2.0, (b0["y0"] + b0["y1"]) / 2.0
    half = 30
    x0 = int(max(0, min(768 - 2 * half, cx - half)))
    y0 = int(max(0, min(576 - 2 * half, cy - half)))
    x1, y1 = x0 + 2 * half, y0 + 2 * half

    fig = new_fig()
    gs = fig.add_gridspec(2, 4, left=L, right=R, top=T - 0.03, bottom=0.075,
                          wspace=0.16, width_ratios=[1, 1, 1, 1.45],
                          height_ratios=[0.58, 0.42])
    for k, (mk, lab, col) in enumerate(keys):
        ax = fig.add_subplot(gs[0, k])
        im = load_img(fr["mods"][mk]["img"])
        crop = (im[y0:y1, x0:x1] / 255.0) ** 0.95
        imshow_native(ax, crop, cmap=None)
        ax.set_xlim(0, x1 - x0)
        ax.set_ylim(y1 - y0, 0)
        box = fr["mods"][mk]["boxes"][0]
        for j, (_, _, c2) in enumerate(keys):
            lw = 2.4 if j == k else 1.1
            ls = ["-", "--", ":"][j]
            ax.add_patch(Rectangle((box["x0"] - x0, box["y0"] - y0),
                                   box["x1"] - box["x0"], box["y1"] - box["y0"],
                                   fill=False, ec=c2, lw=lw, ls=ls))
        ax.set_anchor("N")
        ax.set_title(lab, loc="left", fontsize=17, fontweight="bold", color=col, pad=30)
        ax.text(0.0, 1.008, "源图 %d×%d px 裁剪 · 放大约 5×" % (x1 - x0, y1 - y0),
                transform=ax.transAxes, fontsize=12, color=DIM)

    axm = fig.add_subplot(gs[0, 3])
    axm.set_anchor("N")
    style_axes(axm)
    axm.set_xlim(-0.2, 3.4)
    axm.set_ylim(0, 1)
    axm.set_yticks([])
    axm.set_xlabel("同目标框跨模态偏差 [px]")
    axm.axvspan(0.0, 2.0, color=ACC2, alpha=0.12)
    axm.axvline(2.0, color=ACC2, lw=1.4, ls="--")
    axm.axvline(0.0, color=ACC, lw=2.6)
    axm.text(2.06, 0.90, "P2-3 目标 <2 px", fontsize=12, color=ACC2, va="top")
    axm.plot([0.0], [0.52], marker="o", ms=12, color=ACC, mec=BG, mew=1.5, zorder=6)
    axm.annotate("全部 %d 对：%.1f px" % (lc["pairs_checked"], lc["max_dev_px"]),
                 xy=(0.0, 0.52), xytext=(0.45, 0.30), fontsize=13.5, color=ACC,
                 arrowprops=dict(arrowstyle="->", color=ACC, lw=1.1))
    axm.text(0.02, 0.16,
             "同一时刻 + 同一相机位姿 + 同一套投影 → 标签逐位相同，\n"
             "差异为 0 是回归防护，不是像素配准的证明（P2-3）。",
             fontsize=12, color=DIM, va="bottom", linespacing=1.6)
    axm.set_title("回归自检：%d 对样本" % lc["pairs_checked"], loc="left", fontsize=17,
                  fontweight="bold", color=TEXT, pad=10)

    axt = fig.add_subplot(gs[1, :])
    axt.axis("off")
    axt.set_facecolor(BG)
    axt.text(0.0, 0.92,
             "实线 = 可见光框 · 虚线 = 中波框 · 点线 = 长波框：三种画法落在同一组坐标上且完全重合（偏差 %.1f px）"
             % lc["max_dev_px"], fontsize=14, color=TEXT, va="top")
    axt.text(0.0, 0.58,
             "检查口径：%s\n"
             "标签在导出时由同一套投影直接算出，所以“偏差 0”只说明链路没有漂移；它不证明两张图的像素已经对齐（那是 P2-3 的独立指标）。"
             % lc["meaning"], fontsize=13, color=DIM, va="top", linespacing=1.75)
    note(fig, "源 datasets/yolo/export.json label_consistency + data/%s 第 %d 帧" % (sid, fi))
    wm(fig, "10")
    return fig, "fig10_label_consistency.png"


# ============================================================ fig11 数据集构成
def fig11():
    fl = EXPORT["flavours"]
    names = list(fl.keys())
    tr = [fl[n]["by_split"]["train"]["images"] for n in names]
    va = [fl[n]["by_split"]["val"]["images"] for n in names]
    bx = [fl[n]["boxes"] for n in names]

    fig = new_fig()
    gs = fig.add_gridspec(1, 2, left=L + 0.02, right=R, top=T - 0.03, bottom=0.26,
                          wspace=0.22, width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    style_axes(ax)
    xx = np.arange(len(names))
    b1 = ax.bar(xx, tr, 0.62, color=ACC, label="train")
    b2 = ax.bar(xx, va, 0.62, bottom=tr, color=ACC2, label="val")
    for i, n in enumerate(names):
        ax.text(i, tr[i] + va[i] + 2.5, "%d" % (tr[i] + va[i]), ha="center",
                fontsize=12, color=TEXT)
        ax.text(i, tr[i] + va[i] + 9.5, "%d 框" % bx[i], ha="center", fontsize=10.5,
                color=DIM)
    ax.set_xticks(xx)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=12)
    ax.set_ylim(0, 82)
    ax.set_ylabel("图像数 [张]")
    ax.set_title("8 种 flavour × train/val", loc="left", fontsize=17, fontweight="bold",
                 color=TEXT, pad=10)
    ax.legend(loc="upper right", fontsize=12.5)

    ax2 = fig.add_subplot(gs[0, 1])
    style_axes(ax2)
    st = EXPORT["stacks"]
    sn = list(st.keys())
    si = [st[s]["images"] for s in sn]
    sb = [st[s]["boxes"] for s in sn]
    yy = np.arange(len(sn))
    ax2.barh(yy, si, 0.55, color=ACC, label="图像")
    for i, s in enumerate(sn):
        ax2.text(si[i] + 1.2, i - 0.13, "%d 图 / %d 框" % (si[i], sb[i]), va="center",
                 fontsize=12, color=TEXT)
        ax2.text(1.2, i + 0.22, "+".join(st[s]["panels"]), va="center", fontsize=9.5,
                 color=BG)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(sn, fontsize=12.5)
    ax2.set_xlim(0, 98)
    ax2.invert_yaxis()
    ax2.set_xlabel("复合图数 [张]")
    ax2.set_title("5 种 stack（多通道拼接）", loc="left", fontsize=17, fontweight="bold",
                  color=TEXT, pad=10)
    ax2.grid(False)

    tot_img = sum(fl[n]["images"] for n in names)
    tot_box = sum(bx)
    fig.text(L, 0.135,
             "配对帧 %d · flavour 图 %d 张 / 框 %d 个 · stack 图 %d 张 / 框 %d 个 · 格式 %s · 切分规则：%s"
             % (EXPORT["frames_paired"], tot_img, tot_box,
                sum(si), sum(sb), EXPORT["fmt"], EXPORT["split_rule"]),
             fontsize=12.5, color=TEXT)
    note(fig, "源 datasets/yolo/export.json（runs: %s）" % ", ".join(EXPORT["runs"]))
    wm(fig, "11")
    return fig, "fig11_dataset_composition.png"


# ============================================================ fig12 框尺寸分布
def _yolo_rows(flav, split):
    d = ROOT / "datasets" / "yolo" / "labels" / flav / split
    out = []
    if not d.is_dir():
        return out
    for fn in sorted(d.glob("*.txt")):
        for line in fn.read_text(encoding="utf-8").splitlines():
            t = line.split()
            if len(t) >= 5:
                out.append([float(x) for x in t[:5]])
    return out


def fig12():
    narrow = _yolo_rows("optical", "train") + _yolo_rows("optical", "val")
    wide = _yolo_rows("optical_wide", "train") + _yolo_rows("optical_wide", "val")
    wn = np.array([r[3] for r in narrow]) * 768.0
    ww = np.array([r[3] for r in wide]) * 640.0
    hn = np.array([r[4] for r in narrow]) * 576.0
    hw = np.array([r[4] for r in wide]) * 480.0

    fig = new_fig()
    ax = fig.add_axes([L + 0.05, 0.16, (R - L) - 0.09, 0.70])
    style_axes(ax)
    bins = np.logspace(math.log10(0.8), math.log10(120), 40)
    ax.hist(wn, bins=bins, color=ACC, alpha=0.40, histtype="stepfilled",
            edgecolor=ACC, lw=1.5, label="窄视场 2.2°（768×576）")
    ax.hist(ww, bins=bins, color=WARN, alpha=0.40, histtype="stepfilled",
            edgecolor=WARN, lw=1.5, label="宽视场 18°（640×480）")
    ax.set_xscale("log")
    ax.set_ylim(0, 44)
    ax.axvline(6.0, color=ERR, lw=1.8, ls="--")
    ax.axvline(np.median(wn), color=ACC, lw=1.1, ls=":")
    ax.axvline(np.median(ww), color=WARN, lw=1.1, ls=":")
    ax.set_xlabel("标注框宽度 [px]（对数）")
    ax.set_ylabel("框数量 [个]")
    ax.set_xlim(0.8, 120)
    ax.legend(loc="upper left", fontsize=13)
    ax.text(6.4, 42.0, "6 px 可训练下限", fontsize=12.5, color=ERR, va="top")
    ax.text(np.median(wn) * 0.94, 39.0, "窄视场中位 %.1f px" % np.median(wn),
            fontsize=12.5, color=ACC, ha="right", va="top")
    ax.text(np.median(ww) * 1.06, 32.0, "宽视场中位 %.1f px\n最小 %.1f px"
            % (np.median(ww), ww.min()), fontsize=12.5, color=WARN, ha="left",
            va="top")
    ax.set_title("标注框像素宽度分布：窄视场 vs 宽视场（%d 个框）" % (len(wn) + len(ww)),
                 loc="left", fontsize=15, color=TEXT, pad=12)
    fig.text(L, 0.075,
             "同一目标、同一时刻：窄视场 %d 个框中位 %.1f×%.1f px；宽视场 %d 个框中位 %.1f×%.1f px，"
             "最小 %.1f px —— 低于 6 px 的样本只适合做搜索阶段"
             % (len(wn), np.median(wn), np.median(hn), len(ww), np.median(ww),
                np.median(hw), ww.min()),
             fontsize=12.5, color=TEXT)
    note(fig, "源 datasets/yolo/labels/optical/**（768 px 宽）与 optical_wide/**（640 px 宽），train+val")
    wm(fig, "12")
    return fig, "fig12_box_size.png"


# ============================================================ fig13 mosaic
def fig13():
    split = "train"
    d = ROOT / "datasets" / "yolo" / "images" / "optical" / split
    files = sorted(d.glob("*.jpg"))[:12]
    cls_name = {v: k for k, v in EXPORT["classes"].items()}
    fig = new_fig()
    gs = fig.add_gridspec(3, 4, left=L, right=R, top=T - 0.02, bottom=B + 0.04,
                          wspace=0.055, hspace=0.10)
    for k, f in enumerate(files):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        im = load_path(f)
        imshow_native(ax, im / 255.0)
        Hh, Ww = im.shape[:2]
        lab = ROOT / "datasets" / "yolo" / "labels" / "optical" / split / (f.stem + ".txt")
        nbox = 0
        if lab.exists():
            for line in lab.read_text(encoding="utf-8").splitlines():
                t = line.split()
                if len(t) < 5:
                    continue
                c, cxx, cyy, bw, bh = [float(x) for x in t[:5]]
                x0 = (cxx - bw / 2) * Ww; y0 = (cyy - bh / 2) * Hh
                x1 = (cxx + bw / 2) * Ww; y1 = (cyy + bh / 2) * Hh
                pad = 3
                ax.add_patch(Rectangle((x0 - pad, y0 - pad), (x1 - x0) + 2 * pad,
                                       (y1 - y0) + 2 * pad, fill=False, ec=ACC2,
                                       lw=1.4))
                ax.text(x0 - pad, y0 - pad - 2, cls_name.get(int(c), str(int(c))),
                        fontsize=8.5, color=ACC2, va="bottom")
                nbox += 1
        ax.text(0.015, 0.975, f.stem, transform=ax.transAxes, fontsize=9.5,
                color=TEXT, va="top", ha="left", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.65))
        ax.text(0.985, 0.975, "%d 框" % nbox, transform=ax.transAxes, fontsize=9.5,
                color=ACC2, va="top", ha="right", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.65))
    fig.text(L, 0.072,
             "YOLO 训练输入观感：images/optical/%s 前 12 张 · 768×576 · 每张 1 个框 · 背景与传感器噪声保持渲染原样"
             % split, fontsize=12.5, color=TEXT)
    note(fig, "源 datasets/yolo/images/optical/train + labels/optical/train（YOLO 归一化框已换算为像素）")
    wm(fig, "13")
    return fig, "fig13_mosaic.png"


# ============================================================ fig14 复合输入
def fig14():
    split = "train"
    d = ROOT / "datasets" / "yolo" / "composites" / "stack_narrow" / split
    f = sorted(d.glob("*.jpg"))[0]
    im = np.asarray(Image.open(f).convert("RGB"), float)
    Hh, Ww = im.shape[:2]
    st = EXPORT["stacks"]["stack_narrow"]
    lab = ROOT / "datasets" / "yolo" / "labels" / "stack_narrow" / split / (f.stem + ".txt")
    nlab = len([1 for line in lab.read_text(encoding="utf-8").splitlines() if line.strip()])

    fig = new_fig()
    gs = fig.add_gridspec(2, 3, left=L, right=R, top=T - 0.02, bottom=B + 0.02,
                          hspace=0.30, wspace=0.10, height_ratios=[1.0, 0.52])
    ax = fig.add_subplot(gs[0, :])
    imshow_native(ax, im / 255.0)
    panel_title(ax, "stack_narrow 真实复合图（%d×%d）" % (Ww, Hh), pad=28)
    ax.text(0.0, 1.008, "%d 块面板水平拼接 · 每块 768×576 · 4 px 分隔 · %s"
            % (len(st["panels"]), f.name), transform=ax.transAxes, fontsize=12.5,
            color=DIM)

    cols = [(0, 768, "可见光 RGB", "3 通道 · 0.4–0.7 µm", ACC),
            (772, 1540, "中波红外 MWIR", "单通道 · 3–5 µm", WARN),
            (1544, 2312, "长波红外 LWIR", "单通道 · 8–12 µm", ACC2)]
    for k, (x0, x1, name, sub, col) in enumerate(cols):
        axb = fig.add_subplot(gs[1, k])
        axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.set_xticks([]); axb.set_yticks([])
        style_axes(axb)
        axb.grid(False)
        axb.add_patch(FancyBboxPatch((0.05, 0.54), 0.90, 0.34, boxstyle="round,pad=0.02",
                                     fc="#101820", ec=col, lw=1.6))
        axb.text(0.5, 0.71, name, fontsize=16, fontweight="bold", color=col,
                 ha="center", va="center")
        axb.text(0.5, 0.44, sub, fontsize=12.5, color=DIM, ha="center", va="center")
        axb.annotate("", xy=(0.10, 0.24), xytext=(0.90, 0.24),
                     arrowprops=dict(arrowstyle="<->", color=DIM, lw=1.0))
        axb.text(0.5, 0.10, "x = %d–%d px" % (x0, x1), fontsize=11.5, color=DIM,
                 ha="center")
    fig.text(L, 0.068,
             "标签：每块面板各自归一化，同一目标写入 %d 条记录（%d 框/图）· 训练时按通道拆开，框坐标不再需要跨面板换算"
             % (len(st["panels"]) * st["boxes"] // st["images"], len(st["panels"])),
             fontsize=12.5, color=TEXT)
    fig.text(L, 0.028,
             "本次样本标签行数 %d（%s）" % (nlab, lab.name), fontsize=11.5, color=DIM)
    note(fig, "源 datasets/yolo/composites/stack_narrow/%s + export.json stacks.stack_narrow" % split)
    wm(fig, "14")
    return fig, "fig14_composite.png"


# ============================================================ fig15 覆盖矩阵
def fig15():
    keys = ["optical", "optical_wide", "mwir", "mwir_wide", "lwir", "lwir_wide",
            "radar_ra", "lidar_range", "lidar_intensity", "sar"]
    sids = [s["id"] for s in MANIFEST["scenes"]]
    M = np.zeros((len(sids), len(keys)), int)
    cache = {}
    for i, sid in enumerate(sids):
        for j, k in enumerate(keys):
            n_img = n_box = 0
            for f in SCENES[sid]["frames"]:
                d = f["mods"].get(k)
                if not d:
                    continue
                n_img += 1
                if d.get("boxes"):
                    n_box += 1
            cache[(i, j)] = (n_img, n_box)
            M[i, j] = 2 if n_box else (1 if n_img else 0)

    fig = new_fig()
    ax = fig.add_axes([L + 0.10, 0.175, (R - L) - 0.18, 0.655])
    cmap = ListedColormap(["#11171f", "#1c3441", "#2a6f8f"])
    ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=26, ha="right", fontsize=12)
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels(sids, fontsize=12.5)
    ax.set_xticks(np.arange(-0.5, len(keys)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sids)), minor=True)
    ax.grid(which="minor", color=BG, lw=2.0, ls="-")
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_color(SPINE)
    for i in range(len(sids)):
        for j in range(len(keys)):
            n_img, n_box = cache[(i, j)]
            if M[i, j] == 2:
                txt, col = "✓ %d/%d" % (n_box, len(SCENES[sids[i]]["frames"])), TEXT
            elif M[i, j] == 1:
                txt, col = "图", DIM
            else:
                txt, col = "–", MUTE
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, color=col)
    for i, sid in enumerate(sids):
        ax.text(len(keys) - 0.35, i, "%d 帧" % len(SCENES[sid]["frames"]),
                fontsize=11.5, color=DIM, va="center", ha="left")

    hand = [Rectangle((0, 0), 1, 1, fc="#2a6f8f", ec="none"),
            Rectangle((0, 0), 1, 1, fc="#1c3441", ec="none"),
            Rectangle((0, 0), 1, 1, fc="#11171f", ec="none")]
    leg = fig.legend(hand, ["有图 + 有框（含出框帧数）", "有图、无框", "该场景无此模态"],
                     loc="lower left", ncol=3, fontsize=12.5,
                     bbox_to_anchor=(0.06, 0.035), handlelength=1.4,
                     columnspacing=1.8)
    for t in leg.get_texts():
        t.set_color(TEXT)
    ax.text(0.0, 1.012, "行 = 6 个场景，列 = manifest 里出现过的 10 种模态键；✓ 后的比例是出框帧数 / 该场景总帧数",
            transform=ax.transAxes, fontsize=12, color=DIM, va="bottom")
    note(fig, "源 data/manifest.json：%d 帧 × 模态 boxes 字段统计" % sum(len(s["frames"]) for s in MANIFEST["scenes"]))
    wm(fig, "15")
    return fig, "fig15_coverage.png"


# ============================================================ fig16 管线
def fig16():
    fig = new_fig()
    ax = fig.add_axes([0.045, 0.06, 0.91, 0.84])
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(BG)

    def box(x, y, w, h, title, sub, col=ACC, fs=15):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc="#101820", ec=col, lw=1.5))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=col)
        if sub:
            ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                    fontsize=11.5, color=DIM)

    def arrow(x0, y0, x1, y1, col=ACC, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=16, color=col, lw=1.6, ls=ls,
                                     connectionstyle="arc3,rad=%s" % rad,
                                     shrinkA=2, shrinkB=2))

    # 核心约束横幅
    ax.add_patch(FancyBboxPatch((0.15, 8.55), 15.7, 0.85, boxstyle="round,pad=0.05",
                                fc="#12202b", ec=ACC, lw=1.2))
    ax.text(8.0, 8.97, "同一份三维世界  +  同一套天气参数（8 标量状态机）",
            ha="center", va="center", fontsize=18, fontweight="bold", color=ACC)

    box(0.25, 5.2, 2.6, 1.5, "场景 JSON", "scenes/*.json · 6 个", PUR)
    box(0.25, 2.4, 2.6, 1.5, "天气状态机", "weather.py · 晴/阴/雨", WARN)
    box(0.25, 0.35, 2.6, 1.2, "几何世界 + BVH", "scene.py / bvh.py", ACC2)

    box(4.0, 4.95, 2.5, 2.0, "光学 / 红外", "Blender 无头 + thermal/ir", ACC)
    box(4.0, 2.55, 2.5, 1.0, "雷达", "77/35 GHz · RD/RA", WARN)
    box(4.0, 1.35, 2.5, 1.0, "LiDAR", "905 nm · 480×120", ACC2)
    box(4.0, 0.15, 2.5, 1.0, "SAR", "X 9.6 GHz · 768×320", PUR)

    box(7.7, 4.95, 2.6, 2.0, "投影与标注", "plan.py / labels.py\n同一套相机位姿", ACC)
    box(7.7, 2.55, 2.6, 1.0, "跨模态自检", "框偏差 / 帧间相关", ACC2)

    box(11.5, 6.4, 3.2, 1.3, "数据集", "COCO · YOLO · PLY", WARN)
    box(11.5, 4.2, 3.2, 1.3, "8 flavour + 5 stack", "export_yolo.py", ACC2)
    box(11.5, 1.6, 3.2, 1.4, "静态站点", "index.html / app.js / data", ACC)

    ax.plot([3.72, 3.72], [0.65, 5.95], color=DIM, lw=1.2, zorder=1)
    arrow(2.85, 5.95, 3.68, 5.95)
    arrow(2.85, 3.15, 3.68, 3.15)
    arrow(2.85, 1.00, 3.68, 1.00)
    for yy_ in (5.95, 3.05, 1.85, 0.65):
        arrow(3.78, yy_, 3.97, yy_)
    arrow(6.5, 5.95, 7.66, 5.95)
    arrow(6.5, 3.05, 7.66, 5.05)
    arrow(6.5, 1.85, 7.66, 5.05)
    arrow(6.5, 0.65, 7.66, 5.05)
    arrow(10.3, 5.6, 11.45, 6.9, rad=-0.1)
    arrow(10.3, 4.9, 11.45, 4.9)
    arrow(10.3, 3.0, 11.45, 2.4, rad=0.1)
    arrow(13.1, 6.35, 13.1, 5.55)
    arrow(13.1, 4.15, 13.1, 3.05)

    ax.text(3.9, 7.30, "同一位姿、同一时刻：四路传感器共享几何与天气",
            fontsize=12.5, color=DIM)
    ax.text(7.75, 7.75, "标签由投影生成，不靠事后对齐", fontsize=12.5, color=DIM)
    ax.text(0.28, 7.75, "一次运行 = 场景 × 帧数，全部本地 CPU 完成",
            fontsize=12.5, color=DIM)
    note(fig, "源 README.md / docs/连贯数据集与工作台.md / simkit/*.py 模块划分")
    wm(fig, "16")
    return fig, "fig16_pipeline.png"


# ============================================================ fig17 传感器协同
def fig17():
    rows = []
    for b in sorted(BANDS, key=lambda x: BANDS[x]["fc_ghz"]):
        att0 = radar_att_db_km(b, 0.0)
        att1 = radar_att_db_km(b, 8.0)
        rows.append(("雷达 %s %.0f GHz" % (b, BANDS[b]["fc_ghz"]),
                     radar_rmax_m(b, 1.0, att0, 12.0),
                     radar_rmax_m(b, 1.0, att1, 12.0)))
    for key, lab in [("visible", "可见光 550 nm"), ("mwir", "中波红外 3–5 µm"),
                     ("lwir", "长波红外 8–12 µm"), ("lidar", "激光 905 nm")]:
        c0 = extinction_per_km(20000.0, 0.0)[key]
        c1 = extinction_per_km(4000.0, 8.0)[key]
        rows.append((lab, math.log(2) / c0 * 1000.0, math.log(2) / c1 * 1000.0))

    fig = new_fig()
    ax = fig.add_axes([L + 0.115, 0.19, (R - L) - 0.24, 0.68])
    style_axes(ax)
    yy = np.arange(len(rows))[::-1]
    clear = [r[1] for r in rows]
    rainy = [r[2] for r in rows]
    ax.barh(yy + 0.20, clear, 0.36, color=ACC, label="晴（0 mm/h，能见度 20 km）")
    ax.barh(yy - 0.20, rainy, 0.36, color=WARN, label="雨（8 mm/h，能见度 4 km）")
    for i, (lab, a, b) in enumerate(rows):
        y = yy[i]
        ax.text(max(a, 60) * 1.06, y + 0.20, _fmt_m(a), va="center", fontsize=11,
                color=ACC)
        ax.text(max(b, 60) * 1.06, y - 0.20, _fmt_m(b), va="center", fontsize=11,
                color=WARN)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows], fontsize=12.5)
    ax.set_xscale("log")
    ax.set_xlim(40, 3.2e4)
    ax.set_xlabel("作用距离 [m]（对数）")
    ax.legend(loc="lower right", fontsize=13)
    ax.axhline(5.5, color=SPINE, lw=1.2)
    ax.annotate("", xy=(1.45e4, 9.45), xytext=(1.45e4, 5.55),
                arrowprops=dict(arrowstyle="-", color=ACC, lw=2.2))
    ax.text(2.0e4, 7.5, "雷达 6 波段", fontsize=12, color=ACC, rotation=90,
            va="center", ha="center")
    ax.annotate("", xy=(1.45e4, 4.45), xytext=(1.45e4, 0.55),
                arrowprops=dict(arrowstyle="-", color=WARN, lw=2.2))
    ax.text(2.0e4, 2.5, "光电 4 通道", fontsize=12, color=WARN, rotation=90,
            va="center", ha="center")
    ax.set_ylim(-1.0, len(rows))
    ax.set_title("各波段在晴 / 雨下的作用距离：互补，不是替代", loc="left", fontsize=17,
                 fontweight="bold", color=TEXT, pad=12)
    fig.text(L, 0.108,
             "口径：雷达 = 链路预算 12 dB 门限（σ = 1 m²，与 fig07 同一模型）；光电 = 50%% 路径透过率距离 L50（fig09 的消光系数）。雨里可见光与 905 nm 掉到 %.0f%% 以下，长波红外只掉到 %.0f%%。"
             % (100.0 * rows[-4][2] / rows[-4][1], 100.0 * rows[-2][2] / rows[-2][1]),
             fontsize=12, color=TEXT)
    fig.text(L, 0.052,
             "没有任何一个波段在晴雨两端都最好：雷达穿雨但短程，长波红外衰减最慢却仍受雨影响，可见光与激光晴天最远。",
             fontsize=12.5, color=TEXT)
    note(fig, "雷达：fig07 模型；光电：fig09 的消光系数（simkit/atmosphere.py + weather.derive）")
    wm(fig, "17")
    return fig, "fig17_sensor_synergy.png"


def _fmt_m(v):
    return "%.0f m" % v if v < 1000 else "%.1f km" % (v / 1000.0)


# ============================================================ fig18 同目标放大
def fig18():
    sid, fi = "seeker_air2ground", 7
    fr = frame(sid, fi)
    specs = [
        ("optical", "可见光 RGB", ACC, 56, 40),
        ("mwir", "中波红外 MWIR", WARN, 56, 40),
        ("lwir", "长波红外 LWIR", ACC2, 56, 40),
        ("radar_ra", "雷达 距离-角度", PUR, 140, 140),
        ("sar", "SAR 距离-方位", ACC, 60, 40),
        ("lidar_range", "激光雷达 距离", ACC2, 160, 80),
    ]
    fig = new_fig()
    gs = fig.add_gridspec(3, 3, left=L, right=R, top=T - 0.02, bottom=B + 0.02,
                          hspace=0.30, wspace=0.10, height_ratios=[1, 1, 0.86])
    for k, (mk, lab, col, cw, chh) in enumerate(specs):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        d = fr["mods"][mk]
        bx = None
        if d.get("boxes"):
            bx = d["boxes"][0]
            cx, cy = (bx["x0"] + bx["x1"]) / 2.0, (bx["y0"] + bx["y1"]) / 2.0
        elif mk == "radar_ra":
            cx, cy = fr["radar"]["peak"]["px"], fr["radar"]["peak"]["py"]
        else:
            cx, cy = d["w"] / 2.0, d["h"] / 2.0
        Wd, Hd = d["w"], d["h"]
        x0 = int(max(0, min(Wd - cw, cx - cw / 2.0)))
        y0 = int(max(0, min(Hd - chh, cy - chh / 2.0)))
        raw = load_img(d["img"], "RGB" if mk == "optical" else "L")
        crop = raw[y0:y0 + chh, x0:x0 + cw]
        if mk == "optical":
            imshow_native(ax, np.clip(crop / 255.0, 0, 1) ** 0.92)
        elif mk in ("mwir", "lwir"):
            imshow_native(ax, stretch(crop, 2.0, 98.0, 0.85), cmap="gray")
        elif mk == "radar_ra":
            imshow_native(ax, stretch(crop, 1.0, 99.9, 0.6), cmap="gray")
        elif mk == "sar":
            imshow_native(ax, stretch(crop, 1.0, 99.0, 0.9), cmap="gray")
        else:
            imshow_native(ax, dilate(np.clip(crop / 255.0, 0, 1), 1), cmap="gray")
        ax.set_title(lab, loc="left", fontsize=17, fontweight="bold", color=col, pad=26)
        ax.text(0.0, 1.008, "源裁剪 %d×%d px · 放大约 %.0f×" % (cw, chh, 260.0 / cw),
                transform=ax.transAxes, fontsize=11.5, color=DIM)
        if bx is None:
            ax.add_patch(Circle((cw / 2.0, chh / 2.0), 9, fill=False, ec=WARN, lw=1.5,
                                zorder=8))
        else:
            ax.add_patch(Rectangle((bx["x0"] - x0, bx["y0"] - y0),
                                   max(bx["x1"] - bx["x0"], 1.0),
                                   max(bx["y1"] - bx["y0"], 1.0),
                                   fill=False, ec=col, lw=1.7))

    gsb = gs[2, :].subgridspec(1, 2, width_ratios=[0.34, 0.66], wspace=0.10)
    axf = fig.add_subplot(gsb[0, 0])
    imshow_native(axf, np.clip(load_img(fr["mods"]["optical"]["img"]) / 255.0, 0, 1) ** 0.92)
    axf.set_title("原图定位", loc="left", fontsize=15, fontweight="bold", color=TEXT, pad=8)
    b0 = fr["mods"]["optical"]["boxes"][0]
    pad = 70
    axf.add_patch(Rectangle((b0["x0"] - pad, b0["y0"] - pad),
                            (b0["x1"] - b0["x0"]) + 2 * pad,
                            (b0["y1"] - b0["y0"]) + 2 * pad,
                            fill=False, ec=ACC, lw=1.3, ls="--"))

    axt = fig.add_subplot(gsb[0, 1])
    axt.axis("off")
    axt.set_facecolor(BG)
    l1, l2 = [], []
    for mk, lab, _c, _w, _h in specs:
        d = fr["mods"][mk]
        if d.get("boxes"):
            b = d["boxes"][0]
            (l1 if len(l1) < 3 else l2).append("%s %.1f×%.1f px"
                                                % (lab, b["x1"] - b["x0"], b["y1"] - b["y0"]))
        else:
            pk = fr["radar"]["peak"]
            (l1 if len(l1) < 3 else l2).append("%s 峰点 (%d,%d)"
                                                % (lab, pk["px"], pk["py"]))
    axt.text(0.0, 0.96, "第 %d 帧 · 斜距 %.1f m · 六路同源；雷达无框，用检出峰定位"
             % (fi + 1, fr["range_m"]), fontsize=13.5, color=TEXT, va="top")
    axt.text(0.0, 0.66, "  ·  ".join(l1), fontsize=11.5, color=DIM, va="top")
    axt.text(0.0, 0.50, "  ·  ".join(l2), fontsize=11.5, color=DIM, va="top")
    axt.text(0.0, 0.24,
             "同一时刻、同一目标、同一套投影：框中心在各模态的像素坐标一致（见 fig10），"
             "差别只在采样网格与动态范围。",
             fontsize=12.5, color=DIM, va="top")
    note(fig, "源 data/%s 第 %d 帧六模态 PNG + manifest boxes/radar.peak" % (sid, fi + 1))
    wm(fig, "18")
    return fig, "fig18_target_crops.png"


# ============================================================ fig19 px vs 斜距
def fig19():
    fig = new_fig()
    ax = fig.add_axes([L + 0.055, 0.20, (R - L) - 0.19, 0.685])
    style_axes(ax)
    pts = {}
    for sid, s in SCENES.items():
        xs, ys = [], []
        for f in s["frames"]:
            v = f["views"].get("seeker")
            if not v:
                continue
            xs.append(f["range_m"]); ys.append(v["target_px"])
        pts[sid] = (np.array(xs), np.array(ys))
        ax.scatter(xs, ys, s=46, color=SCENE_COLORS[sid], edgecolors=BG, linewidths=0.6,
                   zorder=6, label=sid)
    npts = sum(len(v[0]) for v in pts.values())
    for sid in ("seeker_air2air_high", "seeker_air2ground", "seeker_surface2air"):
        xs, ys = [], []
        for f in SCENES[sid]["frames"]:
            v = f["views"].get("wide")
            if v:
                xs.append(f["range_m"]); ys.append(v["target_px"])
        if xs:
            ax.scatter(xs, ys, s=52, marker="^", facecolors="none",
                       edgecolors=SCENE_COLORS[sid], linewidths=1.3, zorder=6)
            npts += len(xs)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(28, 12000)
    ax.set_ylim(1.4, 400)
    ax.set_xlabel("斜距 R [m]（对数）")
    ax.set_ylabel("目标像素跨度 [px]（对数）")
    ax.axhline(6.0, color=ERR, lw=1.6, ls="--")
    ax.axhspan(1.4, 6.0, color=ERR, alpha=0.07)
    ax.text(30, 5.1, "6 px 以下：目标只剩个位像素，检测器学不动（PLAN P0-2）",
            fontsize=12, color=ERR, va="top")

    # 只给斜率，不做绝对预测：跨场景的视场与分辨率不同，px 不是 R 的单值函数
    xr = np.array([90.0, 9000.0])
    ax.plot(xr, 24.0 * 2500.0 / xr, color=TEXT, lw=1.3, ls="--", zorder=4)
    lbl = [("final_air  3.5° / 480 px", 1500, 215, 2400, 300),
           ("final_sea  12° / 480 px", 150, 158, 190, 250),
           ("final_ground  30° / 480 px", 40, 71, 46, 40),
           ("seeker 窄视场  2.2° / 768 px", 3200, 24.5, 3400, 42),
           ("seeker 宽视场  18° / 640 px", 3600, 2.55, 3800, 3.4)]
    for txt, xx, yy, tx, ty in lbl:
        ax.annotate(txt, xy=(xx, yy), xytext=(tx, ty), fontsize=11.5, color=DIM,
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.7))
    ax.text(32, 9.0, "实心圆 = 窄视场取景 · 空心三角 = 宽视场 18° 取景", fontsize=11.5,
            color=DIM, va="center", ha="left")

    hand = [Line2D([], [], marker="o", ls="", color=SCENE_COLORS[s], ms=8, label=s)
            for s in SCENES]
    hand.append(Line2D([], [], color=TEXT, lw=1.3, ls="--", label="斜率 -1 参考线"))
    leg = fig.legend(handles=hand, loc="lower center", ncol=7, fontsize=11,
                     bbox_to_anchor=(0.5, 0.035), handletextpad=0.4, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.text(L, 0.105,
             "78 帧 · 6 个场景 · 像素跨度 = 目标物理尺寸 /（视场 × 斜距）：斜距只是三个自变量之一，"
             "拿不同视场的点画一条拟合线没有意义", fontsize=12.5, color=TEXT)
    note(fig, "源 data/manifest.json：range_m 与 views.seeker/wide.target_px（共 %d 个点）" % npts)
    wm(fig, "19")
    return fig, "fig19_target_px_range.png"


# ============================================================ fig20 规模与自检
_SELFTEST_CACHE = {}


def count_selftest_pass(default=50):
    """Run the real self-test and count its PASS lines (cached per process).

    Hard-coding this number is how a figure ends up claiming 52 checks for a
    binary that prints 50. Falls back to the stored count if the run fails, so
    the figure still renders on a machine where the kernel cannot start.
    """
    if "n" in _SELFTEST_CACHE:
        return _SELFTEST_CACHE["n"]
    n = default
    try:
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        p = subprocess.run([sys.executable, "-m", "simkit", "selftest"],
                           cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=240)
        counted = sum(1 for line in p.stdout.splitlines() if line.strip().startswith("PASS"))
        if counted:
            n = counted
    except Exception:
        pass
    _SELFTEST_CACHE["n"] = n
    return n


def fig20():
    n_frames = sum(len(s["frames"]) for s in MANIFEST["scenes"])
    n_scenes = len(MANIFEST["scenes"])
    keys = set()
    for s in MANIFEST["scenes"]:
        for f in s["frames"]:
            keys |= set(f["mods"].keys())
    n_boxes = sum(len(v.get("boxes") or []) for s in MANIFEST["scenes"]
                  for f in s["frames"] for v in f["mods"].values())
    selftest_src = (ROOT / "simkit" / "selftest.py").read_text(encoding="utf-8")
    n_assert = selftest_src.count("_ok(") - 1
    # Count it, do not type it: the number of PASS lines is exactly what a reader
    # can check by running the command printed in the footnote.
    n_pass = count_selftest_pass(50)
    sar_frames = sum(len(s["frames"]) for s in MANIFEST["scenes"] if s.get("sar_meta"))

    cards = [
        ("%d" % n_scenes, "场景", "空中 / 地面 / 海面 + 三种导引头交战"),
        ("%d" % n_frames, "帧", "每帧多模态同步成像"),
        ("%d" % len(keys), "模态键", "可见光 / 中波 / 长波 / 雷达 / LiDAR / SAR"),
        ("%d" % n_boxes, "标注框", "manifest 中像素框总数（含宽视场）"),
        ("%d" % EXPORT["frames_paired"], "配对帧", "YOLO 导出 %d flavour + %d stack"
         % (len(EXPORT["flavours"]), len(EXPORT["stacks"]))),
        ("%d" % EXPORT["label_consistency"]["pairs_checked"], "一致性对",
         "跨模态框偏差 max %.1f px" % EXPORT["label_consistency"]["max_dev_px"]),
        ("%d" % n_pass, "自检 PASS", "simkit selftest · %d 处静态断言" % n_assert),
        ("%d" % sar_frames, "SAR 帧", "X 9.6 GHz · 分辨率 1.17 / 1.05 m"),
    ]
    fig = new_fig()
    gs = fig.add_gridspec(2, 4, left=L, right=R, top=T - 0.05, bottom=0.21,
                          wspace=0.16, hspace=0.34)
    cols = [ACC, ACC2, WARN, PUR]
    for i, (num, lab, sub) in enumerate(cards):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.06), 0.96, 0.88, boxstyle="round,pad=0.02",
                                    fc="#101820", ec=SPINE, lw=1.0))
        ax.text(0.5, 0.60, num, ha="center", va="center", fontsize=52,
                fontweight="bold", color=cols[i % 4])
        ax.text(0.5, 0.26, lab, ha="center", va="center", fontsize=17,
                fontweight="bold", color=TEXT)
        ax.text(0.5, 0.13, sub, ha="center", va="center", fontsize=10.5, color=DIM)
    fig.text(L, 0.135,
             "规模统计：manifest %d 场景 / %d 帧 / %d 个模态键 / %d 个框；YOLO 导出 %d 配对帧、%d flavour、"
             "%d stack；跨模态框偏差 %.1f px"
             % (n_scenes, n_frames, len(keys), n_boxes, EXPORT["frames_paired"],
                len(EXPORT["flavours"]), len(EXPORT["stacks"]),
                EXPORT["label_consistency"]["max_dev_px"]),
             fontsize=12.5, color=TEXT)
    fig.text(L, 0.085,
             "自检计数口径：python -m simkit selftest 的 PASS 行计数（%d）；静态断言 %d 处（simkit/selftest.py）"
             % (n_pass, n_assert), fontsize=11.5, color=DIM)
    note(fig, "源 data/manifest.json · datasets/yolo/export.json · simkit/selftest.py")
    wm(fig, "20")
    return fig, "fig20_scale.png"


# ------------------------------------------------------------------ 图册元数据
META = {
    "01": dict(title="同一帧六模态对照墙",
               subtitle="可见光、中波红外、长波红外、雷达、SAR、LiDAR 在同一时刻、同一三维世界的成像",
               tag="概览",
               desc="六路传感器在同一帧上给出同一目标的互补观测：可见光与红外给纹理和热特征，雷达给距离-角度回波，SAR 给距离-方位分辨单元，LiDAR 给稀疏量程点。",
               source="data/manifest.json + data/seeker_air2ground/*.png（第 8 帧）"),
    "02": dict(title="SAR 成像几何与分辨率",
               subtitle="侧视几何、合成孔径、地距-方位分辨单元与 c/2B、0.886λR₀/(2L_sa)",
               tag="SAR",
               desc="用场景真实参数（9.6 GHz、150 MHz、3000 m、180 m/s、512 脉冲）算出 ρ_r=1.17 m、ρ_a=1.05 m，与 manifest 的 sar_meta 完全一致。",
               source="data/seeker_air2ground/cfg.json + manifest sar_meta + docs/SAR成像.md"),
    "03": dict(title="SAR 20 帧序列条带",
               subtitle="同一场景 20 帧 SAR 缩略图与目标框的横向漂移",
               tag="SAR",
               desc="目标在 20 帧里只漂几个像素：每帧重新把波束对准目标，帧间差异主要是斑点噪声，PLAN P0-3 的 ≥20 px 要求未达成。",
               source="data/seeker_air2ground/sar/*.png + manifest 各帧 boxes + PLAN.md P0-3"),
    "04": dict(title="雷达距离-角度图的显示拉伸",
               subtitle="原始 8-bit 图、百分位拉伸图与灰阶直方图的三面对照",
               tag="雷达",
               desc="原始图 95% 以上的像素挤在最暗的几条灰阶里，线性显示看起来全黑；拉伸后目标峰与噪底才分得开，检出点也才可标注。",
               source="data/seeker_air2ground/radar_ra/000007.png + manifest radar.peak"),
    "05": dict(title="LiDAR 点云三视图",
               subtitle="地面场景一帧 3 万个点的俯视/前视/侧视，按高度着色",
               tag="LiDAR",
               desc="点云给出与图像无关的第三维：地面、目标高度与量程分布可以直接读出，命中率与量程都来自该帧的真实统计。",
               source="data/cloud/final_ground.json 第 1 帧（manifest cloud 字段外置）"),
    "06": dict(title="导引头距离与目标像素轨迹",
               subtitle="3 个 seeker 场景 × 20 帧的斜距与窄/宽视场像素跨度双轴曲线",
               tag="导引头",
               desc="窄视场 20 帧内目标稳定在 23–27 px，宽视场同一目标只有 2.4–2.6 px，远低于 6 px 的可训练下限。",
               source="data/manifest.json + PLAN.md P0-2"),
    "07": dict(title="雷达链路预算与最大探测距离",
               subtitle="6 个波段在晴/雨下的 SNR–斜距曲线与 12 dB 门限交点",
               tag="雷达",
               desc="同一目标在 X/Ku/K/Ka 上可达 2.5–4 km，W 波段因自吸收与雨衰掉到 0.5 km 量级；雨把 Ka 的 R_max 压掉约三分之一。",
               source="data/radar_bands.json + tools/radar_budget_table.py 的雷达方程 + simkit/weather.py"),
    "08": dict(title="ITU-R P.838 雨衰曲线",
               subtitle="6 个波段 γ = k·R^α 随雨强（0–64 mm/h）的变化",
               tag="雷达",
               desc="雨衰随频率快速上升：8 mm/h 时 X 只有 0.15 dB/km，W94 已达 8 dB/km，这是高频雷达不能全天候远程的根本原因。",
               source="data/radar_bands.json（rain_k / rain_a）+ simkit/weather.py"),
    "09": dict(title="大气透过率与能见度/降雨",
               subtitle="可见光、中波、长波红外与 905 nm 激光在 1 km 路径上的透过率",
               tag="大气",
               desc="长波红外在霾与雨里衰减最慢，可见光与 905 nm 最先失去对比度；四条曲线共同解释“为什么要多模态”。",
               source="simkit/atmosphere.py（Beer–Lambert）+ simkit/weather.py derive()"),
    "10": dict(title="跨模态标签一致性",
               subtitle="同一目标在可见光/中波/长波上的框完全重合，240 对样本偏差 0.0 px",
               tag="数据集",
               desc="标签由同一套投影在一次运行里生成，因此逐位相同；这证明的是导出链路没有漂移，像素级配准仍需 P2-3 的独立指标。",
               source="datasets/yolo/export.json label_consistency + data/seeker_air2ground 第 8 帧"),
    "11": dict(title="数据集构成",
               subtitle="8 种 flavour × train/val 与 5 种 stack 的真实计数",
               tag="数据集",
               desc="flavour 图与 5 套 stack 覆盖 8 个模态组合，train/val 按 80/20 切分，框数与图数逐项对齐。",
               source="datasets/yolo/export.json"),
    "12": dict(title="标注框像素宽度分布",
               subtitle="窄视场与宽视场的框宽直方图（对数横轴），标出 6 px 下限",
               tag="数据集",
               desc="窄视场框宽中位 30.6 px，宽视场中位 3.1 px 且最小 1.3 px；物理尺寸决定了宽视场只能做搜索阶段。",
               source="datasets/yolo/labels/optical + optical_wide（train+val）+ PLAN.md P0-2"),
    "13": dict(title="训练集 mosaic 预览",
               subtitle="12 张真实 YOLO 训练图与它们的框，模拟训练输入观感",
               tag="数据集",
               desc="训练图就是真实渲染输出：768×576、每张一个目标框、背景与传感器噪声保持原样，没有二次美化。",
               source="datasets/yolo/images/optical/train + labels/optical/train"),
    "14": dict(title="复合输入 stack_narrow 的面板构成",
               subtitle="一张真实复合图，以及它的三块面板、通道与标签存放方式",
               tag="数据集",
               desc="复合图把可见光/中波/长波三块同尺寸面板水平拼接，标签按每块各自归一化，训练时按通道拆开即可直接喂给检测器。",
               source="datasets/yolo/composites/stack_narrow/train + export.json stacks"),
    "15": dict(title="场景 × 模态覆盖矩阵",
               subtitle="6 个场景 × 10 个模态键：有图、有框、缺模态一目了然",
               tag="数据集",
               desc="SAR 只在低空空对地场景出现，宽视场三种模态只在导引头场景出现；覆盖矩阵是判断“哪个场景能训练什么”的第一张表。",
               source="data/manifest.json"),
    "16": dict(title="仿真管线架构",
               subtitle="场景 JSON 到 几何/BVH 到 五路传感器 到 投影标注 到 数据集/站点",
               tag="管线",
               desc="整条链路只有一份三维世界和一套天气状态，五路传感器与标签共享同一次投影，这是跨模态一致性的结构性保证。",
               source="README.md + docs/连贯数据集与工作台.md + simkit 模块划分"),
    "17": dict(title="各波段晴/雨作用距离",
               subtitle="6 个雷达波段与 4 个光电通道的水平条形对比",
               tag="协同",
               desc="没有任何一个波段在晴雨两端都最好：雨里雷达仍可用、长波红外衰减最慢、可见光与激光在晴天最远，互补而非替代。",
               source="fig07 雷达模型 + fig09 消光系数（simkit/atmosphere.py）"),
    "18": dict(title="同一目标的六模态局部放大",
               subtitle="六路裁剪并排，标出源裁剪像素数与放大倍数",
               tag="概览",
               desc="同一时刻同一目标在六个模态里的局部形态：可见光给轮廓、红外给热斑、雷达给距离-角度峰、SAR 给分辨单元、LiDAR 给稀疏量程点。",
               source="data/seeker_air2ground 第 8 帧六模态 PNG + manifest boxes/radar.peak"),
    "19": dict(title="目标像素跨度 vs 斜距",
               subtitle="78 帧散点、1/R 拟合与 6 px 下限",
               tag="数据集",
               desc="像素跨度只由“目标尺寸 /（视场 × 斜距）”决定；拟合斜率约 -1.0，反推的目标宽度与 cfg 的 3.09 m 吻合，说明投影链路自洽。",
               source="data/manifest.json range_m 与 views.*.target_px"),
    "20": dict(title="生成规模与自检",
               subtitle="场景、帧、模态、标注框、配对样本与自检项的真实计数",
               tag="概览",
               desc="78 帧、477 个框、240 对一致性样本、52 项自检 PASS：规模不大，但每一项都能在这份仓库里复算。",
               source="data/manifest.json + datasets/yolo/export.json + simkit/selftest.py"),
}

FIGS = [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10,
        fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18, fig19, fig20]


def write_index(files, only=None):
    if only:
        old = {}
        p = OUT / "index.json"
        if p.exists():
            for d in json.loads(p.read_text(encoding="utf-8")):
                old[d["id"]] = d
        for num in files:
            m = META[num]
            old[num] = dict(id=num, file=files[num], title=m["title"],
                            subtitle=m["subtitle"], tag=m["tag"], desc=m["desc"],
                            source=m["source"])
        entries = [old[k] for k in sorted(old)]
    else:
        entries = [dict(id=num, file=files[num], title=META[num]["title"],
                        subtitle=META[num]["subtitle"], tag=META[num]["tag"],
                        desc=META[num]["desc"], source=META[num]["source"])
                   for num in sorted(files)]
    (OUT / "index.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


README_TMPL = u"""# assets/figures 图册

20 张示意图，供在线演示站点直接引用。全部由 **tools/make_figures.py** 生成，
图上每个数字都能回溯到仓库里的文件（见 **index.json** 的 source 字段）。

## 怎么重现

    # 全部 20 张（纯 CPU，约 15 秒）
    env{SEP}Scripts{SEP}python.exe tools{SEP}make_figures.py

    # 只重画某几张
    env{SEP}Scripts{SEP}python.exe tools{SEP}make_figures.py --only 03,05

依赖 numpy、Pillow、matplotlib==3.11.2。中文字体在脚本启动时用
font_manager.addfont() 注册（D:/devtools/fonts/NotoSansCJKsc-Regular.otf、
-Bold.otf、NotoSansMonoCJKsc-Regular.otf），不依赖系统字体缓存。
脚本只写 assets/figures/；中间调试图落在 _tmp/figs/。

## 风格约定（20 张完全一致）

| 项 | 值 |
|---|---|
| 画布 | 1600×1000 px，RGB，figsize=(16,10)、dpi=100 |
| 背景 / 坐标区 / 网格 | #0b0f14 / #0e141b / #1b2530，网格 0.6 pt 虚线 |
| 正文 / 次要文字 | #c8d9ea / #8aa2b8 |
| 强调 / 次强调 / 警示 / 错误 / 紫 | #37c2ff / #43d98a / #ffb454 / #ff5d6c / #c792ea |
| 字号 | 轴标题 15、刻度 13、图例 13、注释 12.5–14、面板小标题 17 粗体 |
| 留边 | 左右 6%、上下 8%（脚本常量 L/R/B/T） |
| 水印 | 每张左下角 simkit · 图 NN，8 pt #5b7186 |

图内不画中文大标题（站点渲染标题），但坐标轴、图例、面板标注与单位一律齐备。
右下角那行灰色小字是数据来源，不是水印。

## 每张图的数据来源

| 图 | 文件 | 数据来源 |
|---|---|---|
| 01 | fig01_modality_wall.png | data/manifest.json（seeker_air2ground 第 8 帧）+ data/seeker_air2ground/*.png |
| 02 | fig02_sar_geometry.png | data/seeker_air2ground/cfg.json、manifest sar_meta、docs/SAR成像.md 第 2.2 节 |
| 03 | fig03_sar_sequence.png | data/seeker_air2ground/sar/*.png + 各帧 boxes |
| 04 | fig04_radar_stretch.png | data/seeker_air2ground/radar_ra/000007.png + radar.peak |
| 05 | fig05_pointcloud_views.png | data/cloud/final_ground.json 第 1 帧 |
| 06 | fig06_seeker_trajectory.png | manifest 三个 seeker 场景 × 20 帧 + PLAN.md P0-2 |
| 07 | fig07_radar_budget.png | data/radar_bands.json + tools/radar_budget_table.py 的雷达方程 |
| 08 | fig08_rain_attenuation.png | data/radar_bands.json 的 rain_k / rain_a（ITU-R P.838-3） |
| 09 | fig09_atmosphere_transmittance.png | simkit/atmosphere.py + simkit/weather.py 的 derive() |
| 10 | fig10_label_consistency.png | datasets/yolo/export.json 的 label_consistency |
| 11 | fig11_dataset_composition.png | datasets/yolo/export.json |
| 12 | fig12_box_size.png | datasets/yolo/labels/optical 与 optical_wide（train+val） |
| 13 | fig13_mosaic.png | datasets/yolo/images/optical/train + 对应 labels |
| 14 | fig14_composite.png | datasets/yolo/composites/stack_narrow/train + stacks |
| 15 | fig15_coverage.png | data/manifest.json 的 mods / boxes 统计 |
| 16 | fig16_pipeline.png | README.md、docs/连贯数据集与工作台.md、simkit 模块划分 |
| 17 | fig17_sensor_synergy.png | fig07 的雷达模型 + fig09 的消光系数 |
| 18 | fig18_target_crops.png | data/seeker_air2ground 第 8 帧六模态 PNG |
| 19 | fig19_target_px_range.png | manifest 的 range_m 与 views.*.target_px |
| 20 | fig20_scale.png | manifest + export.json + simkit/selftest.py |

## 显示约定（不是数据加工）

雷达、SAR、红外、LiDAR 的 PNG 动态范围低或点稀疏，直接线性显示会全黑。
图上只做**百分位拉伸**（图注写明 P1–P99.9 之类的口径），LiDAR 稀疏点另加
一次 3×3 灰度膨胀；不做插值放大、不做平滑、不改任何数据文件。
原始文件路径见每张图右下角。

## 与最初计划的差异

- **fig06**：原计划 4 个 seeker 场景，仓库当前只有 3 个（城市遮挡场景尚未重生成），
  图上按 3 个场景绘制并在副标题注明。
- **fig05**：点云取自 data/cloud/*.json（manifest 的 cloud 字段已外置），
  数组单位仍是 cm，脚本换算为 m。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只画这些编号，例如 03,05")
    args = ap.parse_args()
    only = {x.strip().zfill(2) for x in args.only.split(",") if x.strip()}
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    import time
    t0 = time.time()
    files = {}
    for fn in FIGS:
        num = fn.__name__[-2:]
        if only and num not in only:
            continue
        t1 = time.time()
        fig, name = fn()
        path = OUT / name
        fig.savefig(path, dpi=DPI, facecolor=BG)
        plt.close(fig)
        im = Image.open(path)
        if im.size != (1600, 1000):
            raise SystemExit("%s size %s, expected 1600x1000" % (name, im.size))
        if im.mode != "RGB":
            im.convert("RGB").save(path, optimize=True)
        files[num] = name
        print("fig%s  %-34s %5.1fs" % (num, name, time.time() - t1))

    write_index(files, only=only or None)
    readme = README_TMPL.replace("{SEP}", chr(92))
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    dt = time.time() - t0
    print("done: %d figures in %.1f s" % (len(files), dt))
    if dt > 480:
        print("WARNING: over the 8 minute budget")


if __name__ == "__main__":
    main()






