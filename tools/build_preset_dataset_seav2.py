"""seav2: 海域真实感数据集生成器.

针对第一轮 sea 预设的三个真实性缺陷：

1. 舰船骑在岛上 -- look_down 的取景中心 UV 落在了岛屿照片的陆地上，世界原点
   （目标位置）对应照片里的山坡，船"开到了山上"。seav2 逐一在三张海图上重选
   落水取景中心（船与尾迹都必须在水面像素内）。
2. 舰船模型与尺寸 -- 导弹艇换成战列舰改型轮廓并缩到 28 m（约 37 px，原 45 px），
   一艘 20 kn 的快速攻击艇。
3. 没有尾迹 -- 按帧位姿把程序生成的 Kelvin 尾迹（tools/make_wake_texture.py，
   19.47° 楔形 + 横波弧 + 中心湍流）直接烘进底板照片。光学（Cycles）与红外
   （底板采样器）读同一张图，两个模态自动一致，渲染管线零改动。

另外修复红外预览的双标尺问题（simkit/ir.py）：底板灰度与 AGC 预览混用让热目标
在明亮照片前读成黑色剪影。

用法:
    PYTHONPATH=$PWD python tools/build_preset_dataset_seav2.py \
        --dest "D:/work/多模态数据集生成/dataest/1110/seav2" \
        --frames 30 --jobs 4 --threads 5 --force
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import build_preset_dataset_1110 as base          # noqa: E402
from make_wake_texture import wake_alpha, KELVIN_TAN  # noqa: E402

DEST_DEFAULT = r"D:\work\多模态数据集生成\dataest\1110\seav2"

# 落水取景中心（裁剪底板的照片坐标 u 右 / v 下）。相机足迹 714x803 m 占 900 m
# 底板的八成，无平铺接缝的自由度只有 off_u∈[0.35,0.55] / v0∈[0.50,0.61]；
# 在这个带内由水面掩膜搜索选点，使船位与 300 m 尾迹走廊都在水面上。
# 底板用裁剪版（去天空/去码头陆地），保证取景窗内没有照片的天空带。
SEAV2_UV = {
    "03_Sea_wave_abstract_texture_Unsplash.jpg": (0.50, 0.52),   # 全水原片
    "seav2_phi_phi_crop.jpg": (0.39, 0.52),     # 裁剪 u0..0.7 v0.30 起：海湾进安全带、无地平线
    "seav2_miami_crop.jpg": (0.40, 0.55),
}

# 配方引用原始照片 -> seav2 裁剪版（同目录）。临时换表，用完恢复，
# 不影响 1110 构建器对原始照片的引用。open 片全水无需裁剪。
SEAV2_PLATES = {
    "03_Phi_Phi_From_Above_Unsplash.jpg": "seav2_phi_phi_crop.jpg",
    "01_Aerial_photograph_of_the_Port_of_Miami_Container.jpg": "seav2_miami_crop.jpg",
}

# 尾随视角（reverse_course）：尾迹拖在船后、朝向相机一侧的开阔水域。
# island 的配方航向（sign=-1，目标迎面开来）会把尾迹甩向照片上方的岛体；
# 翻转后航向 ≈ 64°±漂移，尾迹拖向左下湾内深水。open 全水无差异，统一翻转
# 保持六个场景的构图一致（尾随视角下尾迹在船与相机之间，最完整）。
REVERSE_COURSE = True

SHIP_MODEL = "军舰/战列舰改型.obj"
SHIP_LEN_M = 28.0                # 快速攻击艇量级，画幅里 ~37 px
SHIP_SCALE = SHIP_LEN_M / 250.7  # 战列舰改型 自然长度 250.7 m
WAKE_LEN_M = 300.0


def bake_wake(photo_path, out_path, ship_xy, heading_deg, speed_mps,
              span_m, off_u, v_photo, opacity=0.92):
    """把 Kelvin 尾迹按帧位姿合成到底板照片上，返回输出路径与诊断信息.

    ship_xy: 船的世界坐标 (x, y)（取片段中点，尾迹随中点对齐使首尾误差各半）
    底板映射（与 ir._plate_view_ground / render_optical 一致）:
        col = (x/span + off_u) * iw
        row = (v_photo - y/span) * ih      # v 轴翻转后的照片行
    """
    im = Image.open(photo_path).convert("RGB")
    iw, ih = im.size
    sx, sy = float(ship_xy[0]), float(ship_xy[1])
    psi = math.radians(float(heading_deg))
    cos_p, sin_p = math.cos(psi), math.sin(psi)
    gsdu = span_m / iw                       # u 向米/像素
    gsdv = span_m / ih                       # v 向米/像素

    ship_col = (sx / span_m + off_u) * iw
    ship_row = (v_photo - sy / span_m) * ih

    # 尾迹影响的世界包围盒：沿航向 [-0.5L-wake, +1.5L]，横向 ±wake*tan*1.35
    along_min, along_max = -(WAKE_LEN_M + 0.5 * SHIP_LEN_M), 1.5 * SHIP_LEN_M
    half_w = WAKE_LEN_M * KELVIN_TAN * 1.35
    # 世界包围盒的四个角（旋转轴对齐包围盒）
    corners = []
    for a in (along_min, along_max):
        for c in (-half_w, half_w):
            corners.append((sx + a * cos_p - c * sin_p,
                            sy + a * sin_p + c * cos_p))
    xs = [sx] + [c[0] for c in corners]
    ys = [sy] + [c[1] for c in corners]
    col0 = max(0, int((min(xs) / span_m + off_u) * iw) - 2)
    col1 = min(iw, int((max(xs) / span_m + off_u) * iw) + 3)
    row0 = max(0, int((v_photo - max(ys) / span_m) * ih) - 2)
    row1 = min(ih, int((v_photo - min(ys) / span_m) * ih) + 3)
    if col1 <= col0 or row1 <= row0:
        return out_path, dict(skipped="wake bbox outside the plate")

    cols = (np.arange(col0, col1) + 0.5) / iw
    rows = (np.arange(row0, row1) + 0.5) / ih
    # 每个像素的世界坐标（米）：u 由列决定，v 由行决定
    wx = np.repeat(((cols - off_u) * span_m)[None, :], rows.size, axis=0)
    wy = np.repeat(((v_photo - rows) * span_m)[:, None], cols.size, axis=1)
    dx = wx - sx
    dy = wy - sy
    along = dx * cos_p + dy * sin_p
    across = -dx * sin_p + dy * cos_p

    px_per_m = 2.5
    rgba, info = wake_alpha(WAKE_LEN_M, speed_mps, SHIP_LEN_M,
                            px_per_m=px_per_m, seed=11)
    sh, sw = rgba.shape[:2]
    # 船中心对应 sprite 列：船头在 sprite 的 u_stern + 0.45L 处
    col_s = info["u_stern_px"] + (along + 0.45 * SHIP_LEN_M) * px_per_m
    row_s = info["h_px"] / 2.0 + across * px_per_m
    c0 = np.clip(np.round(col_s).astype(np.int64), -1, sw)
    r0 = np.clip(np.round(row_s).astype(np.int64), -1, sh)
    valid = (c0 >= 0) & (c0 < sw) & (r0 >= 0) & (r0 < sh)
    a = np.zeros(along.shape, np.float32)
    a[valid] = rgba[r0[valid], c0[valid], 3].astype(np.float32) / 255.0
    a *= float(opacity)

    arr = np.asarray(im, np.float32)
    foam = np.array([236.0, 239.0, 243.0], np.float32)
    arr[row0:row1, col0:col1] = (
        arr[row0:row1, col0:col1] * (1.0 - a[:, :, None])
        + foam[None, None, :] * a[:, :, None])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(out_path)
    return out_path, dict(info=info, ship_px=[round(ship_col, 1), round(ship_row, 1)],
                          lam_m=round(info["lam"], 1), froude=round(info["froude"], 2))


# 每场景的水面掩膜缓存（键 = 变体底板路径）
_MASK_CACHE = {}


def _ok_water_mask(photo_path):
    """水面（且非浪花/雾霾/沙滩）掩膜，用于船位与尾迹走廊的落水判定."""
    if photo_path in _MASK_CACHE:
        return _MASK_CACHE[photo_path]
    a = np.asarray(Image.open(photo_path).convert("RGB"), np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    water = (b > r * 1.05) & (sat > 0.10) & (mx > 0.15)
    surf = (mx > 0.72) & (sat < 0.30)          # 浪花/沙滩
    haze = (mx > 0.60) & (sat < 0.25)          # 地平线雾霾/云
    ok = water & ~surf & ~haze
    _MASK_CACHE[photo_path] = ok
    return ok


# 足迹包围盒（900 m 底板分数，4 角保守值，含 wander 余量）：取景偏移必须让
# 足迹完全落在底板内，否则 IR 采样器 mod 1.0 平铺会出现接缝。
FP_U_MIN, FP_U_MAX = -0.347, +0.447      # x = -312..+402 m
FP_V_MIN, FP_V_MAX = -0.502, +0.391      # y = +452..-352 m


def solve_scene_uv(photo_path, ship_xy, heading_deg):
    """按场景求解 off_u / v0：船位与尾迹走廊落水、足迹不出底板.

    seeker 的 LOS 扫掠让轨道整体偏离世界原点最多 ~200 m（各场景不同），
    固定 UV 会把个别场景的船放在礁石上。规划完成后船的真实位置/航向已知，
    在水面掩膜上反解取景偏移即可。镜像变体扩大了可达水域。
    """
    ok = _ok_water_mask(photo_path)
    ih, iw = ok.shape
    sx, sy = float(ship_xy[0]), float(ship_xy[1])
    rad = math.radians(float(heading_deg) + 180.0)      # astern
    ax_, ay_ = math.cos(rad), math.sin(rad)
    px_, py_ = -ay_, ax_

    best = None
    for off_u in np.arange(0.36, 0.55, 0.01):
        for v0 in np.arange(0.51, 0.60, 0.01):
            # 足迹不出底板
            if not (0.005 <= off_u + FP_U_MIN and off_u + FP_U_MAX <= 0.995):
                continue
            if not (0.005 <= v0 + FP_V_MIN and v0 + FP_V_MAX <= 0.995):
                continue
            su, sv = off_u + sx / 900.0, v0 - sy / 900.0    # 船的照片分数
            cu, cv = int(su * iw), int(sv * ih)
            if not (4 <= cu < iw - 4 and 4 <= cv < ih - 4):
                continue
            shipok = ok[cv - 4:cv + 5, cu - 4:cu + 5].mean()
            if shipok < 0.9:
                continue
            # 尾迹走廊（300 m, 横向 ±150 m）：以船的世界坐标为基准
            wake = 0.0
            n = 0
            for t in np.linspace(15, WAKE_LEN_M, 14):
                for w_ in np.linspace(-150, 150, 7):
                    x = sx + t * ax_ + w_ * px_
                    y = sy + t * ay_ + w_ * py_
                    uu2 = off_u + x / 900.0
                    vv2 = v0 - y / 900.0
                    if 0 <= uu2 < 1 and 0 <= vv2 < 1:
                        wake += ok[int(vv2 * ih), int(uu2 * iw)]
                    n += 1
            wake /= n
            if wake < 0.80:
                continue
            # 足迹内非水占比（越低越好，用网格近似）
            score = 0.55 * wake + 0.45 * shipok
            if best is None or score > best[0]:
                best = (round(float(score), 3), round(float(off_u), 3),
                        round(float(v0), 3), round(float(wake), 2),
                        round(float(shipok), 2))
    return best    # (score, off_u, v0, wake_frac, ship_frac) 或 None


def build_scene_seav2(recipe, frames):
    """配方 -> seav2 场景配置（换模型、规划、按场景解 UV）."""
    saved_override = dict(base.PLATE_OVERRIDE)
    base.PLATE_OVERRIDE.update(SEAV2_PLATES)
    if REVERSE_COURSE:
        recipe = json.loads(json.dumps(recipe))    # never mutate the preset
        recipe.setdefault("view", {})["reverse_course"] = True
    try:
        cfg, applied = base.scene_from_recipe(recipe, frames, "sea")
    finally:
        base.PLATE_OVERRIDE = saved_override

    # The wake is baked into the photograph in PHOTO coordinates; the optical
    # ground mapping must therefore sample the plate WITHOUT the random
    # rotation/offset it applies to unaligned backdrops (scenegen only sets
    # backdrop_aligned for top_down, which left the sea band rotated -- the
    # baked wake then appeared somewhere else entirely in the optical band,
    # while the IR plate sampler, which has no rotation, stayed correct).
    cfg["ground"]["backdrop_aligned"] = True

    # 舰船：换模型 + 缩小
    prim = cfg["objects"][0]
    prim["model"] = os.path.join(base.ROOT, "assets", "models", SHIP_MODEL)
    prim["scale"] = round(SHIP_SCALE, 4)
    applied["ship"] = [SHIP_MODEL, SHIP_LEN_M]

    # 帧位姿 -> 中点船位 + 航向 + 航速（航向由 seeker LOS 状态给出，
    # reverse_course 已在配方层翻转；尾迹永远拖在速度反方向）
    from simkit import plan as planlib
    prim_name = prim["name"]
    states = list(planlib.plan_frames(cfg, frames, int(cfg.get("seed") or 0),
                                      float(cfg["frames"]["dt"])))
    mid = states[frames // 2]
    pose = mid["poses"][prim_name]
    sx, sy = float(pose["pos"][0]), float(pose["pos"][1])
    yaw = float(pose.get("yaw") or 0.0)
    vel = np.asarray(pose.get("velocity_mps") or [0, 0, 0], float)
    speed = float(math.hypot(vel[0], vel[1]))

    # 按场景解 UV：正片 + 镜像两个变体，取分高者
    bg_path = str(cfg["ground"]["backdrop"])
    stem, ext = os.path.splitext(bg_path)
    mirror_path = stem + "_mirror" + ext
    if not os.path.isfile(mirror_path):
        Image.open(bg_path).convert("RGB").transpose(
            Image.FLIP_LEFT_RIGHT).save(mirror_path, quality=92)
    solved = solve_scene_uv(bg_path, (sx, sy), yaw)
    variant, tag = bg_path, "normal"
    if solved is None:
        solved = solve_scene_uv(mirror_path, (sx, sy), yaw)
        variant, tag = mirror_path, "mirror"
    if solved is None:
        raise ValueError("seav2: %s 找不到落水取景组合（ship=%.0f,%.0f heading=%.0f）"
                         % (prim_name, sx, sy, yaw))
    score, off_u, v0, wake_frac, ship_frac = solved
    applied["plate_uv_photo"] = [off_u, v0]
    applied["plate_variant"] = tag
    applied["uv_solve"] = dict(score=score, wake_water=wake_frac,
                               ship_water=ship_frac)

    return cfg, applied, dict(ship_xy=(sx, sy), yaw=yaw, speed=speed,
                              photo_path=variant, prim=prim_name,
                              off_u=off_u, v0=v0)


def main():
    import argparse
    import subprocess
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEST_DEFAULT)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="逗号分隔的场景名单")
    ap.add_argument("--validate", type=int, default=0, metavar="N",
                    help="最小验证：按完整帧数规划/烘焙（UV 求解与最终版一致），"
                         "但每场景只渲染前 N 帧")
    args = ap.parse_args()

    dest = args.dest
    os.makedirs(dest, exist_ok=True)
    names = (args.only.split(",") if args.only else
             ["p_sea_open_active", "p_sea_open_passive",
              "p_sea_island_active", "p_sea_island_passive",
              "p_sea_port_active", "p_sea_port_passive"])

    scenes = []
    pool = []
    for name in names:
        recipe = base.jload(os.path.join(base.ROOT, "configs", "recipes",
                                         "%s.json" % name))
        cfg, applied, wk = build_scene_seav2(recipe, args.frames)

        out_dir = os.path.join(dest, name)
        scene_path = os.path.join(dest, "scenes", "%s.json" % name)
        os.makedirs(os.path.dirname(scene_path), exist_ok=True)

        # 烘尾迹（底板照片 -> seav2/plates）；off_u/v0 来自按场景求解
        plate_out = os.path.join(dest, "plates", "%s.png" % name)
        photo = wk["photo_path"]
        off_u, v0 = wk["off_u"], wk["v0"]
        out, diag = bake_wake(photo, plate_out, wk["ship_xy"], wk["yaw"],
                              wk["speed"], 900.0, off_u, v0)
        cfg["ground"]["backdrop_offset"] = [round(off_u, 4),
                                            round(1.0 - v0, 6)]
        cfg["ground"]["backdrop"] = out
        if isinstance(cfg.get("ir") or {}, dict) and cfg["ir"].get("background_image"):
            cfg["ir"]["background_image"] = out
        cfg["optical"]["camera"]["flare"]["enabled"] = False
        applied["wake"] = {k: v for k, v in diag.items() if k != "info"}
        base.jwrite(scene_path, cfg)
        applied["ship_pos_mid"] = [round(v, 1) for v in wk["ship_xy"]]
        applied["heading_deg"] = applied.get("heading_deg", wk["yaw"])
        scenes.append(dict(name=name, run=os.path.relpath(out_dir, dest).replace("\\", "/"),
                           applied=applied))
        pool.append((name, scene_path, out_dir))

    ok_all = True
    running = []
    for name, scene_path, out_dir in pool:
        if os.path.isdir(out_dir) and not args.force:
            print("SKIP (存在，用 --force 覆盖):", name)
            continue
        t0 = time.time()
        render_n = args.validate if args.validate else args.frames
        base.run_gen(scene_path, out_dir, render_n, args.threads)
        cfg = base.jload(scene_path)
        res = base.verify_scene(out_dir, cfg, render_n)
        base.stamp_optical_meta(out_dir, cfg, render_n)
        if not args.validate:
            base.contact_sheet(out_dir, name, render_n)
        ok = bool(res["passed"])
        ok_all = ok_all and ok
        chk = res["checks"]
        print("[%s] %-26s optical=%d/%d ir=%d/%d rd=%d/%d npy=%d/%d coco=%d/%d  (%.0f s)"
              % ("PASS" if ok else "FAIL", name,
                 res["optical"], render_n, res["ir_lwir"], render_n,
                 res["radar_png"], render_n, res["radar_npy"], render_n,
                 res["coco_anns"], render_n, time.time() - t0))
        if not ok:
            print("      failed checks:", [k for k, v in chk.items() if not v])

    # 索引 + 报告 + README
    idx = dict(
        dataset="seav2 海域真实感数据集（Kelvin 尾迹 / 落水取景 / 28 m 战舰轮廓）",
        frames=(args.validate or args.frames),
        scenes=[dict(name=s["name"], run=s["run"], applied=s["applied"])
                for s in scenes],
        passed=ok_all)
    base.jwrite(os.path.join(dest, "index.json"), idx)
    base.jwrite(os.path.join(dest, "generation_report.json"),
                dict(passed=ok_all, scenes=len(pool),
                     frames_per_scene=args.frames, per_scene=scenes))
    print("\nOVERALL %s -> %s" % ("PASS" if ok_all else "FAIL", dest))


if __name__ == "__main__":
    main()
