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


def build_scene_seav2(recipe, frames):
    """配方 -> seav2 场景配置（换模型、落水 UV、烘尾迹）."""
    # seav2 的落水取景中心与裁剪底板都与 base 不同：临时换表再恢复
    saved_uv = dict(base.PLATE_UV)
    saved_override = dict(base.PLATE_OVERRIDE)
    base.PLATE_UV.update(SEAV2_UV)
    base.PLATE_OVERRIDE.update(SEAV2_PLATES)
    if REVERSE_COURSE:
        recipe = json.loads(json.dumps(recipe))    # never mutate the preset
        recipe.setdefault("view", {})["reverse_course"] = True
    try:
        cfg, applied = base.scene_from_recipe(recipe, frames, "sea")
    finally:
        base.PLATE_UV = saved_uv
        base.PLATE_OVERRIDE = saved_override

    bg_path = str(cfg["ground"]["backdrop"])
    plate_name = os.path.basename(bg_path)
    uv = SEAV2_UV.get(plate_name)
    if not uv:
        raise ValueError("seav2: 底板 %s 没有落水取景点" % plate_name)
    applied["plate_uv_photo"] = [float(uv[0]), float(uv[1])]

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

    return cfg, applied, dict(ship_xy=(sx, sy), yaw=yaw, speed=speed,
                              photo_path=bg_path, prim=prim_name)


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

        # 烘尾迹（底板照片 -> seav2/plates）
        plate_out = os.path.join(dest, "plates", "%s.png" % name)
        photo = wk["photo_path"]
        # 场景里的底板已经被 abs_asset 指到绝对路径
        out, diag = bake_wake(photo, plate_out, wk["ship_xy"], wk["yaw"],
                              wk["speed"], 900.0,
                              cfg["ground"]["backdrop_offset"][0],
                              1.0 - cfg["ground"]["backdrop_offset"][1])
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
        base.run_gen(scene_path, out_dir, args.frames, args.threads)
        cfg = base.jload(scene_path)
        res = base.verify_scene(out_dir, cfg, args.frames)
        base.stamp_optical_meta(out_dir, cfg, args.frames)
        base.contact_sheet(out_dir, name, args.frames)
        ok = bool(res["passed"])
        ok_all = ok_all and ok
        chk = res["checks"]
        print("[%s] %-26s optical=%d/%d ir=%d/%d rd=%d/%d npy=%d/%d coco=%d/%d  (%.0f s)"
              % ("PASS" if ok else "FAIL", name,
                 res["optical"], args.frames, res["ir_lwir"], args.frames,
                 res["radar_png"], args.frames, res["radar_npy"], args.frames,
                 res["coco_anns"], args.frames, time.time() - t0))
        if not ok:
            print("      failed checks:", [k for k, v in chk.items() if not v])

    # 索引 + 报告 + README
    idx = dict(
        dataset="seav2 海域真实感数据集（Kelvin 尾迹 / 落水取景 / 28 m 战舰轮廓）",
        frames=args.frames,
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
