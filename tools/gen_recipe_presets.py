"""Generate the curated workbench recipe presets (2026-11-09).

18 recipes = 3 domains (air / land / sea) x 3 real-photo backdrops x 2 ECM
groups (active / passive). They land in configs/recipes/, which is exactly the
directory the workbench lists through GET /api/recipes and loads through
POST /api/recipe/load -- no front-end registration table to maintain.

Baseline is the verified 海陆空1109v1 look: real photographed plates, 5 frames,
optical + LWIR + radar RD, ~30 px small targets, a light civilian vehicle on a
real road, a wide plate tile so the REPEAT seam leaves the frame, and a
measured-scale RCS instead of the geometric PO estimate.

Usage
    env\\Scripts\\python.exe tools\\gen_recipe_presets.py
    env\\Scripts\\python.exe tools\\gen_recipe_presets.py --check   # validate only
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPES = os.path.join(ROOT, "configs", "recipes")
sys.path.insert(0, ROOT)

from simkit import scenegen  # noqa: E402


# ---------------------------------------------------------------------------
# Domain baselines, transcribed from the delivered v1 scene configs.
# ---------------------------------------------------------------------------
DOMAINS = {
    "air": dict(
        label="空", kind="sky", asset="军用飞机/双发战斗机.obj", cls="fighter",
        scale=0.84, rcs_db=5.0, aspect="pursuit", range_km=2.8, fov_deg=8.0,
        closing_mps=48.0, plate_span=None, sky_card=None,
        uv=None, ground_span_m=None, offset=None,
        note="v1 空域基线：2.8 km / 8° 窄视场 pursuit，目标 28–34 px。"),
    "land": dict(
        label="陆", kind="ground", asset="地面车辆/厢式货车.obj", cls="vehicle",
        scale=1.25, rcs_db=8.0, mat="paint_car_white", aspect="top_down",
        range_km=0.65, fov_deg=10.0, closing_mps=12.0,
        # 400 m is the FRAME; the plate tile is 3x wider so an off-centre
        # photographic centre never drags a REPEAT seam into the picture.
        plate_span=1200.0, sky_card=False,
        # The target is placed at the frame centre (uv 0.5, 0.5); the PLATE is
        # shifted so that the photographed point (0.20, 0.45) -- a straight
        # multi-lane road with lane markings and parked traffic -- lands there.
        # Without the shift the bright grass island at plate centre dominates
        # the frame and the van ends up on grass.
        uv=[0.5, 0.5], ground_span_m=400.0, offset=[0.20, 0.45],
        note="v1 陆域基线：6.75 m 白色厢式货车、8 dBsm、1200 m 宽底板（画面仅 "
             "400 m）消除平铺接缝；正俯视 + 底板偏移把目标钉在多车道沥青上。"),
    "sea": dict(
        label="海", kind="sea", asset="军舰/导弹艇.obj", cls="ship",
        scale=1.42, rcs_db=20.0, aspect="look_down", range_km=2.2,
        fov_deg=12.0, closing_mps=12.0, plate_span=480.0, sky_card=False,
        uv=None, ground_span_m=None, offset=[0.5, 0.5],
        note="v1 海域基线：2.2 km / 12° 斜俯视、480 m 平铺；海面纹理均匀，"
             "不写天空卡避免盖住底板。"),
}

# 3 distinct photographed plates per domain. `file` is a basename inside
# assets/backdrops/<category>/, which is what scenegen._pick_backdrop expects.
BACKDROPS = {
    "air": [
        dict(key="skyclear", label="高空晴空", category="sky",
             file="02_10000_Ft_jpg.jpg", weather="clear"),
        dict(key="cumulus", label="积云层", category="sky",
             file="01_-i---i-_30651918381_jpg.jpg", weather="cloudy"),
        dict(key="overcast", label="航拍云顶", category="sky",
             file="05_2016-08_Paris_Montreal_flight_08_jpg.jpg",
             weather="cloudy"),
    ],
    "land": [
        dict(key="city", label="城市环岛道路", category="city",
             file="06_Drone_view_of_roundabout_Unsplash.jpg", weather="clear"),
        dict(key="rural", label="乡村田野公路", category="rural",
             file="06_2011-08-17_13-24-50_Serbia_Popinci_jpg.jpg",
             weather="clear"),
        dict(key="industrial", label="工业区停车场", category="industrial",
             file="03_Mechelen_aerial_jpg.jpg", weather="cloudy"),
    ],
    "sea": [
        dict(key="open", label="开阔海面", category="open",
             file="03_Sea_wave_abstract_texture_Unsplash.jpg", weather="clear"),
        dict(key="port", label="集装箱港口", category="port",
             file="01_Aerial_photograph_of_the_Port_of_Miami_Container.jpg",
             weather="cloudy"),
        dict(key="island", label="近岸岛礁", category="island",
             file="03_Phi_Phi_From_Above_Unsplash.jpg", weather="clear"),
    ],
}

MODES = ("active", "passive")
MODE_LABEL = {"active": "有源干扰", "passive": "无源干扰"}


# ---------------------------------------------------------------------------
# ECM payloads. Every field below is either a per-kind default from
# simkit.ecm.DEFAULTS or one of the front end's COMMON keys, so loading a
# preset into the workbench and re-submitting it keeps every parameter.
# ---------------------------------------------------------------------------
def _active_entities(domain):
    out = [dict(
        id="jammer_01", kind="noise_jammer", erp_w=200.0,
        jam_bandwidth_mhz=200.0, mode="barrage", band="Ku",
        look_at_sensor=True, sidelobe_db=-25.0, off_band_factor=0.05,
        rcs_m2=0.2, visible=True,
        note="支援干扰机：宽带噪声压制，抬高雷达噪声底"),
        dict(
            id="drfm_01", kind="false_target", count=4,
            range_offset_m=150.0, range_step_m=210.0, power_ratio=1.6,
            band="Ku", visible=True,
            note="DRFM 转发式距离假目标：4 个假距离门")]
    third = {
        "air": dict(id="rgpo_01", kind="range_gate_pull_off", count=1,
                    rate_mps=80.0, max_offset_m=300.0, power_ratio=1.2,
                    band="Ku", visible=True,
                    note="距离波门拖引：把跟踪门从真目标上拖开"),
        "land": dict(id="angle_01", kind="angle_deception", count=1,
                     angle_offset_deg=0.5, power_ratio=1.0, band="Ku",
                     visible=True,
                     note="角度欺骗：对地面跟踪雷达制造测角偏置"),
        "sea": dict(id="vgpo_01", kind="velocity_gate_pull_off", count=1,
                    rate_mps=8.0, max_offset_mps=60.0, power_ratio=1.2,
                    band="Ku", visible=True,
                    note="速度波门拖引：针对反舰雷达的多普勒跟踪"),
    }[domain]
    out.append(third)
    return out


def _passive_entities(domain):
    out = [dict(
        id="chaff_01", kind="chaff_cloud", radius_m=2.0, growth_mps=4.0,
        rcs_m2=100.0, decay_s=6.0, dilution_exponent=2.0,
        fall_mps=1.5, wind=[6.0, 0.0, 0.0], aspect=[1.0, 1.0, 1.0],
        band="Ku", visible=True,
        note="箔条云：体散射无源干扰，不遮挡目标"),
        dict(
            id="corner_01", kind="corner_reflector", size_m=0.3,
            rcs_m2=None, band="Ku", visible=True,
            note="0.3 m 三面角反射器：Ku 波段约 28.8 dBsm 的强点假目标"),
        dict(
            id="decoy_01", kind="decoy", size_m=0.3, rcs_m2=0.5,
            tether_m=80.0, band="Ku", visible=True,
            note="拖曳式无源诱饵，随目标滞后运动")]
    n_flares = 6 if domain == "air" else 2
    for i in range(n_flares):
        r = round(0.18 + 0.02 * (i % 4), 3)
        out.append(dict(
            id="flare_%02d" % (i + 1), kind="infrared_flare",
            radius_m=r, trail_m=round(2.4 + 0.3 * (i % 4), 2),
            temp_c=2000.0, decay_s=3.0, rcs_m2=0.02,
            mat="flare_pellet", band="Ku", visible=True,
            note="抛射式红外诱饵：手-sized 发光弹 + 短尾迹，远小于载机"))
    return out


# ---------------------------------------------------------------------------
def build_recipe(domain_key, bg, mode, seed):
    d = DOMAINS[domain_key]
    obj = dict(asset=d["asset"], cls=d["cls"], scale=d["scale"], count=1)
    if d.get("mat"):
        obj["mat"] = d["mat"]
    obj["rcs_db"] = d["rcs_db"]
    if d["uv"]:
        obj["uv"] = list(d["uv"])

    background = dict(mode="library", category=bg["category"], file=bg["file"])
    if d["plate_span"] is not None:
        background["span_m"] = d["plate_span"]
    if d["sky_card"] is not None:
        background["sky_card"] = d["sky_card"]
    if d["offset"]:
        background["offset"] = list(d["offset"])

    view = dict(range_km=d["range_km"], aspect=d["aspect"],
                fov_deg=d["fov_deg"], views="single",
                width=768, height=576, wide_width=768, wide_height=576,
                closing_mps=d["closing_mps"])
    if d["ground_span_m"]:
        view["ground_span_m"] = d["ground_span_m"]

    name = "p_%s_%s_%s" % (domain_key, bg["key"], mode)
    ents = (_active_entities(domain_key) if mode == "active"
            else _passive_entities(domain_key))
    return {
        "name": name,
        "title": "%s · %s · %s ｜ 5帧 × 三模态" % (
            d["label"], bg["label"], MODE_LABEL[mode]),
        "note": d["note"] + " 干扰组：" + (
            "噪声压制 + DRFM 假目标 + 域专属欺骗" if mode == "active" else
            "箔条云 + 角反射器 + 拖曳诱饵 + 红外诱饵弹"),
        "tags": [domain_key, bg["category"], mode, "v1-baseline"],
        "environment": dict(kind=d["kind"], weather=bg["weather"],
                            background=background),
        "objects": [obj],
        "ecm": dict(mode=mode, entities=ents),
        "view": view,
        "capture": dict(sensors=["optical", "ir", "radar"], frames=5,
                        fps=10.0, seed=seed),
        # Real photo plates already contain their own sun/lamp highlights;
        # the synthetic glare stage would double them into white blobs.
        "effects": dict(plume=True, supersample=2, flare=False),
        "output": dict(preview=True, coco=True, yolo=True, readme=True),
        "_saved_at": "preset",
    }


def jwrite(path, obj):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate existing presets without writing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite presets that already exist")
    args = ap.parse_args()

    os.makedirs(RECIPES, exist_ok=True)
    names = []
    for di, domain_key in enumerate(("air", "land", "sea")):
        for bi, bg in enumerate(BACKDROPS[domain_key]):
            plate = os.path.join(scenegen.BACKDROP_DIR, bg["category"],
                                 bg["file"])
            if not os.path.isfile(plate):
                raise SystemExit("缺少背景板：" + plate)
            for mi, mode in enumerate(MODES):
                seed = 1000 + di * 100 + bi * 10 + mi
                rec = build_recipe(domain_key, bg, mode, seed)
                names.append(rec["name"])
                path = os.path.join(RECIPES, rec["name"] + ".json")
                if not args.check:
                    if os.path.isfile(path) and not args.force:
                        raise SystemExit("已存在（用 --force 覆盖）：" + path)
                    jwrite(path, rec)
                # Validate through the real pipeline, not just JSON syntax.
                norm = scenegen.validate_recipe(rec)
                if norm["ecm"]["mode"] != mode:
                    raise SystemExit("%s: ecm.mode 被改写" % rec["name"])
                print("%-26s %-8s %-12s %-9s ents=%d" % (
                    rec["name"], DOMAINS[domain_key]["label"],
                    bg["label"], MODE_LABEL[mode], len(norm["ecm"]["entities"])))

    print("\n%d 个预设配方%s；目录 %s" % (
        len(names), "（仅校验）" if args.check else " 已写入", RECIPES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
