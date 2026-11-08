#!/usr/bin/env python
"""Adversarial injection suite for the trimodal seeker acceptance gate.

The red team listed four failure families: config drift, modality misalignment,
oversized targets and unstable generation.  A gate that always passes is not a
gate, so every family gets an injected fault and MUST be rejected by
tools/validate_trimodal.py.  This is the "fix coverage" cross-check: it proves
the acceptance contract detects the faults it claims to detect.

Pure-static for config cases (mutate a copy of the scene JSON in memory and run
check_config); the oversize case uses the shipped demo_sea_surface run whose
243 px ship failed the visibility contract.  No rendering, no repo writes.

Usage
    python tools/attack_surface_check.py
"""
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from validate_trimodal import check_config, check_visibility  # noqa: E402


def load(key):
    with open(os.path.join(ROOT, "scenes", "ds20_v2_%s.json" % key),
              encoding="utf-8") as fh:
        return json.load(fh)


def run_config_gate(cfg, tmp_path):
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False)
    checks = check_config(None, tmp_path)
    return [c["check"] for c in checks if not c["ok"]]


def config_cases(key="air_plane"):
    # Run the same injections against BOTH geometry branches: a ground/sea
    # look_down scene needs three同源 plate keys, a sky-only scene needs two
    # and must NOT carry a ground.backdrop (that produced 51% dead-black LWIR).
    base = load(key)
    out = []

    def case(name, mutate):
        c = copy.deepcopy(base)
        mutate(c)
        out.append((name, c))

    case("0x0 尺寸静默回退",
         lambda c: c["clip"]["views"][0].update(width=0, height=0))
    case("sensors.optical 与 clip.views 不一致",
         lambda c: c["sensors"]["optical"].update(width=1024, height=768, fov_deg=6.0))
    case("target_px 门禁被改回 34–68 px",
         lambda c: c["clip"]["seeker"].update(target_px=[34.3, 68.7]))
    case("range_m 脱离 target_px 反解带",
         lambda c: c["clip"]["seeker"].update(range_m=[800, 1200]))
    case("抖动恢复 scenegen 默认 0.35·fov",
         lambda c: c["clip"]["seeker"].update(wander_deg=2.8, los_jitter_deg=0.04))
    case("rig.vibration_deg 非零",
         lambda c: c["clip"]["rig"].update(vibration_deg=0.05))
    case("雷达距离窗不足",
         lambda c: c["sensors"]["radar"].update(n_samples=32))
    case("pfa 与 det_snr_db 双门限",
         lambda c: c["sensors"]["radar"].update(det_snr_db=12.0))
    case("20 帧 / 单 clip 被破坏",
         lambda c: c["clip"].update(frames_per_clip=10))
    case("output.yolo 污染仓库",
         lambda c: c["output"].update(yolo=True))
    case("红外回到双波段 + 大 PSF",
         lambda c: c["ir"].update(bands=["mwir", "lwir"], psf_px=1.3))
    case("实拍背景被移除（退回程序化）",
         lambda c: (c["optical"].pop("sky_image", None),
                    c["ir"].pop("background_image", None),
                    c["ground"].pop("backdrop", None)))
    case("背景不同源",
         lambda c: c["ir"].update(
             background_image=str(c["optical"]["sky_image"]) + ".bak"))
    if key == "air_plane":
        case("无地面场景错挂 ground.backdrop（红外半幅死黑）",
             lambda c: c["ground"].update(
                 backdrop=str(c["optical"]["sky_image"]),
                 backdrop_span_m=400.0, backdrop_span_y_m=300.0))
    case("真实性开关被关回默认",
         lambda c: c.update(atmosphere={"profile": "homogeneous"}))
    return out


def oversize_case():
    """A 243x112 px ship must fail the barely-visible visibility gate."""
    run = os.path.join(ROOT, "out", "demo_sea_surface")
    if not os.path.isdir(run):
        return None
    checks, _ = check_visibility(run)
    return [c["check"] for c in checks if not c["ok"]]


def stability_cases():
    """Seed propagation must be visible in the actual side-car written."""
    out = []
    for key in ("air_plane", "ground_vehicle", "sea_ship"):
        run = os.path.join(r"D:\work\多模态数据集生成\dataest", key)
        sp = os.path.join(run, "_render_cfg.json")
        mp = os.path.join(run, "meta.json")
        if not (os.path.isfile(sp) and os.path.isfile(mp)):
            out.append((key, None, "missing"))
            continue
        with open(sp, encoding="utf-8") as fh:
            side = json.load(fh)
        with open(mp, encoding="utf-8") as fh:
            meta = json.load(fh)
        cfg = load(key)
        ok = side.get("seed") == cfg.get("seed") == meta.get("seed")
        out.append((key, ok, "sidecar=%s meta=%s cfg=%s"
                    % (side.get("seed"), meta.get("seed"), cfg.get("seed"))))
    return out


def main():
    tmp = os.path.join(ROOT, "_tmp", "_attack_cfg.json")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    print("== 配置偏差注入（每个必须被门禁拒绝） ==")
    passed = failed = 0
    for key in ("air_plane", "ground_vehicle"):
        for name, cfg in config_cases(key):
            bad = run_config_gate(cfg, tmp)
            ok = bool(bad)
            passed += ok
            failed += (not ok)
            print("  [%s] %-16s %-36s -> %s"
                  % ("PASS" if ok else "FAIL", key, name,
                     ", ".join(bad[:2]) if bad else "未被拦截"))
    print("\n== 目标过大注入（243x112 px 舰船） ==")
    over = oversize_case()
    if over is None:
        print("  SKIP: out/demo_sea_surface 不存在")
    else:
        ok = bool(over)
        passed += ok
        failed += (not ok)
        print("  [%s] 真实框宽门禁 -> %s" % ("PASS" if ok else "FAIL",
                                              ", ".join(over[:3]) if over else "未被拦截"))
    print("\n== 生成稳定性：seed 透传到 Blender side-car ==")
    for key, ok, detail in stability_cases():
        if ok is None:
            print("  [SKIP] %-16s %s" % (key, detail))
            continue
        passed += bool(ok)
        failed += (not ok)
        print("  [%s] %-16s %s" % ("PASS" if ok else "FAIL", key, detail))
    try:
        os.remove(tmp)
    except OSError:
        pass
    print("\n攻击注入：%d 项被正确拦截，%d 项漏检" % (passed, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
