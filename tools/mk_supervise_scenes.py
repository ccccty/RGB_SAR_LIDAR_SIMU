"""生成三域监督场景（空/陆/海），供 tools/supervise.py 使用。

固定素材与种子，保证每一轮监督比较的是同一条基线。
"""

import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simkit import scenegen

KINDS = [
    ("air", "sky", "军用飞机/双发战斗机.obj", "fighter", "sky"),
    ("ground", "ground", "军用地面车辆/主战坦克.obj", "vehicle", "city"),
    ("sea", "sea", "军舰/导弹艇.obj", "ship", "port"),
]


def build(tag, kind, asset, cls, cat, span=300.0):
    r = scenegen.default_recipe()
    r["name"] = "sup_%s" % tag
    r["environment"].update(kind=kind, weather="clear",
                            background={"mode": "library", "category": cat})
    r["objects"] = [dict(asset=asset, cls=cls, scale=1.0, count=1,
                         uv=[0.45, 0.5])]
    r["view"].update(aspect="top_down", ground_span_m=span, fov_deg=8.0)
    r["capture"].update(sensors=["optical", "ir", "radar"], frames=1, seed=7)
    return scenegen.build_scene(scenegen.validate_recipe(r))


def main():
    out = os.path.join(ROOT, "_tmp")
    os.makedirs(out, exist_ok=True)
    for tag, kind, asset, cls, cat in KINDS:
        cfg = build(tag, kind, asset, cls, cat)
        path = os.path.join(out, "sup_%s.json" % tag)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=1)
        print("  已写 %s" % path)


if __name__ == "__main__":
    main()