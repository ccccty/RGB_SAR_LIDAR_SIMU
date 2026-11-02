"""Download and normalize the vehicle asset library (rotary/fixed-wing UAVs,
helicopters, civil & military aircraft, cars, armour, warships, ships).

Two stages, mirroring tools/fetch_aviation_models.py:
  1. this script fetches the curated GLBs and writes a Blender manifest;
  2. tools/prepare_vehicles.blender.py runs inside Blender to normalize and
     decimate them into assets/models/<category>/<chinese name>.obj.

The curated list below is the source of truth: id comes from poly.pizza, and
length_m is the real-world body length the model is scaled to. axis="major"
takes the longest horizontal extent as the body axis (vehicles, ships);
axis="minor" takes the shortest (aircraft, where the wingspan dominates).

Run:  python tools/fetch_vehicle_models.py            # download only
      python tools/fetch_vehicle_models.py --convert  # + run Blender if present
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "assets", "_cache", "vehicles")
BLENDER = os.path.join(ROOT, "third_party", "blender", "blender.exe")
UA = {"User-Agent": "Mozilla/5.0 (compatible; simkit-asset-fetch/1.0)"}

# (poly.pizza id, category, chinese name, real body length m, axis)
CURATION = [
    # 旋翼无人机
    ("DNbUoMtG3H", "旋翼无人机", "四旋翼无人机", 0.35, "major"),
    ("3Ae_y67lzvd", "旋翼无人机", "四旋翼无人机改型", 0.42, "major"),
    ("dJ9mjQQqDQJ", "旋翼无人机", "小型四旋翼", 0.25, "major"),
    ("6CUQX98vha4", "旋翼无人机", "多旋翼无人机", 0.6, "major"),
    ("dCOUTcu89u6", "旋翼无人机", "无人机机体", 0.45, "major"),
    ("fyB_cRYLcUY", "旋翼无人机", "航拍无人机", 0.3, "major"),
    # 固定翼无人机
    ("3Eio09miiAF", "固定翼无人机", "中空长航时无人机", 8.22, "minor"),
    # 直升机
    ("EQJ2MECUbx", "直升机", "轻型直升机", 9.0, "minor"),
    ("cTzINMr0WdS", "直升机", "通用直升机", 12.0, "minor"),
    ("eb7b31pjGtQ", "直升机", "运输直升机", 15.0, "minor"),
    ("hG2Qr0A3zR", "直升机", "武装直升机", 14.5, "minor"),
    # 民航飞机
    ("a3XrQkLNna9", "民航飞机", "双发窄体客机", 37.6, "minor"),
    ("8ciDd9k8wha", "民航飞机", "四发宽体客机", 70.0, "minor"),
    ("8VysVKMXN2J", "民航飞机", "支线客机", 27.0, "minor"),
    ("9Ev6pklkSYp", "民航飞机", "涡桨支线机", 22.0, "minor"),
    ("2eG17I-VDiG", "民航飞机", "货运飞机", 45.0, "minor"),
    # 军用飞机
    ("6fyLMORhgGK", "军用飞机", "单发战斗机", 15.0, "minor"),
    ("3B3Pa6BHXn1", "军用飞机", "双发战斗机", 19.0, "minor"),
    ("dukcCKsLDrS", "军用飞机", "多用途战斗机", 16.0, "minor"),
    ("bgUY8zN2Bq9", "军用飞机", "高速截击机", 18.0, "minor"),
    ("9iVI9GHMleJ", "军用飞机", "双翼飞机", 7.9, "minor"),
    # 地面车辆
    ("75h3mi6uHuC", "地面车辆", "家用轿车", 4.5, "major"),
    ("eRu2kPYOCa7", "地面车辆", "紧凑型轿车", 4.3, "major"),
    ("Cz6yDaUcM9", "地面车辆", "低多边形轿车", 4.4, "major"),
    ("unqqkULtRU", "地面车辆", "经济型轿车", 4.2, "major"),
    ("BG0KAhmGDt", "地面车辆", "两厢轿车", 4.0, "major"),
    ("1mkmFkAz5v", "地面车辆", "跑车", 4.4, "major"),
    ("xsMtZhBkxL", "地面车辆", "运动型多用途车", 4.8, "major"),
    ("qn4grQgHm8", "地面车辆", "皮卡", 5.4, "major"),
    ("4CPpvEmrMoF", "地面车辆", "城市公交", 12.0, "major"),
    ("aT_24cDaW1a", "地面车辆", "厢式货车", 5.4, "major"),
    # 军用地面车辆
    ("FA5daiyZQq", "军用地面车辆", "主战坦克", 9.8, "major"),
    ("jWS1CLA0RO", "军用地面车辆", "主战坦克改型", 9.5, "major"),
    ("YWSyTmvGI2", "军用地面车辆", "中型坦克", 8.0, "major"),
    ("41Tq_Kf0Tui", "军用地面车辆", "轻型坦克", 6.6, "major"),
    ("uYHpj7lz1J", "军用地面车辆", "坦克", 8.0, "major"),
    ("Ebryot9iKM", "军用地面车辆", "高机动越野车", 4.6, "major"),
    ("y8lBpvMlim", "军用地面车辆", "军用运输卡车", 7.6, "major"),
    # 军舰
    ("aYAmfkUYZjg", "军舰", "战列舰", 220.0, "major"),
    ("cqV6mUkn7Ow", "军舰", "战列舰改型", 250.0, "major"),
    ("wouBxOe3CD", "军舰", "导弹艇", 25.0, "major"),
    ("8PgPdFGg3MO", "军舰", "常规潜艇", 70.0, "major"),
    ("dfSQmeuuYt3", "军舰", "攻击核潜艇", 110.0, "major"),
    # 民用船舶
    ("yq9EKmEmfC", "民用船舶", "豪华游轮", 300.0, "major"),
    ("dgLCxDWhnZQ", "民用船舶", "游轮", 240.0, "major"),
    ("aOikPp4rz68", "民用船舶", "大型游轮", 260.0, "major"),
    ("3AmDGcCu6Ll", "民用船舶", "集装箱船", 200.0, "major"),
    ("9znU2c-dTSR", "民用船舶", "游艇", 28.0, "major"),
    ("6okvxHsSdzO", "民用船舶", "帆船", 12.0, "major"),
    ("cIzO4MBPqI", "民用船舶", "远洋帆船", 45.0, "major"),
    ("5UEl54KsuC", "民用船舶", "小型渔船", 15.0, "major"),
    ("2vKA7JD0JU", "民用船舶", "远洋邮轮", 269.0, "major"),
]

LICENSE_URL = {"CC0": "https://creativecommons.org/publicdomain/zero/1.0/",
               "CC-BY": "https://creativecommons.org/licenses/by/4.0/"}


def fetch(url, timeout=180, tries=5):
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            last = exc
    raise IOError("download failed: %s (%s)" % (url, last))


def page_meta(mid):
    """Scrape title/author/license/glb url from a poly.pizza model page."""
    import re
    html = fetch("https://poly.pizza/m/" + mid, timeout=60).decode("utf-8", "ignore")
    title = ""
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
    if m:
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip()
    author = ""
    m = re.search(r"Free 3D Model By ([A-Za-z0-9_.\- ]{2,40}?)\s*-\s*Poly Pizza", html)
    if m:
        author = m.group(1).strip()
    if re.search(r"publicdomain/zero", html):
        lic = "CC0"
    elif re.search(r"creativecommons\.org/licenses/by", html):
        lic = "CC-BY"
    else:
        lic = "CC0" if "CC0" in html else "UNKNOWN"
    glb = ""
    m = re.search(r"(https://static\.poly\.pizza/[0-9a-fA-F-]+\.glb)", html)
    if m:
        glb = m.group(1)
    return dict(title=title, author=author, license=lic, glb=glb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true", help="run Blender to build the OBJs")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    entries, meta = [], []
    for idx, (mid, cate, cn, length, axis) in enumerate(CURATION, 1):
        raw = os.path.join(CACHE, mid + ".glb")
        info = page_meta(mid)
        if info["license"] == "UNKNOWN" or not info["glb"]:
            raise SystemExit("cannot resolve license/file for %s" % mid)
        if not os.path.exists(raw) or os.path.getsize(raw) < 1024:
            data = fetch(info["glb"])
            if data[:4] != b"glTF":
                raise SystemExit("not a GLB: %s" % mid)
            with open(raw, "wb") as fh:
                fh.write(data)
        entries.append(dict(glb=raw, obj=os.path.join(CACHE, "obj", mid + ".obj"),
                            length_m=length, axis=axis, tri_target=3500))
        meta.append(dict(id=mid, category=cate, name_cn=cn, length_m=length,
                         license=info["license"], author=info["author"],
                         source_title=info["title"],
                         source_page="https://poly.pizza/m/" + mid,
                         glb_url=info["glb"]))
        print("[fetch] %-12s %-10s %-12s %s" % (mid, cate, cn, info["license"]), flush=True)

    manifest = os.path.join(CACHE, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(CACHE, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print("[fetch] %d models cached in %s" % (len(entries), CACHE))

    if not args.convert:
        return
    script = os.path.join(ROOT, "tools", "prepare_vehicles.blender.py")
    if not os.path.exists(BLENDER):
        raise SystemExit("Blender not found at %s; run stage 2 manually" % BLENDER)
    cmd = [BLENDER, "-b", "--factory-startup", "-P", script, "--", manifest]
    print("[fetch] running Blender ...", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())