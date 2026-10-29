"""Merge several simkit runs into one COCO dataset and write a summary report.

    python tools\\merge_dataset.py out\\dataset out\\final_air out\\final_ground out\\final_sea

Every modality that produced an annotations_{mod}_coco.json is merged, so adding
a sensor to the simulator needs no change here. Image ids and annotation ids are
offset per run and per modality so the merged files stay valid COCO.
"""
import json
import os
import shutil
import sys


def load(p):
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def mods_of(run):
    """Modality name -> source image directory, discovered from the run itself."""
    found = {}
    for fn in sorted(os.listdir(run)):
        if not (fn.startswith("annotations_") and fn.endswith(".json")):
            continue
        mod = fn[len("annotations_"):-len("_coco.json")] if fn.endswith("_coco.json") \
            else fn[len("annotations_"):-len(".json")]
        if mod == "optical":
            found[mod] = "images/optical"
        elif mod.startswith("ir_"):
            found[mod] = "images/" + mod
        elif mod == "lidar":
            found[mod] = "lidar/range"
        else:
            found[mod] = "images/" + mod
    return found


def main():
    out_root = sys.argv[1]
    runs = sys.argv[2:]
    if not runs:
        print("usage: merge_dataset.py <out_root> <run> [run...]")
        return 1
    os.makedirs(out_root, exist_ok=True)

    merged = {}
    scene_index = []
    for run in runs:
        name = os.path.basename(os.path.normpath(run))
        meta = load(os.path.join(run, "meta.json")) if os.path.exists(os.path.join(run, "meta.json")) else {}
        stats = load(os.path.join(run, "stats.json")) if os.path.exists(os.path.join(run, "stats.json")) else {}
        per_mod = {}
        for mod, img_rel in mods_of(run).items():
            src = os.path.join(run, "annotations_%s_coco.json" % mod)
            if not os.path.exists(src):
                continue
            coco = load(src)
            target = merged.setdefault(mod, dict(
                info=dict(description="simkit merged (%s)" % mod, version="1.0"),
                images=[], annotations=[], categories=[]))
            cat_map = {c["name"]: c["id"] for c in target["categories"]}
            for c in coco.get("categories", []):
                if c["name"] not in cat_map:
                    nid = len(target["categories"]) + 1
                    target["categories"].append(dict(id=nid, name=c["name"],
                                                     supercategory=c.get("supercategory", "target")))
                    cat_map[c["name"]] = nid
            off_img = (max([i["id"] for i in target["images"]], default=0) + 1) if target["images"] else 1
            off_ann = (max([a["id"] for a in target["annotations"]], default=0) + 1) if target["annotations"] else 1
            local_cat = {c["id"]: c["name"] for c in coco.get("categories", [])}
            id_map = {}
            for img in coco.get("images", []):
                base = os.path.basename(img["file_name"])
                new_id = off_img + img["id"]
                id_map[img["id"]] = new_id
                target["images"].append(dict(
                    id=new_id, file_name="%s/%s/%s" % (name, img_rel, base),
                    width=img["width"], height=img["height"]))
            for a in coco.get("annotations", []):
                if a["image_id"] not in id_map:
                    continue
                nm = local_cat.get(a.get("category_id"))
                if nm is None or nm not in cat_map:
                    continue
                target["annotations"].append(dict(a, id=off_ann, image_id=id_map[a["image_id"]],
                                                  category_id=cat_map[nm]))
                off_ann += 1
            dst_dir = os.path.join(out_root, name, img_rel)
            os.makedirs(dst_dir, exist_ok=True)
            for img in coco.get("images", []):
                base = os.path.basename(img["file_name"])
                s = os.path.join(run, img_rel, base)
                if os.path.exists(s):
                    shutil.copy2(s, os.path.join(dst_dir, base))
            per_mod[mod] = dict(images=len(coco.get("images", [])),
                                annotations=len(coco.get("annotations", [])))
        scene_index.append(dict(name=name, stats=stats, modalities=per_mod,
                                meta_keys=sorted(meta.keys())))

    for k, v in sorted(merged.items()):
        with open(os.path.join(out_root, "coco_%s.json" % k), "w", encoding="utf-8") as fh:
            json.dump(v, fh, ensure_ascii=False, indent=1)
        print("%-14s images=%4d anns=%4d cats=%s" % (
            k, len(v["images"]), len(v["annotations"]),
            [c["name"] for c in v["categories"]]))

    with open(os.path.join(out_root, "scenes.json"), "w", encoding="utf-8") as fh:
        json.dump(scene_index, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
