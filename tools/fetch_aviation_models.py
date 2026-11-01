"""Download official NASA airborne-science aircraft, with source hashes.

The NASA media usage guidelines apply; these are not labelled CC0.
Run tools/prepare_aviation_models.py through Blender after downloading.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "aviation"
BASE = "https://airbornescience.nasa.gov/3d-models/"
MODELS = {"airliner": "B777_LARC", "uav": "SIERRA_ARC"}
EXPECTED = {
    "airliner": "3e6bc7b5cb9140be72b588a035ebb85ccc118d6d295cee8a25f3df5d835fd035",
    "uav": "594fd04140612231d0f7b6776dc7ed322eb06b767f53760742fdc8836154f95b",
}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for role, model in MODELS.items():
        url = BASE + "models/" + model + "_AIR_1026.glb"
        path = DEST / (role + ".glb")
        if not path.exists():
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read()
            if data[:4] != b"glTF":
                raise ValueError("not a GLB: " + url)
        else:
            data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != EXPECTED[role]:
            raise ValueError("source checksum changed for %s; review upstream version before using" % role)
        if not path.exists():
            path.write_bytes(data)
        records.append(dict(role=role, model=model, url=url,
                            source_page=BASE + "?id=" + model,
                            sha256=digest, bytes=len(data), verified_on="2026-11-01",
                            terms="https://www.nasa.gov/nasa-brand-center/images-and-media/",
                            usage="Academic graphical simulation; credit NASA Airborne Science Program. "
                                  "NASA identifiers/trademarks remain protected; no endorsement implied."))
        print(role, len(data), records[-1]["sha256"], flush=True)
    (DEST / "sources.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
