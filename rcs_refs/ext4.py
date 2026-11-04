import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("imarest_warship.pdf")
pg = doc[9]
# dump spans with font size + origin around the formula
d = pg.get_text("dict")
for b in d["blocks"]:
    for l in b.get("lines", []):
        line = "".join(s["text"] for s in l["spans"])
        if "52" in line or "displacement" in line or "formula" in line:
            print("LINE:", repr(line))
            for s in l["spans"]:
                print("   span size=%.1f origin=(%.1f,%.1f) font=%s text=%r" % (s["size"], s["origin"][0], s["origin"][1], s["font"], s["text"]))
