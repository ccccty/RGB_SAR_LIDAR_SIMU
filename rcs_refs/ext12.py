import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("emcos.pdf")
pg = doc[0]
print(pg.rect)
d = pg.get_text("dict")
ys = []
for b in d["blocks"]:
    for l in b.get("lines", []):
        line = "".join(s["text"] for s in l["spans"])
        if any(k in line for k in ["Inshore","Warship","Medium tanker","Container carrier","Ship RCS Table"]):
            print("%.1f %r" % (l["bbox"][1], line[:80]))
