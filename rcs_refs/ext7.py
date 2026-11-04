import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("emcos.pdf")
pg = doc[1]
print(pg.rect)
d = pg.get_text("dict")
for b in d["blocks"]:
    for l in b.get("lines", []):
        line = "".join(s["text"] for s in l["spans"])
        if "Empirical" in line or "Expression" in line:
            for s in l["spans"]:
                print("span org=(%.1f,%.1f) size=%.1f %r" % (s["origin"][0], s["origin"][1], s["size"], s["text"]))
pix = pg.get_pixmap(matrix=pymupdf.Matrix(6,6))
pix.save("emcos_p2.png")
print("page img", pix.width, pix.height)
