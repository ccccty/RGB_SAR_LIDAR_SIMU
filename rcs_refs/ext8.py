import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("emcos.pdf")
pg = doc[1]
for name, rect, z in [("emcos_formula.png", pymupdf.Rect(195, 235, 420, 300), 10),
                      ("emcos_formula2.png", pymupdf.Rect(195, 290, 420, 330), 10)]:
    pix = pg.get_pixmap(matrix=pymupdf.Matrix(z,z), clip=rect)
    pix.save(name); print(name, pix.width, pix.height)
