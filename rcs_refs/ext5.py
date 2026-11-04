import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("imarest_warship.pdf")
pg = doc[9]
clip = pymupdf.Rect(60, 355, 560, 400)
pix = pg.get_pixmap(matrix=pymupdf.Matrix(5,5), clip=clip)
pix.save("formula.png")
print("saved", pix.width, pix.height)
