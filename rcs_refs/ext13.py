import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("emcos.pdf")
pg = doc[0]
pix = pg.get_pixmap(matrix=pymupdf.Matrix(4.2,4.2), clip=pymupdf.Rect(40, 495, 560, 765))
pix.save("rcs_table.png"); print(pix.width, pix.height)
