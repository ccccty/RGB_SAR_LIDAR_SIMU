import sys
from pypdf import PdfReader
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rd = PdfReader("imarest_warship.pdf")
for i,pg in enumerate(rd.pages):
    if i < 3: continue
    t = pg.extract_text() or ""
    print("--- page %d ---" % (i+1))
    print(t[:2600])
