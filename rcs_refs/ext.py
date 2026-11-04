import io, sys
from pypdf import PdfReader
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for name in ["emcos.pdf","imarest_warship.pdf"]:
    print("\n######### " + name)
    try:
        rd = PdfReader(name)
        for i,pg in enumerate(rd.pages):
            t = pg.extract_text() or ""
            print("--- page %d ---" % (i+1))
            print(t[:2500])
    except Exception as e:
        print("ERR", e)
