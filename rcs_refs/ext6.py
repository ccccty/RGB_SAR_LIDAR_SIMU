import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("emcos.pdf")
print("pages", doc.page_count)
for i,pg in enumerate(doc):
    print("\n--- page %d ---" % (i+1))
    print(pg.get_text()[:2500])
