import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for name in ["shipjournal_rcs.pdf","nps_visby.pdf"]:
    doc = pymupdf.open(name)
    print("\n######### " + name + " pages=" + str(doc.page_count))
    for i,pg in enumerate(doc):
        t = pg.get_text()
        print("--- p%d ---" % (i+1))
        print(t[:2000])
        if i >= 5: break
