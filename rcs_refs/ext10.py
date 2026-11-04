import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("nps_visby.pdf")
for i in range(6, doc.page_count):
    print("--- p%d ---" % (i+1))
    print(doc[i].get_text()[:1200])
print("\n\n=========== TNO")
doc2 = pymupdf.open("tno_ship.pdf")
for i in range(min(8, doc2.page_count)):
    print("--- p%d ---" % (i+1))
    print(doc2[i].get_text()[:1500])
