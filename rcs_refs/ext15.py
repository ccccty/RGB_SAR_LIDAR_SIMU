import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("marit_rcs.pdf")
for i in range(6, doc.page_count):
    t = doc[i].get_text()
    print("--- p%d ---" % (i+1)); print(t[:1500])
