import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("davis_irss.pdf")
print("pages", doc.page_count)
for i in range(min(6, doc.page_count)):
    print("--- p%d ---" % (i+1))
    print(doc[i].get_text()[:2200])
