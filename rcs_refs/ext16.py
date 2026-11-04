import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("marit_rcs.pdf")
for i in [5, 11, 12]:
    print("=== p%d ===" % (i+1))
    print(doc[i].get_text(extraction_mode="text")[:3000])
