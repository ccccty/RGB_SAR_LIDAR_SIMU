import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("marit_rcs.pdf")
for i in [5, 11, 12]:
    t = doc[i].get_text()
    print("=== p%d len=%d ===" % (i+1, len(t)))
    print(t[:2600])
