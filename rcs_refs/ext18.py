import sys, pymupdf, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open("trawler.pdf")
print("== TRAWLER pages", doc.page_count)
for i,pg in enumerate(doc):
    t = pg.get_text()
    if re.search(r"[Ll]ength|Beam|Draft|Draught|Tonnage|GT", t):
        print("--- p%d ---" % (i+1)); print(t[:1200])
print("== DAVIS IR numeric lines")
d2 = pymupdf.open("davis_irss.pdf")
for i in [3,4,5]:
    t = d2[i].get_text()
    for line in t.split("\n"):
        if re.search(r"°C|deg C|contrast|ambient|temperature", line, re.I) and len(line) < 200:
            print("p%d: %s" % (i+1, line.strip()))
