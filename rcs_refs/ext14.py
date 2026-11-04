import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for name in ["mitll_rcs.pdf","marit_rcs.pdf"]:
    try:
        doc = pymupdf.open(name)
        print("\n######## " + name + " pages=" + str(doc.page_count))
        for i in range(min(doc.page_count, 8)):
            t = doc[i].get_text()
            if any(k in t for k in ["ship","Ship","boat","m2","m 2"]):
                print("--- p%d ---" % (i+1)); print(t[:1600])
    except Exception as e:
        print("ERR", name, e)
