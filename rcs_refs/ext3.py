import sys
from pypdf import PdfReader
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rd = PdfReader("imarest_warship.pdf")
t = rd.pages[9].extract_text(extraction_mode="layout") or ""
idx = t.find("52")
print(repr(t[max(0,idx-800):idx+800]))
print("=======FULL PAGE10 raw text mode")
print(repr(rd.pages[9].extract_text())[:4000])
