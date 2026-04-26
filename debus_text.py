from pypdf import PdfReader

reader = PdfReader("data/The Algorithm Design Manual by Steven S. Skiena.pdf")

# print first 15 pages — this covers TOC and start of Chapter 1
for i in range(15):
    print(f"\n{'='*60}")
    print(f"PAGE {i+1}")
    print('='*60)
    text = reader.pages[i].extract_text() or ""
    print(text)