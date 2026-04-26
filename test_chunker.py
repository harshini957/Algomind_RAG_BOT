from app.core.chunker import DoclingChunker

chunker = DoclingChunker(
    child_max_tokens=200,
    child_overlap=20,
    source="Algorithm_Design_Manual.pdf"
)

parents, children = chunker.chunk("C:\\Users\\91702\\Desktop\\Naive_rag\\cs_rag\\data\\The Algorithm Design Manual by Steven S. Skiena.pdf")

print("\n--- PARENTS (first 5) ---")
for p in parents[:5]:
    print(f"  chapter : {p.chapter}")
    print(f"  pages   : {p.page_start} → {p.page_end}")
    print(f"  words   : {len(p.text.split())}")
    print()

print("--- CHILDREN (first 5) ---")
for c in children[:5]:
    print(f"  section     : {c.section}")
    print(f"  chapter     : {c.chapter}")
    print(f"  parent_id   : {c.parent_id}")
    print(f"  page        : {c.page}")
    print(f"  chunk_index : {c.chunk_index}")
    print(f"  words       : {len(c.text.split())}")
    print()