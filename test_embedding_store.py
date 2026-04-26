import logging
logging.basicConfig(level=logging.INFO)

from app.core.chunker import DoclingChunker
from app.services.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore

PDF = "data/The Algorithm Design Manual by Steven S. Skiena.pdf"

# 1. Chunk
chunker = DoclingChunker(source=PDF)
parents, children = chunker.chunk(PDF)

# 2. Embed only first chapter's children to test quickly
test_children = [c for c in children if c.chapter_num == 1]
print(f"\nEmbedding {len(test_children)} children from Chapter 1...")

emb_service = EmbeddingService()
embeddings = emb_service.embed_documents([c.text for c in test_children])
print(f"Embedding shape: {len(embeddings)} vectors x {len(embeddings[0])} dims")

# 3. Store
vs = VectorStore()
test_parents = [p for p in parents if p.chapter_num == 1]
vs.store_parents(test_parents)
vs.store_children(test_children, embeddings)

# 4. Test retrieval
print("\nTesting search...")
query_vec = emb_service.embed_query("What is the nearest neighbor heuristic?")
results = vs.search_children(query_vec, top_k=3)

for r in results:
    print(f"\n  score   : {r['score']:.4f}")
    print(f"  section : {r['section']}")
    print(f"  chapter : {r['chapter']}")
    print(f"  page    : {r['page']}")
    print(f"  text    : {r['text'][:120]}...")

# 5. Test parent expansion
parent_ids = list({r['parent_id'] for r in results})
parents_fetched = vs.get_parents_by_ids(parent_ids)
print(f"\nParent expansion: fetched {len(parents_fetched)} parent(s)")
for p in parents_fetched:
    print(f"  chapter : {p['chapter']}")
    print(f"  words   : {len(p['text'].split())}")

# 6. Stats
stats = vs.get_collection_stats()
print(f"\nCollection stats: {stats}")