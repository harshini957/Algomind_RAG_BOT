from app.services.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore

emb = EmbeddingService()
vs  = VectorStore()

# check collection stats first
stats = vs.get_collection_stats()
print(f"parent_chunks : {stats['parent_chunks']}")
print(f"child_chunks  : {stats['child_chunks']}")
print()

# test different BFS phrasings
queries = [
    "Breadth-First Search algorithm and time complexity",
    "BFS graph traversal",
    "breadth first search queue",
    "graph traversal algorithms",
]

for q in queries:
    vec = emb.embed_query(q)
    results = vs.search_children(vec, top_k=5)
    print(f"Query: {q}")
    for r in results:
        print(f"  {r['score']:.4f}  {r['section']}")
    print()