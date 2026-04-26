import logging
logging.basicConfig(level=logging.INFO)

from app.services.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore
from app.services.ingestion_service import IngestionService

PDF = "data/The Algorithm Design Manual by Steven S. Skiena.pdf"

# first clear existing collections so we start fresh
from qdrant_client import QdrantClient
from app.config import settings

print("Clearing existing Qdrant collections...")
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
client.delete_collection(settings.parent_collection)
client.delete_collection(settings.child_collection)
print("Collections cleared.")

# wire up services
emb_service = EmbeddingService()
vs = VectorStore()          # recreates collections fresh
svc = IngestionService(
    embedding_service=emb_service,
    vector_store=vs,
)

# run full ingestion
result = svc.ingest_pdf(PDF)

print("\n--- INGESTION RESULT ---")
for k, v in result.items():
    print(f"  {k}: {v}")

# verify counts
stats = vs.get_collection_stats()
print("\n--- QDRANT COLLECTION STATS ---")
print(f"  parent_chunks : {stats['parent_chunks']} points")
print(f"  child_chunks  : {stats['child_chunks']} points")

# sanity check — parents stored == chapters detected
assert stats['parent_chunks'] == result['parents_stored'], "Parent count mismatch!"
assert stats['child_chunks']  == result['children_stored'], "Child count mismatch!"
print("\n✓ All assertions passed")