from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):

    # --- Groq LLM ---
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Embeddings ---
    embedding_model: str = "nomic-ai/nomic-embed-text-v1"
    embedding_dim: int = 768

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZmNjMzI4MGMtNjU1My00ODcyLWJmYjEtNjI2NjJlOGY1NDk2In0.2GTJB7yRJ28k68zpG732jPyNgzzxJlMmOnR0m2xFHsw" 
    parent_collection: str = "parent_chunks"
    child_collection: str = "child_chunks"

    # --- Chunking ---
    parent_chunk_tokens: int = 1200
    child_chunk_tokens: int = 200
    child_overlap_tokens: int = 20

    # --- Langfuse ---
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_base_url: str = "https://cloud.langfuse.com"
    redis_host: str = "localhost"
    redis_port: int = 6379

    class Config:
        env_file = Path(__file__).parent.parent / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"   # ignore any extra env vars from other packages

settings = Settings()