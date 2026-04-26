import logging
from langfuse import get_client as _get_client
from app.config import settings
import os

logger = logging.getLogger(__name__)

_langfuse_instance = None


def get_langfuse_client():
    global _langfuse_instance
    if _langfuse_instance is None:
        # v4 reads from env vars — set them from our config
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_HOST"]        = settings.langfuse_base_url
        _langfuse_instance = _get_client()
        logger.info("[Langfuse] Client initialised")
    return _langfuse_instance