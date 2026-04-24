"""Embedding module."""
from typing import List


def create_embedder(
    model: str = None,
    use_local: bool = True,
    **kwargs
):
    """
    Create an embedding model for text vectorization.

    Args:
        model: Model name (ignored if use_local=True)
        use_local: If True, use sentence-transformers (local, no API cost)
                   If False, use OpenAI-compatible API
        **kwargs: Additional arguments

    Returns:
        Embedding model instance

    Example:
        # Local embedding (recommended for learning)
        embedder = create_embedder(use_local=True)

        # API-based embedding
        embedder = create_embedder(use_local=False, model="text-embedding-3-small")
    """
    if use_local:
        return create_local_embedder(model)
    else:
        return create_api_embedder(model, **kwargs)


def create_local_embedder(model: str = None):
    """
    Create a local embedding model using sentence-transformers.

    This is free, fast, and works offline. Recommended for learning and development.

    Args:
        model: Model name from sentence-transformers (default: all-MiniLM-L6-v2)

    Returns:
        HuggingFaceEmbedding instance
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Default model: lightweight, high quality
    model_name = model or "sentence-transformers/all-MiniLM-L6-v2"

    return HuggingFaceEmbedding(
        model_name=model_name,
        max_length=512,
    )


def create_api_embedder(model: str = None, **kwargs):
    """
    Create an OpenAI-compatible API-based embedding model.

    Note: LlamaIndex's OpenAIEmbedding only supports:
    - text-embedding-ada-002
    - text-embedding-3-small
    - text-embedding-3-large

    For other providers (Aliyun, MiniMax), use local embeddings instead.
    """
    from llama_index.embeddings.openai import OpenAIEmbedding
    from src.config.settings import (
        ALIYUN_API_KEY,
        DASHSCOPE_BASE_URL,
    )

    model_name = model or "text-embedding-3-small"

    return OpenAIEmbedding(
        model=model_name,
        api_key=kwargs.get("api_key") or ALIYUN_API_KEY,
        base_url=kwargs.get("base_url") or f"{DASHSCOPE_BASE_URL}/embeddings",
    )


def configure_global_embedder(
    model: str = None,
    use_local: bool = True,
    **kwargs
):
    """
    Configure LlamaIndex's global embedding settings.

    Args:
        model: Model name
        use_local: Whether to use local embeddings
        **kwargs: Additional arguments
    """
    from llama_index.core import Settings

    embedder = create_embedder(model=model, use_local=use_local, **kwargs)
    Settings.embed_model = embedder
    return embedder


def get_text_embeddings(texts: List[str], embedder=None) -> List[List[float]]:
    """
    Get embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        embedder: Embedder instance (creates default if None)

    Returns:
        List of embedding vectors
    """
    if embedder is None:
        embedder = create_local_embedder()

    return embedder.get_text_embedding_batch(texts)
