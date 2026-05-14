"""RAG module for document processing and retrieval."""
from src.rag.document_loader import (
    load_documents,
    load_document,
    get_supported_extensions,
)
from src.rag.text_splitter import (
    split_documents,
    split_text,
)
from src.rag.embedder import (
    create_embedder,
    configure_global_embedder,
    get_text_embeddings,
)
from src.rag.vector_store import (
    create_vector_index,
    load_existing_index,
    query_index,
    VectorStoreManager,
)
from src.rag.index_manager import (
    IndexManager,
    get_index_manager,
    search_knowledge_base,
)

__all__ = [
    # Document loader
    "load_documents",
    "load_document",
    "get_supported_extensions",
    # Text splitter
    "split_documents",
    "split_text",
    # Embedder
    "create_embedder",
    "configure_global_embedder",
    "get_text_embeddings",
    # Vector store
    "create_vector_index",
    "load_existing_index",
    "query_index",
    "VectorStoreManager",
    # Index manager
    "IndexManager",
    "get_index_manager",
    "search_knowledge_base",
]
