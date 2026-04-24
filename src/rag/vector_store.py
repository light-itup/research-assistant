"""Vector store module."""
from typing import List, Optional, Dict, Any
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
import chromadb


def create_vector_index(
    documents: List[Document],
    embed_model=None,
    store_locally: bool = False,
    persist_dir: str = None,
    collection_name: str = "research_assistant",
) -> VectorStoreIndex:
    """
    Create a vector index from documents.

    Args:
        documents: List of documents to index
        embed_model: Embedding model (uses global setting if None)
        store_locally: Whether to persist the index to disk
        persist_dir: Directory to persist index (required if store_locally=True)
        collection_name: Name of the ChromaDB collection

    Returns:
        VectorStoreIndex instance

    Example:
        index = create_vector_index(documents)
        retriever = index.as_retriever()
        results = retriever.retrieve("transformer architecture")
    """
    from src.config.settings import CHROMA_DB_DIR
    from src.rag.embedder import create_embedder

    if embed_model is None:
        embed_model = create_embedder()

    if store_locally:
        # Use ChromaDB for persistent storage
        persist_dir = persist_dir or str(CHROMA_DB_DIR)

        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=persist_dir,
        )

        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            storage_context=storage_context,
            show_progress=True,
        )
    else:
        # In-memory index
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            show_progress=True,
        )

    return index


def load_existing_index(
    persist_dir: str = None,
    collection_name: str = "research_assistant",
    embed_model=None,
) -> VectorStoreIndex:
    """
    Load an existing vector index from disk.

    Args:
        persist_dir: Directory where index is persisted
        collection_name: Name of the ChromaDB collection
        embed_model: Embedding model (uses global setting if None)

    Returns:
        VectorStoreIndex instance
    """
    from src.config.settings import CHROMA_DB_DIR
    from src.rag.embedder import create_embedder

    persist_dir = persist_dir or str(CHROMA_DB_DIR)

    if embed_model is None:
        embed_model = create_embedder()

    # Load from ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create index from vector store (without documents)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    return index


def query_index(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Query a vector index.

    Args:
        index: VectorStoreIndex to query
        query: Query string
        top_k: Number of results to return
        filters: Optional metadata filters

    Returns:
        List of retrieved nodes
    """
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters,
    )
    return retriever.retrieve(query)


class VectorStoreManager:
    """
    Manager class for vector store operations.

    Provides a higher-level interface for managing the RAG pipeline.
    """

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = "research_assistant",
        embed_model=None,
    ):
        """
        Initialize the vector store manager.

        Args:
            persist_dir: Directory for index persistence
            collection_name: ChromaDB collection name
            embed_model: Embedding model
        """
        from src.config.settings import CHROMA_DB_DIR
        from src.rag.embedder import create_embedder

        self.persist_dir = persist_dir or str(CHROMA_DB_DIR)
        self.collection_name = collection_name
        self.embed_model = embed_model or create_embedder()
        self.index = None

    def build_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Build or rebuild the index from documents."""
        self.index = create_vector_index(
            documents=documents,
            embed_model=self.embed_model,
            store_locally=True,
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
        )
        return self.index

    def load_index(self) -> VectorStoreIndex:
        """Load an existing index from disk."""
        self.index = load_existing_index(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            embed_model=self.embed_model,
        )
        return self.index

    def query(self, query: str, top_k: int = 5) -> List[Any]:
        """Query the current index."""
        if self.index is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        return query_index(self.index, query, top_k)

    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self.index is not None
