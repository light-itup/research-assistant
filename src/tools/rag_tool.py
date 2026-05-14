"""RAG search tool for knowledge base queries."""
from typing import List, Optional
from langchain_core.tools import tool

from src.rag import (
    load_documents,
    split_documents,
    IndexManager,
)
from src.config.settings import KNOWLEDGE_BASE_DIR


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    Search the local knowledge base for information related to the query.

    This tool searches through uploaded documents, PDFs, and other files
    in the knowledge base to find relevant information.

    NOTE: The knowledge base must be initialized first by running:
        python scripts/init_knowledge_base.py

    Args:
        query: The search query (e.g., "transformer architecture", "attention mechanism")
        top_k: Number of results to return (default: 5)

    Returns:
        A formatted string containing the search results with source information.
        Each result includes the document name, content snippet, and relevance score.

    Example:
        >>> search_knowledge_base("What is a transformer?")
        Found 3 results:

        [1] transformer_intro.txt (score: 0.87)
        ---
        Transformer Architecture Introduction

        The Transformer architecture was introduced in the paper
        "Attention Is All You Need" by Vaswani et al. in 2017...

        ---

        [2] attention_paper.pdf (score: 0.82)
        ---
        Self-Attention Mechanism
        Allows each token to attend to all other tokens...
        ---
    """
    # Get the global index manager
    manager = IndexManager.get_instance()

    # Try to initialize if not ready
    if not manager.is_ready():
        manager.initialize()

    # Check if index is ready
    if not manager.is_ready():
        return (
            "Knowledge base not initialized. "
            "Please run: python scripts/init_knowledge_base.py"
        )

    # Query the index (only vectorizes the query, no document reprocessing)
    try:
        results = manager.search(query, top_k=top_k)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

    if not results:
        return f"No results found for query: '{query}'"

    # Format results
    output = [f"Found {len(results)} results for query: '{query}'\n"]

    for i, node in enumerate(results, 1):
        score = node.score if hasattr(node, 'score') else 'N/A'
        doc_name = node.metadata.get('file_name', 'Unknown') if hasattr(node, 'metadata') else 'Unknown'
        content = node.text[:300].replace('\n', ' ') + '...' if len(node.text) > 300 else node.text

        output.append(f"[{i}] {doc_name} (score: {score:.2f})")
        output.append("---")
        output.append(content)
        output.append("---\n")

    return '\n'.join(output)


@tool
def get_knowledge_base_stats() -> str:
    """
    Get statistics about the knowledge base.

    Returns:
        A string containing the number of documents and their file types.
    """
    try:
        docs = load_documents(directory=str(KNOWLEDGE_BASE_DIR))
    except Exception as e:
        return f"Error loading documents: {str(e)}"

    if not docs:
        return "Knowledge base is empty. Add some documents to data/knowledge_base/"

    # Count file types
    file_types = {}
    total_chunks = 0

    for doc in docs:
        ext = doc.metadata.get('file_type', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
        file_types[ext] = file_types.get(ext, 0) + 1

    # Estimate chunks
    chunks = split_documents(docs, chunk_size=512, chunk_overlap=50)
    total_chunks = len(chunks)

    output = [
        f"Knowledge Base Statistics:",
        f"- Total documents: {len(docs)}",
        f"- Total chunks: {total_chunks}",
        f"- File types:",
    ]

    for ext, count in file_types.items():
        output.append(f"  - {ext}: {count}")

    return '\n'.join(output)


@tool
def add_document_to_knowledge_base(file_path: str) -> str:
    """
    Add a single document to the knowledge base.

    Args:
        file_path: Path to the document file (absolute or relative path)

    Returns:
        A confirmation message with the document details.
    """
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"

    # Supported extensions
    supported = {'.txt', '.md', '.pdf', '.docx', '.pptx', '.html', '.json', '.yaml', '.csv'}
    if path.suffix.lower() not in supported:
        return f"Error: Unsupported file type {path.suffix}. Supported: {', '.join(supported)}"

    # Copy to knowledge base
    import shutil
    dest_path = KNOWLEDGE_BASE_DIR / path.name
    shutil.copy2(path, dest_path)

    return f"Successfully added document: {path.name}\nSaved to: {dest_path}"


@tool
def rebuild_knowledge_base_index() -> str:
    """
    Rebuild the knowledge base vector index.

    Use this when documents have been added or removed and the index needs to be updated.

    Returns:
        Confirmation message with index statistics.
    """
    from src.rag.vector_store import VectorStoreManager

    try:
        docs = load_documents(directory=str(KNOWLEDGE_BASE_DIR))
    except Exception as e:
        return f"Error loading documents: {str(e)}"

    if not docs:
        return "No documents to index."

    manager = VectorStoreManager()
    manager.build_index(docs)

    return (
        f"Knowledge base index rebuilt successfully!\n"
        f"- Documents indexed: {len(docs)}\n"
        f"- Index stored at: {manager.persist_dir}"
    )
