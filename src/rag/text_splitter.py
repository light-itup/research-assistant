"""Text splitter module."""
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into smaller chunks using LlamaIndex SentenceSplitter.

    Args:
        documents: List of documents to split
        chunk_size: Target size of each chunk (in characters, approximately)
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        List of chunked documents (as nodes)
    """
    parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return parser.get_nodes_from_documents(documents)


def split_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Split raw text into chunks.

    Args:
        text: Raw text to split
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # Create a temporary document and split it
    doc = Document(text=text)
    nodes = parser.get_nodes_from_documents([doc])
    return [node.text for node in nodes]
