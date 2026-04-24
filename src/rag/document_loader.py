"""Document loader module."""
from pathlib import Path
from typing import List, Union
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


def load_documents(
    directory: str = None,
    files: List[str] = None,
    recursive: bool = False,
    exclude_hidden: bool = True,
) -> List[Document]:
    """
    Load documents from a directory or specific files.

    Supports: .pdf, .txt, .md, .docx, .pptx, .html, etc.

    Args:
        directory: Directory path to load from
        files: Specific file paths to load (alternative to directory)
        recursive: Whether to search subdirectories
        exclude_hidden: Whether to exclude hidden files (starting with .)

    Returns:
        List of Document objects

    Example:
        # Load all documents from a directory
        docs = load_documents("./data/papers")

        # Load specific files
        docs = load_documents(files=["./paper1.pdf", "./paper2.pdf"])
    """
    if files:
        # Load specific files
        return SimpleDirectoryReader(input_files=files).load_data()

    if directory:
        # Load from directory
        return SimpleDirectoryReader(
            input_dir=directory,
            recursive=recursive,
            exclude_hidden=exclude_hidden,
        ).load_data()

    raise ValueError("Either 'directory' or 'files' must be provided")


def load_document(file_path: str) -> List[Document]:
    """
    Load a single document.

    Args:
        file_path: Path to the file

    Returns:
        List containing one Document
    """
    return SimpleDirectoryReader(input_files=[file_path]).load_data()


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return [
        ".txt", ".md", ".pdf", ".docx", ".doc",
        ".pptx", ".ppt", ".html", ".htm",
        ".json", ".yaml", ".yml", ".csv",
    ]
