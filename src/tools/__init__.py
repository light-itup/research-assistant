"""Tools module for the research assistant."""
from src.tools.rag_tool import (
    search_knowledge_base,
    get_knowledge_base_stats,
    add_document_to_knowledge_base,
    rebuild_knowledge_base_index,
)
from src.tools.web_search_tool import (
    web_search,
    web_search_with_depth,
)
from src.tools.code_tool import (
    execute_python_code,
    explain_code,
    inspect_module,
    generate_sample_code,
)
from src.tools.file_tool import (
    read_file,
    write_file,
    list_directory,
    get_file_info,
    search_files,
)

__all__ = [
    # RAG tools
    "search_knowledge_base",
    "get_knowledge_base_stats",
    "add_document_to_knowledge_base",
    "rebuild_knowledge_base_index",
    # Web search tools
    "web_search",
    "web_search_with_depth",
    # Code tools
    "execute_python_code",
    "explain_code",
    "inspect_module",
    "generate_sample_code",
    # File tools
    "read_file",
    "write_file",
    "list_directory",
    "get_file_info",
    "search_files",
]
