"""Test tools module."""
import os
os.environ.pop('SSL_CERT_FILE', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from src.tools import (
    # RAG tools
    search_knowledge_base,
    get_knowledge_base_stats,
    add_document_to_knowledge_base,
    rebuild_knowledge_base_index,
    # Web search tools
    web_search,
    web_search_with_depth,
    # Code tools
    execute_python_code,
    explain_code,
    inspect_module,
    generate_sample_code,
    # File tools
    read_file,
    write_file,
    list_directory,
    get_file_info,
    search_files,
)


def test_code_tools():
    """Test code-related tools."""
    print("=" * 50)
    print("Testing Code Tools")
    print("=" * 50)

    # Test execute_python_code
    print("\n[1] Testing execute_python_code...")
    result = execute_python_code.invoke("print(1 + 2)")
    print(f"   Result: {result}")
    assert "3" in result

    # Test explain_code
    print("\n[2] Testing explain_code...")
    code = """
def add(a, b):
    '''Add two numbers.'''
    return a + b
"""
    result = explain_code.invoke(code)
    print(f"   Result:\n{result[:200]}...")

    # Test inspect_module
    print("\n[3] Testing inspect_module...")
    result = inspect_module.invoke({"module_name": "os", "member": "path"})
    print(f"   Result:\n{result[:200]}...")

    # Test generate_sample_code
    print("\n[4] Testing generate_sample_code...")
    result = generate_sample_code.invoke("sort list")
    print(f"   Result:\n{result[:300]}...")

    print("\n" + "=" * 50)
    print("Code Tools Test Complete!")
    print("=" * 50)


def test_file_tools():
    """Test file-related tools."""
    print("\n" + "=" * 50)
    print("Testing File Tools")
    print("=" * 50)

    # Test read_file
    print("\n[1] Testing read_file...")
    result = read_file.invoke("src/config/settings.py")
    print(f"   First 300 chars:\n{result[:300]}...")

    # Test list_directory
    print("\n[2] Testing list_directory...")
    result = list_directory.invoke("src", pattern="*.py")
    print(f"   Result:\n{result[:300]}...")

    # Test get_file_info
    print("\n[3] Testing get_file_info...")
    result = get_file_info.invoke("src/config/settings.py")
    print(f"   Result:\n{result[:300]}...")

    # Test search_files
    print("\n[4] Testing search_files...")
    result = search_files.invoke("src", regex="def.*index")
    print(f"   Result:\n{result[:300]}...")

    print("\n" + "=" * 50)
    print("File Tools Test Complete!")
    print("=" * 50)


def test_rag_tools():
    """Test RAG-related tools."""
    print("\n" + "=" * 50)
    print("Testing RAG Tools")
    print("=" * 50)

    # Test get_knowledge_base_stats
    print("\n[1] Testing get_knowledge_base_stats...")
    result = get_knowledge_base_stats.invoke({})
    print(f"   Result: {result[:300]}...")

    # Test search_knowledge_base
    print("\n[2] Testing search_knowledge_base...")
    result = search_knowledge_base.invoke("transformer architecture")
    print(f"   Result:\n{result[:500]}...")

    print("\n" + "=" * 50)
    print("RAG Tools Test Complete!")
    print("=" * 50)


def test_web_search_tools():
    """Test web search tools (requires Tavily API key)."""
    print("\n" + "=" * 50)
    print("Testing Web Search Tools")
    print("=" * 50)

    # Check if API key is configured
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("\n   Skipping web search tests - TAVILY_API_KEY not configured")
        print("   Set TAVILY_API_KEY in .env to enable these tests")
        return

    # Test web_search
    print("\n[1] Testing web_search...")
    result = web_search.invoke("what is RAG in LLM")
    print(f"   Result:\n{result[:500]}...")

    print("\n" + "=" * 50)
    print("Web Search Tools Test Complete!")
    print("=" * 50)


if __name__ == "__main__":
    test_code_tools()
    test_file_tools()
    test_rag_tools()
    test_web_search_tools()
