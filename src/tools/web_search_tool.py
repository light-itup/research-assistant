"""Web search tool using Tavily API."""
from langchain_core.tools import tool

from src.config.settings import TAVILY_API_KEY


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the internet for current information on any topic.

    Use this tool when you need to find up-to-date information that may not be
    in the local knowledge base. Tavily searches across multiple sources and
    returns concise, relevant results.

    Args:
        query: The search query (e.g., "latest GPT-5 news", "2024 machine learning trends")
        max_results: Maximum number of results to return (default: 5, max: 10)

    Returns:
        A formatted string containing search results with titles, URLs, and snippets.

    Example:
        >>> web_search("what is RAG in LLM")
        Search Results for "what is RAG in LLM":

        [1] What is RAG?
        URL: https://aws.amazon.com/what-is/retrieval-augmented-generation/
        Snippet: Retrieval-Augmented Generation (RAG) is an AI framework...
        ---

        [2] RAG Explained
        URL: https://docs.llamaindex.ai/concepts/rag/
        Snippet: RAG combines the power of large language models with external data...
        ---
    """
    if not TAVILY_API_KEY:
        return (
            "Error: Tavily API key not configured. "
            "Please set TAVILY_API_KEY in your .env file."
        )

    try:
        from tavily import TavilyClient
    except ImportError:
        return "Error: tavily-python package not installed. Run: pip install tavily-python"

    # Limit max_results to 10
    max_results = min(max_results, 10)

    client = TavilyClient(api_key=TAVILY_API_KEY)

    try:
        results = client.search(
            query=query,
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
        )
    except Exception as e:
        return f"Error searching the web: {str(e)}"

    if not results.get('results'):
        return f"No results found for query: '{query}'"

    # Format output
    output = [f"Search Results for '{query}':\n"]

    for i, result in enumerate(results['results'], 1):
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        snippet = result.get('content', 'No description')[:300]

        output.append(f"[{i}] {title}")
        output.append(f"URL: {url}")
        output.append(f"Snippet: {snippet}...")
        output.append("---")

    # Include answer if available
    if results.get('answer'):
        output.append(f"\nAI Answer: {results['answer']}")

    return '\n'.join(output)


@tool
def web_search_with_depth(query: str, search_depth: str = "basic") -> str:
    """
    Perform a deeper web search with optional advanced search depth.

    Args:
        query: The search query
        search_depth: Either "basic" (faster) or "advanced" (more comprehensive)

    Returns:
        Search results with optional AI-generated answer.

    Note:
        Advanced search may take longer and use more API credits.
    """
    if not TAVILY_API_KEY:
        return (
            "Error: Tavily API key not configured. "
            "Please set TAVILY_API_KEY in your .env file."
        )

    try:
        from tavily import TavilyClient
    except ImportError:
        return "Error: tavily-python package not installed. Run: pip install tavily-python"

    if search_depth not in ["basic", "advanced"]:
        search_depth = "basic"

    client = TavilyClient(api_key=TAVILY_API_KEY)

    try:
        results = client.search(
            query=query,
            max_results=5,
            search_depth=search_depth,
            include_answer=True,
            include_raw_content=True,
        )
    except Exception as e:
        return f"Error searching the web: {str(e)}"

    if not results.get('results'):
        return f"No results found for query: '{query}'"

    output = [f"Search Results for '{query}' (depth: {search_depth}):\n"]

    for i, result in enumerate(results['results'], 1):
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        snippet = result.get('content', 'No description')[:400]

        output.append(f"[{i}] {title}")
        output.append(f"URL: {url}")
        output.append(f"Content: {snippet}...")
        output.append("---")

    if results.get('answer'):
        output.append(f"\nAI Answer: {results['answer']}")

    return '\n'.join(output)
