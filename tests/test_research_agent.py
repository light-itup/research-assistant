"""Test Research Agent with ReAct pattern."""
import os
import sys
import warnings

# 忽略所有 DeprecationWarning (LangChain 升级过渡期)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 如果想完全抑制 stderr 输出，取消下面的注释
# sys.stderr = open(os.devnull, 'w')

os.environ.pop('SSL_CERT_FILE', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Fix Windows console encoding for emoji
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def safe_print(s):
    """Print safely, replacing problematic characters."""
    if isinstance(s, str):
        print(s.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
    else:
        print(s)


# 测试用例定义
TEST_CASES = {
    "transformer": {
        "name": "Test 1: Transformer Architecture",
        "query": "What is the Transformer architecture? Explain its key components.",
        "tools": ["rag", "web_search"],
    },
    "rnn_comparison": {
        "name": "Test 2: RNN vs Transformer",
        "query": "Compare RNNs with Transformers. What are the main advantages of Transformers?",
        "tools": ["rag", "web_search"],
    },
    "code_explain": {
        "name": "Test 3: Code Explanation",
        "query": "Explain what this Python code does: [x for x in range(10) if x % 2 == 0]",
        "tools": ["explain_code"],
    },
    "code_execute": {
        "name": "Test 3b: Code Execution",
        "query": "Execute this Python code and show me the result: [x for x in range(10) if x % 2 == 0]",
        "tools": ["execute_python_code"],
    },
    "file_read": {
        "name": "Test 4: File Reading",
        "query": "Show me the contents of the text_splitter.py file",
        "tools": ["read_file"],
    },
    "self_attention": {
        "name": "Test 5a: Self-Attention (first turn)",
        "query": "What is self-attention?",
        "tools": ["rag", "web_search"],
    },
    "cross_attention": {
        "name": "Test 5b: Cross-Attention (follow-up)",
        "query": "How does it differ from cross-attention?",
        "tools": ["rag", "web_search"],
    },
    "knowledge_stats": {
        "name": "Test 6: Knowledge Base Stats",
        "query": "How many documents are in my knowledge base?",
        "tools": ["get_knowledge_base_stats"],
    },
}


def run_research_agent_test(
    test_names: list = None,
    tools_config: dict = None,
    verbose: bool = True,
) -> dict:
    """
    运行 Research Agent 测试

    Args:
        test_names: 要运行的测试名称列表，如 ["transformer", "code_explain"]
                   如果为 None，运行所有测试
        tools_config: 自定义工具配置，格式 {"rag": search_knowledge_base, ...}
        verbose: 是否打印详细输出

    Returns:
        dict: 测试结果，格式 {test_name: result, ...}

    Example:
        # 运行单个测试
        run_research_agent_test(["transformer"])

        # 运行多个测试
        run_research_agent_test(["transformer", "code_explain"])

        # 运行全部测试
        run_research_agent_test()
    """
    from src.agents.research_agent import create_research_agent
    from src.tools.rag_tool import search_knowledge_base, get_knowledge_base_stats
    from src.tools.web_search_tool import web_search
    from src.tools.file_tool import read_file, list_directory
    from src.tools.code_tool import execute_python_code, explain_code

    # 默认工具配置
    if tools_config is None:
        tools_config = {
            "rag": search_knowledge_base,
            "get_knowledge_base_stats": get_knowledge_base_stats,
            "web_search": web_search,
            "read_file": read_file,
            "list_directory": list_directory,
            "execute_python_code": execute_python_code,
            "explain_code": explain_code,
        }

    # 默认运行所有测试
    if test_names is None:
        test_names = list(TEST_CASES.keys())

    print("=" * 70)
    print("Research Agent Test Suite")
    print("=" * 70)
    print(f"\nTests to run: {test_names}")

    # 构建工具列表
    available_tools = list(tools_config.values())
    print(f"\nTools loaded: {len(available_tools)}")

    # 创建 Agent
    print("\nCreating Agent...")
    agent = create_research_agent(
        tools=available_tools,
        verbose=verbose,
    )
    print("Agent created!\n")

    results = {}

    for test_name in test_names:
        if test_name not in TEST_CASES:
            print(f"[!] Unknown test: {test_name}, skipping...")
            continue

        test_case = TEST_CASES[test_name]
        print("=" * 70)
        print(test_case["name"])
        print("=" * 70)

        query = test_case["query"]
        print(f"Query: {query}\n")

        try:
            result = agent.run(query)
            results[test_name] = result

            if verbose:
                safe_print(f"Result:\n{result[:500]}...")
            else:
                print(f"Result: [len={len(result)} chars]")
            print()

        except Exception as e:
            print(f"[!] Error in {test_name}: {e}")
            results[test_name] = f"Error: {e}"

    print("=" * 70)
    print("Tests completed!")
    print("=" * 70)

    return results


# 默认执行入口
if __name__ == "__main__":
    # 运行所有测试
    # run_research_agent_test()

    # 或者运行指定测试：
    # run_research_agent_test(["transformer"])
    # run_research_agent_test(["code_execute"])
    run_research_agent_test(["file_read"])
