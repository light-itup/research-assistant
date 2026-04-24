"""Code execution and interpretation tools."""
import ast
import inspect
import io
import sys
from typing import Any, Dict, Optional
from langchain_core.tools import tool


@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code and return the output.

    This tool can run Python code snippets for testing, calculation, or
    data manipulation. It captures both stdout output and returned values.

    Args:
        code: Python code to execute. Should be a complete, runnable snippet.

    Returns:
        The stdout output and any returned value from the code.

    Example:
        >>> execute_python_code("print(1 + 2)")
        3

        >>> execute_python_code("import math; math.sqrt(16)")
        4.0

        >>> execute_python_code("def add(a, b): return a + b; add(1, 2)")
        3
    """
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    try:
        # Execute the code
        result = exec(code, {'__name__': '__main__'})

        # Get stdout
        output = captured_output.getvalue()

        # Restore stdout
        sys.stdout = old_stdout

        # Format result
        if output:
            return output.rstrip()
        elif result is None:
            return "Code executed successfully (no output)"
        else:
            return str(result)

    except Exception as e:
        sys.stdout = old_stdout
        return f"Error executing code: {type(e).__name__}: {str(e)}"


@tool
def explain_code(code: str) -> str:
    """
    Explain what a piece of Python code does.

    This tool analyzes Python code and provides a natural language
    explanation of its functionality, inputs, and outputs.

    Args:
        code: Python code to explain

    Returns:
        A detailed explanation of what the code does.

    Example:
        >>> explain_code("sorted([3, 1, 2])")
        This code sorts a list of numbers in ascending order.
        ...
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error in code: {str(e)}"

    explanations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            explanations.append(_explain_function(node))
        elif isinstance(node, ast.ClassDef):
            explanations.append(_explain_class(node))
        elif isinstance(node, ast.Assign):
            explanations.append(_explain_assignment(node))
        elif isinstance(node, ast.Expr):
            explanations.append(_explain_expression(node))

    if not explanations:
        return "Could not analyze the code structure."

    return '\n'.join(explanations)


def _explain_function(node: ast.FunctionDef) -> str:
    """Explain a function definition."""
    args = [arg.arg for arg in node.args.args]
    returns = ast.unparse(node.returns) if node.returns else "None"

    lines = [
        f"Function: {node.name}",
        f"  Arguments: {', '.join(args) if args else 'none'}",
        f"  Returns: {returns}",
    ]

    if ast.get_docstring(node):
        lines.append(f"  Docstring: {ast.get_docstring(node)}")

    return '\n'.join(lines)


def _explain_class(node: ast.ClassDef) -> str:
    """Explain a class definition."""
    lines = [f"Class: {node.name}"]

    if ast.get_docstring(node):
        lines.append(f"  Docstring: {ast.get_docstring(node)}")

    # List methods
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
    if methods:
        lines.append(f"  Methods: {', '.join(m.name for m in methods)}")

    return '\n'.join(lines)


def _explain_assignment(node: ast.Assign) -> str:
    """Explain an assignment."""
    targets = [ast.unparse(t) for t in node.targets]
    value = ast.unparse(node.value)
    return f"Assignment: {' = '.join(targets)} = {value}"


def _explain_expression(node: ast.Expr) -> str:
    """Explain an expression."""
    return f"Expression: {ast.unparse(node.value)}"


@tool
def inspect_module(module_name: str, member: Optional[str] = None) -> str:
    """
    Inspect a Python module or its members.

    Args:
        module_name: The module name (e.g., "langchain", "llamaindex")
        member: Optional specific member to inspect (function, class, etc.)

    Returns:
        Information about the module or specified member.

    Example:
        >>> inspect_module("langchain", "create_react_agent")
        Function: create_react_agent
        Signature: create_react_agent(llm, tools, prompt, ...)
        ...
    """
    try:
        import importlib
        module = importlib.import_module(module_name)
    except ImportError:
        return f"Error: Module '{module_name}' not found or not installed."

    if member:
        if not hasattr(module, member):
            return f"Error: Module '{module_name}' has no member '{member}'."

        obj = getattr(module, member)
        return _inspect_object(member, obj)
    else:
        # List module members
        members = [name for name in dir(module) if not name.startswith('_')]
        return f"Module: {module_name}\n\nExported members ({len(members)}):\n" + '\n'.join(f"  - {m}" for m in members[:50])


def _inspect_object(name: str, obj: Any) -> str:
    """Inspect a specific object."""
    obj_type = type(obj).__name__
    lines = [f"{name} ({obj_type})"]

    # Try to get signature for functions/callables
    if callable(obj) and hasattr(obj, '__signature__'):
        sig = inspect.signature(obj)
        lines.append(f"Signature: {sig}")

    # Try to get docstring
    if hasattr(obj, '__doc__') and obj.__doc__:
        doc = obj.__doc__.strip()[:200]
        lines.append(f"Docstring: {doc}...")

    return '\n'.join(lines)


@tool
def generate_sample_code(task: str, language: str = "python") -> str:
    """
    Generate sample code for common tasks.

    Args:
        task: Description of the task (e.g., "read PDF", "web search", "sort list")
        language: Programming language (default: "python")

    Returns:
        Sample code that accomplishes the described task.
    """
    task_lower = task.lower()
    samples = {
        "read pdf": '''```python
from src.rag import load_documents

# Load PDF documents from a directory
docs = load_documents(directory="data/papers/")
for doc in docs:
    print(f"Loaded: {doc.metadata.get('file_name', 'unknown')}")
    print(f"Content preview: {doc.text[:200]}...")
```''',
        "web search": '''```python
from tavily import TavilyClient

client = TavilyClient(api_key="your-api-key")
results = client.search(query="your search query", max_results=5)

for r in results["results"]:
    print(f"Title: {r["title"]}")
    print(f"URL: {r["url"]}")
    print(f"Snippet: {r["content"]}")
    print("---")
```''',
        "sort list": '''```python
# Sort a list in ascending order
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = sorted(numbers)
print(sorted_numbers)  # [1, 1, 2, 3, 4, 5, 6, 9]

# Sort descending
sorted_desc = sorted(numbers, reverse=True)
print(sorted_desc)  # [9, 6, 5, 4, 3, 2, 1, 1]
```''',
        "file read": '''```python
from pathlib import Path

# Read a text file
content = Path("example.txt").read_text(encoding="utf-8")
print(content)

# Read line by line
for line in Path("example.txt").read_text(encoding="utf-8").splitlines():
    print(line)
```''',
    }

    # Find best match
    for key, code in samples.items():
        if key in task_lower:
            return f"Sample code for '{task}':\n\n{code}"

    return (
        f"No sample available for '{task}'. "
        f"Available tasks: {', '.join(samples.keys())}"
    )
