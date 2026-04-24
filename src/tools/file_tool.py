"""File reading and writing tools."""
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool

from src.config.settings import PROJECT_ROOT


@tool
def read_file(file_path: str, max_lines: int = 100, offset: int = 0) -> str:
    """
    Read the contents of a file.

    Args:
        file_path: Path to the file (absolute or relative to project root)
        max_lines: Maximum number of lines to read (default: 100)
        offset: Line number to start reading from (default: 0)

    Returns:
        The file contents as a string, with line numbers for reference.

    Example:
        >>> read_file("src/rag/document_loader.py")
        (shows file contents with line numbers)
    """
    path = _resolve_path(file_path)

    if not path.exists():
        return f"Error: File not found: {file_path}"

    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except Exception as e:
        return f"Error reading file: {str(e)}"

    total_lines = len(lines)
    lines = lines[offset:offset + max_lines]

    if not lines:
        return f"File is empty or offset {offset} is beyond end of file."

    output = [f"File: {path} ({total_lines} total lines)\n"]

    start_line = offset + 1
    for i, line in enumerate(lines, start_line):
        output.append(f"{i:4d} | {line}")

    if offset + max_lines < total_lines:
        output.append(f"\n... (showing lines {offset + 1}-{offset + max_lines} of {total_lines})")

    return '\n'.join(output)


@tool
def write_file(file_path: str, content: str, append: bool = False) -> str:
    """
    Write content to a file.

    Args:
        file_path: Path to the file (absolute or relative to project root)
        content: Content to write
        append: If True, append to existing file; if False, overwrite (default: False)

    Returns:
        Confirmation message with the path and byte count.

    Warning:
        This tool can overwrite existing files. Use append=True to add content safely.
    """
    path = _resolve_path(file_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if append:
            path.write_text(content, encoding='utf-8', append=True)
            action = "Appended"
        else:
            path.write_text(content, encoding='utf-8')
            action = "Written"

        size = path.stat().st_size
        return f"{action} {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def list_directory(directory_path: str = ".", pattern: str = "*", recursive: bool = False) -> str:
    """
    List files in a directory.

    Args:
        directory_path: Directory to list (default: project root)
        pattern: Glob pattern to filter files (default: "*")
        recursive: If True, list recursively (default: False)

    Returns:
        A formatted list of files with their sizes and types.

    Example:
        >>> list_directory("src", "*.py")
        (shows files in src directory)
    """
    path = _resolve_path(directory_path)

    if not path.exists():
        return f"Error: Directory not found: {directory_path}"

    if not path.is_dir():
        return f"Error: Not a directory: {directory_path}"

    if recursive:
        files = list(path.rglob(pattern))
    else:
        files = list(path.glob(pattern))

    files = [f for f in files if f.is_file()]

    if not files:
        return f"No files found matching '{pattern}' in {path}"

    files.sort(key=lambda f: f.stat().st_size, reverse=True)

    output = [f"Files in {path} (matching '{pattern}'):\n"]

    for f in files:
        size = f.stat().st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"

        rel_path = f.relative_to(path)
        output.append(f"- {rel_path} ({size_str})")

    output.append(f"\nTotal: {len(files)} files")

    return '\n'.join(output)


@tool
def get_file_info(file_path: str) -> str:
    """
    Get detailed information about a file.

    Args:
        file_path: Path to the file

    Returns:
        File metadata including size, creation/modification dates, and type.
    """
    path = _resolve_path(file_path)

    if not path.exists():
        return f"Error: File not found: {file_path}"

    stat = path.stat()

    output = [
        f"File: {path}",
        f"Type: {path.suffix or '(no extension)'}",
        f"Size: {stat.st_size} bytes ({stat.st_size / 1024:.2f} KB)",
        f"Created: {stat.st_ctime}",
        f"Modified: {stat.st_mtime}",
        f"Accessed: {stat.st_atime}",
    ]

    text_extensions = {'.txt', '.md', '.py', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.js'}
    if path.is_file() and path.suffix.lower() in text_extensions:
        try:
            first_lines = path.read_text(encoding='utf-8').splitlines()[:5]
            output.append("\nFirst few lines:")
            for i, line in enumerate(first_lines, 1):
                truncated = line[:80] + '...' if len(line) > 80 else line
                output.append(f"  {i}: {truncated}")
        except Exception:
            pass

    return '\n'.join(output)


@tool
def search_files(directory_path: str = ".", pattern: str = None, regex: str = None) -> str:
    """
    Search for text within files.

    Args:
        directory_path: Directory to search in (default: project root)
        pattern: Simple glob pattern for file names (e.g., "*.py")
        regex: Regular expression pattern to search within file contents

    Returns:
        Matching lines with file names and line numbers.

    Example:
        >>> search_files("src", regex="def.*search")
        (shows matching lines)
    """
    import re

    path = _resolve_path(directory_path)

    if not path.exists():
        return f"Error: Directory not found: {directory_path}"

    try:
        search_pattern = re.compile(regex) if regex else None
    except re.error as e:
        return f"Invalid regex: {str(e)}"

    if pattern:
        files = list(path.rglob(pattern))
    else:
        extensions = {'.py', '.txt', '.md', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.js'}
        files = [f for f in path.rglob('*') if f.is_file() and f.suffix.lower() in extensions]

    matches = []

    for file_path in files:
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.splitlines()

            for i, line in enumerate(lines, 1):
                if search_pattern and search_pattern.search(line):
                    display_line = line[:120] + '...' if len(line) > 120 else line
                    matches.append(f"{file_path}:{i}: {display_line}")
        except Exception:
            continue

    if not matches:
        return f"No matches found for pattern: {regex or pattern or '(all files)'}"

    output = [f"Found {len(matches)} matches:\n"]
    output.extend(matches[:50])

    if len(matches) > 50:
        output.append(f"\n... and {len(matches) - 50} more matches")

    return '\n'.join(output)


def _resolve_path(file_path: str) -> Path:
    """Resolve a file path relative to project root."""
    path = Path(file_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path
