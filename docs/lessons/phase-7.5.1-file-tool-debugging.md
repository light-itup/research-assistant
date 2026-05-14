# Phase 7.5.1: File Tool + ReAct Agent 问题排查

**日期**: 2026/05/12
**问题**: file_read 测试失败，Agent 格式错误 + 工具参数解析失败

---

## 问题 1: Directory not found (JSON 参数传递)

**现象**:
```
Action: list_directory
Action Input: {"directory_path": ".", "pattern": "*.py", "recursive": true}
Error: Directory not found: {"directory_path": ".", "pattern": "*.py", "recursive": true}
```

**原因**: Agent 把函数参数打包成 JSON 字符串传入，但工具函数直接使用该字符串作为路径

**修复**: 在 `file_tool.py` 的所有函数开头添加 JSON 解析：

```python
@tool
def list_directory(directory_path: str = ".", pattern: str = "*", recursive: bool = False) -> str:
    # Handle JSON string input from agent
    if directory_path.startswith('{'):
        import json
        try:
            kwargs = json.loads(directory_path)
            directory_path = kwargs.get('directory_path', directory_path)
            pattern = kwargs.get('pattern', pattern)
            recursive = kwargs.get('recursive', recursive)
        except json.JSONDecodeError:
            pass

    path = _resolve_path(directory_path)
    # ... 正常逻辑
```

**适用函数**: `read_file`, `write_file`, `list_directory`, `get_file_info`, `search_files`

---

## 问题 2: Invalid Format: Missing 'Action:' after 'Observation:'

**现象**: Agent 输出包含 `Observation:` 但 ReAct Agent 不识别此格式

**原因**: Prompt 中写了 `After receiving the tool result, respond with: Observation: [...]`，但 LangChain ReAct agent 不使用 Observation 关键字

**修复**: 修改 prompt 格式，移除 Observation：

```
**After receiving a tool result, continue with:**
Thought: [Your reasoning about the observation]
Action: [next_tool_name or Final Answer]
Action Input: [...]
```

---

## 问题 3: Missing prompt variables {'"code"', '"file_path"'}

**现象**:
```
Error: 'Input to PromptTemplate is missing variables {"code", "file_path"}.
Expected: ["code", "file_path", "agent_scratchpad", ...]
```

**原因**: Tool description 中的 JSON 示例（如 `{"code": "..."}`）被 LangChain 解释为 prompt 变量

**修复**: 在 `create_react_prompt()` 中清理 tool description：

```python
import re
tool_desc_clean = re.sub(r'\{[^{}]*\}', '', tool_desc)
tools_desc.append(f"- **{tool_name}**: {tool_desc_clean.strip()}")
```

---

## 问题 4: Tool docstring 中的 JSON 示例导致 Prompt 解析失败

**原因**: 代码示例在 docstring 中包含 `{"code": "..."}` 这样的 JSON，被 prompt 模板错误解析

**修复策略**:
1. 简化所有 tool 的 docstring，移除 JSON 示例
2. 在 prompt 中使用文字描述替代 JSON 格式示例

**修改位置**:
- `code_tool.py`: `execute_python_code`, `explain_code` - 移除 `{"code": "..."}` 示例
- `file_tool.py`: 移除所有 `>>> read_file(...)` 格式的示例
- `research_agent.py`: 清理 Code Execution Guidelines 中的 JSON 格式

---

## 关键教训

1. **Agent 可能发送 JSON 字符串作为单个参数** - 工具函数需要检测并解析
2. **LangChain prompt 模板会把 `{xxx}` 当成变量** - description 中的 JSON 示例必须移除
3. **ReAct Agent 使用 `Observation:` 但 prompt 模板需自定义** - 确认实际使用的格式
4. **测试时用最小例子** - 先跑通 `file_read` 再扩展到其他工具

---

## 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `src/tools/file_tool.py` | 5 个函数添加 JSON 输入检测 |
| `src/tools/code_tool.py` | 简化 docstring，移除 JSON 示例 |
| `src/agents/research_agent.py` | 清理 tool description，简化 prompt 格式 |

---

## 验证方式

```bash
PYTHONPATH=. python tests/test_research_agent.py
# 应正常完成 file_read 测试
```