# Phase 7.6: Memory 持久化

**日期**: 2026/05/13
**问题**: 之前的 memory (chat_history) 和 intermediate_steps (scratchpad) 都在内存中，程序结束就丢失

---

## 问题背景

1. **chat_history**: 每次调用 agent 后保存到 memory，但 memory 是内存对象，进程结束丢失
2. **intermediate_steps (scratchpad)**: 只在 AgentExecutor 的 while 循环中存在，调用结束丢失
3. **用户需求**: 想要看到完整的 agent 执行过程

---

## 解决方案

### 1. 持久化 Chat History

**新增文件**: `src/memory/persistent_memory.py`

```python
class FileBasedChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a JSON file."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.messages: List[BaseMessage] = []
        self._load()

    def _save(self):
        """Save messages to file."""
        data = [
            {'type': 'Human' if isinstance(m, HumanMessage) else 'AI', 'content': m.content}
            for m in self.messages
        ]
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
```

**存储位置**: `data/memory/chat_history.json`

---

### 2. 持久化 Scratchpad History

**新增文件**: `src/memory/scratchpad_history.py`

```python
class AgentScratchpadHistory:
    """Store agent scratchpad (intermediate steps) history to a JSON file."""

    def add_entry(self, query: str, scratchpad: str, final_answer: str = ""):
        self.entries.append({
            'query': query,
            'scratchpad': scratchpad,
            'final_answer': final_answer,
        })
        self._save()
```

**存储位置**: `data/memory/scratchpad_history.json`

---

### 3. 在 Agent 中集成

修改 `ResearchAgent`:

```python
class ResearchAgent:
    def __init__(self, ...):
        # 初始化 scratchpad 历史
        self.scratchpad_history = AgentScratchpadHistory(scratchpad_file)

    def run(self, query: str) -> str:
        result = self.executor.invoke({"input": query})

        # 保存 scratchpad 到文件
        if hasattr(self, 'scratchpad_history'):
            intermediate_steps = result.get('intermediate_steps', [])
            scratchpad_str = format_steps_to_scratchpad(intermediate_steps)
            final_answer = result.get('output', '')
            self.scratchpad_history.add_entry(query, scratchpad_str, final_answer)

        return result.get("output", "No output generated")
```

---

### 4. 关键: return_intermediate_steps=True

```python
self.executor = AgentExecutor(
    agent=self.agent,
    tools=tools,
    verbose=verbose,
    max_iterations=max_iterations,
    memory=self.memory,
    handle_parsing_errors=True,
    return_intermediate_steps=True,  # 必须开启才能获取 intermediate_steps
)
```

---

## 文件结构

```
data/memory/
├── chat_history.json      # 用户和 AI 的对话记录
└── scratchpad_history.json # Agent 的思考过程记录
```

---

## 管理工具

```bash
# 查看 chat history
python scripts/manage_memory.py --show-chat

# 查看 scratchpad 历史
python scripts/manage_memory.py --show-scratchpad

# 清空所有历史
python scripts/manage_memory.py --clear
```

---

## 关键教训

1. **AgentExecutor 默认不返回 intermediate_steps**
   - 必须设置 `return_intermediate_steps=True`
   - 否则 `result.get('intermediate_steps', [])` 永远为空

2. **scratchpad 格式化**
   ```python
   def format_steps_to_scratchpad(intermediate_steps):
       lines = []
       for action, observation in intermediate_steps:
           lines.append(action.log)  # "Thought: ... Action: ... Action Input: ..."
           lines.append(f"\nObservation: {observation}\n")
       return ''.join(lines)
   ```
   - `action.log` 包含 LLM 原始输出
   - `observation` 是工具执行结果

3. **save_context vs save scratchpad**
   - `save_context` 是 Chain 调用 memory 保存对话
   - scratchpad 是手动从 result 中提取并保存的

---

## 修改的文件

| 文件 | 说明 |
|------|------|
| `src/memory/persistent_memory.py` | 新增 - 文件持久化 chat history |
| `src/memory/scratchpad_history.py` | 新增 - scratchpad 持久化 |
| `src/agents/research_agent.py` | 添加 scratchpad_history 初始化和保存 |
| `scripts/manage_memory.py` | 更新 - 支持查看/清空 scratchpad |