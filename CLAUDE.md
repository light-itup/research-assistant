# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Research Assistant using LangChain + LlamaIndex with RAG + Agent architecture. The agent uses ReAct (Reasoning + Acting) pattern with multiple tools: RAG search, web search, code execution, and file operations.

## Common Commands

```bash
# Initialize knowledge base (run once or after docs change)
python scripts/init_knowledge_base.py

# Run agent tests
python -m tests.test_research_agent

# Manage memory
python scripts/manage_memory.py --show-chat        # view chat history
python scripts/manage_memory.py --show-scratchpad  # view agent thinking process
python scripts/manage_memory.py --clear            # clear all history
```

## Architecture

```
research_agent.py
    ├── create_react_agent() → creates Runnable (prompt | llm | ReActSingleInputOutputParser)
    ├── AgentExecutor (while loop)
    │   └── calls agent.plan() each iteration
    │       ├── LLM outputs: Thought/Action/Action Input
    │       ├── ReActSingleInputOutputParser extracts tool name + input via regex
    │       ├── Tool executes → returns observation
    │       └── accumulated as intermediate_steps for next iteration
    ├── memory (ConversationBufferMemory) - chat history persistence
    └── scratchpad_history - agent thinking process persistence
```

## Key Files

- `src/agents/research_agent.py` - ReAct Agent implementation
- `src/rag/index_manager.py` - Singleton index manager (avoids rebuilding on every query)
- `src/memory/persistent_memory.py` - File-based chat history
- `src/memory/scratchpad_history.py` - File-based scratchpad persistence
- `scripts/init_knowledge_base.py` - One-time index initialization

## Important Patterns

### ReAct Output Parsing
LangChain's `ReActSingleInputOutputParser` uses regex to parse LLM output:
```regex
Action\s*\d*\s*:[\s]*(.*?)Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)
```
To use custom labels (e.g., `ToolCall` instead of `Action`), you must provide a custom `AgentOutputParser`.

### Tool JSON Parameters
LLM may send JSON objects as tool input (e.g., `{"directory_path": ".", "pattern": "*.py"}`). File tools handle this by JSON parsing at function start.

### return_intermediate_steps
Must set `return_intermediate_steps=True` on `AgentExecutor` to capture intermediate steps for scratchpad history.

### Prompt Template Variables
Tool descriptions with JSON examples like `{param}` are interpreted as template variables. Use regex to strip them:
```python
tool_desc_clean = re.sub(r'\{[^{}]*\}', '', tool_desc)
```

## Experience Documents

Key lessons are stored in `docs/lessons/`:
- `phase-7.5.1-file-tool-debugging.md` - Tool debugging issues
- `phase-7.6-memory-persistence.md` - Memory persistence implementation