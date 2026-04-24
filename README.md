# Research Assistant

An AI-powered research assistant built with LangChain and LlamaIndex.

## Features

- **RAG Knowledge Base**: Upload documents (PDF, TXT, MD) to build a searchable knowledge base
- **Web Search**: Search the internet for latest information
- **Code Assistant**: Explain and debug code
- **Multi-turn Conversation**: Remember context across conversations

## Architecture

- LangChain: Agent orchestration, tool definitions, memory management
- LlamaIndex: Document processing, RAG pipeline
- Streamlit: Web UI

## Project Structure

```
research-assistant/
├── src/
│   ├── agents/          # Agent implementations
│   ├── tools/           # Tool definitions
│   ├── rag/             # RAG pipeline
│   ├── memory/          # Conversation memory
│   └── config/          # Configuration
├── data/
│   └── knowledge_base/  # Document storage
└── tests/
```

## Setup

1. Create conda environment:
```bash
conda create -n research-assistant python=3.11
conda activate research-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys in `.env`:
```
OPENAI_API_KEY=sk-xxx
TAVILY_API_KEY=tvly-xxx
```

4. Run:
```bash
streamlit run streamlit_app.py
```

## Learning Goals

This project demonstrates:
- RAG (Retrieval Augmented Generation) with LlamaIndex
- Agent design with LangChain
- Tool calling and ReAct reasoning
- Conversation memory management
