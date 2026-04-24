"""Config module."""
from src.config.settings import (
    PROJECT_ROOT,
    DATA_DIR,
    KNOWLEDGE_BASE_DIR,
    CHROMA_DB_DIR,
    ALIYUN_API_KEY,
    MINIMAX_API_KEY,
    TAVILY_API_KEY,
    DASHSCOPE_BASE_URL,
    LLM_MODEL,
    LLM_MODEL_THINKING,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from src.config.llm import create_llm, create_chat_completion

__all__ = [
    # Settings
    "PROJECT_ROOT",
    "DATA_DIR",
    "KNOWLEDGE_BASE_DIR",
    "CHROMA_DB_DIR",
    "ALIYUN_API_KEY",
    "MINIMAX_API_KEY",
    "TAVILY_API_KEY",
    "DASHSCOPE_BASE_URL",
    "LLM_MODEL",
    "LLM_MODEL_THINKING",
    "EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    # LLM
    "create_llm",
    "create_chat_completion",
]
