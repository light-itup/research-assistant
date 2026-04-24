"""Application configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY", "")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# LLM settings - Aliyun DashScope (OpenAI compatible)
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-flash")
LLM_MODEL_THINKING = os.getenv("LLM_MODEL_THINKING", "qwen-flash")

# MiniMax (可选，用于需要 extended thinking 的场景)
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

# Embedding settings - Aliyun
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# RAG settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
