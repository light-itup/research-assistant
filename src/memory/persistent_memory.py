"""Conversation memory with file-based persistence."""
import json
import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_classic.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)


class FileBasedChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a JSON file."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.messages: List[BaseMessage] = []
        self._load()

    def _load(self):
        """Load messages from file."""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.messages = [
                        HumanMessage(content=m['content']) if m['type'] == 'Human'
                        else AIMessage(content=m['content'])
                        for m in data
                    ]
                logger.info(f"Loaded {len(self.messages)} messages from {self.file_path}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load memory file: {e}, starting fresh")
                self.messages = []

    def _save(self):
        """Save messages to file."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {'type': 'Human' if isinstance(m, HumanMessage) else 'AI', 'content': m.content}
            for m in self.messages
        ]
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))
        self._save()

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))
        self._save()

    def clear(self) -> None:
        self.messages = []
        self._save()

    @property
    def messages_list(self) -> List[BaseMessage]:
        return self.messages


class PersistentResearchAssistantMemory(ConversationBufferMemory):
    """
    Research assistant memory with file-based persistence.

    Automatically saves conversation history to a JSON file and loads
    on initialization.
    """

    def __init__(
        self,
        max_history: int = 20,
        memory_file: Optional[Path] = None,
    ):
        """
        Initialize persistent conversation memory.

        Args:
            max_history: Maximum number of conversation turns to remember
            memory_file: Path to the memory file. Defaults to data/memory/chat_history.json
        """
        from src.config.settings import PROJECT_ROOT

        if memory_file is None:
            memory_file = PROJECT_ROOT / "data" / "memory" / "chat_history.json"

        super().__init__(
            chat_memory=FileBasedChatMessageHistory(memory_file),
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
            input_key="input",
        )
        object.__setattr__(self, 'max_history', max_history)
        object.__setattr__(self, 'memory_file', memory_file)

    def save_context(self, inputs: dict[str, any], outputs: dict[str, str]) -> None:
        """Save context from this conversation to the file-based chat history.

        Args:
            inputs: Dictionary with 'input' key containing user message
            outputs: Dictionary with 'output' key containing AI response
        """
        from langchain_core.messages import HumanMessage, AIMessage

        user_input = inputs.get('input', '')
        ai_output = outputs.get('output', '')

        # Add messages to the file-based chat history
        if user_input:
            self.chat_memory.add_user_message(user_input)
        if ai_output:
            self.chat_memory.add_ai_message(ai_output)

    def add_user_message(self, message: str) -> None:
        self.chat_memory.add_user_message(message)

    def add_ai_message(self, message: str) -> None:
        self.chat_memory.add_ai_message(message)

    def add_tool_result(self, tool_name: str, result: str) -> None:
        self.add_ai_message(f"[Tool Used: {tool_name}] Result: {result[:200]}...")

    def get_history(self) -> List[BaseMessage]:
        return list(self.chat_memory.messages)

    def clear(self) -> None:
        """Clear all conversation history from memory and file."""
        self.chat_memory.clear()

    @property
    def memory_file_path(self) -> Path:
        return self.memory_file