"""Agent scratchpad history with file-based persistence."""
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.agents import AgentAction

logger = logging.getLogger(__name__)


class AgentScratchpadHistory:
    """Store agent scratchpad (intermediate steps) history to a JSON file."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.entries: List[dict] = []
        self._load()

    def _load(self):
        """Load entries from file."""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.entries = json.load(f)
                logger.info(f"Loaded {len(self.entries)} scratchpad entries from {self.file_path}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load scratchpad file: {e}, starting fresh")
                self.entries = []

    def _save(self):
        """Save entries to file."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def add_entry(self, query: str, scratchpad: str, final_answer: str = ""):
        """Add a scratchpad entry.

        Args:
            query: The user query
            scratchpad: The full scratchpad content (concatenated Thought/Action/Observation)
            final_answer: The final answer from agent
        """
        self.entries.append({
            'query': query,
            'scratchpad': scratchpad,
            'final_answer': final_answer,
            'timestamp': str(Path(self.file_path).stat().st_mtime if self.file_path.exists() else 0)
        })
        # Keep only last 50 entries
        if len(self.entries) > 50:
            self.entries = self.entries[-50:]
        self._save()

    def clear(self):
        """Clear all entries."""
        self.entries = []
        self._save()

    @property
    def history(self) -> List[dict]:
        return self.entries


def format_steps_to_scratchpad(intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
    """Format intermediate steps to a readable scratchpad string.

    Args:
        intermediate_steps: List of (action, observation) tuples

    Returns:
        Formatted scratchpad string
    """
    if not intermediate_steps:
        return ""

    lines = []
    for action, observation in intermediate_steps:
        # action.log contains the original LLM output: "Thought: ... Action: ... Action Input: ..."
        lines.append(action.log)
        lines.append(f"\nObservation: {observation}\n")

    return ''.join(lines)