"""CLI tool to manage persistent memory and scratchpad history."""
import sys
import argparse

# Add project root to path
sys.path.insert(0, '.')

from src.memory.persistent_memory import PersistentResearchAssistantMemory
from src.memory.scratchpad_history import AgentScratchpadHistory


def main():
    parser = argparse.ArgumentParser(description="Manage persistent chat memory and scratchpad")
    parser.add_argument('--clear', action='store_true', help='Clear all conversation history and scratchpad')
    parser.add_argument('--show-chat', action='store_true', help='Show current chat history')
    parser.add_argument('--show-scratchpad', action='store_true', help='Show scratchpad history')
    args = parser.parse_args()

    memory = PersistentResearchAssistantMemory()
    scratchpad = AgentScratchpadHistory(
        memory.memory_file.parent / "scratchpad_history.json"
    )

    if args.clear:
        memory.clear()
        scratchpad.clear()
        print("Conversation history and scratchpad cleared.")
    elif args.show_chat:
        messages = memory.get_history()
        if not messages:
            print("No conversation history.")
        else:
            print(f"Conversation history ({len(messages)} messages):")
            for i, msg in enumerate(messages, 1):
                role = "Human" if msg.type == "human" else "AI"
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                print(f"  [{i}] {role}: {content}")
    elif args.show_scratchpad:
        history = scratchpad.history
        if not history:
            print("No scratchpad history.")
        else:
            print(f"Scratchpad history ({len(history)} entries):\n")
            for i, entry in enumerate(history, 1):
                print(f"=== Entry {i} ===")
                print(f"Query: {entry['query'][:80]}...")
                scratchpad_preview = entry.get('scratchpad', '')[:200]
                print(f"Scratchpad (preview): {scratchpad_preview}...")
                print()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()