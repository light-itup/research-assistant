"""Research Agent with ReAct (Reasoning + Acting) pattern."""
from typing import List, Optional, Callable
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.llm import create_llm
from src.config.settings import LLM_MODEL
from src.memory.persistent_memory import PersistentResearchAssistantMemory
from src.memory.scratchpad_history import AgentScratchpadHistory, format_steps_to_scratchpad


# ReAct Agent System Prompt
SYSTEM_PROMPT = """You are a Research Assistant powered by AI. Your goal is to help users with:

1. **Knowledge Base Questions** - Searching and explaining concepts from uploaded documents
2. **Web Search** - Finding up-to-date information from the internet
3. **Code Understanding** - Explaining and executing Python code
4. **File Operations** - Reading and writing files in the project

## Available Tools

You have access to the following tools:

{tools}

## Response Format

You must respond with ONE of the following formats ONLY:

**To use a tool:**
```
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: [json object with parameter names and values]
```

**After receiving a tool result, continue with:**
```
Thought: [Your reasoning about the observation]
Action: [next_tool_name or Final Answer]
Action Input: [...]
```

**To give the final answer:**
```
Thought: I now have enough information to answer.
Final Answer: [Your comprehensive answer here]
```

## Critical Format Rules

1. EVERY tool use MUST have exactly one Action line followed by exactly one Action Input line
2. NEVER add extra colons after Action
3. The Action Input must be a JSON object
4. NEVER include any other text between Thought and Action
5. NEVER include any other text between Action and Action Input

## Code Execution Guidelines

When using `execute_python_code` or `explain_code`:
- The `code` parameter should be the **raw Python code as a string**
- For expressions like list comprehensions, put quotes around them
- Example: Action Input with code as a string variable
- Result: `[0, 1, 4, 9, 16]`
"""


def create_react_prompt(
    tools: List[Callable],
    system_message: str = SYSTEM_PROMPT
) -> PromptTemplate:
    """
    Create a ReAct-style prompt template.

    Args:
        tools: List of tools available to the agent
        system_message: Custom system prompt

    Returns:
        PromptTemplate for ReAct agent
    """
    # Build tools description - avoid JSON braces that confuse prompt template parser
    tools_desc = []
    for tool in tools:
        tool_name = getattr(tool, 'name', tool.__class__.__name__)
        tool_desc = getattr(tool, 'description', 'No description')
        # Strip JSON examples from description to avoid prompt template conflicts
        import re
        tool_desc_clean = re.sub(r'\{[^{}]*\}', '', tool_desc)
        tools_desc.append(f"- **{tool_name}**: {tool_desc_clean.strip()}")

    tools_section = '\n'.join(tools_desc)

    template = f"""{system_message}

## Available Tools

{tools_section}

## Tool Names

{{tool_names}}

## Conversation History

{{chat_history}}

## Current User Question

{{input}}

## Agent Scratchpad (Tool Execution History)

{{agent_scratchpad}}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "chat_history", "agent_scratchpad", "tool_names"]
    )

    return prompt


class ResearchAgent:
    """
    Research Assistant Agent using ReAct (Reasoning + Acting) pattern.

    The agent follows this loop:
    1. Thought - Reason about what to do next
    2. Action - Choose and execute a tool
    3. Observation - See the result
    4. Repeat until ready for Final Answer
    """

    def __init__(
        self,
        tools: List[Callable],
        model: str = LLM_MODEL,
        verbose: bool = True,
        max_iterations: int = 10,
        memory: Optional[PersistentResearchAssistantMemory] = None,
    ):
        """
        Initialize the Research Agent.

        Args:
            tools: List of LangChain tools to make available
            model: LLM model name
            verbose: Whether to print thought process
            max_iterations: Max ReAct loop iterations
            memory: Conversation memory for multi-turn dialogue (persistent)
        """
        from src.config.settings import PROJECT_ROOT

        self.tools = tools
        self.verbose = verbose
        self.max_iterations = max_iterations

        # Initialize LLM
        self.llm = create_llm(model=model)

        # Initialize memory (use persistent memory by default)
        self.memory = memory or PersistentResearchAssistantMemory()

        # Initialize scratchpad history (stores intermediate steps)
        scratchpad_file = PROJECT_ROOT / "data" / "memory" / "scratchpad_history.json"
        self.scratchpad_history = AgentScratchpadHistory(scratchpad_file)

        # Create ReAct prompt
        self.prompt = create_react_prompt(tools)

        # Create the agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=self.prompt,
        )

        # Create agent executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=verbose,
            max_iterations=max_iterations,
            memory=self.memory,
            handle_parsing_errors=True,
            return_intermediate_steps=True,  # Enable to capture for scratchpad history
        )

    def run(self, query: str) -> str:
        """
        Run the agent on a query.

        Args:
            query: User question

        Returns:
            Agent's final answer
        """
        # Get intermediate steps after execution
        # AgentExecutor returns {"output": ..., "intermediate_steps": ...}
        result = self.executor.invoke({"input": query})

        # Save scratchpad to file
        if hasattr(self, 'scratchpad_history'):
            intermediate_steps = result.get('intermediate_steps', [])
            scratchpad_str = format_steps_to_scratchpad(intermediate_steps)
            final_answer = result.get('output', '')
            self.scratchpad_history.add_entry(query, scratchpad_str, final_answer)

        return result.get("output", "No output generated")

    async def run_async(self, query: str) -> str:
        """
        Run the agent asynchronously.

        Args:
            query: User question

        Returns:
            Agent's final answer
        """
        result = await self.executor.ainvoke({"input": query})
        return result.get("output", "No output generated")

    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()

    def get_memory(self) -> List:
        """Get conversation history."""
        return self.memory.load_memory_variables({})


def create_research_agent(
    tools: List[Callable],
    model: str = LLM_MODEL,
    verbose: bool = True,
    memory: Optional[PersistentResearchAssistantMemory] = None,
) -> ResearchAgent:
    """
    Factory function to create a Research Agent.

    Args:
        tools: List of LangChain tools
        model: LLM model name
        verbose: Whether to print thought process
        memory: Conversation memory for multi-turn dialogue (optional, uses persistent by default)

    Returns:
        Configured ResearchAgent instance
    """
    return ResearchAgent(
        tools=tools,
        model=model,
        verbose=verbose,
        memory=memory,
    )
