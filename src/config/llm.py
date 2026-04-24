"""LLM initialization module."""
from openai import OpenAI
from src.config.settings import (
    ALIYUN_API_KEY,
    DASHSCOPE_BASE_URL,
    LLM_MODEL,
    LLM_MODEL_THINKING,
)


def create_llm(
    model: str = None,
    thinking: bool = False,
    api_key: str = None,
    base_url: str = None,
    **kwargs
) -> OpenAI:
    """
    Create an OpenAI-compatible LLM client.

    Args:
        model: Model name. Defaults to settings.LLM_MODEL
        thinking: Whether to enable deep thinking mode
        api_key: API key. Defaults to ALIYUN_API_KEY
        base_url: Base URL. Defaults to DASHSCOPE_BASE_URL
        **kwargs: Additional arguments passed to OpenAI client

    Returns:
        OpenAI client instance
    """
    actual_model = model or LLM_MODEL
    if thinking and LLM_MODEL_THINKING:
        actual_model = LLM_MODEL_THINKING

    return OpenAI(
        api_key=api_key or ALIYUN_API_KEY,
        base_url=base_url or DASHSCOPE_BASE_URL,
        **kwargs
    )


def create_chat_completion(model: str = None, thinking: bool = False, **kwargs):
    """
    Create a chat completion using Aliyun DashScope.

    This is a convenience function that creates an LLM client and returns
    a chat completion in one call.
    """
    client = create_llm(model=model, thinking=thinking)
    return client.chat.completions.create(
        model=model or (LLM_MODEL_THINKING if thinking else LLM_MODEL),
        **kwargs
    )
