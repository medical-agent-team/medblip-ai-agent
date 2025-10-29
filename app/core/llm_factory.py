"""
Centralized LLM factory for consistent model configuration across all agents.
Supports both OpenAI API and vLLM endpoints.
"""

import os
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler


def get_llm(
    temperature: float = 0.7,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    max_tokens: int = 1024,
    **kwargs
) -> ChatOpenAI:
    """
    Get configured LLM instance with support for vLLM and OpenAI endpoints.

    Args:
        temperature: Model temperature (0.0-1.0). Default 0.7
        model: Model name override. If None, uses OPENAI_MODEL env var
        api_key: API key override. If None, uses OPENAI_API_KEY env var
        callbacks: List of callback handlers (e.g., Langfuse)
        **kwargs: Additional arguments passed to ChatOpenAI

    Returns:
        Configured ChatOpenAI instance

    Environment Variables:
        OPENAI_API_BASE: Base URL for API endpoint (default: OpenAI)
                        Example: http://your-vllm-server/v1
        OPENAI_API_KEY: API key (use "dummy" for vLLM)
        OPENAI_MODEL: Model name (default: gpt-4o-mini)
                     Example for vLLM: "gpt-oss-20b"
    """
    # Get configuration from environment with fallbacks
    # Handle empty strings in env vars by using `or` chaining
    base_url = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or "dummy"
    resolved_model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    # Initialize callbacks list
    resolved_callbacks = callbacks or []

    # Create LLM instance
    llm = ChatOpenAI(
        api_key=resolved_api_key,
        base_url=base_url,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=resolved_callbacks,
        **kwargs
    )

    return llm


def get_llm_for_agent(
    agent_type: str,
    api_key: Optional[str] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
) -> ChatOpenAI:
    """
    Get LLM configured for specific agent type with appropriate temperature.

    Args:
        agent_type: Type of agent (admin, doctor, supervisor, radiology)
        api_key: Optional API key override
        callbacks: Optional callback handlers

    Returns:
        Configured ChatOpenAI instance with agent-specific settings
    """
    # Agent-specific temperature settings
    temperature_map = {
        "admin": 0.7,      # Creative patient-friendly translation
        "doctor": 0.7,     # Creative diagnostic reasoning
        "supervisor": 0.3,  # Consistent consensus evaluation
        "radiology": 0.3,   # Precise medical consultation
        "generic": 0.7,     # Default for generic agent
    }

    max_tokens_map = {
        "admin": 4096,
        "doctor": 700,
        "supervisor": 600,
        "radiology": 1024,
        "generic": 1024,
    }

    temperature = temperature_map.get(agent_type, 0.7)
    max_tokens = max_tokens_map.get(agent_type, 1024)

    return get_llm(
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        callbacks=callbacks
    )
