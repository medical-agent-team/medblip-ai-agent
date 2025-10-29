"""
Centralized LLM factory for consistent model configuration across all agents.
Supports both OpenAI API and vLLM endpoints.
"""

import os
import logging
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


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
        max_tokens: Maximum tokens to generate. Default 1024.
                   IMPORTANT: vLLM requires this to be set explicitly.
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

    # Log configuration details
    logger.info("=" * 80)
    logger.info("🔍 [LLM Factory] Creating ChatOpenAI instance...")
    logger.info(f"   Model: {resolved_model}")
    logger.info(f"   Base URL: {base_url}")
    logger.info(f"   API Key: {resolved_api_key[:15]}... (truncated)")
    logger.info(f"   Temperature: {temperature}")
    logger.info(f"   Max Tokens: {max_tokens}")
    logger.info(f"   Callbacks: {[type(cb).__name__ for cb in resolved_callbacks]}")

    # Create LLM instance
    try:
        llm = ChatOpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=resolved_callbacks,
            **kwargs
        )
        logger.info("✅ [LLM Factory] ChatOpenAI instance created successfully")
    except Exception as e:
        logger.error(f"❌ [LLM Factory] Failed to create ChatOpenAI: {type(e).__name__}: {str(e)}")
        raise

    # CRITICAL: Test the LLM immediately after creation
    logger.info("🧪 [LLM Factory] Testing LLM with simple invocation...")
    try:
        test_response = llm.invoke([HumanMessage(content="Hello")])

        if not test_response:
            logger.error("❌ [LLM Factory] Test failed: Response is None")
            raise ValueError("LLM test invocation returned None")

        if not hasattr(test_response, 'content'):
            logger.error(f"❌ [LLM Factory] Test failed: Response has no 'content' attribute. Type: {type(test_response)}")
            raise ValueError(f"LLM response missing 'content' attribute: {type(test_response)}")

        if not test_response.content:
            logger.error("❌ [LLM Factory] Test failed: Response content is empty")
            raise ValueError("LLM test invocation returned empty content")

        logger.info(f"✅ [LLM Factory] Test successful! Response length: {len(test_response.content)} chars")
        logger.info(f"   First 100 chars: {test_response.content[:100]}")

    except Exception as e:
        logger.error(f"❌ [LLM Factory] Test invocation FAILED: {type(e).__name__}: {str(e)}")
        logger.error(f"   This LLM instance will NOT work properly!")
        logger.error(f"   Base URL: {base_url}")
        logger.error(f"   Model: {resolved_model}")
        logger.error(f"   Callbacks: {[type(cb).__name__ for cb in resolved_callbacks]}")
        raise RuntimeError(f"LLM test invocation failed: {str(e)}") from e

    logger.info("=" * 80)
    return llm


def get_llm_for_agent(
    agent_type: str,
    api_key: Optional[str] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
) -> ChatOpenAI:
    """
    Get LLM configured for specific agent type with appropriate temperature and max_tokens.

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

    # Agent-specific max_tokens settings
    # IMPORTANT: vLLM requires max_tokens to be set explicitly
    max_tokens_map = {
        "admin": 4096,      # Patient summaries can be lengthy
        # Keep doctor/supervisor outputs within limits to avoid truncation on vLLM.
        # Empirically, 700 tokens provides enough headroom for structured responses
        # while staying well below the server-side cap.
        "doctor": 700,      # Diagnostic opinions and critiques
        "supervisor": 600,  # Consensus analysis with structured output
        "radiology": 1024,  # Structured medical findings
        "generic": 2048,    # Default for generic agent
    }

    temperature = temperature_map.get(agent_type, 0.7)
    max_tokens = max_tokens_map.get(agent_type, 1024)

    return get_llm(
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        callbacks=callbacks
    )
