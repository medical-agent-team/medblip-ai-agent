"""
Observability module for Langfuse integration.
Provides tracing and monitoring for multi-agent medical consultation system.
"""

import os
import logging
from typing import Optional, List
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


def get_langfuse_handler() -> Optional[BaseCallbackHandler]:
    """
    Get Langfuse callback handler if enabled.

    Returns:
        CallbackHandler instance if Langfuse is enabled, None otherwise

    Environment Variables:
        LANGFUSE_ENABLED: Set to "true" to enable Langfuse tracing
        LANGFUSE_PUBLIC_KEY: Langfuse public API key
        LANGFUSE_SECRET_KEY: Langfuse secret API key
        LANGFUSE_HOST: Langfuse host URL (default: https://cloud.langfuse.com)
    """
    if os.getenv("LANGFUSE_ENABLED", "false").lower() != "true":
        logger.info("Langfuse tracing is disabled")
        return None

    try:
        from langfuse.callback import CallbackHandler

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            logger.warning(
                "Langfuse is enabled but LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY "
                "is not set. Tracing will be disabled."
            )
            return None

        handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

        logger.info(f"Langfuse tracing enabled at {host}")
        return handler

    except ImportError:
        logger.warning(
            "Langfuse is enabled but langfuse package is not installed. "
            "Run 'make add PKG=langfuse' to install."
        )
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse handler: {e}")
        return None


def get_callbacks() -> List[BaseCallbackHandler]:
    """
    Get all enabled callback handlers.

    Returns:
        List of active callback handlers (Langfuse, etc.)
    """
    callbacks = []

    # Add Langfuse handler if enabled
    langfuse_handler = get_langfuse_handler()
    if langfuse_handler:
        callbacks.append(langfuse_handler)

    return callbacks


def configure_langchain_tracing():
    """
    Configure LangChain tracing settings.
    Call this early in application startup (e.g., in main.py).
    """
    # Enable verbose LangChain logging if Langfuse is enabled
    if os.getenv("LANGFUSE_ENABLED", "false").lower() == "true":
        os.environ["LANGCHAIN_VERBOSE"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"  # Disable LangSmith
        logger.info("LangChain verbose logging enabled for Langfuse")
    else:
        os.environ["LANGCHAIN_VERBOSE"] = os.getenv("LANGCHAIN_VERBOSE", "false")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
