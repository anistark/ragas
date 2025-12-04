import typing as t

from ragas.llms.adapters.instructor import InstructorAdapter
from ragas.llms.adapters.litellm import LiteLLMAdapter

ADAPTERS = {
    "instructor": InstructorAdapter(),
    "litellm": LiteLLMAdapter(),
}


def get_adapter(name: str) -> t.Any:
    """
    Get adapter by name.

    Args:
        name: Adapter name ("instructor" or "litellm")

    Returns:
        StructuredOutputAdapter instance

    Raises:
        ValueError: If adapter name is unknown
    """
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTERS.keys())}")
    return ADAPTERS[name]


def auto_detect_adapter(client: t.Any, provider: str) -> str:
    """
    Auto-detect best adapter for client/provider combination.

    Logic:
    1. If client is from litellm module → use litellm
    2. If provider is gemini/google → use litellm
    3. If self-hosted LLM detected (localhost/127.0.0.1 in base_url) → use litellm
    4. Default → use instructor

    Args:
        client: Pre-initialized client
        provider: Provider name

    Returns:
        Adapter name ("instructor" or "litellm")
    """
    # Check if client is LiteLLM
    if hasattr(client, "__class__"):
        if "litellm" in client.__class__.__module__:
            return "litellm"

    # Check provider
    if provider.lower() in ("google", "gemini"):
        return "litellm"

    # Detect self-hosted LLMs by checking base_url
    # Self-hosted LLMs often have unreliable tool calling support,
    # so we use LiteLLM which falls back to JSON mode automatically
    if hasattr(client, "_base_url"):
        try:
            base_url = str(client._base_url)
            # Check for localhost or local IP addresses
            if any(
                indicator in base_url.lower()
                for indicator in ["localhost", "127.0.0.1", "0.0.0.0"]
            ):
                return "litellm"
        except (AttributeError, TypeError):
            # If we can't get base_url, just continue
            pass

    # Default
    return "instructor"


__all__ = [
    "get_adapter",
    "auto_detect_adapter",
    "ADAPTERS",
]
