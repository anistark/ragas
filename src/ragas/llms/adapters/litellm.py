import typing as t

from ragas.llms.adapters.base import StructuredOutputAdapter

if t.TYPE_CHECKING:
    from ragas.llms.litellm_llm import LiteLLMStructuredLLM


class LiteLLMAdapter(StructuredOutputAdapter):
    """
    Adapter using LiteLLM for structured outputs.

    Supports: All 100+ LiteLLM providers (Gemini, Ollama, vLLM, Groq, etc.)
    """

    def create_llm(
        self,
        client: t.Any,
        model: str,
        provider: str,
        **kwargs,
    ) -> "LiteLLMStructuredLLM":
        """
        Create LiteLLMStructuredLLM instance.

        Args:
            client: Pre-initialized client
            model: Model name
            provider: Provider name
            **kwargs: Additional model arguments

        Returns:
            LiteLLMStructuredLLM instance
        """
        from ragas.llms.base import _get_instructor_client
        from ragas.llms.litellm_llm import LiteLLMStructuredLLM

        # Wrap the client with instructor for structured output support
        # This is necessary for both sync and async clients
        patched_client = _get_instructor_client(client, provider)

        return LiteLLMStructuredLLM(
            client=patched_client,
            model=model,
            provider=provider,
            original_client=client,  # Pass original for JSON mode fallback
            **kwargs,
        )
