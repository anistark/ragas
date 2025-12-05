import asyncio
import inspect
import logging
import threading
import typing as t

from ragas._analytics import LLMUsageEvent, track
from ragas.llms.base import InstructorBaseRagasLLM, InstructorTypeVar

logger = logging.getLogger(__name__)


class LiteLLMStructuredLLM(InstructorBaseRagasLLM):
    """
    LLM wrapper using LiteLLM for structured outputs.

    Works with all 100+ LiteLLM-supported providers including Gemini,
    Ollama, vLLM, Groq, and many others.

    The LiteLLM client should be initialized with structured output support.
    """

    def __init__(
        self,
        client: t.Any,
        model: str,
        provider: str,
        original_client: t.Optional[t.Any] = None,
        **kwargs,
    ):
        """
        Initialize LiteLLM structured LLM.

        Args:
            client: LiteLLM client instance (instructor-wrapped)
            model: Model name (e.g., "gemini-2.0-flash")
            provider: Provider name
            original_client: Original unwrapped client (for fallback recreation)
            **kwargs: Additional model arguments (temperature, max_tokens, etc.)
        """
        self.client = client
        self.model = model
        self.provider = provider
        self.model_args = kwargs
        self.original_client = original_client  # Store for fallback

        # Check if client is async-capable at initialization
        self.is_async = self._check_client_async()

    def _check_client_async(self) -> bool:
        """Determine if the client is async-capable.

        Handles multiple cases:
        1. Direct async clients (e.g., litellm Router with acompletion)
        2. Instructor-wrapped AsyncInstructor clients
        3. Instructor-wrapped Instructor clients (need to check underlying client)
        """
        try:
            # Check if this is an AsyncInstructor wrapper (instructor.AsyncInstructor)
            if self.client.__class__.__name__ == "AsyncInstructor":
                return True

            # Check for direct async completion method (e.g., litellm Router)
            if hasattr(self.client, "acompletion"):
                is_coroutine = inspect.iscoroutinefunction(self.client.acompletion)
                if is_coroutine:
                    return True

            # Check for async chat completion (works with instructor-wrapped OpenAI clients)
            if hasattr(self.client, "chat") and hasattr(
                self.client.chat, "completions"
            ):
                if hasattr(self.client.chat.completions, "create"):
                    if inspect.iscoroutinefunction(self.client.chat.completions.create):
                        return True

            # For instructor-wrapped sync clients that wrap async underlying clients,
            # check if the wrapped client has async methods
            if hasattr(self.client, "client"):
                # This is an instructor-wrapped client, check the underlying client
                underlying = self.client.client
                if hasattr(underlying, "acompletion"):
                    is_coroutine = inspect.iscoroutinefunction(underlying.acompletion)
                    if is_coroutine:
                        return True

            # For instructor-wrapped clients, also check the closure of create_fn
            # This handles cases where the underlying client is stored in a closure
            # (e.g., when instructor.from_litellm wraps a litellm Router)
            if (
                hasattr(self.client, "create_fn")
                and hasattr(self.client.create_fn, "__closure__")
                and self.client.create_fn.__closure__
            ):
                for cell in self.client.create_fn.__closure__:
                    try:
                        obj = cell.cell_contents
                        # Check if the closure object has acompletion (e.g., litellm Router)
                        if hasattr(obj, "acompletion"):
                            if inspect.iscoroutinefunction(obj.acompletion):
                                return True
                    except (ValueError, AttributeError):
                        # cell_contents might not be accessible, or object might not have acompletion
                        pass

            return False
        except (AttributeError, TypeError):
            return False

    def _fallback_to_json_mode(self) -> None:
        """
        Fallback to JSON mode when tool calling fails.

        Recreates the instructor client with Mode.JSON instead of Mode.TOOLS.
        This is used when self-hosted LLMs return multiple tool calls or don't
        properly support OpenAI's function calling protocol.
        """
        if self.original_client is None:
            logger.warning(
                "Cannot fallback to JSON mode: original_client not available. "
                "Continuing with current client."
            )
            return

        try:
            import instructor

            logger.warning(
                f"Model {self.model} returned multiple tool calls or doesn't support "
                f"tool calling properly. Falling back to JSON mode (Mode.JSON)."
            )

            # Recreate the client with JSON mode
            if self.provider.lower() == "openai":
                self.client = instructor.from_openai(
                    self.original_client, mode=instructor.Mode.JSON
                )
            elif self.provider.lower() == "litellm":
                self.client = instructor.from_litellm(
                    self.original_client, mode=instructor.Mode.JSON
                )
            else:
                # For other providers, try generic approach with JSON mode
                self.client = instructor.from_openai(
                    self.original_client, mode=instructor.Mode.JSON
                )

            logger.info(f"Successfully switched to JSON mode for {self.model}")

        except Exception as e:
            logger.error(f"Failed to fallback to JSON mode: {e}")
            # Continue with existing client
            pass

    def _run_async_in_current_loop(self, coro: t.Awaitable[t.Any]) -> t.Any:
        """Run an async coroutine in the current event loop if possible.

        This handles Jupyter environments correctly by using a separate thread
        when a running event loop is detected.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()

            if loop.is_running():
                # If the loop is already running (like in Jupyter notebooks),
                # we run the coroutine in a separate thread with its own event loop
                result_container: t.Dict[str, t.Any] = {
                    "result": None,
                    "exception": None,
                }

                def run_in_thread():
                    # Create a new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        # Run the coroutine in this thread's event loop
                        result_container["result"] = new_loop.run_until_complete(coro)
                    except Exception as e:
                        # Capture any exceptions to re-raise in the main thread
                        result_container["exception"] = e
                    finally:
                        # Clean up the event loop
                        new_loop.close()

                # Start the thread and wait for it to complete
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()

                # Re-raise any exceptions that occurred in the thread
                if result_container["exception"]:
                    raise result_container["exception"]

                return result_container["result"]
            else:
                # Standard case - event loop exists but isn't running
                return loop.run_until_complete(coro)

        except RuntimeError:
            # If we get a runtime error about no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                # Clean up
                loop.close()
                asyncio.set_event_loop(None)

    def generate(
        self, prompt: str, response_model: t.Type[InstructorTypeVar]
    ) -> InstructorTypeVar:
        """Generate a response using the configured LLM.

        For async clients, this will run the async method in the appropriate event loop.

        Args:
            prompt: Input prompt
            response_model: Pydantic model for structured output

        Returns:
            Instance of response_model with generated data
        """
        messages = [{"role": "user", "content": prompt}]  # type: ignore

        # If client is async, use the appropriate method to run it
        if self.is_async:
            result = self._run_async_in_current_loop(
                self.agenerate(prompt, response_model)
            )
        else:
            # Call LiteLLM with structured output, with fallback to JSON mode
            try:
                result = self.client.chat.completions.create(  # type: ignore[call-overload]
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    response_model=response_model,
                    **self.model_args,
                )
            except Exception as e:
                # Check if this is the multiple tool calls error
                error_message = str(e)
                if "multiple tool calls" in error_message.lower() or (
                    "InstructorRetryException" in type(e).__name__
                    and "multiple tool calls" in error_message.lower()
                ):
                    # Fallback to JSON mode and retry
                    logger.info(
                        "Retrying with JSON mode due to multiple tool calls error"
                    )
                    self._fallback_to_json_mode()

                    # Retry with JSON mode
                    result = self.client.chat.completions.create(  # type: ignore[call-overload]
                        model=self.model,
                        messages=messages,  # type: ignore[arg-type]
                        response_model=response_model,
                        **self.model_args,
                    )
                else:
                    # Re-raise other exceptions
                    raise

        # Track the usage
        track(
            LLMUsageEvent(
                provider=self.provider,
                model=self.model,
                llm_type="litellm",
                num_requests=1,
                is_async=self.is_async,
            )
        )
        return result

    async def agenerate(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
    ) -> InstructorTypeVar:
        """Asynchronously generate a response using the configured LLM.

        Args:
            prompt: Input prompt
            response_model: Pydantic model for structured output

        Returns:
            Instance of response_model with generated data
        """
        messages = [{"role": "user", "content": prompt}]  # type: ignore

        # If client is not async, raise a helpful error
        if not self.is_async:
            raise TypeError(
                "Cannot use agenerate() with a synchronous client. Use generate() instead."
            )

        # Call LiteLLM async with structured output, with fallback to JSON mode
        try:
            result = await self.client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                response_model=response_model,
                **self.model_args,
            )
        except Exception as e:
            # Check if this is the multiple tool calls error
            error_message = str(e)
            if "multiple tool calls" in error_message.lower() or (
                "InstructorRetryException" in type(e).__name__
                and "multiple tool calls" in error_message.lower()
            ):
                # Fallback to JSON mode and retry
                logger.info("Retrying with JSON mode due to multiple tool calls error")
                self._fallback_to_json_mode()

                # Retry with JSON mode
                result = await self.client.chat.completions.create(  # type: ignore[call-overload]
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    response_model=response_model,
                    **self.model_args,
                )
            else:
                # Re-raise other exceptions
                raise

        # Track the usage
        track(
            LLMUsageEvent(
                provider=self.provider,
                model=self.model,
                llm_type="litellm",
                num_requests=1,
                is_async=True,
            )
        )
        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model!r}, "
            f"provider={self.provider!r}, "
            f"is_async={self.is_async})"
        )
