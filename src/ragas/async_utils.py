"""Async utils."""

import asyncio
import logging
from typing import Any, Coroutine, List, Optional

from tqdm.auto import tqdm

from ragas.utils import batched

logger = logging.getLogger(__name__)


def safe_nest_asyncio_apply() -> bool:
    """
    Safely apply nest_asyncio only if compatible with the current event loop.

    Returns:
        bool: True if nest_asyncio was applied, False if skipped due to incompatibility

    Raises:
        ImportError: If nest_asyncio is required but not installed
        ValueError: If an unexpected error occurs during patching
    """
    try:
        import nest_asyncio
    except ImportError as e:
        raise ImportError(
            "It seems like your running this in a jupyter-like environment. "
            "Please install nest_asyncio with `pip install nest_asyncio` to make it work."
        ) from e

    # Check if we're dealing with uvloop which doesn't support nest_asyncio patching
    try:
        loop = asyncio.get_running_loop()
        loop_type_name = type(loop).__name__

        # uvloop.Loop and other non-standard event loops can't be patched
        if "uvloop" in loop_type_name.lower() or "uvloop" in str(type(loop)):
            logger.debug(
                f"Skipping nest_asyncio.apply() for incompatible loop type: {type(loop)}"
            )
            return False

    except RuntimeError:
        # No running loop, safe to proceed
        pass

    try:
        nest_asyncio.apply()
        logger.debug("Successfully applied nest_asyncio patch")
        return True
    except ValueError as e:
        error_msg = str(e)
        if "Can't patch loop of type" in error_msg:
            # This is the uvloop incompatibility - not an error, just log and skip
            logger.debug(
                f"Skipping nest_asyncio.apply() due to incompatible event loop: {error_msg}"
            )
            return False
        else:
            # Some other ValueError we didn't expect
            raise


def is_uvloop_event_loop():
    """Check if uvloop is being used as the event loop policy."""
    policy = asyncio.get_event_loop_policy()
    return "uvloop" in str(type(policy)) or "uvloop" in policy.__class__.__module__


def safe_asyncio_run(coro):
    """
    Safely run a coroutine, handling both standard asyncio and uvloop.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    # Check if we already have a running event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're in a running loop, we can't use asyncio.run
        # For uvloop, nest_asyncio doesn't work, so we use loop.run_until_complete
        if is_uvloop_event_loop():
            logger.debug("Using loop.run_until_complete for uvloop compatibility")
            return loop.run_until_complete(coro)
        else:
            # Standard asyncio loop - try to apply nest_asyncio and use asyncio.run
            if safe_nest_asyncio_apply():
                logger.debug("Using asyncio.run with nest_asyncio patch")
                return asyncio.run(coro)
            else:
                # Fallback to loop.run_until_complete if nest_asyncio failed
                logger.debug("Using loop.run_until_complete as fallback")
                return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop
        if is_uvloop_event_loop():
            # Create and use uvloop explicitly
            logger.debug("Creating new uvloop for coroutine execution")
            try:
                import uvloop

                loop = uvloop.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            except ImportError:
                # Fallback if uvloop import fails
                logger.debug("uvloop import failed, using standard asyncio.run")
                return asyncio.run(coro)
        else:
            # Standard asyncio - use asyncio.run directly
            logger.debug("Using standard asyncio.run")
            return asyncio.run(coro)


def run_async_tasks(
    tasks: List[Coroutine],
    batch_size: Optional[int] = None,
    show_progress: bool = True,
    progress_bar_desc: str = "Running async tasks",
) -> List[Any]:
    """
    Execute async tasks with optional batching and progress tracking.

    NOTE: Order of results is not guaranteed!

    Args:
        tasks: List of coroutines to execute
        batch_size: Optional size for batching tasks. If None, runs all concurrently
        show_progress: Whether to display progress bars
    """

    async def _run():
        total_tasks = len(tasks)
        results = []

        # If no batching, run all tasks concurrently with single progress bar
        if not batch_size:
            with tqdm(
                total=total_tasks,
                desc=progress_bar_desc,
                disable=not show_progress,
            ) as pbar:
                for future in asyncio.as_completed(tasks):
                    result = await future
                    results.append(result)
                    pbar.update(1)
            return results

        # With batching, show nested progress bars
        batches = batched(tasks, batch_size)  # generator
        n_batches = (total_tasks + batch_size - 1) // batch_size
        with (
            tqdm(
                total=total_tasks,
                desc=progress_bar_desc,
                disable=not show_progress,
                position=0,
                leave=True,
            ) as overall_pbar,
            tqdm(
                total=batch_size,
                desc=f"Batch 1/{n_batches}",
                disable=not show_progress,
                position=1,
                leave=False,
            ) as batch_pbar,
        ):
            for i, batch in enumerate(batches, 1):
                batch_pbar.reset(total=len(batch))
                batch_pbar.set_description(f"Batch {i}/{n_batches}")
                for future in asyncio.as_completed(batch):
                    result = await future
                    results.append(result)
                    overall_pbar.update(1)
                    batch_pbar.update(1)

        return results

    results = safe_asyncio_run(_run())
    return results
