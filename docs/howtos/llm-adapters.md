# LLM Adapters: Using Multiple Structured Output Backends

Ragas supports multiple structured output backends through an adapter pattern. This guide explains how to use different adapters for different LLM providers.

## Overview

Ragas uses adapters to handle structured output from different LLM providers:

- **Instructor Adapter**: Works with OpenAI, Anthropic, Azure, Groq, Mistral, Cohere, and many others
- **LiteLLM Adapter**: Works with all 100+ LiteLLM-supported providers (Gemini, Ollama, vLLM, Bedrock, etc.)

The framework automatically selects the best adapter for your provider, but you can also choose explicitly.

## Quick Start

### Automatic Adapter Selection (Recommended)

Let Ragas auto-detect the best adapter:

```python
from ragas.llms import llm_factory
from openai import OpenAI

# For OpenAI - automatically uses Instructor adapter
client = OpenAI(api_key="...")
llm = llm_factory("gpt-4o-mini", client=client)
```

```python
from ragas.llms import llm_factory
import google.generativeai as genai

# For Gemini - automatically uses LiteLLM adapter
genai.configure(api_key="...")
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
```

### Explicit Adapter Selection

Choose a specific adapter if you need more control:

```python
from ragas.llms import llm_factory

# Force using Instructor adapter
llm = llm_factory("gpt-4o", client=client, adapter="instructor")

# Force using LiteLLM adapter
llm = llm_factory("gemini-2.0-flash", client=client, adapter="litellm")
```

## Auto-Detection Logic

When `adapter="auto"` (default), Ragas uses this logic:

1. **Check client type**: If client is from `litellm` module → use LiteLLM adapter
2. **Check provider**: If provider is `google` or `gemini` → use LiteLLM adapter
3. **Check base_url**: If `localhost`, `127.0.0.1`, or `0.0.0.0` detected → use LiteLLM adapter (for self-hosted LLMs)
4. **Default**: Use Instructor adapter for all other cases

```python
from ragas.llms.adapters import auto_detect_adapter

# See which adapter will be used
adapter_name = auto_detect_adapter(client, "google")
print(adapter_name)  # Output: "litellm"

adapter_name = auto_detect_adapter(client, "openai")
print(adapter_name)  # Output: "instructor"

# Self-hosted detection
from openai import AsyncOpenAI
self_hosted_client = AsyncOpenAI(base_url="http://localhost:8000/v1")
adapter_name = auto_detect_adapter(self_hosted_client, "openai")
print(adapter_name)  # Output: "litellm" (auto-detected!)
```

## Provider-Specific Examples

### OpenAI

```python
from openai import OpenAI
from ragas.llms import llm_factory

client = OpenAI(api_key="your-key")
llm = llm_factory("gpt-4o", client=client)
# Uses Instructor adapter automatically
```

### Anthropic Claude

```python
from anthropic import Anthropic
from ragas.llms import llm_factory

client = Anthropic(api_key="your-key")
llm = llm_factory("claude-3-sonnet", provider="anthropic", client=client)
# Uses Instructor adapter automatically
```

### Google Gemini (with google-generativeai - Recommended)

```python
import google.generativeai as genai
from ragas.llms import llm_factory

genai.configure(api_key="your-key")
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
# Uses LiteLLM adapter automatically for google provider
```

### Google Gemini (with LiteLLM Proxy - Advanced)

```python
from openai import OpenAI
from ragas.llms import llm_factory

# Requires running: litellm --model gemini-2.0-flash
client = OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"  # LiteLLM proxy endpoint
)
llm = llm_factory("gemini-2.0-flash", client=client, adapter="litellm")
# Uses LiteLLM adapter explicitly
```

### Self-Hosted / Local Models (vLLM, Ollama, Text Generation Inference)

For self-hosted LLMs, **use the LiteLLM adapter** as it automatically falls back to JSON mode when function calling is not properly supported:

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

# vLLM, Ollama, or any OpenAI-compatible self-hosted endpoint
client = AsyncOpenAI(
    api_key="your-key",  # or "ollama" for Ollama
    base_url="http://localhost:8000/v1"  # Your self-hosted endpoint
)

# Recommended: Use LiteLLM adapter for self-hosted models
llm = llm_factory(
    "custom-model",
    provider="openai",
    client=client,
    adapter="litellm"  # Important: Use LiteLLM for self-hosted
)
# LiteLLM automatically detects localhost and uses the appropriate adapter
```

**Why LiteLLM for self-hosted?**
- Self-hosted models often don't properly implement OpenAI's function calling protocol
- LiteLLM automatically falls back to JSON mode when tool calling fails
- Supports 100+ providers including vLLM, Ollama, Text Generation Inference, etc.

**Note:** If you use `localhost`, `127.0.0.1`, or `0.0.0.0` in your `base_url`, Ragas will automatically select the LiteLLM adapter for you!

### AWS Bedrock

```python
from openai import OpenAI
from ragas.llms import llm_factory

# Use LiteLLM proxy for Bedrock
# Note: Set up LiteLLM with Bedrock credentials first
client = OpenAI(
    api_key="",  # Bedrock uses IAM auth
    base_url="http://0.0.0.0:4000"  # LiteLLM proxy endpoint
)
llm = llm_factory("claude-3-sonnet", client=client, adapter="litellm")
```

### Groq

```python
from groq import Groq
from ragas.llms import llm_factory

client = Groq(api_key="your-key")
llm = llm_factory("mixtral-8x7b", provider="groq", client=client)
# Uses Instructor adapter automatically
```

### Mistral

```python
from mistralai import Mistral
from ragas.llms import llm_factory

client = Mistral(api_key="your-key")
llm = llm_factory("mistral-large", provider="mistral", client=client)
# Uses Instructor adapter automatically
```

### Cohere

```python
from cohere import Cohere
from ragas.llms import llm_factory

client = Cohere(api_key="your-key")
llm = llm_factory("command-r-plus", provider="cohere", client=client)
# Uses Instructor adapter automatically
```

## Adapter Selection Guide

Choose your adapter based on your needs:

### Use Instructor Adapter if:
- Using OpenAI, Anthropic, Azure, Groq, Mistral, or Cohere
- Provider is natively supported by Instructor
- You want the most stable, well-tested option
- Provider doesn't require special handling
- Using official API endpoints (not self-hosted)

### Use LiteLLM Adapter if:
- Using Google Gemini
- **Using self-hosted LLMs (vLLM, Ollama, Text Generation Inference, etc.)** ← Recommended!
- Using local models that may not implement function calling properly
- Using providers with 100+ options (Bedrock, etc.)
- You need maximum provider compatibility
- Auto-detection selects it for your provider
- Your LLM returns "multiple tool calls" errors

**For self-hosted LLMs:** The LiteLLM adapter automatically falls back to JSON mode when function calling is unreliable, making it much more robust for custom deployments.

## Working with Adapters Directly

### Get Available Adapters

```python
from ragas.llms.adapters import ADAPTERS

print(ADAPTERS)
# Output: {
#     "instructor": InstructorAdapter(),
#     "litellm": LiteLLMAdapter()
# }
```

### Get Specific Adapter

```python
from ragas.llms.adapters import get_adapter

instructor = get_adapter("instructor")
litellm = get_adapter("litellm")

# Create LLM using adapter directly
llm = instructor.create_llm(client, "gpt-4o", "openai")
```

## Advanced Usage

### Model Arguments

All adapters support the same model arguments:

```python
llm = llm_factory(
    "gpt-4o",
    client=client,
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
)
```

### Async Support

Both adapters support async operations:

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

async_client = AsyncOpenAI(api_key="...")
llm = llm_factory("gpt-4o", client=async_client)

# Async generation
response = await llm.agenerate(prompt, ResponseModel)
```

### Custom Providers with LiteLLM

LiteLLM supports many providers beyond what Instructor covers. Use the LiteLLM proxy approach:

```python
from openai import OpenAI
from ragas.llms import llm_factory

# Set up LiteLLM proxy first:
# litellm --model grok-1  (for xAI)
# litellm --model deepseek-chat  (for DeepSeek)
# etc.

client = OpenAI(
    api_key="your-provider-api-key",
    base_url="http://0.0.0.0:4000"  # LiteLLM proxy endpoint
)

# xAI Grok
llm = llm_factory("grok-1", client=client, adapter="litellm")

# DeepSeek
llm = llm_factory("deepseek-chat", client=client, adapter="litellm")

# Together AI
llm = llm_factory("mistral-7b", client=client, adapter="litellm")
```

## Complete Evaluation Example

```python
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerCorrectness,
)

# Initialize LLM with your provider
import google.generativeai as genai
genai.configure(api_key="...")
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# Create evaluation dataset
data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris"],
    "contexts": [["France is in Europe. Paris is its capital."]],
    "ground_truth": ["Paris"]
}
dataset = Dataset.from_dict(data)

# Define metrics
metrics = [
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm),
]

# Evaluate
results = evaluate(dataset, metrics=metrics)
print(results)
```

## Troubleshooting

### "Multiple tool calls" or "Instructor does not support multiple tool calls"

This error typically occurs with self-hosted LLMs that don't properly implement OpenAI's function calling protocol.

**Solution:** Use the LiteLLM adapter instead:

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

client = AsyncOpenAI(
    api_key="your-key",
    base_url="http://localhost:8000/v1"  # Your self-hosted endpoint
)

# Fix: Use LiteLLM adapter
llm = llm_factory(
    "custom-model",
    client=client,
    provider="openai",
    adapter="litellm"  # This solves the issue!
)
```

**Why this happens:**
- Self-hosted models may return multiple tool calls when Instructor expects only one
- Or they may not follow the exact function calling format that Instructor requires
- LiteLLM is more forgiving and automatically falls back to JSON mode

**Note:** If using `localhost` in your base_url, this should auto-detect and use LiteLLM automatically!

### "Unknown adapter: xyz"

Make sure you're using a valid adapter name:

```python
# Valid: "instructor" or "litellm"
llm = llm_factory("model", client=client, adapter="instructor")

# Invalid: "dspy" (not yet implemented)
# llm = llm_factory("model", client=client, adapter="dspy")  # Error!
```

### "Failed to initialize provider client"

Ensure:
1. Your client is properly initialized
2. Your API key is valid
3. The provider is supported by the adapter

```python
# Check if adapter can handle your provider
from ragas.llms.adapters import auto_detect_adapter
adapter = auto_detect_adapter(client, "my-provider")
print(f"Will use: {adapter}")
```

### Adapter Mismatch

Auto-detection handles most cases, but explicit selection can help:

```python
# If auto-detection picks the wrong adapter:
llm = llm_factory(
    "model",
    provider="provider-name",
    client=client,
    adapter="litellm"  # Explicit override
)
```

## Migration Guide

### From Text-Only to Structured Output

If you're upgrading from text-only LLM usage:

```python
# Before (deprecated)
# from ragas.llms import LangchainLLMWrapper
# llm = LangchainLLMWrapper(langchain_llm)

# After (new way)
from ragas.llms import llm_factory
llm = llm_factory("gpt-4o", client=client)
```

### Switching Providers

To switch from OpenAI to Gemini:

```python
# Before: OpenAI
from openai import OpenAI
client = OpenAI(api_key="...")
llm = llm_factory("gpt-4o", client=client)

# After: Gemini (similar code pattern!)
import google.generativeai as genai
genai.configure(api_key="...")
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
# Adapter automatically switches to LiteLLM for google provider
```

## See Also

- [Gemini Integration Guide](./integrations/gemini.md) - Detailed Gemini setup
- [LLM Factory Reference](./llm-factory.md) - Complete API reference
- [Metrics Documentation](../concepts/metrics/index.md) - Using metrics with LLMs
