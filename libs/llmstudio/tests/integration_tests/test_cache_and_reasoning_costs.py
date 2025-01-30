import asyncio
import os
from pprint import pprint

import pytest
from dotenv import load_dotenv
from llmstudio_core.providers import LLMCore

load_dotenv()

# input prompt has to be >1024 tokens to auto cache on OpenAI
input_prompt = """
What is Lorem Ipsum? json
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

Why do we use it?
It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).


Where does it come from?
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.

What is Lorem Ipsum? json
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

Why do we use it?
It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).


Where does it come from?
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.

What is Lorem Ipsum? json
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

Why do we use it?
It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).


Where does it come from?
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.

After reading this answer just in one line, saying hello how are you in latin.
"""


def run_provider(provider, model, api_key, **kwargs):
    print(f"\n\n###RUNNING for <{provider}>, <{model}> ###")
    llm = LLMCore(provider=provider, api_key=api_key, **kwargs)

    metrics = {}

    print("\nAsync Non-Stream")

    chat_request = build_chat_request(model, chat_input=input_prompt, is_stream=False)

    response_async = asyncio.run(llm.achat(**chat_request))
    pprint(response_async)
    metrics["async non-stream"] = response_async.metrics

    print("\nAsync Stream")

    async def async_stream():
        chat_request = build_chat_request(
            model, chat_input=input_prompt, is_stream=True
        )

        response_async = await llm.achat(**chat_request)
        async for p in response_async:
            if not p.metrics:
                print("that: ", p.chat_output_stream)
            if p.metrics:
                pprint(p)
                metrics["async stream"] = p.metrics

    asyncio.run(async_stream())

    print("\nSync Non-Stream")
    chat_request = build_chat_request(model, chat_input=input_prompt, is_stream=False)

    response_sync = llm.chat(**chat_request)
    pprint(response_sync)
    metrics["sync non-stream"] = response_sync.metrics

    print("\nSync Stream")
    chat_request = build_chat_request(model, chat_input=input_prompt, is_stream=True)

    response_sync_stream = llm.chat(**chat_request)
    for p in response_sync_stream:
        if p.metrics:
            pprint(p)
            metrics["sync stream"] = p.metrics

    print(f"\n\n###REPORT for <{provider}>, <{model}> ###")
    return metrics


def build_chat_request(
    model: str, chat_input: str, is_stream: bool, max_tokens: int = 1000
):
    if model == "o1-preview" or model == "o1-mini":
        chat_request = {
            "chat_input": chat_input,
            "model": model,
            "is_stream": is_stream,
            "retries": 0,
            "parameters": {"max_completion_tokens": max_tokens},
        }
    else:
        chat_request = {
            "chat_input": chat_input,
            "model": model,
            "is_stream": is_stream,
            "retries": 0,
            "parameters": {
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "functions": None,
            },
        }
    return chat_request


# Fixture for provider-model pairs
@pytest.fixture(
    scope="module", params=[("openai", "gpt-4o-mini"), ("openai", "o1-mini")]
)
def provider_model(request):
    return request.param


# Fixture for metrics, computes them once per provider-model pair
@pytest.fixture(scope="module")
def metrics(provider_model):
    provider, model = provider_model
    print(f"Running provider {provider} with model {model}")
    return run_provider(
        provider=provider, model=model, api_key=os.environ["OPENAI_API_KEY"]
    )


# Test cache metrics (runs for all models)
def test_metrics_cache(provider_model, metrics):
    provider, model = provider_model
    at_least_one_cached = False
    for current_metrics in metrics.values():
        print(current_metrics)
        assert current_metrics["input_tokens"] > current_metrics["cached_tokens"]
        if current_metrics["cached_tokens"] > 0:
            at_least_one_cached = True
    assert at_least_one_cached == True
    print(f"All Cache Tests Passed for {provider} - {model}")


# Test reasoning metrics (only runs for o1-mini and o1-preview)
def test_metrics_reasoning(provider_model, metrics):
    provider, model = provider_model

    # Skip test if model is not o1-mini or o1-preview
    if model not in ["o1-mini", "o1-preview"]:
        pytest.skip(f"Reasoning metrics test not applicable for model {model}")

    for current_metrics in metrics.values():
        assert current_metrics["reasoning_tokens"] > 0
        assert current_metrics["reasoning_tokens"] < current_metrics["total_tokens"]
        assert (
            current_metrics["input_tokens"]
            + current_metrics["output_tokens"]
            + current_metrics["reasoning_tokens"]
            == current_metrics["total_tokens"]
        )
    print(f"All Reasoning Tests Passed for {provider} - {model}")


def usage_when_max_tokens_reached():
    """
    Usefull to test handling of Usage in other finish_reason scenarios
    """
    provider, model = ("openai", "o1-mini")
    api_key = os.environ["OPENAI_API_KEY"]

    llm = LLMCore(provider=provider, api_key=api_key)
    chat_request = build_chat_request(
        model, chat_input=input_prompt, is_stream=False, max_tokens=7
    )
    response = asyncio.run(llm.achat(**chat_request))

    assert response.metrics["async non-stream"]["reasoning_tokens"] > 0
    assert response.metrics["sync non-stream"]["reasoning_tokens"] > 0
    assert response.metrics["async stream"]["reasoning_tokens"] > 0
    assert response.metrics["sync stream"]["reasoning_tokens"] > 0
    print(f"All Max Tokens Usage Reached Tests Passed for {provider} - {model}")
