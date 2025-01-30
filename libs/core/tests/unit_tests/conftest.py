from unittest.mock import MagicMock

import pytest
from llmstudio_core.providers.azure import AzureProvider
from llmstudio_core.providers.provider import ProviderCore


class MockProvider(ProviderCore):
    async def aparse_response(self, response, **kwargs):
        return response

    def parse_response(self, response, **kwargs):
        return response

    def _output_to_string(self, output):
        # Handle string inputs
        if isinstance(output, str):
            return output
        if output.choices[0].finish_reason == "stop":
            return output.choices[0].message.content
        return ""

    def validate_request(self, request):
        # For testing, simply return the request
        return request

    async def agenerate_client(self, request):
        # For testing, return an async generator
        async def async_gen():
            yield {}

        return async_gen()

    def generate_client(self, request):
        # For testing, return a generator
        def gen():
            yield {}

        return gen()

    @staticmethod
    def _provider_config_name():
        return "mock_provider"


@pytest.fixture
def mock_provider():
    config = MagicMock()
    config.models = {
        "test_model": MagicMock(input_token_cost=0.01, output_token_cost=0.02)
    }
    config.id = "mock_provider"
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: x.split()  # Simple tokenizer mock
    return MockProvider(config=config, tokenizer=tokenizer)


class MockAzureProvider(AzureProvider):
    async def aparse_response(self, response, **kwargs):
        return response

    async def agenerate_client(self, request):
        # For testing, return an async generator
        async def async_gen():
            yield {}

        return async_gen()

    @staticmethod
    def _provider_config_name():
        return "mock_azure_provider"


@pytest.fixture
def mock_azure_provider():
    config = MagicMock()
    config.id = "mock_azure_provider"
    api_key = "key"
    api_endpoint = "http://azureopenai.com"
    api_version = "2025-01-01-preview"
    return MockAzureProvider(
        config=config,
        api_endpoint=api_endpoint,
        api_key=api_key,
        api_version=api_version,
    )
