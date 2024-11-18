from unittest.mock import MagicMock

import pytest
from llmstudio_core.providers.provider import ProviderCore
from llmstudio_core.providers.azure import AzureProvider


class MockProvider(ProviderCore):
    async def aparse_response(self, response, **kwargs):
        return response

    def parse_response(self, response, **kwargs):
        return response

    def output_to_string(self, output):
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
    base_url = 'mock_url.com'
    return MockAzureProvider(config=config, base_url=base_url)