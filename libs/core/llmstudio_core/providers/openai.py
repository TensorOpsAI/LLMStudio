import os
from typing import Any, AsyncGenerator, Coroutine, Generator

import openai
from llmstudio_core.exceptions import ProviderError
from llmstudio_core.providers.provider import ChatRequest, ProviderCore, provider
from openai import OpenAI


@provider
class OpenAIProvider(ProviderCore):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.API_KEY = self.API_KEY if self.API_KEY else os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=self.API_KEY)

    @staticmethod
    def _provider_config_name():
        return "openai"

    def validate_request(self, request: ChatRequest):
        return ChatRequest(**request)

    async def agenerate_client(
        self, request: ChatRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an OpenAI client"""
        return self.generate_client(request=request)

    def generate_client(self, request: ChatRequest) -> Generator:
        """Generate an OpenAI client"""

        try:
            return self._client.chat.completions.create(
                model=request.model,
                messages=(
                    [{"role": "user", "content": request.chat_input}]
                    if isinstance(request.chat_input, str)
                    else request.chat_input
                ),
                stream=True,
                stream_options={"include_usage": True},
                **request.parameters,
            )
        except openai._exceptions.APIError as e:
            raise ProviderError(str(e))

    async def aparse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        result = self.parse_response(response=response, **kwargs)
        for chunk in result:
            yield chunk

    def parse_response(self, response: Generator, **kwargs) -> Generator:
        for chunk in response:
            yield chunk.model_dump()
