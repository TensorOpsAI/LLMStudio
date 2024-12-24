import time
import uuid
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import tiktoken
from llmstudio_core.exceptions import ProviderError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import Function
from pydantic import BaseModel, ValidationError

provider_registry = {}


def provider(cls):
    """Decorator to register a new provider."""
    provider_registry[cls._provider_config_name()] = cls

    return cls


class ChatRequest(BaseModel):
    chat_input: Any
    model: str
    is_stream: Optional[bool] = False
    retries: Optional[int] = 0
    parameters: Optional[dict] = {}

    def __init__(self, **data):
        super().__init__(**data)
        base_model_fields = self.model_fields.keys()
        additional_params = {
            k: v for k, v in data.items() if k not in base_model_fields
        }
        self.parameters.update(additional_params)


class Provider(ABC):
    END_TOKEN = "<END_TOKEN>"

    def __init__(
        self,
        config: Any,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        base_url: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.config = config
        self.API_KEY = api_key
        self.api_endpoint = api_endpoint
        self.api_version = api_version
        self.base_url = base_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.tokenizer = tokenizer if tokenizer else self._get_tokenizer()
        self.count = 0

    @abstractmethod
    async def achat(
        self,
        chat_input: Any,
        model: str,
        is_stream: Optional[bool] = False,
        retries: Optional[int] = 0,
        parameters: Optional[dict] = {},
        **kwargs,
    ) -> Coroutine[Any, Any, Union[ChatCompletionChunk, ChatCompletion]]:
        raise NotImplementedError("Providers needs to have achat method implemented.")

    @abstractmethod
    def chat(
        self,
        chat_input: Any,
        model: str,
        is_stream: Optional[bool] = False,
        retries: Optional[int] = 0,
        parameters: Optional[dict] = {},
        **kwargs,
    ) -> Union[ChatCompletionChunk, ChatCompletion]:
        raise NotImplementedError("Providers needs to have chat method implemented.")

    @staticmethod
    @abstractmethod
    def _provider_config_name():
        raise NotImplementedError(
            "Providers need to implement the '_provider_config_name' property."
        )


class ProviderCore(Provider):
    END_TOKEN = "<END_TOKEN>"

    @abstractmethod
    def validate_request(self, request: ChatRequest):
        raise NotImplementedError("Providers need to implement the 'validate_request'.")

    @abstractmethod
    async def agenerate_client(
        self, request: ChatRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate the provider's client"""
        raise NotImplementedError("Providers need to implement the 'agenerate_client'.")

    @abstractmethod
    def generate_client(self, request: ChatRequest) -> Generator:
        """Generate the provider's client"""
        raise NotImplementedError("Providers need to implement the 'generate_client'.")

    @abstractmethod
    async def _aparse_response(self, response: AsyncGenerator, **kwargs) -> Any:
        raise NotImplementedError("ProviderCore needs a aparse_response method.")

    @abstractmethod
    def _parse_response(self, response: AsyncGenerator, **kwargs) -> Any:
        raise NotImplementedError("ProviderCore needs a parse_response method.")

    def validate_model(self, request: ChatRequest):
        if request.model not in self.config.models:
            raise ProviderError(
                f"Model {request.model} is not supported by {self.config.name}"
            )

    async def achat(
        self,
        chat_input: Any,
        model: str,
        is_stream: Optional[bool] = False,
        retries: Optional[int] = 0,
        parameters: Optional[dict] = {},
        **kwargs,
    ):
        """
        Asynchronously establishes a chat connection with the provider’s API, handling retries,
        request validation, and streaming response options.

        Parameters
        ----------
        chat_input : Any
            The input data for the chat request, such as a string or dictionary, to be sent to the API.
        model : str
            The identifier of the model to be used for the chat request.
        is_stream : Optional[bool], default=False
            Flag to indicate if the response should be streamed. If True, returns an async generator
            for streaming content; otherwise, returns the complete response.
        retries : Optional[int], default=0
            Number of retry attempts on error. Retries will be attempted for specific HTTP errors like rate limits.
        parameters : Optional[dict], default={}
            Additional configuration parameters for the request, such as temperature or max tokens.
        **kwargs
            Additional keyword arguments to customize the request.

        Returns
        -------
        Union[AsyncGenerator, Any]
            - If `is_stream` is True, returns an async generator yielding response chunks.
            - If `is_stream` is False, returns the first complete response chunk.

        Raises
        ------
        ProviderError
            - Raised if the request validation fails or if all retry attempts are exhausted.
            - Also raised for unexpected exceptions during request handling.
        """
        try:
            request = self.validate_request(
                dict(
                    chat_input=chat_input,
                    model=model,
                    is_stream=is_stream,
                    retries=retries,
                    parameters=parameters,
                    **kwargs,
                )
            )
        except ValidationError as e:
            raise ProviderError(str(e))

        self.validate_model(request)

        for _ in range(request.retries + 1):
            try:
                start_time = time.time()
                response = await self.agenerate_client(request)

                if request.is_stream:
                    return self.ahandle_response_stream(request, response, start_time)
                else:
                    return self.ahandle_response_non_stream(request, response, start_time)
            except Exception as e:
                raise ProviderError(str(e))
        raise ProviderError("Too many requests")
    
    class ChatCompletionLLMstudio(ChatCompletion):
        chat_input: str
        """The input prompt for the chat completion."""

        chat_output: str
        """The final response generated by the model."""

        chat_output_stream: str
        """Incremental chunks of the response for streaming."""

        context: List[dict]
        """The conversation history or context provided to the model."""

        provider: str
        """Identifier for the backend service provider."""

        deployment: Optional[str]
        """Information about the deployment configuration used."""

        timestamp: float
        """The timestamp when the response was generated."""

        parameters: dict
        """The parameters used in the request."""

        metrics: Optional["ProviderCore.Metrics"]
        """Performance and usage metrics calculated for the response."""
        
    class ChatCompletionChunkLLMstudio(ChatCompletionChunk):
        chat_input: str
        """The input prompt for the chat completion."""
        
        chat_output: str
        """The final response generated by the model."""

        chat_output_stream: str
        """Incremental chunks of the response for streaming."""

        context: List[dict]
        """The conversation history or context provided to the model."""

        provider: str
        """Identifier for the backend service provider."""

        deployment: Optional[str]
        """Information about the deployment configuration used."""

        timestamp: float
        """The timestamp when the chunk was generated."""

        parameters: dict
        """The parameters used in the request."""

        metrics: Optional["ProviderCore.Metrics"]
        """Performance and usage metrics calculated for the response chunk."""

    def chat(
        self,
        chat_input: Any,
        model: str,
        is_stream: Optional[bool] = False,
        retries: Optional[int] = 0,
        parameters: Optional[dict] = {},
        **kwargs,
    ):
        """
        Establishes a chat connection with the provider’s API, handling retries, request validation,
        and streaming response options.

        Parameters
        ----------
        chat_input : Any
            The input data for the chat request, often a string or dictionary, to be sent to the API.
        model : str
            The model identifier for selecting the model used in the chat request.
        is_stream : Optional[bool], default=False
            Flag to indicate if the response should be streamed. If True, the function returns a generator
            for streaming content. Otherwise, it returns the complete response.
        retries : Optional[int], default=0
            Number of retry attempts on error. Retries will be attempted on specific HTTP errors like rate limits.
        parameters : Optional[dict], default={}
            Additional configuration parameters for the request, such as temperature or max tokens.
        **kwargs
            Additional keyword arguments that can be passed to customize the request.

        Returns
        -------
        Union[Generator, Any]
            - If `is_stream` is True, returns a generator that yields chunks of the response.
            - If `is_stream` is False, returns the first complete response chunk.

        Raises
        ------
        ProviderError
            - Raised if the request validation fails or if the request fails after the specified number of retries.
            - Also raised on other unexpected exceptions during request handling.
        """
        try:
            request = self.validate_request(
                dict(
                    chat_input=chat_input,
                    model=model,
                    is_stream=is_stream,
                    retries=retries,
                    parameters=parameters,
                    **kwargs,
                )
            )
        except ValidationError as e:
            raise ProviderError(str(e))

        self.validate_model(request)

        for _ in range(request.retries + 1):
            try:
                start_time = time.time()
                response = self.generate_client(request) # post to provider

                if request.is_stream:
                    return self.handle_response_stream(request, response, start_time)
                else:
                    return self.handle_response_non_stream(request, response, start_time)
            except Exception as e:
                raise ProviderError(str(e))
        raise ProviderError("Too many requests")

    async def ahandle_response_stream(
        self, request: ChatRequest, response: AsyncGenerator, start_time: float
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously handles the response from an API, processing response chunks for either
        streaming or non-streaming responses.

        Buffers response chunks for non-streaming responses to output one single message. For streaming responses sends incremental chunks.

        Parameters
        ----------
        request : ChatRequest
            The chat request object, which includes input data, model name, and streaming options.
        response : AsyncGenerator
            The async generator yielding response chunks from the API.
        start_time : float
            The timestamp when the response handling started, used for latency calculations.

        Yields
        ------
        ChatCompletionChunk
            yields `ChatCompletionChunk` objects with incremental response chunks for streaming.
        """
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0
        chunks = []

        async for chunk in self._aparse_response(response, request=request):
            token_count += 1
            current_time = time.time()
            first_token_time = first_token_time or current_time
            if previous_token_time is not None:
                token_times.append(current_time - previous_token_time)
            previous_token_time = current_time

            chunks.append(chunk)
            chunk = chunk[0] if isinstance(chunk, tuple) else chunk
            model = chunk.get("model")
            if chunk.get("choices")[0].get("finish_reason") != "stop":
                chat_output = chunk.get("choices")[0].get("delta").get("content")
                chunk = {
                    **chunk,
                    "id": str(uuid.uuid4()),
                    "chat_input": (
                        request.chat_input
                        if isinstance(request.chat_input, str)
                        else request.chat_input[-1]["content"]
                    ),
                    "chat_output": "",
                    "chat_output_stream": chat_output if chat_output else "",
                    "context": (
                        [{"role": "user", "content": request.chat_input}]
                        if isinstance(request.chat_input, str)
                        else request.chat_input
                    ),
                    "provider": self.config.id,
                    "model": (
                        request.model
                        if model and model.startswith(request.model)
                        else (model or request.model)
                    ),
                    "deployment": (
                        model
                        if model and model.startswith(request.model)
                        else (request.model if model != request.model else None)
                    ),
                    "timestamp": time.time(),
                    "parameters": request.parameters,
                    "metrics": None,
                }
                yield self.ChatCompletionChunkLLMstudio(**chunk)

        chunks = [chunk[0] if isinstance(chunk, tuple) else chunk for chunk in chunks]
        model = next(chunk["model"] for chunk in chunks if chunk.get("model"))

        response, output_string = self.join_chunks(chunks)

        metrics = self.calculate_metrics_stream(
            request.chat_input,
            response,
            request.model,
            start_time,
            time.time(),
            first_token_time,
            token_times,
            token_count,
        )

        response = {
            **chunk,
            "id": str(uuid.uuid4()),
            "chat_input": (
                request.chat_input
                if isinstance(request.chat_input, str)
                else request.chat_input[-1]["content"]
            ),
            "chat_output": output_string,
            "chat_output_stream": "",
            "context": (
                [{"role": "user", "content": request.chat_input}]
                if isinstance(request.chat_input, str)
                else request.chat_input
            ),
            "provider": self.config.id,
            "model": (
                request.model
                if model and model.startswith(request.model)
                else (model or request.model)
            ),
            "deployment": (
                model
                if model and model.startswith(request.model)
                else (request.model if model != request.model else None)
            ),
            "timestamp": time.time(),
            "parameters": request.parameters,
            "metrics": metrics,
        }

        yield self.ChatCompletionChunkLLMstudio(**response)
            
    async def ahandle_response_non_stream(
        self, request: ChatRequest, response: AsyncGenerator, start_time: float
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError(
            "The method ahandle_response_non_stream must be implemented in a subclass."
        )

    def handle_response_non_stream(
        self, request: ChatRequest, response: Generator, start_time: float
    ) -> ChatCompletionLLMstudio:
        """
        Processes non-streaming API responses and constructs a complete response object.

        This method handles the entire API response as a single unit (non-streaming mode),
        calculating metrics and returning a structured `ChatCompletionLLMstudio` instance.

        Parameters
        ----------
        request : ChatRequest
            The original request details, including input data, model name, and parameters.
        response : Generator
            A generator yielding response chunks from the API.
        start_time : float
            The timestamp when response processing started, used for latency and metrics calculations.

        Returns
        -------
        ChatCompletionLLMstudio
            A structured response object containing the full output, metrics, and context details.
        """    
        if self.config.id != 'openai':
            response: ChatCompletion = self._parse_response(response, request=request)

        model = request.model

        metrics = self.calculate_metrics_non_stream(
            input = request.model,
            output = response,
            model = model,
            start_time = start_time,
            end_time = time.time()
        )

        response = {
            **response.model_dump(),
            "id": str(uuid.uuid4()),
            "chat_input": (
                request.chat_input
                if isinstance(request.chat_input, str)
                else request.chat_input[-1]["content"]
            ),
            "chat_output": response.choices[0].message.content,
            "chat_output_stream": "",
            "context": (
                [{"role": "user", "content": request.chat_input}]
                if isinstance(request.chat_input, str)
                else request.chat_input
            ),
            "provider": self.config.id,
            "model": response.model,
            "deployment": (
                model
                if model and model.startswith(request.model)
                else (request.model if model != request.model else None)
            ),
            "timestamp": time.time(),
            "parameters": request.parameters,
            "metrics": metrics,
        }
        return self.ChatCompletionLLMstudio(**response)

    def handle_response_stream(
        self, request: ChatRequest, response: Generator, start_time: float
    ) -> Generator:
        """
        Processes API response chunks to build a structured, complete response, yielding
        each chunk if streaming is enabled.

        If streaming, each chunk is yielded as soon as it’s processed. Otherwise, all chunks
        are combined and yielded as a single response at the end.

        Parameters
        ----------
        request : ChatRequest
            The original request details, including model, input, and streaming preference.
        response : Generator
            A generator yielding partial response chunks from the API.
        start_time : float
            The start time for measuring response timing.

        Yields
        ------
        ChatCompletionChunk
            yields each `ChatCompletionChunk` as it’s processed.

        """
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0
        chunks = []

        for chunk in self._parse_response(response, request=request):
            token_count += 1
            current_time = time.time()
            first_token_time = first_token_time or current_time
            if previous_token_time is not None:
                token_times.append(current_time - previous_token_time)
            previous_token_time = current_time

            chunks.append(chunk)
            chunk = chunk[0] if isinstance(chunk, tuple) else chunk
            model = chunk.get("model")
            if chunk.get("choices")[0].get("finish_reason") != "stop":
                chat_output = chunk.get("choices")[0].get("delta").get("content")
                chunk = {
                    **chunk,
                    "id": str(uuid.uuid4()),
                    "chat_input": (
                        request.chat_input
                        if isinstance(request.chat_input, str)
                        else request.chat_input[-1]["content"]
                    ),
                    "chat_output": "",
                    "chat_output_stream": chat_output if chat_output else "",
                    "context": (
                        [{"role": "user", "content": request.chat_input}]
                        if isinstance(request.chat_input, str)
                        else request.chat_input
                    ),
                    "provider": self.config.id,
                    "model": (
                        request.model
                        if model and model.startswith(request.model)
                        else (model or request.model)
                    ),
                    "deployment": (
                        model
                        if model and model.startswith(request.model)
                        else (request.model if model != request.model else None)
                    ),
                    "timestamp": time.time(),
                    "parameters": request.parameters,
                    "metrics": None,
                }
                yield self.ChatCompletionChunkLLMstudio(**chunk)

        chunks = [chunk[0] if isinstance(chunk, tuple) else chunk for chunk in chunks]
        model = next(chunk["model"] for chunk in chunks if chunk.get("model"))

        response, output_string = self.join_chunks(chunks)

        metrics = self.calculate_metrics_stream(
            request.chat_input,
            response,
            request.model,
            start_time,
            time.time(),
            first_token_time,
            token_times,
            token_count,
        )

        response = {
            **chunk,
            "id": str(uuid.uuid4()),
            "chat_input": (
                request.chat_input
                if isinstance(request.chat_input, str)
                else request.chat_input[-1]["content"]
            ),
            "chat_output": output_string,
            "chat_output_stream": "",
            "context": (
                [{"role": "user", "content": request.chat_input}]
                if isinstance(request.chat_input, str)
                else request.chat_input
            ),
            "provider": self.config.id,
            "model": (
                request.model
                if model and model.startswith(request.model)
                else (model or request.model)
            ),
            "deployment": (
                model
                if model and model.startswith(request.model)
                else (request.model if model != request.model else None)
            ),
            "timestamp": time.time(),
            "parameters": request.parameters,
            "metrics": metrics,
        }

        yield self.ChatCompletionChunkLLMstudio(**response)

    def join_chunks(self, chunks):
        """
        Combine multiple response chunks from the model into a single, structured response.
        Handles tool calls, function calls, and standard text completion based on the
        purpose indicated by the final chunk.

        Parameters
        ----------
        chunks : List[Dict]
            A list of partial responses (chunks) from the model.

        Returns
        -------
        Tuple[ChatCompletion, str]
            - `ChatCompletion`: The structured response based on the type of completion
            (tool calls, function call, or text).
            - `str`: The concatenated content or arguments, depending on the completion type.

        Raises
        ------
        Exception
            If there is an issue constructing the response, an exception is raised.
        """

        finish_reason = chunks[-1].get("choices")[0].get("finish_reason")
        if finish_reason == "tool_calls":
            tool_calls = {}
            for chunk in chunks:
                try:
                    data = chunk.get("choices")[0].get("delta").get("tool_calls")[0]
                    tool_calls.setdefault(data["index"], []).append(data)
                except TypeError:
                    continue

            tool_call_ids = [t[0].get("id") for t in tool_calls.values()]
            tool_call_names = [
                t[0].get("function").get("name") for t in tool_calls.values()
            ]
            tool_call_types = [
                t[0].get("function").get("type", "function")
                for t in tool_calls.values()
            ]

            tool_call_arguments_all = []
            for t in tool_calls.values():
                tool_call_arguments_all.append(
                    "".join(
                        chunk.get("function", {}).get("arguments", "") for chunk in t
                    )
                )

            tool_calls_parsed = [
                ChatCompletionMessageToolCall(
                    id=tool_call_id,
                    function=Function(
                        arguments=tool_call_arguments, name=tool_call_name
                    ),
                    type=tool_call_type,
                )
                for tool_call_arguments, tool_call_name, tool_call_type, tool_call_id in zip(
                    tool_call_arguments_all,
                    tool_call_names,
                    tool_call_types,
                    tool_call_ids,
                )
            ]

            try:
                return (
                    ChatCompletion(
                        id=chunks[-1].get("id"),
                        created=chunks[-1].get("created"),
                        model=chunks[-1].get("model"),
                        object="chat.completion",
                        choices=[
                            Choice(
                                finish_reason="tool_calls",
                                index=0,
                                logprobs=None,
                                message=ChatCompletionMessage(
                                    content=None,
                                    role="assistant",
                                    function_call=None,
                                    tool_calls=tool_calls_parsed,
                                ),
                            )
                        ],
                    ),
                    str(tool_call_names + tool_call_arguments_all),
                )
            except Exception as e:
                raise e
        elif finish_reason == "function_call":
            function_calls = [
                chunk.get("choices")[0].get("delta").get("function_call")
                for chunk in chunks[:-1]
                if chunk.get("choices")
                and chunk.get("choices")[0].get("delta")
                and chunk.get("choices")[0].get("delta").get("function_call")
            ]

            function_call_name = function_calls[0].get("name")

            function_call_arguments = ""
            for chunk in function_calls:
                function_call_arguments += chunk.get("arguments")

            return (
                ChatCompletion(
                    id=chunks[-1].get("id"),
                    created=chunks[-1].get("created"),
                    model=chunks[-1].get("model"),
                    object="chat.completion",
                    choices=[
                        Choice(
                            finish_reason="function_call",
                            index=0,
                            logprobs=None,
                            message=ChatCompletionMessage(
                                content=None,
                                role="assistant",
                                tool_calls=None,
                                function_call=FunctionCall(
                                    arguments=function_call_arguments,
                                    name=function_call_name,
                                ),
                            ),
                        )
                    ],
                ),
                function_call_arguments,
            )

        elif finish_reason == "stop" or finish_reason == "length":
            if self.__class__.__name__ in ("OpenAIProvider", "AzureProvider"):
                start_index = 1
            else:
                start_index = 0

            stop_content = "".join(
                filter(
                    None,
                    [
                        chunk.get("choices")[0].get("delta").get("content")
                        for chunk in chunks[start_index:]
                    ],
                )
            )

            return (
                ChatCompletion(
                    id=chunks[-1].get("id"),
                    created=chunks[-1].get("created"),
                    model=chunks[-1].get("model"),
                    object="chat.completion",
                    choices=[
                        Choice(
                            finish_reason="stop",
                            index=0,
                            logprobs=None,
                            message=ChatCompletionMessage(
                                content=stop_content,
                                role="assistant",
                                function_call=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                ),
                stop_content,
            )
            
    class Metrics(BaseModel):
        input_tokens: int
        """Number of tokens in the input."""

        output_tokens: int
        """Number of tokens in the output."""

        total_tokens: int
        """Total token count (input + output)."""

        cost_usd: float
        """Total cost of the response in USD."""

        latency_s: float
        """Total time taken for the response, in seconds."""

        time_to_first_token_s: Optional[float] = None
        """Time to receive the first token, in seconds."""

        inter_token_latency_s: Optional[float] = None
        """Average time between tokens, in seconds. Defaults to None if not provided."""

        tokens_per_second: Optional[float] = None
        """Processing rate of tokens per second. Defaults to None if not provided."""
        
        def __getitem__(self, key: str) -> Any:
            """
            Allows subscriptable access to class fields.

            Parameters
            ----------
            key : str
                The name of the field to retrieve.

            Returns
            -------
            Any
                The value of the specified field.

            Raises
            ------
            KeyError
                If the key does not exist.
            """
            try:
                return getattr(self, key)
            except AttributeError:
                raise KeyError(f"'{key}' not found in MetricsStream.")

        def __iter__(self):
            """
            Allows iteration over the class fields as key-value pairs.
            """
            return iter(self.model_dump().items())

        def __len__(self):
            """
            Returns the number of fields in the class.
            """
            return len(self.model_fields)

        def keys(self):
            """
            Returns the keys of the fields.
            """
            return self.model_fields.keys()

        def values(self):
            """
            Returns the values of the fields.
            """
            return self.model_dump().values()

        def items(self):
            """
            Returns the key-value pairs of the fields.
            """
            return self.model_dump().items()
        
    def calculate_metrics_stream(
        self,
        input: Any,
        output: Any,
        model: str,
        start_time: float,
        end_time: float,
        first_token_time: float,
        token_times: Tuple[float, ...],
        token_count: int,
    ) -> Metrics:
        """
        Calculates performance and cost metrics for a model response based on timing
        information, token counts, and model-specific costs.

        Parameters
        ----------
        input : Any
            The input provided to the model, used to determine input token count.
        output : Any
            The output generated by the model, used to determine output token count.
        model : str
            The model identifier, used to retrieve model-specific configuration and costs.
        start_time : float
            The timestamp marking the start of the model response.
        end_time : float
            The timestamp marking the end of the model response.
        first_token_time : float
            The timestamp when the first token was received, used for latency calculations.
        token_times : Tuple[float, ...]
            A tuple of time intervals between received tokens, used for inter-token latency.
        token_count : int
            The total number of tokens processed in the response.

        Returns
        -------
        Metrics
            A `Metrics` instance containing calculated performance and cost metrics.
        """
        model_config = self.config.models[model]

        # Token counts
        input_tokens = len(self.tokenizer.encode(self.input_to_string(input)))
        output_tokens = len(self.tokenizer.encode(self.output_to_string(output)))

        # Cost calculations
        input_cost = self.calculate_cost(input_tokens, model_config.input_token_cost)
        output_cost = self.calculate_cost(output_tokens, model_config.output_token_cost)
        total_cost = input_cost + output_cost

        # Latency calculations
        total_time = end_time - start_time
        time_to_first_token = first_token_time - start_time
        inter_token_latency = sum(token_times) / len(token_times) if token_times else 0.0
        tokens_per_second = token_count / total_time if total_time > 0 else 0.0

        return self.Metrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=total_cost,
            latency_s=total_time,
            time_to_first_token_s=time_to_first_token,
            inter_token_latency_s=inter_token_latency,
            tokens_per_second=tokens_per_second,
        )


    def calculate_metrics_non_stream(
        self,
        input: Any,
        output: Any,
        model: str,
        start_time: float,
        end_time: float
    ) -> Metrics:
        """
        Calculates performance and cost metrics for a non-streaming model response.

        This method evaluates the input and output token counts, calculates associated costs,
        and measures latency based on provided timestamps. It returns a structured `Metrics`
        object containing these performance metrics.

        Parameters
        ----------
        input : Any
            The input provided to the model, used to determine input token count if not
            explicitly available in the response.
        output : Any
            The output generated by the model, used to determine output token count if not
            explicitly available in the response.
        model : str
            The model identifier, used to retrieve model-specific configurations and costs.
        start_time : float
            The timestamp marking the start of the model response, used for latency calculations.
        end_time : float
            The timestamp marking the end of the model response, used for latency calculations.

        Returns
        -------
        Metrics
            A `Metrics` object containing calculated performance and cost metrics, including:
            - `input_tokens`: Number of tokens in the input.
            - `output_tokens`: Number of tokens in the output.
            - `total_tokens`: Total token count (input + output).
            - `cost_usd`: Total cost of the response in USD.
            - `latency_s`: Total time taken for the response, in seconds.

        Notes
        -----
        - Token counts are retrieved from the `output.usage` field if available; otherwise,
        they are calculated using the tokenizer.
        - Additional cost calculations such as cache cost and reasoning cost are currently
        placeholders and need implementation.
        """
        model_config = self.config.models[model]

        # Token counts
        input_tokens = (getattr(output.usage, "input_tokens", None) or len(self.tokenizer.encode(self.input_to_string(input))))

        output_tokens = (getattr(output.usage, "output_tokens", None) or len(self.tokenizer.encode(self.output_to_string(output))))


        # Cost calculations
        input_cost = self.calculate_cost(token_count=input_tokens, token_cost=model_config.input_token_cost)
        output_cost = self.calculate_cost(token_count=output_tokens, token_cost=model_config.output_token_cost)
        """ TODO
        cache_cost = self.calculate_cache_cost(
            token_count=input_tokens, 
            token_cost=model_config.cache_token_cost
        )
        
        reasoning_cost = self.calculate_reasoning_cost(
            token_count=output_tokens, 
            token_cost=model_config.reasoning_token_cost
        )
        """
        
        total_cost = input_cost + output_cost

        total_time = end_time - start_time

        return self.Metrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=total_cost,
            latency_s=total_time
        )

    def calculate_cost(
        self, token_count: int, token_cost: Union[float, List[Dict[str, Any]]]
    ) -> float:
        """
        Calculates the cost for a given number of tokens based on a fixed cost per token
        or a variable rate structure.

        If `token_cost` is a fixed float, the total cost is `token_count * token_cost`.
        If `token_cost` is a list, it checks each range and calculates cost based on the applicable range's rate.

        Parameters
        ----------
        token_count : int
            The total number of tokens for which the cost is being calculated.
        token_cost : Union[float, List[Dict[str, Any]]]
            Either a fixed cost per token (as a float) or a list of dictionaries defining
            variable cost ranges. Each dictionary in the list represents a range with
            'range' (a tuple of minimum and maximum token counts) and 'cost' (cost per token) keys.

        Returns
        -------
        float
            The calculated cost based on the token count and cost structure.
        """
        if isinstance(token_cost, list):
            for cost_range in token_cost:
                if token_count >= cost_range.range[0] and (
                    cost_range.range[1] is None or token_count <= cost_range.range[1]
                ):
                    return cost_range.cost * token_count
        else:
            return token_cost * token_count
        return 0

    def input_to_string(self, input):
        """
        Converts an input, which can be a string or a structured list of messages, into a single concatenated string.

        Parameters
        ----------
        input : Any
            The input data to be converted. This can be:
            - A simple string, which is returned as-is.
            - A list of message dictionaries, where each dictionary may contain `content`, `role`,
            and nested items like `text` or `image_url`.

        Returns
        -------
        str
            A concatenated string representing the text content of all messages,
            including text and URLs from image content if present.
        """
        if isinstance(input, str):
            return input
        else:
            result = []
            for message in input:
                if message.get("content") is not None:
                    if isinstance(message["content"], str):
                        result.append(message["content"])
                    elif (
                        isinstance(message["content"], list)
                        and message.get("role") == "user"
                    ):
                        for item in message["content"]:
                            if item.get("type") == "text":
                                result.append(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                url = item.get("image_url", {}).get("url", "")
                                result.append(url)
            return "".join(result)

    def output_to_string(self, output):
        """
        Extracts and returns the content or arguments from the output based on
        the `finish_reason` of the first choice in `output`.

        Parameters
        ----------
        output : Any
            The model output object, expected to have a `choices` attribute that should contain a `finish_reason` indicating the type of output
            ("stop", "tool_calls", or "function_call") and corresponding content or arguments.

        Returns
        -------
        str
            - If `finish_reason` is "stop": Returns the message content.
            - If `finish_reason` is "tool_calls": Returns the arguments for the first tool call.
            - If `finish_reason` is "function_call": Returns the arguments for the function call.
        """
        if output.choices[0].finish_reason == "stop":
            return output.choices[0].message.content
        elif output.choices[0].finish_reason == "tool_calls":
            return output.choices[0].message.tool_calls[0].function.arguments
        elif output.choices[0].finish_reason == "function_call":
            return output.choices[0].message.function_call.arguments

    def get_end_token_string(self, metrics: Dict[str, Any]) -> str:
        return f"{self.END_TOKEN},input_tokens={metrics['input_tokens']},output_tokens={metrics['output_tokens']},cost_usd={metrics['cost_usd']},latency_s={metrics['latency_s']:.5f},time_to_first_token_s={metrics['time_to_first_token_s']:.5f},inter_token_latency_s={metrics['inter_token_latency_s']:.5f},tokens_per_second={metrics['tokens_per_second']:.2f}"

    def _get_tokenizer(self):
        return {}.get(self.config.id, tiktoken.get_encoding("cl100k_base"))
