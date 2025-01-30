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
from llmstudio_core.providers.data_structures import (
    ChatCompletionChunkLLMstudio,
    ChatCompletionLLMstudio,
    Metrics,
)
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
    async def aparse_response(self, response: AsyncGenerator, **kwargs) -> Any:
        raise NotImplementedError("ProviderCore needs a aparse_response method.")

    @abstractmethod
    def parse_response(self, response: AsyncGenerator, **kwargs) -> Any:
        raise NotImplementedError("ProviderCore needs a parse_response method.")

    @abstractmethod
    def get_usage(self, response: AsyncGenerator, **kwargs) -> Any:
        raise NotImplementedError("ProviderCore needs a get_usage method.")

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
                response_handler = self._ahandle_response(request, response, start_time)

                if request.is_stream:
                    return response_handler
                else:
                    return await response_handler.__anext__()
            # except HTTPException as e:
            #     if e.status_code == 429:
            #         continue  # Retry on rate limit error
            #     else:
            #         raise e  # Raise other HTTP exceptions
            except Exception as e:
                raise ProviderError(str(e))
        raise ProviderError("Too many requests")

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
                response = self.generate_client(request)
                response_handler = self._handle_response(request, response, start_time)

                if request.is_stream:
                    return response_handler
                else:
                    return response_handler.__next__()
            # except HTTPExceptio as e:
            #     if e.status_code == 429:
            #         continue  # Retry on rate limit error
            #     else:
            #         raise e  # Raise other HTTP exceptions
            except Exception as e:
                raise ProviderError(str(e))
        raise ProviderError("Too many requests")

    async def _ahandle_response(
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
        Union[ChatCompletionChunk, ChatCompletion]
            - If `request.is_stream` is True, yields `ChatCompletionChunk` objects with incremental
            response chunks for streaming.
            - If `request.is_stream` is False, yields a final `ChatCompletion` object after processing all chunks.
        """
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0
        chunks = []
        is_next_usage = False
        usage = {}

        async for chunk in self.aparse_response(response, request=request):
            token_count += 1
            current_time = time.time()
            first_token_time = first_token_time or current_time
            if previous_token_time is not None:
                token_times.append(current_time - previous_token_time)
            previous_token_time = current_time

            if is_next_usage:
                usage = self.get_usage(chunk)
                break

            chunks.append(chunk)
            finish_reason = chunk.get("choices")[0].get("finish_reason")
            if finish_reason:
                is_next_usage = True

            if request.is_stream:
                chunk = chunk[0] if isinstance(chunk, tuple) else chunk
                model = chunk.get("model")
                chat_output = chunk.get("choices")[0].get("delta").get("content")
                chunk = {
                    **chunk,
                    "id": str(uuid.uuid4()),
                    "chat_input": (
                        request.chat_input
                        if isinstance(request.chat_input, str)
                        else request.chat_input[-1]["content"]
                    ),
                    "chat_output": None,
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
                yield ChatCompletionChunkLLMstudio(**chunk)

        chunks = [chunk[0] if isinstance(chunk, tuple) else chunk for chunk in chunks]
        model = next(chunk["model"] for chunk in chunks if chunk.get("model"))

        response, output_string = self._join_chunks(chunks)

        metrics = self._calculate_metrics(
            request.chat_input,
            response,
            request.model,
            start_time,
            time.time(),
            first_token_time,
            token_times,
            token_count,
            is_stream=request.is_stream,
            usage=usage,
        )

        response = {
            **(chunk if request.is_stream else response.model_dump()),
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

        if request.is_stream:
            yield ChatCompletionChunkLLMstudio(**response)
        else:
            yield ChatCompletionLLMstudio(**response)

    def _handle_response(
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
        Union[ChatCompletionChunk, ChatCompletion]
            If streaming (`is_stream=True`), yields each `ChatCompletionChunk` as it’s processed.
            Otherwise, yields a single `ChatCompletion` with the full response data.

        """
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0
        chunks = []
        is_next_usage = False
        usage = {}

        for chunk in self.parse_response(response, request=request):
            token_count += 1
            current_time = time.time()
            first_token_time = first_token_time or current_time
            if previous_token_time is not None:
                token_times.append(current_time - previous_token_time)
            previous_token_time = current_time

            if is_next_usage:
                usage = self.get_usage(chunk)
                break

            chunks.append(chunk)
            finish_reason = chunk.get("choices")[0].get("finish_reason")
            if finish_reason:
                is_next_usage = True

            if request.is_stream:
                chunk = chunk[0] if isinstance(chunk, tuple) else chunk
                model = chunk.get("model")

                chat_output = chunk.get("choices")[0].get("delta").get("content")
                chunk = {
                    **chunk,
                    "id": str(uuid.uuid4()),
                    "chat_input": (
                        request.chat_input
                        if isinstance(request.chat_input, str)
                        else request.chat_input[-1]["content"]
                    ),
                    "chat_output": None,
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
                yield ChatCompletionChunkLLMstudio(**chunk)

        chunks = [chunk[0] if isinstance(chunk, tuple) else chunk for chunk in chunks]
        model = next(chunk["model"] for chunk in chunks if chunk.get("model"))

        response, output_string = self._join_chunks(chunks)

        metrics = self._calculate_metrics(
            request.chat_input,
            response,
            request.model,
            start_time,
            time.time(),
            first_token_time,
            token_times,
            token_count,
            is_stream=request.is_stream,
            usage=usage,
        )

        response = {
            **(chunk if request.is_stream else response.model_dump()),
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

        if request.is_stream:
            yield ChatCompletionChunkLLMstudio(**response)
        else:
            yield ChatCompletionLLMstudio(**response)

    def _join_chunks(self, chunks):
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

    def _calculate_metrics(
        self,
        input: Any,
        output: Any,
        model: str,
        start_time: float,
        end_time: float,
        first_token_time: float,
        token_times: Tuple[float, ...],
        token_count: int,
        is_stream: bool,
        usage: Dict = {},
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
        """

        model_config = self.config.models[model]

        # Token counts
        cached_tokens = 0
        reasoning_tokens = 0
        input_tokens = len(self.tokenizer.encode(self._input_to_string(input)))
        output_tokens = len(self.tokenizer.encode(self._output_to_string(output)))
        total_tokens = input_tokens + output_tokens

        # Cost calculations
        input_cost = self._calculate_cost(input_tokens, model_config.input_token_cost)
        output_cost = self._calculate_cost(
            output_tokens, model_config.output_token_cost
        )
        total_cost_usd = input_cost + output_cost

        if usage:
            if getattr(model_config, "cached_token_cost", None):
                cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
                cached_savings = self._calculate_cost(
                    cached_tokens, model_config.cached_token_cost
                )
                total_cost_usd -= cached_savings

            reasoning_tokens = usage.get("completion_tokens_details", {}).get(
                "reasoning_tokens", None
            )
            if reasoning_tokens:
                total_tokens += reasoning_tokens
                reasoning_cost = self._calculate_cost(
                    reasoning_tokens, model_config.output_token_cost
                )  # billed as output tokens
                print(f"Reasoning Cost: {reasoning_cost}")
                total_cost_usd += reasoning_cost

        # Latency calculations
        total_time = end_time - start_time

        if is_stream:
            time_to_first_token = first_token_time - start_time
            inter_token_latency = (
                sum(token_times) / len(token_times) if token_times else 0.0
            )
            tokens_per_second = token_count / total_time if total_time > 0 else 0.0

            return Metrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                cost_usd=total_cost_usd,
                latency_s=total_time,
                time_to_first_token_s=time_to_first_token,
                inter_token_latency_s=inter_token_latency,
                tokens_per_second=tokens_per_second,
            )
        else:
            return Metrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                cost_usd=total_cost_usd,
                latency_s=total_time,
            )

    def _calculate_cost(
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

    def get_usage(self, chunk) -> Dict:
        """
        Gets Usage Object from chunk - usually the last one.
        Returns an empty dictionary if usage does not exist or is None.
        """
        if not chunk or "usage" not in chunk or chunk["usage"] is None:
            return {}
        return dict(chunk["usage"])

    def _input_to_string(self, input):
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

    def _output_to_string(self, output):
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
