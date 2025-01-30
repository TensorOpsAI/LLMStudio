import ast
import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Generator, Union

import openai
from llmstudio_core.exceptions import ProviderError
from llmstudio_core.providers.provider import ChatRequest, ProviderCore, provider
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)


@provider
class AzureProvider(ProviderCore):
    def __init__(
        self,
        config,
        api_key=None,
        api_endpoint=None,
        api_version=None,
        base_url=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.API_KEY = api_key or os.getenv("AZURE_API_KEY")
        self.API_ENDPOINT = api_endpoint
        self.API_VERSION = api_version or os.getenv("AZURE_API_VERSION")
        self.BASE_URL = base_url
        self.is_llama = False
        self.has_tools_functions = False

        if self.BASE_URL and (self.API_ENDPOINT is None):
            self._client = OpenAI(
                api_key=self.API_KEY,
                base_url=self.BASE_URL,
            )
        else:
            self._client = AzureOpenAI(
                api_key=self.API_KEY,
                azure_endpoint=self.API_ENDPOINT,
                api_version=self.API_VERSION,
            )

    @staticmethod
    def _provider_config_name():
        return "azure"

    def validate_request(self, request: ChatRequest):
        return ChatRequest(**request)

    async def agenerate_client(self, request: ChatRequest) -> Any:
        """Generate an AzureOpenAI client"""
        return self.generate_client(request=request)

    def generate_client(self, request: ChatRequest) -> Any:
        """
        Generates an AzureOpenAI client for processing a chat request.

        This method prepares and configures the arguments required to create a client
        request to AzureOpenAI's chat completions API. It determines model-specific
        configurations (e.g., whether tools or functions are enabled) and combines
        these with the base arguments for the API call.

        Args:
            request (ChatRequest): The chat request object containing the model,
                                parameters, and other necessary details.

        Returns:
            Any: The result of the chat completions API call.

        Raises:
            ProviderError: If there is an issue with the API connection or an error
                        returned from the API.
        """

        self.is_llama = "llama" in request.model.lower()
        self.is_openai = "gpt" in request.model.lower()
        self.has_tools = request.parameters.get("tools") is not None
        self.has_functions = request.parameters.get("functions") is not None

        try:
            messages = self.prepare_messages(request)

            tool_args = {}
            if not self.is_llama and self.has_tools and self.is_openai:
                tool_args = {
                    "tools": request.parameters.get("tools"),
                    "tool_choice": "auto" if request.parameters.get("tools") else None,
                }

            function_args = {}
            if not self.is_llama and self.has_functions and self.is_openai:
                function_args = {
                    "functions": request.parameters.get("functions"),
                    "function_call": "auto"
                    if request.parameters.get("functions")
                    else None,
                }

            base_args = {
                "model": request.model,
                "messages": messages,
                "stream": True,
            }

            combined_args = {
                **base_args,
                **tool_args,
                **function_args,
                **request.parameters,
            }
            return self._client.chat.completions.create(**combined_args)

        except openai._exceptions.APIConnectionError as e:
            raise ProviderError(f"There was an error reaching the endpoint: {e}")

        except openai._exceptions.APIStatusError as e:
            raise ProviderError(e.response.json())

    def prepare_messages(self, request: ChatRequest):
        if self.is_llama and (self.has_tools or self.has_functions):
            user_message = self.convert_to_openai_format(request.chat_input)
            content = "<|begin_of_text|>"
            content = self.build_llama_system_message(
                user_message,
                content,
                request.parameters.get("tools"),
                request.parameters.get("functions"),
            )
            content = self.build_llama_conversation(user_message, content)
            return [{"role": "user", "content": content}]
        else:
            return (
                [{"role": "user", "content": request.chat_input}]
                if isinstance(request.chat_input, str)
                else request.chat_input
            )

    async def aparse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        result = self.parse_response(response=response, **kwargs)
        for chunk in result:
            yield chunk

    def parse_response(self, response: AsyncGenerator, **kwargs) -> Any:
        """
        Processes a generator response and yields processed chunks.

        If `is_llama` is True and tools or functions are enabled, it processes the response
        using `handle_tool_response`. Otherwise, it processes each chunk and yields only those
        containing "choices".

        Args:
            response (Generator): The response generator to process.
            **kwargs: Additional arguments for tool handling.

        Yields:
            Any: Processed response chunks.
        """
        if self.is_llama and (self.has_tools or self.has_functions):
            for chunk in self.handle_tool_response(response, **kwargs):
                if chunk:
                    yield chunk
        else:
            for chunk in response:
                c = chunk.model_dump()
                if c.get("choices"):
                    yield c

    def handle_tool_response(self, response: AsyncGenerator, **kwargs) -> Generator:
        """
        Asynchronously handles tool responses by parsing the content for function calls or tool activations.
        It processes the response chunks to extract and execute embedded function or tool calls, then yields
        the processed chunks or the results of these calls.

        Args:
            response: An asynchronous generator yielding response chunks from the tool or function.
            **kwargs: Additional keyword arguments that may be required for processing.

        Yields:
            An asynchronous generator of strings or processed chunks, depending on whether the response
            contains function or tool calls that need to be executed or just plain content to be returned.
        """

        function_call_buffer = ""
        saving = False
        normal_call_chunks = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                if (
                    "§" in chunk.choices[0].delta.content
                    or "<|python_tag|>" in chunk.choices[0].delta.content
                ):
                    saving = True

            if saving:
                function_call_buffer += chunk.choices[0].delta.content
                if chunk.choices[0].finish_reason == "stop":
                    cleaned_buffer = function_call_buffer.replace("§", "").replace(
                        "<|python_tag|>", ""
                    )
                    result_dict = eval(cleaned_buffer)

                    if self.has_functions:

                        # Create first chunk
                        first_chunk = self.create_tool_first_chunk(kwargs)
                        yield first_chunk

                        name_chunk = self.create_function_name_chunk(
                            result_dict["name"], kwargs
                        )
                        yield name_chunk

                        parameters = json.dumps(result_dict["parameters"])
                        args_chunk = self.create_function_argument_chunk(
                            parameters, kwargs
                        )
                        yield args_chunk

                        finish_chunk = self.create_function_finish_chunk(kwargs)
                        yield finish_chunk

                    if self.has_tools:
                        name_chunk = self.create_tool_name_chunk(
                            result_dict["name"], kwargs
                        )
                        yield name_chunk

                        parameters = json.dumps(result_dict["parameters"])
                        args_chunk = self.create_tool_argument_chunk(parameters, kwargs)
                        yield args_chunk

                        finish_chunk = self.create_tool_finish_chunk(kwargs)
                        yield finish_chunk

            else:
                normal_call_chunks.append(chunk)
                if chunk.choices[0].finish_reason == "stop":
                    for chunk in normal_call_chunks:
                        normal_call_chunks.append(chunk)
                if chunk.choices[0].finish_reason == "stop":
                    for chunk in normal_call_chunks:
                        yield chunk.model_dump()

    def create_tool_name_chunk(self, function_name: str, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id=str(uuid.uuid4()),
                                function=ChoiceDeltaToolCallFunction(
                                    name=function_name,
                                    arguments="",
                                    type="function",
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
        ).model_dump()

    def create_function_name_chunk(self, function_name: str, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=None,
                        function_call=ChoiceDeltaFunctionCall(
                            arguments="", name=function_name
                        ),
                        role="assistant",
                        tool_calls=None,
                        refusal=None,
                    ),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
            system_fingerprint=None,
            usage=None,
        ).model_dump()

    def create_tool_finish_chunk(self, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(),
                    finish_reason="tool_calls",
                    index=0,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
        ).model_dump()

    def create_tool_argument_chunk(self, content: str, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(
                                    arguments=content,
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
        ).model_dump()

    def create_function_argument_chunk(self, content: str, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=None,
                        function_call=ChoiceDeltaFunctionCall(
                            arguments=content, name=None
                        ),
                        role=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
            system_fingerprint=None,
            usage=None,
        ).model_dump()

    def create_tool_first_chunk(self, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=None,
                        function_call=None,
                        role="assistant",
                        tool_calls=None,
                    ),
                    index=0,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
            usage=None,
        ).model_dump()

    def create_function_finish_chunk(self, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=None, function_call=None, role=None, tool_calls=None
                    ),
                    finish_reason="function_call",
                    index=0,
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
            system_fingerprint=None,
            usage=None,
        ).model_dump()

    def convert_to_openai_format(self, message: Union[str, list]) -> list:
        if isinstance(message, str):
            return [{"role": "user", "content": message}]
        return message

    def build_llama_system_message(
        self, openai_message: list, llama_message: str, tools: list, functions: list
    ) -> str:
        """
        Builds a complete system message for Llama based on OpenAI's message, tools, and functions.

        If a system message is present in the OpenAI message, it is included in the result.
        Otherwise, a default system message is used. Additional tool and function instructions
        are appended if provided.

        Args:
            openai_message (list): List of OpenAI messages.
            llama_message (str): The message to prepend to the system message.
            tools (list): List of tools to include in the system message.
            functions (list): List of functions to include in the system message.

        Returns:
            str: The formatted system message combined with Llama message.
        """
        system_message = ""
        system_message_found = False
        for message in openai_message:
            if message["role"] == "system" and message["content"] is not None:
                system_message_found = True
                system_message = f"""
        <|start_header_id|>system<|end_header_id|>
        {message['content']}
        """
        if not system_message_found:
            system_message = """
      <|start_header_id|>system<|end_header_id|>
      You are a helpful AI assistant.
      """

        if tools:
            system_message = system_message + self.build_tool_instructions(tools)

        if functions:
            system_message = system_message + self.build_function_instructions(
                functions
            )

        end_tag = "\n<|eot_id|>"
        return llama_message + system_message + end_tag

    def build_tool_instructions(self, tools: list) -> str:
        """
        Builds a detailed instructional prompt for tools available to the assistant.

        This function generates a message describing the available tools, focusing on tools
        of type "function." It explains to the LLM how to use each tool and provides an example of the
        correct response format for function calls.

        Args:
            tools (list): A list of tool dictionaries, where each dictionary contains tool
            details such as type, function name, description, and parameters.

        Returns:
            str: A formatted string detailing the tool instructions and usage examples.
        """
        tool_prompt = """
    You have access to the following tools:
    """

        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                tool_prompt += (
                    f"Use the function '{func['name']}' to '{func['description']}':\n"
                )
                params_info = json.dumps(func["parameters"], indent=4)
                tool_prompt += f"Parameters format:\n{params_info}\n\n"

        tool_prompt += """
If you choose to use a function to produce this response, ONLY reply in the following format with no prefix or suffix:
§{"type": "function", "name": "FUNCTION_NAME", "parameters": {"PARAMETER_NAME": PARAMETER_VALUE}}
IMPORTANT: IT IS VITAL THAT YOU NEVER ADD A PREFIX OR A SUFFIX TO THE FUNCTION CALL.

Here is an example of the output I desiere when performing function call:
§{"type": "function", "name": "python_repl_ast", "parameters": {"query": "print(df.shape)"}}
NOTE: There is no prefix before the symbol '§' and nothing comes after the call is done.

    Reminder:
    - Function calls MUST follow the specified format.
    - Only call one function at a time.
    - Required parameters MUST be specified.
    - Put the entire function call reply on one line.
    - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
    - If you have already called a tool and got the response for the users question please reply with the response.
    """

        return tool_prompt

    def build_function_instructions(self, functions: list) -> str:
        """
        Builds a detailed instructional prompt for available functions.

        This method creates a message describing the functions accessible to the assistant.
        It includes the function name, description, and required parameters, along with
        specific guidelines for calling functions.

        Args:
            functions (list): A list of function dictionaries, each containing details such as
            name, description, and parameters.

        Returns:
            str: A formatted string with instructions on using the provided functions.
        """
        function_prompt = """
You have access to the following functions:
"""

        for func in functions:
            function_prompt += (
                f"Use the function '{func['name']}' to: '{func['description']}'\n"
            )
            params_info = json.dumps(func["parameters"], indent=4)
            function_prompt += f"{params_info}\n\n"

        function_prompt += """
If you choose to use a function to produce this response, ONLY reply in the following format with no prefix or suffix:
§{"type": "function", "name": "FUNCTION_NAME", "parameters": {"PARAMETER_NAME": PARAMETER_VALUE}}

Here is an example of the output I desiere when performing function call:
§{"type": "function", "name": "python_repl_ast", "parameters": {"query": "print(df.shape)"}}

Reminder:
- Function calls MUST follow the specified format.
- Only call one function at a time.
- NEVER call more than one function at a time.
- Required parameters MUST be specified.
- Put the entire function call reply on one line.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
- If you have already called a function and got the response for the user's question, please reply with the response.
"""
        return function_prompt

    def build_llama_conversation(self, openai_message: list, llama_message: str) -> str:
        """
        Appends the OpenAI message to the Llama message while formatting OpenAI messages.

        This function iterates through a list of OpenAI messages and formats them for inclusion
        in a Llama message. It handles user messages that might include nested content (lists of
        messages) by safely evaluating the content. System messages are skipped.

        Args:
            openai_message (list): A list of dictionaries representing the OpenAI messages. Each
                                dictionary should have "role" and "content" keys.
            llama_message (str): The initial Llama message to which the conversation is appended.

        Returns:
            str: The Llama message with the conversation appended.
        """
        conversation_parts = []
        for message in openai_message:
            if message["role"] == "system":
                continue
            elif message["role"] == "user" and isinstance(message["content"], str):
                try:
                    content_as_list = ast.literal_eval(message["content"])
                    if isinstance(content_as_list, list):
                        for nested_message in content_as_list:
                            conversation_parts.append(
                                self.format_message(nested_message)
                            )
                    else:
                        conversation_parts.append(self.format_message(message))
                except (ValueError, SyntaxError):
                    conversation_parts.append(self.format_message(message))
            else:
                conversation_parts.append(self.format_message(message))

        return llama_message + "".join(conversation_parts)

    def format_message(self, message: dict) -> str:
        """
        Formats a single message dictionary into a structured string for a conversation.

        The formatting depends on the content of the message, such as tool calls,
        function calls, or simple user/assistant messages. Each type of message
        is formatted with specific headers and tags.

        Args:
            message (dict): A dictionary containing message details. Expected keys
                            include "role", "content", and optionally "tool_calls",
                            "tool_call_id", or "function_call".

        Returns:
            str: A formatted string representing the message. Returns an empty
            string if the message cannot be formatted.
        """
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                return f"""
        <|start_header_id|>assistant<|end_header_id|>
        <function={function_name}>{arguments}</function>
        <|eom_id|>
        """
        elif "tool_call_id" in message:
            tool_response = message["content"]
            return f"""
    <|start_header_id|>ipython<|end_header_id|>
    {tool_response}
    <|eot_id|>
    """
        elif "function_call" in message:
            function_name = message["function_call"]["name"]
            arguments = message["function_call"]["arguments"]
            return f"""
        <|start_header_id|>assistant<|end_header_id|>
        <function={function_name}>{arguments}</function>
        <|eom_id|>
        """
        elif (
            message["role"] in ["assistant", "user"] and message["content"] is not None
        ):
            return f"""
    <|start_header_id|>{message['role']}<|end_header_id|>
    {message['content']}
    <|eot_id|>
    """
        elif message["role"] == "function":
            function_response = message["content"]
            return f"""
    <|start_header_id|>ipython<|end_header_id|>
    {function_response}
    <|eot_id|>
    """
        return ""
