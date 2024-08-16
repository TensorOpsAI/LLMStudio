import asyncio
import json
import os
import time
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

import requests
from fastapi import HTTPException
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import BaseModel, Field, ValidationError

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class VertexParameters(BaseModel):
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    top_k: Optional[float] = Field(default=1, ge=0, le=1)
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_output_tokens: Optional[float] = Field(default=8192, ge=0, le=8192)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class VertexAIRequest(ChatRequest):
    parameters: Optional[VertexParameters] = VertexParameters()
    tools: Any = None
    chat_input: Union[str, List[Dict[str, Any]]]


# Define Pydantic model for OpenAI tool parameter
class OpenAIParameter(BaseModel):
    type: str
    description: Optional[str] = None


# Define Pydantic model for OpenAI tool parameters container
class OpenAIParameters(BaseModel):
    type: str
    properties: Dict[str, OpenAIParameter]
    required: List[str]


# Define Pydantic model for OpenAI tool function
class OpenAIToolFunction(BaseModel):
    name: str
    description: str
    parameters: OpenAIParameters


# Define Pydantic model for OpenAI tool
class OpenAITool(BaseModel):
    type: str
    function: OpenAIToolFunction


# Define Pydantic model for VertexAI tool parameter
class VertexAIParameter(BaseModel):
    type: str
    description: str


# Define Pydantic model for VertexAI tool parameters container
class VertexAIParameters(BaseModel):
    type: str
    properties: Dict[str, VertexAIParameter]
    required: List[str]


# Define Pydantic model for VertexAI function declaration
class VertexAIFunctionDeclaration(BaseModel):
    name: str
    description: str
    parameters: VertexAIParameters


# Define Pydantic model for VertexAI functions container
class VertexAI(BaseModel):
    function_declarations: List[VertexAIFunctionDeclaration]


@provider
class VertexAIProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    def validate_request(self, request: VertexAIRequest):
        return VertexAIRequest(**request)

    async def generate_client(
        self, request: VertexAIRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Initialize Vertex AI"""

        try:
            # Init genai
            api_key = request.api_key or self.GOOGLE_API_KEY
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:streamGenerateContent?alt=sse"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }

            # Convert the chat input into VertexAI format
            tool_payload = self.process_tools(request.tools)
            message = self.convert_openai_to_vertexai(request.chat_input, tool_payload)

            # Generate content
            return await asyncio.to_thread(
                requests.post, url, headers=headers, json=message, stream=True
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        for chunk in response.iter_content(chunk_size=None):

            chunk = json.loads(chunk.decode("utf-8").lstrip("data: "))
            chunk = chunk.get("candidates")[0].get("content")

            # Check if it is a function call
            if (
                "functionCall" in chunk["parts"][0]
                and chunk["parts"][0]["functionCall"] is not None
            ):
                first_chunk = ChatCompletionChunk(
                    id="chatcmpl-9woLM1b1qGErhTbXA3UBQf2FhUAho",
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
                )
                yield first_chunk.model_dump()

                for index, functioncall in enumerate(chunk["parts"]):

                    name_chunk = ChatCompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            Choice(
                                delta=ChoiceDelta(
                                    role="assistant",
                                    tool_calls=[
                                        ChoiceDeltaToolCall(
                                            index=index,
                                            id="call_" + str(uuid.uuid4())[:29],
                                            function=ChoiceDeltaToolCallFunction(
                                                name=functioncall["functionCall"][
                                                    "name"
                                                ],
                                                arguments="",
                                                type="function",
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason=None,
                                index=index,
                            )
                        ],
                        created=int(time.time()),
                        model=kwargs.get("request").model,
                        object="chat.completion.chunk",
                    )
                    yield name_chunk.model_dump()

                    args_chunk = ChatCompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            Choice(
                                delta=ChoiceDelta(
                                    tool_calls=[
                                        ChoiceDeltaToolCall(
                                            index=index,
                                            function=ChoiceDeltaToolCallFunction(
                                                arguments=json.dumps(
                                                    functioncall["functionCall"]["args"]
                                                ),
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason=None,
                                index=index,
                            )
                        ],
                        created=int(time.time()),
                        model=kwargs.get("request").model,
                        object="chat.completion.chunk",
                    )
                    yield args_chunk.model_dump()

                final_chunk = ChatCompletionChunk(
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
                )
                yield final_chunk.model_dump()
            # Check if it is a normal call
            elif chunk.get("parts")[0].get("text"):
                # Parse google chunk response into ChatCompletionChunk
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(
                            delta=ChoiceDelta(
                                content=chunk.get("parts")[0].get("text"),
                                role="assistant",
                            ),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                ).model_dump()

                # Create the closing chunk
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                ).model_dump()

    def convert_openai_to_vertexai(self, input_data, tool_payload):
        # Check if the input is a simple string
        if isinstance(input_data, str):
            # Return a Vertex AI formatted message with a user message
            return {
                "system_instruction": {
                    "parts": {
                        "text": "You are a helpful assistant"  # Default system instruction
                    }
                },
                "contents": [{"role": "user", "parts": [{"text": input_data}]}],
                "tools": tool_payload,  # Use the parsed object instead of the JSON string
                "tool_config": {"function_calling_config": {"mode": "AUTO"}},
            }

        # Validate if input_data is a list and each element is a dictionary with the correct structure
        if not isinstance(input_data, list) or not all(
            isinstance(msg, dict) and "role" in msg and "content" in msg
            for msg in input_data
        ):
            raise ValueError(
                "Input must be a list of dictionaries, each containing 'role' and 'content' keys."
            )

        # Initialize the Vertex AI format if the input is not a simple string
        vertexai_format = {
            "system_instruction": {
                "parts": {
                    "text": "You are a helpful assistant"  # Default system instruction
                }
            },
            "contents": [],
            "tools": tool_payload,  # Use the parsed object instead of the JSON string
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
        }

        # Loop through the OpenAI formatted messages
        for message in input_data:
            if message["role"] == "system":
                # Set the system instruction if provided, otherwise keep the default
                vertexai_format["system_instruction"]["parts"]["text"] = (
                    message["content"] or "You are a helpful assistant"
                )
            elif message["role"] == "user":
                # Append the user message to the contents list in Vertex AI format
                vertexai_format["contents"].append(
                    {"role": "user", "parts": [{"text": message["content"]}]}
                )
            elif message["role"] == "assistant":
                if message["content"] is None and "tool_calls" in message:
                    # Parse the tool call into Vertex AI format
                    tool_call = message["tool_calls"][0]
                    vertexai_format["contents"].append(
                        {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": tool_call["function"]["name"],
                                        "args": json.loads(
                                            tool_call["function"]["arguments"]
                                        ),
                                    }
                                }
                            ],
                        }
                    )
                else:
                    # Append the assistant message to the contents list in Vertex AI format
                    vertexai_format["contents"].append(
                        {"role": "model", "parts": [{"text": message["content"]}]}
                    )
            elif message["role"] == "tool":
                # Add the function call and response to the system instruction
                function_name = message["name"]
                response = message["content"]
                vertexai_format["system_instruction"]["parts"][
                    "text"
                ] += f"\nYou have called {function_name} and got the following response: {response}."
            else:
                raise ValueError(
                    f"Invalid role: {message['role']}. Expected 'system', 'user', 'assistant', or 'tool'."
                )

        return vertexai_format

    def process_tools(
        self, tools: Optional[Union[List[Dict], Dict]]
    ) -> Optional[VertexAI]:
        if tools is None:
            return None

        try:
            # Try to parse as OpenAI format
            parsed_tools = (
                [OpenAITool(**tool) for tool in tools]
                if isinstance(tools, list)
                else [OpenAITool(**tools)]
            )
            # Convert to VertexAI format
            function_declarations = []
            for tool in parsed_tools:
                function = tool.function
                properties = {
                    name: VertexAIParameter(
                        type=param.type, description=param.description or ""
                    )
                    for name, param in function.parameters.properties.items()
                }
                function_decl = VertexAIFunctionDeclaration(
                    name=function.name,
                    description=function.description,
                    parameters=VertexAIParameters(
                        type=function.parameters.type,
                        properties=properties,
                        required=function.parameters.required,
                    ),
                )
                function_declarations.append(function_decl)
            return VertexAI(function_declarations=function_declarations).model_dump()
        except ValidationError:
            # If the format is not OpenAI, attempt to validate as VertexAI format
            try:
                return VertexAI(**tools).model_dump()
            except ValidationError:
                # If it fails to validate as VertexAI, throw an error
                raise ValueError(
                    "Invalid tool format. Tool data must be in OpenAI or VertexAI format."
                )
