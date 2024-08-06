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

import google.generativeai as genai
from fastapi import HTTPException
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
)
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class VertexAIParameters(BaseModel):
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    top_k: Optional[float] = Field(default=1, ge=0, le=1)
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_output_tokens: Optional[float] = Field(default=8192, ge=0, le=8192)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class VertexAIRequest(ChatRequest):
    parameters: Optional[VertexAIParameters] = VertexAIParameters()
    functions: Optional[List[Dict[str, Any]]] = None
    chat_input: Union[str, List[Dict[str, Any]]]


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
            genai.configure(api_key=request.api_key or self.GOOGLE_API_KEY)

            # Check if chat_input is a string
            if isinstance(request.chat_input, str):
                message = request.chat_input

            # Check if the message is in the OpenAI format
            elif isinstance(request.chat_input, list) and all(
                isinstance(item, dict) and "role" in item and "content" in item
                for item in request.chat_input
            ):
                system_message, message = self.parse_openai_messages_with_functions(
                    request.chat_input
                )

            # Check if has functions and is in OpenAI format
            if request.functions and all(
                isinstance(item, dict) and "role" in item and "content" in item
                for item in request.chat_input
            ):
                message = self.generate_prompt_with_functions(message, system_message)

            if request.functions:
                model = genai.GenerativeModel(request.model, tools=[request.functions])
            else:
                model = genai.GenerativeModel(request.model)

            # Generate content
            return await asyncio.to_thread(model.generate_content, message, stream=True)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def generate_prompt_with_functions(self, conversation, system_message):
        # Create the prompt
        prompt = (
            f"Here are your system instructions: {system_message}\n"
            if system_message
            else ""
            "Based on the conversation so far, please answer the last questions the user made. "
            "Here is the conversation:\n"
            f"{conversation}\n\n"
        )
        return prompt

    def parse_openai_messages_with_functions(self, messages):
        system_message = ""
        conversation = []

        for message in messages:
            role = message["role"]
            content = message.get("content")
            function_call = message.get("function_call")

            if role == "system":
                system_message = content
            elif role == "user":
                conversation.append(f"User: {content}")
            elif role == "assistant":
                if function_call:
                    # Handle function call messages
                    function_name = function_call.get("name")
                    arguments = function_call.get("arguments", "{}")
                    conversation.append(
                        f"Assistant called function '{function_name}' with arguments {arguments}"
                    )
                else:
                    conversation.append(f"Assistant: {content}")
            elif role == "function":
                function_name = message.get("name")
                conversation.append(
                    f"Function '{function_name}' responded with: {content}"
                )

        # Join the conversation parts with newlines
        conversation_str = "\n".join(conversation)

        return system_message, conversation_str

    def genereate_tool_messege(self, chat_input: List[Dict[str, Any]]) -> str:
        question = next(
            (entry["content"] for entry in chat_input if entry["role"] == "user"), ""
        )
        tools = [
            {
                "tool_name": entry["name"],
                "tool_description": "Description not provided",
                "tool_response": entry["content"],
            }
            for entry in chat_input
            if entry["role"] == "function"
        ]

        tools_str = "\n".join(
            f"Tool{i+1}: {tool['tool_name']}, Description{i+1}:{tool['tool_description']}, Response{i+1}:{tool['tool_response']}"
            for i, tool in enumerate(tools)
        )

        return f"""
        Please answer the following question: {question}

        This is the tool responses I got so far:

        {tools_str}

        Call any other tool if you think is necessary. Otherwise please answer the question based on the tool responses you got.
        """

    @staticmethod
    def parse_string_to_dict(args):
        return json.dumps(
            {
                key: int(value.ListFields()[0][1])
                if isinstance(value.ListFields()[0][1], (int, float))
                and value.ListFields()[0][1] == int(value.ListFields()[0][1])
                else value.ListFields()[0][1]
                for key, value in args.items()
            }
        ).replace(" ", "")

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        for chunk in response:

            # Check if it is a function call
            if chunk.parts[0].function_call:
                name = chunk.parts[0].__dict__["_pb"].function_call.name
                args = chunk.parts[0].__dict__["_pb"].function_call.args.fields
                openai_args = self.parse_string_to_dict(args)
                str(uuid.uuid4())

                # First chunk
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(
                            delta=ChoiceDelta(
                                role="assistant",
                                function_call=ChoiceDeltaFunctionCall(
                                    name=name, arguments=""
                                ),
                            ),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                ).model_dump()

                # Midle chunk with arguments
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(
                            delta=ChoiceDelta(
                                role=None,
                                function_call=ChoiceDeltaFunctionCall(
                                    arguments=openai_args, name=None
                                ),
                            ),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                ).model_dump()

                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(
                            delta=ChoiceDelta(), finish_reason="function_call", index=0
                        )
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                ).model_dump()

            # Check if it is a normal call
            elif chunk.parts[0].text:
                # Parse google chunk response into ChatCompletionChunk
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(
                            delta=ChoiceDelta(content=chunk.text, role="assistant"),
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
