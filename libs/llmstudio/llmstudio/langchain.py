from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from langchain.schema.messages import BaseMessage
from langchain.schema.output import ChatGeneration, ChatResult
from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from llmstudio.providers import LLM
from openai import BaseModel


class ChatLLMstudio(BaseChatModel):
    llm: LLM
    model: str
    is_stream: bool = False
    retries: int = 0
    parameters: dict = {}

    @property
    def _llm_type(self):
        return "LLMstudio"

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts

    def _create_chat_result(self, response: Any) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("metrics", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": response["model"],
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                "auto" to automatically determine which function to call
                with the option to not call any function, "any" to enforce that some
                function is called, or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "any", "none")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, dict) and (len(formatted_tools) != 1):
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, dict) and (
                formatted_tools[0]["function"]["name"]
                != tool_choice["function"]["name"]
            ):
                raise ValueError(
                    f"Tool choice {tool_choice} was specified, but the only "
                    f"provided tool was {formatted_tools[0]['function']['name']}."
                )
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        messages_dicts = self._create_message_dicts(messages, [])
        response = self.llm.chat(
            messages_dicts,
            model=kwargs.get("model", self.model),
            is_stream=kwargs.get("is_stream", self.is_stream),
            retries=kwargs.get("retries", self.retries),
            parameters=kwargs.get("parameters", self.parameters),
            **kwargs,
        )
        return self._create_chat_result(response)
