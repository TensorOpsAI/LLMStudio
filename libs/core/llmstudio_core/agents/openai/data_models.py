from enum import Enum
from typing import Dict, List, Optional

from llmstudio_core.agents.data_models import (
    AgentBase,
    CreateAgentRequest,
    ResultBase,
    RunAgentRequest,
    RunBase,
)
from openai.types.beta.thread import Thread
from pydantic import BaseModel


class ToolResources(BaseModel):
    """Resources required by tools e.g. code_interpreter and file_search."""

    file_ids: Optional[List[str]] = None  # For code_interpreter
    vector_store_ids: Optional[List[str]] = None  # For file_search


class ResponseFormatType(str, Enum):
    """Enum for response format types."""

    AUTO = "auto"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class ResponseFormat(BaseModel):
    """Defines how the model should format its output."""

    type: ResponseFormatType
    json_schema: Optional[Dict] = None  # Required if type=json_schema


class OpenAIAgent(AgentBase):
    thread_id: Optional[Thread]
    tool_resources: Optional[ToolResources]
    temperature: Optional[float]
    top_p: Optional[float]
    response_format: Optional[ResponseFormat]


class OpenAIRun(RunBase):
    thread_id: str
    run_id: str


class OpenAIResult(ResultBase):
    thread_id: str


class OpenAICreateAgentRequest(CreateAgentRequest):
    tool_resources: ToolResources
    temperature: Optional[float]
    top_p: Optional[float]
    response_format: Optional[ResponseFormat]


class OpenAIRunAgentRequest(RunAgentRequest):
    thread_id: Optional[str]
