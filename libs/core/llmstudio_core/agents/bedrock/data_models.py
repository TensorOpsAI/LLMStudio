from llmstudio_core.agents.data_models import (
    AgentBase,
    CreateAgentRequest,
    ResultBase,
    RunAgentRequest,
    RunBase,
)


class BedrockAgent(AgentBase):
    agentResourceRoleArn: str
    agentStatus: str
    agentVersion: str
    agentArn: str


class BedrockRun(RunBase):
    session_id: str
    response: dict


class BedrockResult(ResultBase):
    session_id: str


class BedrockCreateAgentRequest(CreateAgentRequest):
    agent_resourcerole_arn: str
    agent_alias: str


class BedrockRunAgentRequest(RunAgentRequest):
    session_id: str
