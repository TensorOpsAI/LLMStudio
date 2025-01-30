import pytest
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from llmstudio.langchain import ChatLLMstudio
from llmstudio.providers import LLM


@tool
def power_disco_ball(power: bool) -> bool:
    """Powers the spinning disco ball."""
    print(f"Disco ball is {'spinning!' if power else 'stopped.'}")
    return True


@tool
def start_music(energetic: bool, loud: bool, bpm: int) -> str:
    """Play some music matching the specified parameters."""
    print(f"Starting music! {energetic=} {loud=}, {bpm=}")
    return "Never gonna give you up."


@tool
def dim_lights(brightness: float) -> bool:
    """Dim the lights."""
    print(f"Lights are now set to {brightness:.0%}")
    return True


def assistant_functions(chat_llm: ChatLLMstudio, question: str) -> str:
    tools = [power_disco_ball, start_music, dim_lights]

    prompt = hub.pull("hwchase17/openai-functions-agent")

    agent = create_openai_functions_agent(llm=chat_llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
    )

    response = agent_executor.invoke({"input": question})

    return response


@pytest.mark.parametrize(
    "provider, model",
    [
        ("openai", "gpt-4o-mini"),
    ],
)
def test_function_calling(provider, model):
    llm = LLM(provider=provider)
    chat_llm = ChatLLMstudio(llm=llm, model=model, parameters={"temperature": 0})
    response = assistant_functions(chat_llm, "Turn this into a party!")
    assert len(response["intermediate_steps"]) > 0


def assistant_tools(chat_llm: ChatLLMstudio, question: str) -> str:
    tools = [power_disco_ball, start_music, dim_lights]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_openai_tools_agent(chat_llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
    )

    response = agent_executor.invoke({"input": question})

    return response


@pytest.mark.parametrize(
    "provider, model",
    [
        ("openai", "gpt-4o-mini"),
    ],
)
def test_tool_calling(provider, model):
    llm = LLM(provider=provider)
    chat_llm = ChatLLMstudio(llm=llm, model=model, parameters={"temperature": 0})
    response = assistant_tools(chat_llm, "Turn this into a party!")
    assert len(response["intermediate_steps"]) > 0
