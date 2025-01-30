# %% [markdown]
# # Langchain integration

# %% [markdown]
# ### LLMstudio setup

# %%
import os
from llmstudio.langchain import ChatLLMstudio
from llmstudio.providers import LLM

llm = LLM(provider="openai")
chat_llm = ChatLLMstudio(llm=llm, model = "gpt-4o-mini", parameters={"temperature":0})
# chat_llm = ChatLLMstudio(model_id='vertexai/gemini-1.5-flash', temperature=0)

# %% [markdown]
# ### Langchain setup

# %%
from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent, AgentExecutor

from langchain.agents.openai_functions_agent.base import (
    create_openai_functions_agent,
)
from langchain import hub

# # %%
# print("\n", chat_llm.invoke('Hello'))

# # %% [markdown]
# # ### Example 1: Train ticket

# # %%
# @tool
# def get_departure(ticket_number: str):
#     """Use this to fetch the departure time of a train"""
#     return "12:00 AM"

# @tool
# def buy_ticket(destination: str):
#     """Use this to buy a ticket"""
#     return "Bought ticket number 123456"


# def assistant(question: str)->str:
#     tools = [get_departure, buy_ticket]
#     print(tools)

#     #rebuild agent with new tools
#     agent_executor = initialize_agent(
#         tools, chat_llm, agent=AgentType.OPENAI_FUNCTIONS, verbose = True, debug = True
#     )

#     response = agent_executor.invoke({"input": question})

#     return response

# # %%
# assistant('When does my train depart? My ticket is 1234')


# # %%
# assistant('Buy me a ticket to Madrid and tell the departure time')

# # %% [markdown]
# # ### Example 2: Start a party

# # %%
# @tool
# def power_disco_ball(power: bool) -> bool:
#     """Powers the spinning disco ball."""
#     print(f"Disco ball is {'spinning!' if power else 'stopped.'}")
#     return True

# @tool
# def start_music(energetic: bool, loud: bool, bpm: int) -> str:
#     """Play some music matching the specified parameters.
#     """
#     print(f"Starting music! {energetic=} {loud=}, {bpm=}")
#     return "Never gonna give you up."

# @tool
# def dim_lights(brightness: float) -> bool:
#     """Dim the lights.
#     """
#     print(f"Lights are now set to {brightness:.0%}")
#     return True


# # %%
# def assistant(question: str)->str:
#     tools = [power_disco_ball, start_music, dim_lights]
#     print(tools)

#     #rebuild agent with new tools
#     agent_executor = initialize_agent(
#         tools, chat_llm, agent=AgentType.OPENAI_FUNCTIONS, verbose = True, debug = True
#     )

#     response = agent_executor.invoke(
#         {
#             "input": question
#         }
#     )

#     return response

# # %%
# assistant('Turn this into a party!')


# # azure
# from llmstudio.providers import LLM
# llm = LLM(provider="azure", 
#           api_key=os.environ["AZURE_API_KEY"], 
#           api_version=os.environ["AZURE_API_VERSION"],
#           api_endpoint=os.environ["AZURE_API_ENDPOINT"])
# chat_llm = ChatLLMstudio(llm=llm, model = "gpt-4o-mini", parameters={"temperature":0})

# vertex

# chat_llm = ChatLLMstudio(model_id='vertexai/gemini-1.5-flash', temperature=0)

# # %% [markdown]
# # ### Langchain setup

# # %%
# from langchain.tools import tool
# from langchain.agents import AgentType, initialize_agent

# # %%
# print("\n", chat_llm.invoke('Hello'))

# # %% [markdown]
# # ### Example 1: Train ticket

# # %%
# @tool
# def get_departure(ticket_number: str):
#     """Use this to fetch the departure time of a train"""
#     return "12:00 AM"

# @tool
# def buy_ticket(destination: str):
#     """Use this to buy a ticket"""
#     return "Bought ticket number 123456"


# def assistant(question: str)->str:
#     tools = [get_departure, buy_ticket]
#     print(tools)

#     #rebuild agent with new tools
#     agent_executor = initialize_agent(
#         tools, chat_llm, agent=AgentType.OPENAI_FUNCTIONS, verbose = True, debug = True
#     )

#     response = agent_executor.invoke({"input": question})

#     return response

# # %%
# assistant('When does my train depart? My ticket is 1234')


# # %%
# assistant('Buy me a ticket to Madrid and tell the departure time')

# # %% [markdown]
# # ### Example 2: Start a party

# %%
@tool
def power_disco_ball(power: bool) -> bool:
    """Powers the spinning disco ball."""
    print(f"Disco ball is {'spinning!' if power else 'stopped.'}")
    return True

@tool
def start_music(energetic: bool, loud: bool, bpm: int) -> str:
    """Play some music matching the specified parameters.
    """
    print(f"Starting music! {energetic=} {loud=}, {bpm=}")
    return "Never gonna give you up."

@tool
def dim_lights(brightness: float) -> bool:
    """Dim the lights.
    """
    print(f"Lights are now set to {brightness:.0%}")
    return True


# %%
def assistant(question: str)->str:
    tools = [power_disco_ball, start_music, dim_lights]
    print(tools)

    #rebuild agent with new tools - This is the old outdated way of using agents in langchain
    #agent_executor = initialize_agent(
    #    tools, chat_llm, agent=AgentType.OPENAI_FUNCTIONS, verbose = True, debug = True
    #) 
    prompt = hub.pull("hwchase17/openai-functions-agent")
    
    agent = create_openai_functions_agent(llm=chat_llm, tools=tools, prompt=prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
    )

    response = agent_executor.invoke(
        {
            "input": question
        }
    )

    return response

# %%
assistant('Turn this into a party!')


# azure
@tool
def power_disco_ball(power: bool) -> bool:
    """Powers the spinning disco ball."""
    print(f"Disco ball is {'spinning!' if power else 'stopped.'}")
    return True

@tool
def start_music(energetic: bool, loud: bool, bpm: int) -> str:
    """Play some music matching the specified parameters.
    """
    print(f"Starting music! {energetic=} {loud=}, {bpm=}")
    return "Never gonna give you up."

@tool
def dim_lights(brightness: float) -> bool:
    """Dim the lights.
    """
    print(f"Lights are now set to {brightness:.0%}")
    return True

from langchain.agents import AgentExecutor, create_tool_calling_agent, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate

def assistant_new(question: str,chat_llm)->str:
    tools = [power_disco_ball, start_music, dim_lights]
    print(tools)

    #rebuild agent with new tools
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)
    agent = create_openai_tools_agent(chat_llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    response = agent_executor.invoke(
        {
            "input": question
        }
    )

    return response

from llmstudio.providers import LLM
llm = LLM(provider="azure", 
          api_key=os.environ["AZURE_API_KEY"], 
          api_version=os.environ["AZURE_API_VERSION"],
          api_endpoint=os.environ["AZURE_API_ENDPOINT"])
chat_llm = ChatLLMstudio(llm=llm, model = "gpt-4o-mini", parameters={"temperature":0})

print("\n\nresult:\n", assistant_new("Turn this into a party!",chat_llm),"\n")


print("###### vertex")
import os
from llmstudio.langchain import ChatLLMstudio
from llmstudio.providers import LLM

llm = LLM(provider="vertexai", 
          api_key=os.environ["GOOGLE_API_KEY"])
chat_llm = ChatLLMstudio(llm=llm, model = "gemini-1.5-pro-latest", parameters={"temperature":0})

from langchain.tools import tool

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

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate


def assistant_vertex(question: str) -> str:
    tools = [power_disco_ball, start_music, dim_lights]
    print(tools)

    # Rebuild agent with new tools
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(chat_llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    response = agent_executor.invoke({"input": question})

    return response


print("\n\nresult:\n", assistant_vertex("Turn this into a party!"),"\n")




print("###### bedrock anthropic")
import os
from llmstudio.langchain import ChatLLMstudio
from llmstudio.providers import LLM

llm = LLM(provider="bedrock", 
          region=os.environ["BEDROCK_REGION"],
          secret_key=os.environ["BEDROCK_SECRET_KEY"],
          access_key=os.environ["BEDROCK_ACCESS_KEY"])
chat_llm = ChatLLMstudio(llm=llm, model = "anthropic.claude-3-5-sonnet-20240620-v1:0", parameters={"temperature":0})

from langchain.tools import tool

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

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate


def assistant_bedrock_anthropic(question: str) -> str:
    tools = [power_disco_ball, start_music, dim_lights]
    print(tools)

    # Rebuild agent with new tools
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(chat_llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    response = agent_executor.invoke({"input": question})

    return response


print("\n\nresult:\n", assistant_bedrock_anthropic("Turn this into a party!"),"\n")


print("###### proxy")
from llmstudio.server import start_servers
start_servers()
from llmstudio_tracker.tracker import TrackingConfig

from llmstudio.providers import LLM
from llmstudio_proxy.provider import ProxyConfig

# from llmstudio_core.providers import LLMCore as LLM
# from llmstudio.providers import LLM

llm = LLM(provider="openai", 
          proxy_config=ProxyConfig(host="0.0.0.0", port="8001"),
          tracking_config=TrackingConfig(host="0.0.0.0", port="8002"),
          session_id="proxy-chat-model")
chat_llm = ChatLLMstudio(llm=llm, model = "gpt-4o-mini", parameters={"temperature":0})


from langchain.tools import tool

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

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate


def assistant_proxy(question: str) -> str:
    tools = [power_disco_ball, start_music, dim_lights]
    print(tools)

    # Rebuild agent with new tools
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(chat_llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    response = agent_executor.invoke({"input": question})

    return response


print("\n\nresult:\n", assistant_proxy("Turn this into a party!"),"\n")

llm = LLM(provider="openai", 
          proxy_config=ProxyConfig(host="0.0.0.0", port="8001"),
          tracking_config=TrackingConfig(host="0.0.0.0", port="8002"))
chat_llm = ChatLLMstudio(llm=llm, model = "gpt-4o-mini", parameters={"temperature":0})
print("\n\nresult:\n", assistant_proxy("Turn this into a party!"),"\n")
