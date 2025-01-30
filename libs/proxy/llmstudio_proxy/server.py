import json
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Union

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llmstudio_core.providers import _load_providers_config
from llmstudio_core.providers.provider import provider_registry
from llmstudio_proxy.config import ENGINE_HOST, ENGINE_PORT
from llmstudio_proxy.utils import get_current_version
from pydantic import BaseModel, Field

ENGINE_HEALTH_ENDPOINT = "/health"
ENGINE_TITLE = "LLMstudio Proxy API"
ENGINE_DESCRIPTION = "The core API for LLM interactions"
ENGINE_VERSION = get_current_version()
ENGINE_BASE_ENDPOINT = "/api/engine"

_proxy_server_started = False


class CostRange(BaseModel):
    range: List[Optional[int]]
    cost: float


class ModelConfig(BaseModel):
    mode: str
    max_tokens: Optional[int] = Field(default=None, alias="max_completion_tokens")
    max_completion_tokens: Optional[int] = None
    input_token_cost: Union[float, List["CostRange"]]
    cached_token_cost: Optional[Union[float, List["CostRange"]]] = None
    output_token_cost: Union[float, List["CostRange"]]


class ProviderConfig(BaseModel):
    id: str
    name: str
    chat: bool
    embed: bool
    keys: Optional[List[str]] = None
    models: Optional[Dict[str, ModelConfig]] = None
    parameters: Optional[Dict[str, Any]] = None


class ProxyConfig(BaseModel):
    providers: Dict[str, ProviderConfig]


def create_proxy_app(
    started_event: Event, config: ProxyConfig = _load_providers_config()
) -> FastAPI:
    app = FastAPI(
        title=ENGINE_TITLE,
        description=ENGINE_DESCRIPTION,
        version=ENGINE_VERSION,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get(ENGINE_HEALTH_ENDPOINT)
    def health_check():
        """Health check endpoint to ensure the API is running."""
        return {"status": "healthy", "message": "Proxy is up and running"}

    @app.get(f"{ENGINE_BASE_ENDPOINT}/providers")
    def get_providers():
        """Return all providers supported."""
        return list(config.providers.keys())

    @app.get(f"{ENGINE_BASE_ENDPOINT}/models")
    def get_models(provider: Optional[str] = None):
        """Return all models supported with the provider as a key."""
        all_models = {}
        for provider_name, provider_config in config.providers.items():
            if provider and provider_name != provider:
                continue
            if provider_config.models:
                all_models[provider_name] = {}
                all_models[provider_name]["name"] = provider_config.name
                all_models[provider_name]["models"] = list(
                    provider_config.models.keys()
                )
        return all_models[provider] if provider else all_models

    def create_chat_handler(provider_config):
        async def chat_handler(request: Request):
            """Endpoint for chat functionality."""
            provider_class = provider_registry.get(f"{provider_config.name}".lower())
            provider_instance = provider_class(provider_config)
            request_dict = await request.json()

            result = await provider_instance.achat(**request_dict)
            if request_dict.get("is_stream", False):

                async def result_generator():
                    async for chunk in result:
                        yield json.dumps(chunk.dict())

                return StreamingResponse(
                    result_generator(), media_type="application/json"
                )
            return result

        return chat_handler

    for provider_name, provider_config in config.providers.items():
        if provider_config.chat:
            app.post(f"{ENGINE_BASE_ENDPOINT}/chat/{provider_name}")(
                create_chat_handler(provider_config)
            )

    @app.get(f"{ENGINE_BASE_ENDPOINT}/parameters")
    def get_parameters(provider: str, model: Optional[str] = None):
        """Return parameters for a given provider and model in an array JSON format."""
        provider_config = config.providers.get(provider)
        if not provider_config:
            return {"error": f"Provider {provider} not found"}, 404
        parameters = provider_config.parameters
        parameters_array = [{"id": key, **value} for key, value in parameters.items()]
        return parameters_array

    @app.post("/api/export")
    async def export(request: Request):
        data = await request.json()
        csv_content = ""

        if len(data) > 0:
            csv_content += ";".join(data[0].keys()) + "\n"
            for execution in data:
                csv_content += (
                    ";".join([json.dumps(value) for value in execution.values()]) + "\n"
                )

        headers = {"Content-Disposition": "attachment; filename=parameters.csv"}
        return StreamingResponse(
            iter([csv_content]), media_type="text/csv", headers=headers
        )

    @app.on_event("startup")
    async def startup_event():
        started_event.set()
        print(f"Running LLMstudio Proxy on http://{ENGINE_HOST}:{ENGINE_PORT} ")

    return app


def run_proxy_app(started_event: Event):
    try:
        proxy = create_proxy_app(started_event)
        uvicorn.run(
            proxy,
            host=ENGINE_HOST,
            port=ENGINE_PORT,
            log_level="warning",
        )
    except Exception as e:
        print(f"Error running LLMstudio Proxy: {e}")


def is_server_running(url, path="/health"):
    try:
        response = requests.get(f"{url}{path}")
        if response.status_code == 200 and response.json().get("status") == "healthy":
            return True
    except requests.ConnectionError:
        pass
    return False


def start_server_component(host, port, run_func, server_name):
    if not is_server_running(url=f"http://{host}:{port}"):
        started_event = Event()
        thread = Thread(target=run_func, daemon=True, args=(started_event,))
        thread.start()
        started_event.wait()  # wait for startup, this assumes the event is set somewhere
        return thread
    else:
        print(f"{server_name} server already running on {host}:{port}")
        return None


def setup_engine_server():
    global _proxy_server_started
    proxy_thread = None
    if not _proxy_server_started:
        proxy_thread = start_server_component(
            ENGINE_HOST, ENGINE_PORT, run_proxy_app, "Proxy"
        )
        _proxy_server_started = True
    return proxy_thread
