import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError


class OpenAIToolParameters(BaseModel):
    type: str
    properties: Dict
    required: List[str]


class OpenAIToolFunction(BaseModel):
    name: str
    description: str
    parameters: OpenAIToolParameters


class OpenAITool(BaseModel):
    type: str
    function: OpenAIToolFunction


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


class EngineConfig(BaseModel):
    providers: Dict[str, ProviderConfig]


def _load_providers_config() -> EngineConfig:
    # TODO read from github
    default_config_path = Path(os.path.join(os.path.dirname(__file__), "config.yaml"))
    local_config_path = Path(os.getcwd(), "config.yaml")

    def _merge_configs(config1, config2):
        for key in config2:
            if key in config1:
                if isinstance(config1[key], dict) and isinstance(config2[key], dict):
                    _merge_configs(config1[key], config2[key])
                elif isinstance(config1[key], list) and isinstance(config2[key], list):
                    config1[key].extend(config2[key])
                else:
                    config1[key] = config2[key]
            else:
                config1[key] = config2[key]
        return config1

    try:
        default_config_data = yaml.safe_load(default_config_path.read_text())
        local_config_data = (
            yaml.safe_load(local_config_path.read_text())
            if local_config_path.exists()
            else {}
        )
        config_data = _merge_configs(default_config_data, local_config_data)
        return EngineConfig(**config_data)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing YAML configuration: {e}")
    except ValidationError as e:
        raise RuntimeError(f"Error in configuration data: {e}")
