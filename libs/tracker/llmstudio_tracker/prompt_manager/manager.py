import json

import requests
from llmstudio_tracker.prompt_manager.schemas import PromptDefault
from llmstudio_tracker.tracker import TrackingConfig


class PromptManager:
    def __init__(self, tracking_config: TrackingConfig):
        self.tracking_url = tracking_config.url
        self._session = requests.Session()

    def add_prompt(self, prompt: PromptDefault):
        req = self._session.post(
            f"{self.tracking_url}/api/tracking/add/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=prompt.model_dump_json(),
            timeout=100,
        )
        return req

    def delete_prompt(self, prompt: PromptDefault):
        req = self._session.delete(
            f"{self.tracking_url}/api/tracking/delete/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=prompt.model_dump_json(),
            timeout=100,
        )
        return req

    def update_prompt(self, prompt: PromptDefault):
        req = self._session.patch(
            f"{self.tracking_url}/api/tracking/update/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=prompt.model_dump_json(),
            timeout=100,
        )
        return req

    def get_prompt(
        self,
        prompt_id: str = None,
        name: str = None,
        model: str = None,
        provider: str = None,
    ):

        data = {
            "prompt_id": prompt_id,
            "name": name,
            "model": model,
            "provider": provider,
        }

        req = self._session.get(
            f"{self.tracking_url}/api/tracking/get/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
            data=json.dumps(data),
        )
        return req
