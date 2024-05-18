import json
import os
import time
from typing import Optional
from pydantic import BaseModel

import requests



class OpenAIChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[dict]
    usage: Optional[dict] = None
    system_fingerprint: Optional[str] = None


class OpenAIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.api_base_url = "https://api.openai.com"

    def _handle_streaming_response(self, response):
        # https://platform.openai.com/docs/api-reference/chat/create?lang=curl
        for line in response.iter_lines():
            if line:
                event_data = line.decode("utf-8")
                if event_data.startswith("data:"):
                    # Skip the "data:" prefix
                    event_data = event_data[5:].strip()

                    if event_data == "[DONE]":
                        break

                    data = json.loads(event_data)
                    event = OpenAIChatResponse(**data)

                    yield event

    def api_request(self, endpoint: str, data: dict, retry=3) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

        response = requests.post(
            self.api_base_url + endpoint,
            headers=headers,
            data=json.dumps(data),
        )

        if response.status_code == 200:
            if data.get("stream"):
                return self._handle_streaming_response(response)
            else:
                return OpenAIChatResponse(**response.json())
        else:
            if response.status_code == 429 and retry > 0:
                # rate limited, retry if possible
                time.sleep((4 - retry) * 10)
                return self.api_request(endpoint, data, retry - 1)
            else:
                raise Exception(
                    f"OpenAI API request failed with status code {response.status_code} {response.text}"
                )

    def chat(self, messages, **kwargs) -> dict:
        data = {
            "messages": [msg.model_dump(exclude_none=True) for msg in messages],
            **kwargs,
        }
        return self.api_request(
            "/v1/chat/completions",
            data,
        )
