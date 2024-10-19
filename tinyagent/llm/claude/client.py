import json
import os
import time
from pydantic import BaseModel
import requests

from tinyagent.schema import TokenUsage


class ChatStreamEvent(BaseModel):
    type: str
    payload: dict


class AnthropicClient:
    API_BASE_URL = "https://api.anthropic.com"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.token_usage = TokenUsage()

    def _handle_streaming_response(self, response):
        # https://docs.anthropic.com/en/api/messages-streaming#basic-streaming-request
        for line in response.iter_lines():
            if line:
                event_data = line.decode("utf-8")
                if event_data.startswith("data:"):
                    # Skip the "data:" prefix
                    data = json.loads(event_data[5:])
                    event = ChatStreamEvent(type=data.get("type"), payload=data)

                    yield event

                    if event.type == "message_stop":
                        break

    def api_request(self, endpoint: str, data: dict, retry=3) -> dict:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "anthropic-beta": "tools-2024-05-16",
        }

        response = requests.post(
            self.API_BASE_URL + endpoint,
            headers=headers,
            data=json.dumps(data),
        )

        if response.status_code == 200:
            if data.get("stream"):
                return self._handle_streaming_response(response)
            else:
                return response.json()
        else:
            if response.status_code == 429 and retry > 0:
                # rate limited, retry if possible
                time.sleep((4 - retry) * 10)
                return self.api_request(endpoint, data, retry - 1)
            else:
                raise Exception(
                    f"Anthropic API request failed with status code {response.status_code} {response.text}"
                )

    def chat(self, messages, **kwargs):
        data = {
            "messages": [msg.model_dump(exclude_none=True) for msg in messages],
            **kwargs,
        }
        return self.api_request(
            "/v1/messages",
            data,
        )
