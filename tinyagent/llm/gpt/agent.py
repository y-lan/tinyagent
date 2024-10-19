"""
GPTAgent
"""

import json
from typing import List, Optional, Type

from pydantic import BaseModel, ConfigDict

from tinyagent.base import BaseAgent
from tinyagent.llm.gpt.client import OpenAIChatResponse, OpenAIClient
from tinyagent.schema import (
    BaseConfig,
    ChatResponse,
    Message,
    Role,
    TextContent,
    TokenUsage,
    Tool,
    ToolUseContent,
    ToolUseResultMessage,
)
from tinyagent.tools.tool import build_function_signature
from tinyagent.utils import get_param


def _parse_openai_message(msg):
    role = Role(msg["role"])
    content = msg["content"]

    if isinstance(content, str):
        return Message(
            role=role,
            content=[TextContent(text=content)],
        )

    if "tool_calls" in msg:
        message = Message(
            role=role,
            content=None,
            tool_calls=[ToolUseContent(**tool) for tool in msg["tool_calls"]],
        )
        return message

    raise ValueError(f"Unsupported content type: {msg}")


def _assemble_chat_response(raw: OpenAIChatResponse):
    return ChatResponse(
        message=_parse_openai_message(raw.choices[0]["message"]),
        model=raw.model,
        finish_reason=raw.choices[0]["finish_reason"],
        usage=TokenUsage(
            prompt=raw.usage["prompt_tokens"],
            completion=raw.usage["completion_tokens"],
        ),
    )


class AgentConfig(BaseConfig):
    """
    Configuration for the GPT agent.
    """

    model_name: str = "gpt-4o-mini-2024-07-18"
    frequency_penalty: float = 0
    use_azure: bool = False
    azure_endpoint: str = None
    response_format: Optional[Type[BaseModel]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    model_config = ConfigDict(
        extra="forbid",
    )


class GPTAgent(BaseAgent):
    """
    GPTAgent, an agent that uses the OpenAI GPT API to chat.
    """

    def __init__(
        self,
        config: AgentConfig = None,
        tools: list[Tool] = None,
        api_key=None,
        logger=None,
        **kwargs,
    ):
        if config is None:
            config = AgentConfig(**kwargs)

        super().__init__(config=config, tools=tools, logger=logger, **kwargs)

        self.client = OpenAIClient(api_key=api_key)

    def _handle_stream(self, stream):
        message = None
        choice_cache = None
        for event in stream:
            if self.config.verbose:
                self.logger.debug("Stream event: %s", event)

            if message is None:
                message = OpenAIChatResponse(
                    id=event.id,
                    object="chat.completion",
                    created=event.created,
                    model=event.model,
                    choices=[],
                    usage=None,
                    system_fingerprint=event.system_fingerprint,
                )

            if len(event.choices) > 0:
                choice = event.choices[0]
                if "delta" in choice:
                    if choice_cache is None:
                        choice_cache = choice
                        choice_cache["message"] = choice.pop("delta")
                        if choice_cache["message"].get("content"):
                            self.event_manager.publish_new_chat_token(
                                choice_cache["message"]["content"]
                            )
                    elif "content" in choice["delta"]:
                        if choice_cache["message"].get("content"):
                            choice_cache["message"]["content"] += choice["delta"][
                                "content"
                            ]
                        else:
                            choice_cache["message"]["content"] = choice["delta"][
                                "content"
                            ]
                        self.event_manager.publish_new_chat_token(
                            choice["delta"]["content"]
                        )
                    elif "tool_calls" in choice["delta"]:
                        if choice_cache["message"].get("tool_calls"):
                            for func in choice["delta"]["tool_calls"]:
                                idx = func["index"]
                                if idx < len(choice_cache["message"]["tool_calls"]):
                                    choice_cache["message"]["tool_calls"][idx][
                                        "function"
                                    ]["arguments"] += func["function"]["arguments"]
                                else:
                                    choice_cache["message"]["tool_calls"].append(func)
                        else:
                            choice_cache["message"]["tool_calls"] = choice["delta"][
                                "tool_calls"
                            ]
                if choice.get("finish_reason"):
                    choice_cache["finish_reason"] = choice["finish_reason"]

            if event.usage:
                message.usage = event.usage

        message.choices.append(choice_cache)

        return message

    def _client_chat(self, messages, **params):
        res = self.client.chat(messages, **params)

        message = None
        if params.get("stream"):
            message = self._handle_stream(res)
        else:
            message = res

        return _assemble_chat_response(message)

    def _process_chat(self, messages, **params):
        response = self._client_chat(messages, **params)

        tool_result_content = []
        if response.message.tool_calls:
            for content in response.message.tool_calls:
                if isinstance(content, ToolUseContent):
                    try:
                        args = json.loads(content.function.arguments)
                        tool_result = self.run_tool(content.function.name, args)
                    except json.JSONDecodeError:
                        self.logger.error(
                            f"Invalid JSON in tool arguments: {content.function.arguments}"
                        )
                        tool_result = f"Error: Invalid JSON in tool arguments: {content.function.arguments}"
                    tool_result_content.append(
                        {
                            "tool_call_id": content.id,
                            "name": content.function.name,
                            "content": tool_result,
                        }
                    )

            if tool_result_content:
                messages.append(response.message)
                for content in tool_result_content:
                    messages.append(
                        ToolUseResultMessage(
                            tool_call_id=content["tool_call_id"],
                            name=content["name"],
                            content=content["content"],
                        )
                    )
                return self._process_chat(messages, **params)

        return response

    def _chat(
        self,
        messages: List[Message],
        **kwargs,
    ):
        messages = self._assemble_request_messages(messages)

        params = {
            "model": self.config.model_name,
            "max_completion_tokens": get_param(
                kwargs, ["max_tokens", "max_completion_tokens"], self.config.max_tokens
            ),
            "temperature": get_param(kwargs, "temperature", self.config.temperature),
            "top_p": get_param(kwargs, "top_p", self.config.top_p),
            "frequency_penalty": self.config.frequency_penalty,
            "seed": self.config.seed,
            "stream": get_param(kwargs, "stream", self.config.stream),
        }

        if self.config.json_output:
            if self.config.response_format:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "math_response",
                        "strict": True,
                        "schema": self.config.response_format.model_json_schema(),
                    },
                }
            else:
                params["response_format"] = {"type": "json_object"}

        if params["stream"]:
            params["stream_options"] = {"include_usage": True}

        if self.config.use_tools and self.tools:
            tools = [build_function_signature(tool) for tool in self.tools.values()]
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = self._process_chat(messages, **params)

        response.input_message = (
            messages[1] if messages[0].role == Role.SYSTEM.value else messages[0]
        )

        return response
