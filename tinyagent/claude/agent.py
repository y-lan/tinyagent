import base64
import imghdr
import json
import mimetypes
import os
from typing import Any, Union
import urllib

from tinyagent.base import BaseAgent
from tinyagent.claude.client import AnthropicClient
from tinyagent.tools import build_function_signature
from tinyagent.utils import get_param, replace_magic_placeholders
from tinyagent.schema import (
    BaseConfig,
    BaseContent,
    ChatResponse,
    Message,
    Role,
    TextContent,
    TokenUsage,
)


def _create_image_block(data: str, media_type: str = "image/jpeg"):
    return ImageContent(source=dict(type="base64", media_type=media_type, data=data))


def _create_image_content(image_path):
    if image_path.startswith("data:image/jpeg;base64,"):
        return _create_image_block(image_path.split(",")[1])

    parsed = urllib.parse.urlparse(image_path)

    if parsed.scheme in ("http", "https"):
        mime_type, _ = mimetypes.guess_type(image_path)
        with urllib.request.urlopen(image_path) as response:
            base64_image = base64.b64encode(response.read()).decode("utf-8")
            return _create_image_block(base64_image, mime_type or "image/jpeg")

    elif os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            image_type = imghdr.what(image_path)
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            return _create_image_block(base64_image, f"image/{image_type}")

    raise Exception(f"Invalid image path: {image_path}")


def _assemble_chat_response(raw: dict):
    return ChatResponse(
        message=Message(
            role=Role.ASSISTANT,
            content=[_parse_claude_content(content) for content in raw["content"]],
        ),
        model=raw.get("model"),
        finish_reason="completed",
        usage=TokenUsage(
            prompt=raw["usage"]["input_tokens"],
            completion=raw["usage"]["output_tokens"],
        ),
    )


def _parse_claude_content(content):
    if content["type"] == "text":
        return TextContent(**content)
    elif content["type"] == "image":
        return ImageContent(**content)
    elif content["type"] == "tool_use":
        return ToolUseContent(**content)
    else:
        raise Exception(f"Unsupported content type: {content['type']}")


class ImageContentSource(BaseContent):
    type: str
    media_type: str
    data: str


class ImageContent(BaseContent):
    type: str = "image"
    source: ImageContentSource


class ToolUseContent(BaseContent):
    type: str = "tool_use"
    id: str
    name: str
    input: dict


class ToolResultContent(BaseContent):
    type: str = "tool_result"
    tool_use_id: str
    content: str


class AgentConfig(BaseConfig):
    model_name: str = "claude-3-haiku-20240307"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ClaudeAgent(BaseAgent):
    def __init__(
        self,
        config: AgentConfig = None,
        tools: list[Any] = None,
        api_key=None,
        logger=None,
        **kwargs,
    ):
        if config is None:
            config = AgentConfig(**kwargs)

        super().__init__(config=config, tools=tools, logger=logger, **kwargs)

        self.client = AnthropicClient(api_key=api_key)

    def _get_system_msg(self):
        if self.config.enable_magic_placeholders:
            return replace_magic_placeholders(self.config.system_prompt)
        else:
            return self.config.system_prompt

    def _assemble_request_messages(self, user_contents):
        msgs = []
        if self.config.enable_history:
            msgs.extend(self.history)

        msgs.append(Message(role=Role.USER, content=user_contents))

        return msgs

    def _handle_stream(self, stream):
        message = None
        content_block_cache = None
        for event in stream:
            if self.config.verbose:
                self.logger.debug(f"Stream event: {event}")
            match event.type:
                case "message_start":
                    message = dict(
                        id=event.payload["message"]["id"],
                        model=event.payload["message"]["model"],
                        role=event.payload["message"]["role"],
                        type="message",
                        content=[],
                        usage=event.payload["message"]["usage"],
                    )
                case "content_block_start":
                    message["content"].append(event.payload["content_block"])
                case "content_block_delta":
                    match event.payload["delta"].get("type"):
                        case "text_delta":
                            if content_block_cache is None:
                                content_block_cache = event.payload["delta"]
                            else:
                                content_block_cache["text"] += event.payload["delta"][
                                    "text"
                                ]
                            self.event_manager.publish_new_chat_token(
                                event.payload["delta"]["text"]
                            )
                        case "input_json_delta":
                            if content_block_cache is None:
                                content_block_cache = event.payload["delta"]
                            else:
                                content_block_cache["partial_json"] += event.payload[
                                    "delta"
                                ]["partial_json"]
                case "content_block_stop":
                    match content_block_cache.get("type"):
                        case "input_json_delta":
                            message["content"][-1]["input"] = json.loads(
                                content_block_cache["partial_json"]
                            )
                            content_block_cache = None
                        case "text_delta":
                            message["content"][-1]["text"] = content_block_cache["text"]
                            content_block_cache = None
                        case _:
                            raise Exception(
                                f"Unsupported content block type: {content_block_cache['type']}"
                            )
                case "message_delta":
                    message["usage"]["output_tokens"] = event.payload["usage"][
                        "output_tokens"
                    ]
                case "message_stop":
                    pass

        return message

    def _client_chat(self, messages, **params):
        res = self.client.chat(messages, **params)

        message = None
        if params.get("stream"):
            message = self._handle_stream(res)
        else:
            message = res

        return _assemble_chat_response(message)

    def _chat(
        self,
        user_input,
        image=None,
        return_complex=False,
        **kwargs,
    ) -> Union[str, ChatResponse]:
        user_contents = [TextContent(text=user_input)]
        if image is not None:
            user_contents.append(_create_image_content(image))

        messages = self._assemble_request_messages(user_contents)

        params = dict(
            model=self.config.model_name,
            max_tokens=get_param(kwargs, "max_tokens", self.config.max_tokens),
            temperature=get_param(kwargs, "temperature", self.config.temperature),
            system=self._get_system_msg(),
            stream=get_param(kwargs, "stream", self.config.stream),
        )

        if self.config.use_tools and self.tools:
            tools = []
            for _, tool in self.tools.items():
                signature = build_function_signature(tool)["function"]
                signature["input_schema"] = signature.pop("parameters")
                tools.append(signature)

            params["tools"] = tools

        response = self._client_chat(messages, **params)

        tool_result_content = []
        for i, content in enumerate(response.message.content):
            if isinstance(content, ToolUseContent):
                tool = self.tools[content.name]
                self.event_manager.publish_use_tool(content.name, content.input)
                self.logger.debug(
                    f"Using tool: {content.name} with input: {content.input}"
                )
                tool_response = tool.run(**content.input)
                tool_result_content.append(
                    ToolResultContent(tool_use_id=content.id, content=tool_response)
                )

        if tool_result_content:
            messages.append(response.message)
            messages.append(Message(role=Role.USER, content=tool_result_content))
            response = self._client_chat(messages, **params)

        response.input_message = messages[0]
        self.event_manager.publish_finish_chat(response)

        return response if return_complex else response.message.content[0].text
