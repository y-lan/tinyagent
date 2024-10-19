import json
from typing import Any, List, Union

from tinyagent.base import BaseAgent
from tinyagent.llm.claude.client import AnthropicClient
from tinyagent.llm.claude.schema import (
    ClaudeImageContent,
    ToolResultContent,
    ToolUseContent,
)
from tinyagent.tools.tool import build_function_signature
from tinyagent.utils import (
    convert_image_to_base64_uri,
    get_param,
    replace_magic_placeholders,
)
from tinyagent.schema import (
    BaseConfig,
    ChatResponse,
    ImageContent,
    Message,
    Role,
    TextContent,
    TokenUsage,
)


def _convert_message_to_claude_format(message: Message):
    for i in range(len(message.content)):
        if isinstance(message.content[i], ImageContent):
            message.content[i] = ClaudeImageContent.from_image_content(
                message.content[i]
            )
    return message


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


class AgentConfig(BaseConfig):
    model_name: str = "claude-3-haiku-20240307"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ClaudeAgent(BaseAgent):
    client: AnthropicClient

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

    def _assemble_request_messages(self, messages: List[Message]):
        msgs = []
        if self.config.enable_history:
            msgs.extend(self.history)
        msgs.extend(messages)
        return msgs

    def _handle_stream(self, stream, prefill_response=None):
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
                                # hack for prefill_response, attach to the first content block
                                if prefill_response:
                                    content_block_cache["text"] = (
                                        prefill_response + content_block_cache["text"]
                                    )
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

    def _client_chat(self, messages, prefill_response=None, **params):
        res = self.client.chat(messages, **params)

        message = None
        if params.get("stream"):
            message = self._handle_stream(res, prefill_response)
        else:
            message = res
            if prefill_response:
                message["content"][0]["text"] = (
                    prefill_response + message["content"][0]["text"]
                )

        response = _assemble_chat_response(message)

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
            return self._client_chat(messages, **params)

        return response

    def _chat(
        self,
        messages: List[Message],
        return_complex=False,
        prefill_response=None,
        **kwargs,
    ) -> Union[str, ChatResponse]:
        messages = self._assemble_request_messages(messages)

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

        if (
            prefill_response is None
            and self.config.json_output
            and not params.get("tools")
        ):
            prefill_response = "{"

        if prefill_response:
            messages.append(
                Message(
                    role=Role.ASSISTANT, content=[TextContent(text=prefill_response)]
                )
            )

        messages = [_convert_message_to_claude_format(message) for message in messages]
        response = self._client_chat(
            messages, prefill_response=prefill_response, **params
        )

        response.input_message = messages[0]
        self.event_manager.publish_finish_chat(response)

        return response if return_complex else response.message.content[0].text
