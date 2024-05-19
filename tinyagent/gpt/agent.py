"""
GPTAgent
"""
import base64
import json
import os
import urllib

from pydantic import BaseModel, ConfigDict

from tinyagent.base import BaseAgent
from tinyagent.gpt.client import OpenAIChatResponse, OpenAIClient
from tinyagent.schema import (
    BaseConfig,
    BaseContent,
    ChatResponse,
    Message,
    Role,
    TextContent,
    TokenUsage,
    Tool,
)
from tinyagent.tools import build_function_signature
from tinyagent.utils import get_param, replace_magic_placeholders


def _create_image_content(image_path):
    parsed = urllib.parse.urlparse(image_path)

    if parsed.scheme in ("http", "https"):
        return ImageContent(image_url={"url": image_path})

    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            return ImageContent(
                image_url={"url": f"data:image/jpeg;base64,{base64_image}"}
            )
    elif image_path.startswith("data:image/jpeg;base64,"):
        return ImageContent(image_url={"url": image_path})
    else:
        raise ValueError(f"Invalid image path: {image_path}")


def _parse_openai_message(msg):
    role = Role(msg["role"])
    content = msg["content"]

    if isinstance(content, str):
        return Message(
            role=role,
            content=[TextContent(text=content)],
        )

    if "tool_calls" in msg:
        return Message(
            role=role,
            content=None,
            tool_calls=[ToolUseContent(**tool) for tool in msg["tool_calls"]],
        )

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


class ImageContent(BaseContent):
    """
    Represents an image content block.
    image_url is a dictionary with a single key "url" that contains the URL of the image.
    """

    type: str = "image_url"
    image_url: dict


class FunctionCall(BaseModel):
    """
    Represents a function call in the tool use message.
    arguments is a JSON string representing the dictionary of arguments.
    """

    name: str
    arguments: str


class ToolUseContent(BaseContent):
    """
    Represents a tool use content block.
    """

    type: str = "function"
    id: str
    function: FunctionCall


class ToolUseMessage(Message):
    """
    Represents a tool use message.
    tool_call_id is the ID of the tool call.
    name is the name of the tool.
    content is the result of the tool.
    """

    model_config = ConfigDict(use_enum_values=True)
    tool_call_id: str
    name: str
    content: str


class AgentConfig(BaseConfig):
    """
    Configuration for the GPT agent.
    """

    model_name: str = "gpt-3.5-turbo-0125"
    frequency_penalty: float = 0
    use_azure: bool = False
    azure_endpoint: str = None

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

    def _get_system_msg(self):
        if self.config.enable_magic_placeholders:
            return Message.from_text(
                Role.SYSTEM, replace_magic_placeholders(self.config.system_prompt)
            )
        return Message.from_text(Role.SYSTEM, self.config.system_prompt)

    def _assemble_request_messages(self, user_contents):
        msgs = [self._get_system_msg()]
        if self.config.enable_history:
            msgs.extend(self.history)

        msgs.append(Message(role=Role.USER, content=user_contents))
        return msgs

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
                        if choice_cache["message"]["content"]:
                            self.event_manager.publish_new_chat_token(
                                choice_cache["message"]["content"]
                            )
                    elif "content" in choice["delta"]:
                        choice_cache["message"]["content"] += choice["delta"]["content"]
                        self.event_manager.publish_new_chat_token(
                            choice["delta"]["content"]
                        )
                    elif "tool_calls" in choice["delta"]:
                        for i, func in enumerate(choice["delta"]["tool_calls"]):
                            choice_cache["message"]["tool_calls"][i]["function"][
                                "arguments"
                            ] += func["function"]["arguments"]
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

    def _chat(
        self,
        user_input,
        image=None,
        return_complex=False,
        **kwargs,
    ):
        user_contents = [TextContent(text=user_input)]
        if image is not None:
            user_contents.append(_create_image_content(image))

        messages = self._assemble_request_messages(user_contents)

        params = {
            "model": self.config.model_name,
            "max_tokens": get_param(kwargs, "max_tokens", self.config.max_tokens),
            "temperature": get_param(kwargs, "temperature", self.config.temperature),
            "frequency_penalty": self.config.frequency_penalty,
            "top_p": self.config.top_p,
            "seed": self.config.seed,
            "stream": get_param(kwargs, "stream", self.config.stream),
        }

        if self.config.json_output:
            params["response_format"] = {"type": "json_object"}

        if params["stream"]:
            params["stream_options"] = {"include_usage": True}

        if self.config.use_tools and self.tools:
            tools = [build_function_signature(tool) for tool in self.tools.values()]
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = self._client_chat(messages, **params)

        tool_result_content = []
        if response.message.tool_calls:
            for content in response.message.tool_calls:
                if isinstance(content, ToolUseContent):
                    tool = self.tools[content.function.name]
                    tool_result = tool.run(**json.loads(content.function.arguments))
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
                        ToolUseMessage(
                            role=Role.TOOL,
                            tool_call_id=content["tool_call_id"],
                            name=content["name"],
                            content=content["content"],
                        )
                    )
                response = self._client_chat(messages, **params)

        response.input_message = (
            messages[1] if messages[0].role == Role.SYSTEM else messages[0]
        )
        self.event_manager.publish_finish_chat(response)

        return response if return_complex else response.message.content[0].text
