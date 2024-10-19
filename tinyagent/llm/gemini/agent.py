import json
from typing import List, Optional, Union

from tinyagent import BaseAgent
from tinyagent.llm.claude.schema import ToolResultContent
from tinyagent.llm.gemini.client import GeminiClient
from tinyagent.llm.gemini.schema import GeminiResponse, GeminiTextPart, GeminiToolMode
from tinyagent.schema import (
    BaseConfig,
    ChatResponse,
    Message,
    Role,
    TokenUsage,
    Tool,
    ToolUseContent,
    ToolUseResultMessage,
)
from tinyagent.utils import get_param


def _assemble_chat_response(raw: GeminiResponse, **kwargs):
    return ChatResponse(
        message=raw.candidates[0].content.to_standard_message(),
        model=kwargs.get("model"),
        finish_reason=raw.candidates[0].finish_reason or "STOP",
        usage=TokenUsage(
            prompt=raw.usage_metadata.prompt_token_count,
            completion=raw.usage_metadata.candidates_token_count,
        ),
    )


class AgentConfig(BaseConfig):
    model_name: str = "gemini-1.5-flash-latest"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GeminiAgent(BaseAgent):
    client: GeminiClient

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
        self.client = GeminiClient(api_key=api_key)

    def _handle_stream(self, stream):
        response = None
        for event in stream:
            if self.config.verbose:
                self.logger.debug(f"Stream event: {event}")

            if event.candidates[0].content.parts and isinstance(
                event.candidates[0].content.parts[0], GeminiTextPart
            ):
                self.event_manager.publish_new_chat_token(
                    event.candidates[0].content.parts[0].text
                )

            if response is None:
                response = event
            else:
                response.update(event)
        return response

    def _process_chat(self, messages, **params):
        response = self.client.chat(messages, **params)

        message = None
        if params.get("stream"):
            message = self._handle_stream(response)
        else:
            message = response

        response = _assemble_chat_response(message, **params)

        tool_result_content = []
        for i, content in enumerate(response.message.content):
            if isinstance(content, ToolUseContent):
                args = json.loads(content.function.arguments)
                tool_result = self.run_tool(content.function.name, args)
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
            if params.get("tool_mode"):
                params.pop("tool_mode")
            return self._process_chat(messages, **params)

        return response

    def _chat(
        self,
        messages: List[Message],
        tool_mode: Optional[GeminiToolMode] = None,
        **kwargs,
    ) -> Union[str, ChatResponse]:
        messages = self._assemble_request_messages(messages, include_system=False)

        params = {
            "model": self.config.model_name,
            "stream": get_param(kwargs, "stream", self.config.stream),
            "json_output": get_param(kwargs, "json_output", self.config.json_output),
            "temperature": get_param(kwargs, "temperature", self.config.temperature),
            "max_output_tokens": get_param(
                kwargs, ["max_output_tokens", "max_tokens"], self.config.max_tokens
            ),
        }
        if tool_mode:
            params["tool_mode"] = tool_mode

        system_msg = self._get_system_msg()
        if system_msg:
            params["system_instruction"] = system_msg

        if self.config.use_tools and self.tools:
            params["tools"] = list(self.tools.values())

        res = self._process_chat(messages, **params)
        res.input_message = messages[0]
        return res
