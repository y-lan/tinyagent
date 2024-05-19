from abc import ABC, abstractmethod
import logging
from typing import Any, Optional, Union

from tinyagent.schema import (
    BaseConfig,
    ChatResponse,
    Event,
    Message,
    Role,
    TokenUsage,
)
from tinyagent.common import SimpleEventManager


class LLMEventManager(SimpleEventManager):
    def on_new_chat_token(self, callback):
        self.subscribe(Event.NEW_CHAT_TOKEN, lambda data: callback(data["content"]))

    def publish_new_chat_token(self, content: str):
        self.publish(Event.NEW_CHAT_TOKEN, content=content)

    def on_finish_chat(self, callback):
        self.subscribe(
            Event.FINISH_CHAT,
            lambda data: callback(data["response"]),
        )

    def publish_finish_chat(self, repsponse: ChatResponse):
        self.publish(Event.FINISH_CHAT, response=repsponse)

    def publish_use_tool(self, tool_name: str, args: dict):
        self.publish(Event.USE_TOOL, tool_name=tool_name, args=args)


class BaseAgent(ABC):
    def __init__(self, config: BaseConfig, tools: list[Any], logger=None, **kwargs):
        self.config = config

        if isinstance(tools, list):
            tools = {tool.name: tool for tool in tools}
        elif not isinstance(tools, dict):
            tools = {}
        self.tools = tools

        self.token_usage = TokenUsage()
        self.history = []
        self.event_manager = LLMEventManager()

        if logger is None:
            logger = logging.getLogger(__name__)
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        self.logger = logger
        if config.verbose:
            self.logger.setLevel(logging.DEBUG)

        def on_finish_chat(response: ChatResponse):
            self.token_usage.update(response.usage)
            self.history.append(response.input_message)
            self.history.append(response.message)

        self.on_finish_chat(on_finish_chat)

    def get_token_usage(self) -> TokenUsage:
        return self.token_usage

    def on_new_chat_token(self, callback):
        if not self.config.stream:
            self.logger.warning(
                "Event.NEW_CHAT_TOKEN is only available when stream=True"
            )
        self.event_manager.on_new_chat_token(callback)

    def on_finish_chat(self, callback):
        self.event_manager.on_finish_chat(callback)

    def _add_history(self, role, content):
        if not self.config.enable_history:
            return

        if isinstance(content, str):
            message = Message.from_text(role, content)
        elif isinstance(content, list):
            message = Message(role, content)
        self.history.append(message)

    def add_ai_history(self, history_message):
        self._add_history(Role.ASSISTANT, history_message)

    def add_user_history(self, history_message):
        self._add_history(Role.USER, history_message)

    @abstractmethod
    def _assemble_request_messages(self, user_contents):
        pass

    def chat(
        self,
        user_input: str,
        image: Optional[str] = None,
        return_complex: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> Union[str, ChatResponse]:
        if kwargs:
            user_input = user_input.format(**kwargs)

        return self._chat(
            user_input=user_input,
            image=image,
            return_complex=return_complex,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

    @abstractmethod
    def _chat(
        self,
        user_input: str,
        image: Optional[str] = None,
        return_complex: bool = False,
        **kwargs,
    ) -> Union[str, ChatResponse]:
        pass
