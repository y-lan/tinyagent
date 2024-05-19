from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Type

from pydantic import BaseModel, ConfigDict, SerializeAsAny


class BaseConfig(BaseModel):
    model_name: str
    system_prompt: str = "You are a helpful assistant"

    # LLM
    temperature: float = 0
    top_p: float = 1
    seed: int = None
    max_tokens: int = 1024

    # system
    json_output: bool = False
    enable_history: bool = False
    max_retry: int = 16
    timeout: int = 60
    stream: bool = False
    enable_magic_placeholders: bool = True
    verbose: bool = False

    # tools
    execute_tools: bool = True
    use_tools: bool = True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",  # no extra fields allowed
        protected_namespaces=(),
    )


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class BaseContent(BaseModel, ABC):
    type: str


class TextContent(BaseContent):
    type: str = "text"
    text: str


class Message(BaseModel):
    role: Role
    content: Optional[list[SerializeAsAny[BaseContent]]] = None
    tool_calls: Optional[list] = None

    @staticmethod
    def from_text(role, text):
        return Message(role=role, content=[TextContent(text=text)])

    def get_text(self):
        if self.content:
            for content in self.content:
                if content.type == "text":
                    return content.text
        return ""

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)


class TokenUsage(BaseModel):
    prompt: int = 0
    completion: int = 0

    def update(self, usage=None, input: int = None, output: int = None):
        if usage:
            self.prompt += usage.prompt
            self.completion += usage.completion
        else:
            self.prompt += input
            self.completion += output


class ChatResponse(BaseModel):
    message: Message
    usage: TokenUsage
    model: str
    finish_reason: str
    input_message: Optional[Message] = None


class Event(Enum):
    NEW_CHAT_TOKEN = 1
    USE_TOOL = 2
    FINISH_CHAT = 3


class Tool(BaseModel, ABC):
    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None  # noqa: F821

    def run(self, *args, **kwargs):
        return str(self._run(*args, **kwargs))

    @abstractmethod
    def _run(self, *args, **kwargs):
        pass
