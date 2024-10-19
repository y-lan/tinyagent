from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Type

from pydantic import BaseModel, ConfigDict, SerializeAsAny

from tinyagent.utils import convert_image_to_base64_uri


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


class ImageSource(BaseModel):
    # data:image/jpeg;base64,{base64_image}
    # https://example.com/image.png
    url: str


class ImageContent(BaseContent):
    type: str = "image_url"
    image_url: ImageSource


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

    index: int
    type: str = "function"
    id: str
    function: FunctionCall


class Message(BaseModel):
    role: Role
    content: Optional[list[SerializeAsAny[BaseContent]]] = None
    tool_calls: Optional[list] = None

    @staticmethod
    def from_text(role: Role, text: str):
        return Message(role=role, content=[TextContent(text=text)])

    @staticmethod
    def from_image(role: Role, image_url: str):
        image_base64 = convert_image_to_base64_uri(image_url)
        return Message(
            role=role, content=[ImageContent(image_url=ImageSource(url=image_base64))]
        )

    def get_text(self):
        if self.content:
            for content in self.content:
                if content.type == "text":
                    return content.text
        return ""

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)


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
