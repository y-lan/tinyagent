from pydantic import BaseModel, ConfigDict
from tinyagent.schema import BaseContent, Message


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

    index: int
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