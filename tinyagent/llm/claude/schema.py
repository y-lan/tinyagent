from tinyagent.schema import BaseContent, ImageContent
from tinyagent.utils import convert_image_to_base64_uri


class ClaudeImageContentSource(BaseContent):
    type: str
    media_type: str
    data: str


class ClaudeImageContent(BaseContent):
    type: str = "image"
    source: ClaudeImageContentSource

    @staticmethod
    def from_image_content(image_content: ImageContent):
        image_base64_url = convert_image_to_base64_uri(image_content.image_url.url)
        mime_type = image_base64_url.split(";")[0].split(":")[1]
        image_base64 = image_base64_url.split(",")[1]

        return ClaudeImageContent(
            source=ClaudeImageContentSource(
                type="base64",
                media_type=mime_type,
                data=image_base64,
            )
        )


class ToolUseContent(BaseContent):
    type: str = "tool_use"
    id: str
    name: str
    input: dict


class ToolResultContent(BaseContent):
    type: str = "tool_result"
    tool_use_id: str
    content: str
