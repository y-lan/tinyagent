from abc import ABC, abstractmethod
from enum import Enum
import json
from typing import Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field

from tinyagent.schema import (
    AudioContent,
    AudioSource,
    BaseContent,
    FunctionCall,
    ImageContent,
    ImageSource,
    Message,
    Role,
    TextContent,
    ToolUseContent,
    ToolUseResultMessage,
)


class GeminiRole(Enum):
    USER = "user"
    ASSISTANT = "model"
    FUNCTION = "function"

    @staticmethod
    def from_role(role: Role | str) -> "GeminiRole":
        if isinstance(role, str):
            role = Role(role)

        if role == Role.USER:
            return GeminiRole.USER
        elif role == Role.ASSISTANT:
            return GeminiRole.ASSISTANT
        elif role == Role.TOOL:
            return GeminiRole.FUNCTION
        else:
            raise ValueError(f"Unsupported role: {role}")

    def to_role(self) -> Role:
        if self == GeminiRole.USER:
            return Role.USER
        elif self == GeminiRole.FUNCTION:
            return Role.TOOL
        else:
            return Role.ASSISTANT


class GeminiContentPart(BaseModel, ABC):
    @abstractmethod
    def to_standard_content(self) -> BaseContent:
        raise NotImplementedError


class GeminiTextPart(GeminiContentPart):
    text: str

    def update(self, another: "GeminiTextPart"):
        self.text += another.text

    def to_standard_content(self) -> TextContent:
        return TextContent(text=self.text)


class GeminiInlineData(BaseModel):
    data: str
    """
    One of the following MIME types:
    - image/png
    - image/jpeg
    - image/webp
    - image/heic
    - image/heif
    """
    mime_type: str = Field(alias="mimeType")
    model_config = ConfigDict(populate_by_name=True)


class GeminiFileData(BaseModel):
    file_uri: str = Field(alias="fileUri")
    mime_type: str = Field(alias="mimeType")
    model_config = ConfigDict(populate_by_name=True)


class GeminiFileDataPart(GeminiContentPart):
    file_data: GeminiFileData

    def update(self, another: "GeminiFileDataPart"):
        raise NotImplementedError

    def to_standard_content(self) -> AudioContent:
        return AudioContent(
            input_audio=AudioSource(
                data=self.file_data.file_uri,
                format=self.file_data.mime_type,
            )
        )


class GeminiInlinePart(GeminiContentPart):
    inlineData: GeminiInlineData

    def update(self, another: "GeminiInlinePart"):
        raise NotImplementedError

    def to_standard_content(self) -> ImageContent:
        return ImageContent(image_url=ImageSource(url=self.inlineData.data))


class GeminiToolCall(BaseModel):
    name: str
    args: dict


class GeminiToolCallPart(GeminiContentPart):
    function_call: GeminiToolCall = Field(alias="functionCall")

    def update(self, another: "GeminiToolCallPart"):
        raise NotImplementedError

    @staticmethod
    def from_tool_use_content(content: ToolUseContent):
        return GeminiToolCallPart(
            functionCall=GeminiToolCall(
                name=content.function.name,
                args=json.loads(content.function.arguments),
            )
        )

    def to_standard_content(self) -> ToolUseContent:
        return ToolUseContent(
            index=-1,
            id=str(uuid.uuid4()),
            function=FunctionCall(
                name=self.function_call.name,
                arguments=json.dumps(self.function_call.args),
            ),
        )


class GeminiToolResponseContent(BaseModel):
    name: str
    content: any

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GeminiToolResponse(BaseModel):
    name: str
    response: GeminiToolResponseContent


class GeminiToolResponsePart(GeminiContentPart):
    function_response: GeminiToolResponse

    def to_standard_content(self) -> ToolUseResultMessage:
        raise NotImplementedError


class GeminiContent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    parts: Optional[
        list[
            GeminiTextPart
            | GeminiInlinePart
            | GeminiFileDataPart
            | GeminiToolCallPart
            | GeminiToolResponsePart
        ]
    ] = None
    role: Optional[GeminiRole] = None

    def update(self, another: "GeminiContent"):
        if another.parts:
            idx_to_extend = 0
            if isinstance(another.parts[0], type(self.parts[-1])):
                self.parts[-1].update(another.parts[0])
                idx_to_extend = 1

            self.parts.extend(another.parts[idx_to_extend:])

    def to_standard_message(self) -> Message:
        content = [part.to_standard_content() for part in self.parts]
        return Message(role=GeminiRole(self.role).to_role(), content=content)

    @staticmethod
    def from_tool_use_message(message: ToolUseResultMessage):
        return GeminiContent(
            parts=[
                GeminiToolResponsePart(
                    function_response=GeminiToolResponse(
                        name=message.name,
                        response=GeminiToolResponseContent(
                            name=message.name, content=message.content
                        ),
                    )
                )
            ],
            role=GeminiRole.FUNCTION,
        )


class GeminiToolMode(Enum):
    NONE = "NONE"
    ANY = "ANY"
    AUTO = "AUTO"


class GeminiResponseCandidate(BaseModel):
    content: GeminiContent
    # STOP, etc
    # https://cloud.google.com/vertex-ai/generative-ai/docs/reference/python/latest/vertexai.preview.generative_models.FinishReason
    finish_reason: Optional[str] = Field(alias="finishReason", default=None)
    safety_ratings: Optional[list[dict]] = Field(alias="safetyRatings", default=None)
    model_config = ConfigDict(populate_by_name=True)

    def update(self, another: "GeminiResponseCandidate"):
        if another.finish_reason:
            self.finish_reason = another.finish_reason
        if another.safety_ratings:
            self.safety_ratings = another.safety_ratings
        if another.content:
            if self.content:
                self.content.update(another.content)
            else:
                self.content = another.content


class GeminiUsage(BaseModel):
    prompt_token_count: int = Field(alias="promptTokenCount")
    candidates_token_count: int = Field(alias="candidatesTokenCount")
    total_token_count: int = Field(alias="totalTokenCount")
    model_config = ConfigDict(populate_by_name=True)


class GeminiResponse(BaseModel):
    candidates: list[GeminiResponseCandidate]
    usage_metadata: GeminiUsage = Field(alias="usageMetadata")
    model_config = ConfigDict(populate_by_name=True)

    def update(self, another: "GeminiResponse"):
        for i, candidate in enumerate(another.candidates):
            if i < len(self.candidates):
                self.candidates[i].update(candidate)
            else:
                self.candidates.append(candidate)
        if another.usage_metadata:
            self.usage_metadata = another.usage_metadata
