from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from tinyagent.schema import Role


class GeminiRole(Enum):
    USER = "user"
    ASSISTANT = "model"

    @staticmethod
    def from_role(role: Role) -> "GeminiRole":
        if role == Role.USER:
            return GeminiRole.USER
        else:
            return GeminiRole.ASSISTANT


class GeminiTextPart(BaseModel):
    text: str


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


class GeminiFileDataPart(BaseModel):
    file_data: GeminiFileData


class GeminiInlinePart(BaseModel):
    inlineData: GeminiInlineData


class GeminiContent(BaseModel):
    parts: list[GeminiTextPart | GeminiInlinePart | GeminiFileDataPart]
    role: Optional[str] = None


class GeminiResponseCandidate(BaseModel):
    content: GeminiContent
    # STOP
    finish_reason: str = Field(alias="finishReason")
    safety_ratings: list[dict] = Field(alias="safetyRatings")
    model_config = ConfigDict(populate_by_name=True)


class GeminiUsage(BaseModel):
    prompt_token_count: int = Field(alias="promptTokenCount")
    candidates_token_count: int = Field(alias="candidatesTokenCount")
    total_token_count: int = Field(alias="totalTokenCount")
    model_config = ConfigDict(populate_by_name=True)


class GeminiResponse(BaseModel):
    candidates: list[GeminiResponseCandidate]
    usage_metadata: GeminiUsage = Field(alias="usageMetadata")
    model_config = ConfigDict(populate_by_name=True)
