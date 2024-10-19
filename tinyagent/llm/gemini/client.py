import os
from pydantic import BaseModel
import requests
from tinyagent.llm.gemini.schema import (
    GeminiContent,
    GeminiInlineData,
    GeminiInlinePart,
    GeminiResponse,
    GeminiRole,
    GeminiTextPart,
)
from tinyagent.schema import BaseContent, ImageContent, Message, Role
from tinyagent.utils import convert_image_to_base64_uri


def _serialize_content(content: BaseContent):
    if content.type == "text":
        return GeminiTextPart(text=content.text)
    elif isinstance(content, ImageContent):
        base64_uri = convert_image_to_base64_uri(content.image_url.url)
        mime_type = base64_uri.split(";")[0].split(":")[1]
        base64_data = base64_uri.split(",")[1]
        return GeminiInlinePart(
            inlineData=GeminiInlineData(mimeType=mime_type, data=base64_data)
        )
    else:
        return content.model_dump(exclude_none=True)


class GeminiClient(BaseModel):
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/models"

    safetySettings: list[dict] = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
            "threshold": "BLOCK_NONE",
        },
    ]

    def __init__(self, api_key: str = None, **data):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        super().__init__(api_key=api_key, **data)

    def chat(
        self,
        messages: list[Message] = [],
        system_instruction: str | None = None,
        model: str = "gemini-1.5-flash-latest",
        **kwargs,
    ) -> dict:
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"

        # Convert messages to the format expected by the Gemini API
        contents = [
            {
                "role": GeminiRole.from_role(msg.role).value,
                "parts": [
                    _serialize_content(content).model_dump() for content in msg.content
                ],
            }
            for msg in messages
        ]

        payload = {
            "contents": contents,
            "safetySettings": self.safetySettings,
            **kwargs,  # Include any additional parameters passed to the method
        }

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content[0].text

        if system_instruction:
            payload["system_instruction"] = GeminiContent(
                parts=[GeminiTextPart(text=system_instruction)],
            ).model_dump(exclude_none=True)

        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        return GeminiResponse(**data)
