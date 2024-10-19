import json
import logging
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
    GeminiToolCallPart,
    GeminiToolMode,
)
from tinyagent.schema import (
    BaseContent,
    ImageContent,
    Message,
    Role,
    TextContent,
    Tool,
    ToolUseContent,
    ToolUseResultMessage,
)
from tinyagent.tools.tool import build_function_signature
from tinyagent.utils import convert_image_to_base64_uri

logger = logging.getLogger(__name__)


def _serialize_content(content: BaseContent):
    if isinstance(content, TextContent):
        return GeminiTextPart(text=content.text)
    elif isinstance(content, ImageContent):
        base64_uri = convert_image_to_base64_uri(content.image_url.url)
        mime_type = base64_uri.split(";")[0].split(":")[1]
        base64_data = base64_uri.split(",")[1]
        return GeminiInlinePart(
            inlineData=GeminiInlineData(mimeType=mime_type, data=base64_data)
        )
    elif isinstance(content, ToolUseContent):
        return GeminiToolCallPart.from_tool_use_content(content)
    else:
        raise ValueError(f"Unsupported content type: {type(content)}")


def _convert_to_gemini_message(message: Message):
    if isinstance(message, ToolUseResultMessage):
        return GeminiContent.from_tool_use_message(message)
    else:
        return GeminiContent(
            parts=[_serialize_content(content) for content in message.content],
            role=GeminiRole.from_role(message.role),
        )


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

    def _build_tool_signature(self, tool: Tool):
        signature = build_function_signature(tool)["function"]

        for _, value in signature["parameters"]["properties"].items():
            if "title" in value:
                value.pop("title")

        return signature

    def _handle_streaming_response(self, response):
        for line in response.iter_lines():
            if line:
                event_data = line.decode("utf-8")
                if event_data.startswith("data:"):
                    data = json.loads(event_data[5:])
                    data = GeminiResponse(**data)
                    yield data

    def chat(
        self,
        messages: list[Message] = [],
        system_instruction: str | None = None,
        stream: bool = False,
        model: str = "gemini-1.5-flash-latest",
        tool_mode: GeminiToolMode = GeminiToolMode.AUTO,
        tools: list[Tool] = None,
        json_output: bool = False,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        **kwargs,
    ) -> dict:
        if stream:
            url = f"{self.base_url}/{model}:streamGenerateContent?alt=sse&key={self.api_key}"
        else:
            url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"

        # Convert messages to the format expected by the Gemini API
        contents = [
            _convert_to_gemini_message(msg).model_dump(exclude_none=True)
            for msg in messages
        ]

        payload = {
            "contents": contents,
            "safetySettings": self.safetySettings,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
            **kwargs,  # Include any additional parameters passed to the method
        }

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content[0].text

        if system_instruction:
            payload["system_instruction"] = GeminiContent(
                parts=[GeminiTextPart(text=system_instruction)],
            ).model_dump(exclude_none=True)

        if tools:
            payload["tools"] = [
                {
                    "function_declarations": [
                        self._build_tool_signature(tool) for tool in tools
                    ]
                }
            ]
            payload["tool_config"] = {
                "function_calling_config": {
                    "mode": tool_mode.value,
                }
            }

            if json_output:
                logger.warning(
                    "Gemini does not support strict json output when tools are used, "
                    "will disable json output"
                )
                json_output = False

        if json_output:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        ## print(json.dumps(payload, indent=2))

        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        if stream:
            return self._handle_streaming_response(response)
        else:
            data = response.json()
            return GeminiResponse(**data)
