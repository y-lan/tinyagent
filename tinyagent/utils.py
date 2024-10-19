import base64
import imghdr
import os
from tinyagent.common.time import get_dt_local
import urllib
import mimetypes
from urllib.request import urlopen


def replace_magic_placeholders(prompt: str) -> str:
    prompt = prompt.replace("__DATE__", get_dt_local("%Y-%m-%d (%a)"))

    return prompt


def get_param(kwargs: dict, key: str | list[str], default):
    if isinstance(key, list):
        for k in key:
            if k in kwargs and kwargs[k] is not None:
                return kwargs[k]
    else:
        if key in kwargs and kwargs[key] is not None:
            return kwargs[key]
    return default


def convert_image_to_base64_uri(image_path: str) -> str:
    if image_path.startswith("data:image/"):
        return image_path

    parsed = urllib.parse.urlparse(image_path)
    if parsed.scheme in ("http", "https"):
        mime_type, _ = mimetypes.guess_type(image_path)
        with urlopen(image_path) as response:
            base64_image = base64.b64encode(response.read()).decode("utf-8")
            return f"data:{mime_type};base64,{base64_image}"

    elif os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            image_type = imghdr.what(image_path)
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            return f"data:image/{image_type};base64,{base64_image}"

    raise ValueError(f"Image not found: {image_path}")


def convert_audio_to_base64(audio_path: str) -> str:
    if audio_path.startswith("data:audio/"):
        return audio_path

    parsed = urllib.parse.urlparse(audio_path)
    if parsed.scheme in ("http", "https"):
        mime_type, _ = mimetypes.guess_type(audio_path)
        with urlopen(audio_path) as response:
            base64_audio = base64.b64encode(response.read()).decode("utf-8")
            return f"data:{mime_type};base64,{base64_audio}"

    elif os.path.exists(audio_path):
        mime_type, _ = mimetypes.guess_type(audio_path)
        with open(audio_path, "rb") as audio_file:
            base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{base64_audio}"

    raise ValueError(f"Audio file not found: {audio_path}")
