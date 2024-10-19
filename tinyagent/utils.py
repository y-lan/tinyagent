import base64
import imghdr
import os
from tinyagent.common.time import get_dt_local


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

    import urllib
    import mimetypes

    parsed = urllib.parse.urlparse(image_path)
    if parsed.scheme in ("http", "https"):
        mime_type, _ = mimetypes.guess_type(image_path)
        with urllib.request.urlopen(image_path) as response:
            base64_image = base64.b64encode(response.read()).decode("utf-8")
            return f"data:{mime_type};base64,{base64_image}"

    elif os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            image_type = imghdr.what(image_path)
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            return f"data:image/{image_type};base64,{base64_image}"

    raise ValueError(f"Image not found: {image_path}")
