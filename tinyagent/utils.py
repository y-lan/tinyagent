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
