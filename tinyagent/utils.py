from tinyagent.common.time import get_dt_local


def replace_magic_placeholders(prompt: str) -> str:
    prompt = prompt.replace("__DATE__", get_dt_local("%Y-%m-%d (%a)"))

    return prompt
