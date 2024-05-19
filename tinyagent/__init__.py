from tinyagent.base import BaseAgent
from tinyagent.gpt.agent import GPTAgent
from tinyagent.claude.agent import ClaudeAgent


def get_agent(name, **kwargs):
    if name.startswith("gpt"):
        from tinyagent.gpt.agent import GPTAgent

        if name == "gpt":
            return GPTAgent(**kwargs)

        return GPTAgent(model_name=name, **kwargs)

    elif name.startswith("claude"):
        from tinyagent.claude.agent import ClaudeAgent

        if name == "claude":
            return ClaudeAgent(**kwargs)

        return ClaudeAgent(model_name=name, **kwargs)

    raise Exception(f"Unsupported agent name: {name}")


__all__ = ["BaseAgent", "get_agent", "GPTAgent", "ClaudeAgent"]
