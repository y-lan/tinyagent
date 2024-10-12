from tinyagent.base import BaseAgent
from tinyagent.gpt.agent import GPTAgent
from tinyagent.claude.agent import ClaudeAgent


def get_agent(name, provider=None, **kwargs):
    if provider == "openai" or name.startswith("gpt") or name.startswith("o1"):
        from tinyagent.gpt.agent import GPTAgent

        if name == "gpt":
            return GPTAgent(**kwargs)

        return GPTAgent(model_name=name, **kwargs)

    elif provider == "anthropic" or name.startswith("claude"):
        from tinyagent.claude.agent import ClaudeAgent

        if name == "claude":
            return ClaudeAgent(**kwargs)

        return ClaudeAgent(model_name=name, **kwargs)

    raise Exception(f"Unsupported agent name: {name}")


__all__ = ["BaseAgent", "get_agent", "GPTAgent", "ClaudeAgent"]
