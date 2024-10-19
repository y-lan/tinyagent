from tinyagent.base import BaseAgent
from tinyagent.llm.gpt.agent import GPTAgent
from tinyagent.llm.claude.agent import ClaudeAgent


def get_agent(name, provider=None, **kwargs):
    if provider == "openai" or name.startswith("gpt") or name.startswith("o1"):
        from tinyagent.llm.gpt.agent import GPTAgent

        if name == "gpt":
            return GPTAgent(**kwargs)

        return GPTAgent(model_name=name, **kwargs)

    elif provider == "anthropic" or name.startswith("claude"):
        from tinyagent.llm.claude.agent import ClaudeAgent

        if name == "claude":
            return ClaudeAgent(**kwargs)

        return ClaudeAgent(model_name=name, **kwargs)

    elif provider == "gemini":
        from tinyagent.llm.gemini.agent import GeminiAgent

        if name == "gemini":
            return GeminiAgent(**kwargs)

        return GeminiAgent(model_name=name, **kwargs)

    raise Exception(f"Unsupported agent name: {name}")


__all__ = ["BaseAgent", "get_agent", "GPTAgent", "ClaudeAgent", "GeminiAgent"]
