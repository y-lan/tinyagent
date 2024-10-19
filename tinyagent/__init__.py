from tinyagent.base import BaseAgent
from tinyagent.llm.gpt.agent import GPTAgent
from tinyagent.llm.gpt.client import OpenAIClient
from tinyagent.llm.claude.agent import ClaudeAgent
from tinyagent.llm.claude.client import AnthropicClient
from tinyagent.llm.gemini.client import GeminiClient
from tinyagent.schema import Message, Role


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


__all__ = [
    "BaseAgent",
    "get_agent",
    "GPTAgent",
    "OpenAIClient",
    "ClaudeAgent",
    "AnthropicClient",
    "GeminiAgent",
    "GeminiClient",
    "Message",
    "Role",
]
