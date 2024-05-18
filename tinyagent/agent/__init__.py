
def get_agent(agent_type: str, **kwargs):
    if agent_type == "openai":
        # from mosaicpy.llm.openai.agent import OpenAIAgent

        return OpenAIAgent(**kwargs)
    elif agent_type == "anthropic":
        from mosaicpy.llm.anthropic.agent import AnthropicAgent

        return AnthropicAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
