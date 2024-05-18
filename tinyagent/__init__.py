def get_agent(name, **kwargs):
    if name.startswith("gpt"):
        from tinyagent.agent.gpt.agent import GPTAgent

        if name == "gpt":
            return GPTAgent(**kwargs)
        else:
            return GPTAgent(model_name=name, **kwargs)

    elif name.startswith("claude"):
        from tinyagent.agent.claude.agent import ClaudeAgent

        if name == "claude":
            return ClaudeAgent(**kwargs)
        else:
            return ClaudeAgent(model_name=name, **kwargs)

    else:
        raise Exception(f"Unsupported agent name: {name}")
