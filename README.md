# tinyagent

**A tiny LLVM-based agent with minimal dependencies, written in Python**

- *Currently only pydantic and requests are necessary*

## Quick Install
```bash
pip install tinyagent
```

## Get Started
```python
# export OPENAI_API_KEY=sk-...

from tinyagent import get_agent
agent = get_agent('gpt-4o')

agent.chat('translate tinyagent to Japanese')

agent.chat('explain this image',
  image='https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Doll_face_silver_Persian_2.jpg/1280px-Doll_face_silver_Persian_2.jpg')
```

## Supports

| LLM    | text | image | too call  | streaming |
|:-------:|:----:|:-----:|:--------:|:--------------:|
| OpenAI | ○    | ○     | ○        | ○              |
| Claude | ○    | ○     | ○        | ○              |
