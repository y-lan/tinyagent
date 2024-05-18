import json
import unittest
from tinyagent.agent.base import BaseAgent
from tinyagent.agent.claude.agent import ClaudeAgent
from tinyagent.agent.schema import ChatResponse
from tinyagent.agent.tools import CalculatorTool


class TestClaudeAgent(unittest.TestCase):
    def _get_agent(self, **kwargs) -> BaseAgent:
        return ClaudeAgent(**kwargs)

    def test_basic(self):
        agent = self._get_agent()
        response = agent.chat("What is 2+2?")
        assert isinstance(response, str)
        assert '4' in response

        response = agent.chat("What is 2+2?", return_complex=True)
        assert isinstance(response, ChatResponse)
        assert response.model == 'claude-3-haiku-20240307'

    def test_history(self):
        agent = self._get_agent(enable_history=True)
        agent.chat("hello, my favorite color is blue")
        response = agent.chat("What is my favorite color?")
        assert 'blue' in response

    def test_stream(self):
        agent = self._get_agent(stream=True)
        stream_event_count = 0

        def increment_count(_):
            nonlocal stream_event_count
            stream_event_count += 1
        agent.on_new_chat_token(increment_count)

        response = agent.chat("What is 2+2?")
        assert isinstance(response, str)
        assert '4' in response
        assert stream_event_count > 0

    def test_image(self):
        agent = self._get_agent()
        response = agent.chat("what's the word in the image?",
                              image='./tests/resources/test.jpg')
        assert isinstance(response, str)
        assert 'tiny' in response

    def test_tool(self):
        calculator = CalculatorTool()
        agent = self._get_agent(tools=[calculator], json_output=True)
        raw = agent.chat(
            "What is 23213 * 2323? answer in json format with the key 'result'")
        res = json.loads(raw)
        assert res['result'] == 23213 * 2323


if __name__ == '__main__':
    unittest.main()
