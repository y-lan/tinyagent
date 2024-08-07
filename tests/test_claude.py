"""
Unit tests for the ClaudeAgent class from the tinyagent.claude module.
"""

import json
import unittest
from tinyagent.base import BaseAgent
from tinyagent.claude.agent import ClaudeAgent
from tinyagent.schema import ChatResponse
from tinyagent.tools import CalculatorTool


class TestClaudeAgent(unittest.TestCase):
    def _get_agent(self, **kwargs) -> BaseAgent:
        return ClaudeAgent(**kwargs)

    def _get_stream_content_collector(self, output):
        assert output == {}

        def on_new_chat_token(content):
            if output.get("event_count") is None:
                output["event_count"] = 0
            output["event_count"] += 1

            if output.get("content") is None:
                output["content"] = ""
            output["content"] += content

        return on_new_chat_token

    def test_basic(self):
        agent = self._get_agent()
        response = agent.chat("What is 2+2?")
        assert isinstance(response, str)
        assert "4" in response

        response = agent.chat("What is 2+2?", return_complex=True)
        assert isinstance(response, ChatResponse)
        assert response.model == "claude-3-haiku-20240307"

    def test_history(self):
        agent = self._get_agent(enable_history=True)
        agent.chat("hello, my favorite color is blue")
        response = agent.chat("What is my favorite color?")
        assert "blue" in response

    def test_stream(self):
        agent = self._get_agent(stream=True)
        stream_output = {}
        agent.on_new_chat_token(self._get_stream_content_collector(stream_output))

        response = agent.chat("What is 2+2?")
        assert isinstance(response, str)
        assert "4" in response
        assert stream_output["event_count"] > 0
        self.assertEqual(response, stream_output["content"])

    def test_image(self):
        agent = self._get_agent()
        response = agent.chat(
            "what's the word in the image?", image="./tests/resources/test.jpg"
        )
        assert isinstance(response, str)
        assert "tiny" in response

    def test_tool(self):
        calculator = CalculatorTool()
        agent = self._get_agent(tools=[calculator], json_output=True)
        raw = agent.chat(
            "What is 23213 * 2323? answer in json format with the key 'result'"
        )
        res = json.loads(raw)
        assert res["result"] == 23213 * 2323

    def test_json_output(self):
        agent = self._get_agent(
            json_output=True,
            system_prompt="You are a helpful assistant; you will answer in JSON format with the key 'result'",
        )
        response = agent.chat("What is 2+2?")
        assert isinstance(response, str)
        assert json.loads(response)["result"] == 4

    def test_prefill_response(self):
        agent = self._get_agent()
        response = agent.chat("What is 2+2?", prefill_response="i think the answer is")
        assert isinstance(response, str)
        assert response.startswith("i think the answer is")

        agent = self._get_agent(stream=True)
        stream_output = {}
        agent.on_new_chat_token(self._get_stream_content_collector(stream_output))
        response = agent.chat("What is 2+2?", prefill_response="i think the answer is")
        assert stream_output["content"].startswith("i think the answer is")


if __name__ == "__main__":
    unittest.main()
