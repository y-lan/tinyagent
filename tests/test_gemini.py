"""
Unit tests for the GeminiAgent class from the tinyagent.llm.gemini module.
"""

import json
import unittest
from tinyagent.base import BaseAgent
from tinyagent.llm.gemini.agent import GeminiAgent
from tinyagent.llm.gemini.schema import GeminiToolMode
from tinyagent.schema import ChatResponse
from tinyagent.tools.calculator import CalculatorTool
from tinyagent.tools.current_time import CurrentTimeTool
from tinyagent.tools.tavily import TavilySearchTool


class TestGeminiAgent(unittest.TestCase):
    def _get_agent(self, **kwargs) -> BaseAgent:
        return GeminiAgent(**kwargs)

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
        assert response.model == "gemini-1.5-flash-latest"

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
            "What is 23213 * 2323? "
            "Answer in JSON format with the key 'result' and the int value as the result of the calculation. "
            "Return the JSON payload only, without any other text or JSON code block markers.",
            tool_mode=GeminiToolMode.ANY,
        )
        if raw.startswith("```json"):
            raw = raw.lstrip("```json").rstrip("```").strip()
        res = json.loads(raw)
        assert res["result"] == 23213 * 2323

    def test_multiple_tool(self):
        current_time = CurrentTimeTool()
        searcher = TavilySearchTool()
        agent = self._get_agent(
            model_name="gemini-1.5-pro-latest",
            tools=[current_time, searcher],
            stream=True,
            temperature=0.0,
        )
        tool_call_count = 0

        def on_tool_call(name, args):
            nonlocal tool_call_count
            tool_call_count += 1

        agent.on_tool_call(on_tool_call)
        raw = agent.chat(
            "Use a tool to find today's date, then use another tool to search for the temperature in Tokyo today",
            tool_mode=GeminiToolMode.ANY,
        )
        assert len(raw) > 0
        assert tool_call_count == 2

    def test_json_output(self):
        agent = self._get_agent(
            json_output=True,
            system_prompt="You are a helpful assistant; you will answer in JSON format with the key 'result' and the int value as the result of the calculation",
        )
        response = agent.chat("What is 2+2?")
        assert isinstance(response, str)
        assert json.loads(response)["result"] == 4


if __name__ == "__main__":
    unittest.main()
