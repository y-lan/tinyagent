"""
Unit tests for the GPTAgent class from the tinyagent.gpt module.
"""

import json
from typing import List
import unittest

from pydantic import BaseModel, ConfigDict

from tinyagent.base import BaseAgent
from tinyagent.gpt.agent import GPTAgent
from tinyagent.schema import ChatResponse
from tinyagent.tools import CalculatorTool, TavilySearchTool


class TestGPTAgent(unittest.TestCase):
    def _get_agent(self, **kwargs) -> BaseAgent:
        return GPTAgent(**kwargs)

    def test_basic(self):
        agent = self._get_agent()
        response = agent.chat("What is 2+2?")
        assert isinstance(response, str)
        assert "4" in response

        response = agent.chat("What is 2+2?", return_complex=True)
        assert isinstance(response, ChatResponse)
        assert response.model == "gpt-4o-mini-2024-07-18"

    def test_history(self):
        agent = self._get_agent(enable_history=True)
        agent.chat("hello, my favorite color is blue")
        response = agent.chat("What is my favorite color?")
        assert "blue" in response

    def test_stream(self):
        agent = self._get_agent(stream=True)

        stream_event_count = 0
        stream_content = ""

        def on_new_chat_token(content):
            nonlocal stream_event_count, stream_content
            stream_event_count += 1
            stream_content += content

        agent.on_new_chat_token(on_new_chat_token)

        response = agent.chat("What is 2+2?")
        assert isinstance(response, str)
        assert "4" in response
        assert stream_event_count > 0
        self.assertEqual(response, stream_content)

    def test_image(self):
        agent = self._get_agent(model_name="gpt-4o")
        response = agent.chat(
            "what's the word in the image?", image="./tests/resources/test.jpg"
        )
        assert isinstance(response, str)
        assert "tiny" in response

    def test_calculator(self):
        calculator = CalculatorTool()
        agent = self._get_agent(tools=[calculator], json_output=True, stream=True)
        raw = agent.chat(
            "What is 23213 * 2323? answer in json format with the key 'result'"
        )
        res = json.loads(raw)
        assert res["result"] == 23213 * 2323

    def test_searcher(self):
        searcher = TavilySearchTool()
        agent = self._get_agent(tools=[searcher], stream=True)
        raw = agent.chat("Who is the Mayor of Meguro-ku, Tokyo in 2023?")
        assert "Eiji Aoki" in raw

    def test_structured_response(self):
        class Step(BaseModel):
            explanation: str
            output: str

            model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

        class MathResponse(BaseModel):
            steps: List[Step]
            final_answer: str
            model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

        agent = self._get_agent(
            system_prompt="You are a helpful math tutor.",
            json_output=True,
            response_format=MathResponse,
        )
        res = MathResponse.model_validate(agent.chat("solve 8x + 31 = 2"))

        assert "-3.625" in res.final_answer


if __name__ == "__main__":
    unittest.main()
