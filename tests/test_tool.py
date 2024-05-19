from typing import Type
from pydantic import BaseModel, Field
from tinyagent.tools import build_function_signature
from tinyagent.schema import Tool


class MockToolSchema(BaseModel):
    param1: str = Field(..., title="Parameter 1", description="A string parameter")
    param2: int = Field(..., title="Parameter 2", description="An integer parameter")


class MockTool(Tool):
    name: str = "TestTool"
    description: str = "A test tool"
    args_schema: Type[BaseModel] = MockToolSchema

    def _run(self, param1: str, param2: int = 1):
        return param1 + str(param2)


def test_build_function_signature_with_args_schema():
    test_tool = MockTool()
    signature = build_function_signature(test_tool)
    assert signature["function"]["name"] == "TestTool"
    assert signature["function"]["description"] == "A test tool"
    assert "properties" in signature["function"]["parameters"]
    assert "required" in signature["function"]["parameters"]
