from typing import Type
from pydantic import BaseModel, Field
from tinyagent.tools.current_time import CurrentTimeTool
from tinyagent.tools.tavily import TavilySearchTool
from tinyagent.tools.tool import (
    build_function_signature,
)
from tinyagent.schema import Tool
from datetime import datetime as dt


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


def test_tavily_search_tool():
    searcher = TavilySearchTool()
    result = searcher._run("the president of the United States?", limit=1)
    assert len(result) > 0


def test_current_time_tool():
    current_time = CurrentTimeTool()
    result1 = current_time._run(timezone="Asia/Tokyo")
    result2 = current_time._run()
    result3 = current_time._run(timezone="America/New_York")

    # Helper function to parse datetime and timezone
    def parse_datetime_with_tz(dt_string):
        dt_part, tz_part = dt_string.rsplit(" ", 1)
        parsed_dt = dt.strptime(dt_part, "%Y-%m-%d %H:%M:%S")
        return parsed_dt, tz_part

    # Parse results
    dt1, tz1 = parse_datetime_with_tz(result1)
    dt2, tz2 = parse_datetime_with_tz(result2)
    dt3, tz3 = parse_datetime_with_tz(result3)

    # Assert that Tokyo time is ahead of UTC, which is ahead of New York time
    assert dt1 > dt2 > dt3, "Time order is incorrect"

    # Additional assertions to check if the timezones are correct
    assert tz1 == "JST", "Tokyo timezone not found in result1"
    assert tz2 == "UTC", "UTC timezone not found in result2"
    assert tz3 in ["EDT", "EST"], "New York timezone not found in result3"
