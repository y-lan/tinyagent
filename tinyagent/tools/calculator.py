from typing import Type
from pydantic import BaseModel, Field

from tinyagent.schema import Tool


class CalculatorSchema(BaseModel):
    expr: str = Field(..., description="The expression to evaluate")


class CalculatorTool(Tool):
    name: str = "Calculator"
    description: str = "A simple calculator"
    args_schema: Type[BaseModel] = CalculatorSchema

    def _run(self, expr: str):
        safe_chars = set("0123456789+-*/(). ")
        if not all(char in safe_chars for char in expr):
            raise ValueError("Invalid characters in expression")

        try:
            result = eval(expr)
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}")

        return result
