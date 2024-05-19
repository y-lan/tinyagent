from typing import Type

from pydantic import BaseModel, Field
from tinyagent.schema import Tool


TYPE_MAP = {
    int: "number",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "list",
    dict: "object",
    tuple: "object",
    set: "object",
    frozenset: "object",
    type(None): "null",
}


def build_function_signature(func: Tool):
    func_signature = {
        "name": func.name,
        "description": func.description,
        "parameters": {
            "type": "object",
        },
    }

    # check if func has args method
    if hasattr(func, "args_schema") and func.args_schema is not None:
        schema = func.args_schema.model_json_schema()
        func_signature["parameters"]["properties"] = schema["properties"]
        func_signature["parameters"]["required"] = schema["required"]
    else:
        import inspect

        _run_signature = inspect.signature(func._run)
        exclude_params = set(["return", "run_manager", "args", "kwargs"])

        params = {}
        required_params = []

        for name, param in _run_signature.parameters.items():
            if param in exclude_params:
                continue

            type_ = param.annotation
            assert (
                type_ in TYPE_MAP
            ), f"Unsupported type: {type_} for auto-generation of function signature"

            params[name] = dict(type=TYPE_MAP.get(type_))

            if param.default == inspect._empty:
                required_params.append(name)

        func_signature["parameters"]["properties"] = params
        func_signature["parameters"]["required"] = required_params

    return {"type": "function", "function": func_signature}


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
