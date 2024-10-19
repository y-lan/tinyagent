from typing import Type
from pydantic import BaseModel, Field
from datetime import datetime
import pytz
from tinyagent.schema import Tool


class CurrentTimeSchema(BaseModel):
    timezone: str = Field(..., description="The timezone for the current time")


class CurrentTimeTool(Tool):
    name: str = "current_time"
    description: str = "Get the current date and time in a specified timezone"
    args_schema: Type[BaseModel] = CurrentTimeSchema

    def _run(self, timezone: str = "UTC"):
        try:
            tz = pytz.timezone(timezone)
            current_time = datetime.now(tz)
            return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        except pytz.exceptions.UnknownTimeZoneError:
            raise ValueError(f"Invalid timezone: {timezone}")
