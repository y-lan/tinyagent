import os
from typing import Optional, Type
from pydantic import BaseModel, Field, PrivateAttr
import requests

from tinyagent.schema import Tool


class TavilySearchSchema(BaseModel):
    query: str = Field(..., description="The query to search")


class TavilySearchTool(Tool):
    name: str = "TavilySearch"
    description: str = "A tool for searching the web"
    args_schema: Type[BaseModel] = TavilySearchSchema
    _api_key: str = PrivateAttr(default=None)

    def __init__(self):
        super().__init__()
        self._api_key = os.environ.get("TAVILY_API_KEY")

    def _run(self, query: str, limit: int = 10) -> str:
        data = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "max_results": limit,
            "include_domains": [],
            "exclude_domains": [],
        }

        try:
            response = requests.post("https://api.tavily.com/search", json=data)
            response.raise_for_status()
            results = response.json()

            formatted_results = []
            for result in results.get("results", []):
                formatted_results.append(
                    f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content']}\n"
                )

            return "\n\n".join(formatted_results)
        except requests.RequestException as e:
            return f"Error occurred while searching: {str(e)}"
