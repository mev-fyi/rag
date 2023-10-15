from typing import Any, Optional, cast
from llama_index.tools import QueryEngineTool

from src.Llama_index_sandbox.custom_react_agent.tools.tool_output import CustomToolOutput


class CustomQueryEngineTool(QueryEngineTool):
    def call(self, input: Any) -> CustomToolOutput:
        query_str = cast(str, input)
        response = self._query_engine.query(query_str)
        return CustomToolOutput(
            content=str(response),
            tool_name=self.metadata.name,
            raw_input={"input": input},
            raw_output=response,
        )

    async def acall(self, input: Any) -> CustomToolOutput:
        query_str = cast(str, input)
        response = await self._query_engine.aquery(query_str)
        return CustomToolOutput(
            content=str(response),
            tool_name=self.metadata.name,
            raw_input={"input": input},
            raw_output=response,
        )



