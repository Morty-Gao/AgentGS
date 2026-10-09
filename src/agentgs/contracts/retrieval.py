"""Tool retrieval interfaces; graph access is pure after lazy initialization."""

from typing import Any, Protocol


class ToolRetriever(Protocol):
    async def retrieve(self, query: Any) -> list[list[str]]: ...
    def get_tool_subgraph(
        self, name: str, depth: int = 1, limit: int = 1000
    ) -> list[list[str]]: ...
    def get_tool_info(self, relations: list[list[str]]) -> dict[str, Any]: ...
