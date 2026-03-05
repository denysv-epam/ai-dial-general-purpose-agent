import json
from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.base import BaseTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class MCPTool(BaseTool):

    def __init__(self, client: MCPClient, mcp_tool_model: MCPToolModel):
        self.client = client
        self.mcp_tool_model = mcp_tool_model

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments or "{}")
        content = await self.client.call_tool(self.mcp_tool_model.name, arguments)

        stage = tool_call_params.stage
        if isinstance(content, str):
            stage.append_content(content)
            return content

        content_json = json.dumps(content, indent=2)
        stage.append_content(f"```json\n\r{content_json}\n\r```\n\r")
        return content_json

    @property
    def name(self) -> str:
        return self.mcp_tool_model.name

    @property
    def description(self) -> str:
        return self.mcp_tool_model.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self.mcp_tool_model.parameters
