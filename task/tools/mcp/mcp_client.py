from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent, ReadResourceResult, TextResourceContents, BlobResourceContents
from pydantic import AnyUrl

from task.tools.mcp.mcp_tool_model import MCPToolModel


class MCPClient:
    """Handles MCP server connection and tool execution"""

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None

    @classmethod
    async def create(cls, mcp_server_url: str) -> 'MCPClient':
        """Async factory method to create and connect MCPClient"""
        instance = cls(mcp_server_url)
        await instance.connect()
        return instance

    async def connect(self):
        """Connect to MCP server"""
        if self.session:
            return

        self._streams_context = streamablehttp_client(self.server_url)
        read_stream, write_stream, _ = await self._streams_context.__aenter__()

        self._session_context = ClientSession(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()

        init_result = await self.session.initialize()
        if hasattr(init_result, "model_dump_json"):
            print(init_result.model_dump_json(indent=2))
        else:
            print(init_result)


    async def get_tools(self) -> list[MCPToolModel]:
        """Get available tools from MCP server"""
        if not self.session:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        tools = await self.session.list_tools()
        return [
            MCPToolModel(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.inputSchema or {"type": "object", "properties": {}},
            )
            for tool in tools.tools
        ]

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a tool on the MCP server"""
        if not self.session:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        tool_result: CallToolResult = await self.session.call_tool(tool_name, tool_args)
        content_blocks = tool_result.content or []
        if not content_blocks:
            return ""

        text_blocks = [
            block.text for block in content_blocks if isinstance(block, TextContent)
        ]
        if len(text_blocks) == len(content_blocks):
            return "\n".join(text_blocks)

        first_block = content_blocks[0]
        if isinstance(first_block, TextContent):
            return first_block.text

        return content_blocks

    async def get_resource(self, uri: AnyUrl) -> str | bytes:
        """Get specific resource content"""
        if not self.session:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        result: ReadResourceResult = await self.session.read_resource(uri)
        contents = result.contents or []
        if not contents:
            return ""

        text_chunks: list[str] = []
        for content in contents:
            if isinstance(content, TextResourceContents):
                text_chunks.append(content.text)
            elif isinstance(content, BlobResourceContents):
                return content.blob

        return "\n".join(text_chunks)

    async def close(self):
        """Close connection to MCP server"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

        self.session = None
        self._session_context = None
        self._streams_context = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        return False
