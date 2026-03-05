import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import GeneralPurposeAgent
from task.prompts import SYSTEM_PROMPT
from task.tools.base import BaseTool
from task.tools.deployment.image_generation_tool import ImageGenerationTool
from task.tools.files.file_content_extraction_tool import FileContentExtractionTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool import MCPTool
from task.tools.py_interpreter.python_code_interpreter_tool import (
    PythonCodeInterpreterTool,
)
from task.tools.rag.document_cache import DocumentCache
from task.tools.rag.rag_tool import RagTool

DIAL_ENDPOINT = os.getenv("DIAL_ENDPOINT", "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
# DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'claude-haiku-4-5')


class GeneralPurposeAgentApplication(ChatCompletion):

    def __init__(self):
        self.tools: list[BaseTool] = []

    async def _get_mcp_tools(self, url: str) -> list[BaseTool]:
        tools: list[BaseTool] = []
        mcp_client = await MCPClient.create(url)
        mcp_tools = await mcp_client.get_tools()
        for tool in mcp_tools:
            tools.append(MCPTool(mcp_client, tool))
        return tools

    async def _create_tools(self) -> list[BaseTool]:
        document_cache = DocumentCache.create()
        tools: list[BaseTool] = [
            ImageGenerationTool(DIAL_ENDPOINT),
            FileContentExtractionTool(DIAL_ENDPOINT),
            RagTool(DIAL_ENDPOINT, DEPLOYMENT_NAME, document_cache),
        ]
        tools.extend(await self._get_mcp_tools("http://localhost:8051/mcp"))
        return tools

    async def chat_completion(self, request: Request, response: Response) -> None:
        if not self.tools:
            self.tools = await self._create_tools()

        with response.create_single_choice() as choice:
            agent = GeneralPurposeAgent(
                endpoint=DIAL_ENDPOINT,
                system_prompt=SYSTEM_PROMPT,
                tools=self.tools,
            )
            await agent.handle_request(
                choice=choice,
                deployment_name=DEPLOYMENT_NAME,
                request=request,
                response=response,
            )


app: DIALApp = DIALApp()
agent_app = GeneralPurposeAgentApplication()
app.add_chat_completion("general-purpose-agent", agent_app)


if __name__ == "__main__":
    uvicorn.run(app, port=5030, host="0.0.0.0")
