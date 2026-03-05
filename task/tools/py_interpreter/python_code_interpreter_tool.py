import base64
import json
from typing import Any, Optional

from aidial_client import Dial
from aidial_sdk.chat_completion import Message, Attachment
from pydantic import AnyUrl

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class PythonCodeInterpreterTool(BaseTool):
    """
    Uses https://github.com/khshanovskyi/mcp-python-code-interpreter PyInterpreter MCP Server.

    ⚠️ Pay attention that this tool will wrap all the work with PyInterpreter MCP Server.
    """

    def __init__(
            self,
            mcp_client: MCPClient,
            mcp_tool_models: list[MCPToolModel],
            tool_name: str,
            dial_endpoint: str,
    ):
        """
        :param tool_name: it must be actual name of tool that executes code. It is 'execute_code'.
            https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L303
        """
        self.dial_endpoint = dial_endpoint
        self.mcp_client = mcp_client
        self._code_execute_tool: Optional[MCPToolModel] = None
        for tool_model in mcp_tool_models:
            if tool_model.name == tool_name:
                self._code_execute_tool = tool_model
                break

        if not self._code_execute_tool:
            raise RuntimeError(
                f"PythonCodeInterpreterTool requires MCP tool '{tool_name}'"
            )

    @classmethod
    async def create(
            cls,
            mcp_url: str,
            tool_name: str,
            dial_endpoint: str,
    ) -> 'PythonCodeInterpreterTool':
        """Async factory method to create PythonCodeInterpreterTool"""
        mcp_client = await MCPClient.create(mcp_url)
        mcp_tools = await mcp_client.get_tools()
        return cls(mcp_client, mcp_tools, tool_name, dial_endpoint)

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        if not self._code_execute_tool:
            raise RuntimeError("Python code execution tool is not configured")
        return self._code_execute_tool.name

    @property
    def description(self) -> str:
        if not self._code_execute_tool:
            raise RuntimeError("Python code execution tool is not configured")
        return self._code_execute_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        if not self._code_execute_tool:
            raise RuntimeError("Python code execution tool is not configured")
        return self._code_execute_tool.parameters

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments or "{}")
        code = arguments.get("code")
        session_id = arguments.get("session_id")

        stage = tool_call_params.stage
        stage.append_content("## Request arguments: \n")
        if not code:
            stage.append_content("## Response: \n")
            stage.append_content("No code provided.\n\r")
            return "Error: code is required."

        stage.append_content(f"```python\n\r{code}\n\r```\n\r")

        if session_id and str(session_id) != "0":
            stage.append_content(f"**session_id**: {session_id}\n\r")
        else:
            stage.append_content("New session will be created\n\r")

        if not self._code_execute_tool:
            raise RuntimeError("Python code execution tool is not configured")

        tool_result = await self.mcp_client.call_tool(
            self._code_execute_tool.name,
            arguments,
        )

        if isinstance(tool_result, str):
            result_payload = tool_result
        else:
            result_payload = json.dumps(tool_result)

        execution_result = _ExecutionResult.model_validate(json.loads(result_payload))

        if execution_result.files:
            dial_client = Dial(
                base_url=self.dial_endpoint,
                api_key=tool_call_params.api_key,
            )
            files_home = dial_client.my_appdata_home()
            if not files_home:
                raise RuntimeError("Unable to resolve appdata home for uploads")

            for file_ref in execution_result.files:
                file_name = file_ref.name
                mime_type = file_ref.mime_type
                resource = await self.mcp_client.get_resource(AnyUrl(file_ref.uri))

                is_text = mime_type.startswith("text/") or mime_type in {
                    "application/json",
                    "application/xml",
                }
                if is_text:
                    if isinstance(resource, bytes):
                        file_bytes = resource
                    else:
                        file_bytes = str(resource).encode("utf-8")
                else:
                    if isinstance(resource, bytes):
                        try:
                            file_bytes = base64.b64decode(resource, validate=True)
                        except Exception:
                            file_bytes = resource
                    else:
                        file_bytes = base64.b64decode(str(resource))

                upload_url = f"files/{(files_home / file_name).as_posix()}"
                dial_client.files.upload(
                    upload_url,
                    file=(file_name, file_bytes, mime_type),
                )

                attachment = Attachment(
                    url=upload_url,
                    type=mime_type,
                    title=file_name,
                )
                stage.add_attachment(attachment)
                tool_call_params.choice.add_attachment(attachment)
                execution_result.output.append(
                    f"Uploaded file: {upload_url}"
                )

        if execution_result.output:
            execution_result.output = [
                output[:1000] for output in execution_result.output
            ]

        stage.append_content(
            f"```json\n\r{execution_result.model_dump_json(indent=2)}\n\r```\n\r"
        )

        return execution_result.model_dump_json()
