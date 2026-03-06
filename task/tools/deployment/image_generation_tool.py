from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class ImageGenerationTool(DeploymentTool):

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        result = await super()._execute(tool_call_params)
        if not isinstance(result, Message):
            return result

        attachments = []
        if result.custom_content and result.custom_content.attachments:
            attachments = result.custom_content.attachments

        if attachments:
            for attachment in attachments:
                if attachment.type in {"image/png", "image/jpeg"} and attachment.url:
                    tool_call_params.choice.append_content(
                        f"\n\r![image]({attachment.url})\n\r"
                    )

        if not result.content:
            result.content = (
                "The image has been successfully generated according to request and shown to user!"
            )

        return result

    @property
    def deployment_name(self) -> str:
        return "dall-e-3"

    @property
    def name(self) -> str:
        return "image_generation_tool"

    @property
    def description(self) -> str:
        return (
            "Generates an image from a detailed prompt using the DALL-E-3 deployment. "
            "Use this tool when the user asks to create or generate an image."
        )
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Extensive description of the image that should be generated.",
                },
                "size": {
                    "type": "string",
                    "description": "Image size. Examples: 1024x1024, 1024x1792, 1792x1024.",
                },
                "quality": {
                    "type": "string",
                    "description": "Image quality. Examples: standard, hd.",
                },
                "style": {
                    "type": "string",
                    "description": "Image style. Examples: vivid, natural.",
                },
            },
            "required": ["prompt"],
        }
