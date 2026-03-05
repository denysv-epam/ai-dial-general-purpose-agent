from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class ImageGenerationTool(DeploymentTool):

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        #TODO:
        # In this override impl we just need to add extra actions, we need to propagate attachment to the Choice since
        # in DeploymentTool they were propagated to the stage only as files. The main goal here is show pictures in chat
        # (DIAL Chat support special markdown to load pictures from DIAL bucket directly to the chat)
        # ---
        # 1. Call parent function `_execute` and get result
        # 2. If attachments are present then filter only "image/png" and "image/jpeg"
        # 3. Append then as content to choice in such format `f"\n\r![image]({attachment.url})\n\r")`
        # 4. After iteration through attachment if message content is absent add such instruction:
        #    'The image has been successfully generated according to request and shown to user!'
        #    Sometimes models are trying to add generated pictures as well to content (choice), with this instruction
        #    we are notifing LLLM that it was done (but anyway sometimes it will try to add file 😅)
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
        # TODO: provide tool parameters JSON Schema:
        #  - prompt is string, description: "Extensive description of the image that should be generated.", required
        #  - there are 3 optional parameters: https://platform.openai.com/docs/guides/image-generation?image-generation-model=dall-e-3#customize-image-output
        #  - Sample: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/dall-e?tabs=dalle-3#call-the-image-generation-api
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
