import json
from abc import ABC, abstractmethod
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent
from aidial_sdk.chat_completion.request import Attachment

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


class DeploymentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    @property
    def tool_parameters(self) -> dict[str, Any]:
        return {}

    @property
    def system_prompt(self) -> str | None:
        return None

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        #TODO:
        # 1. Load arguments with `json`
        # 2. Get `prompt` from arguments (by default we provide `prompt` for each deployment tool, use this param name as standard)
        # 3. Delete `prompt` from `arguments` (there can be provided additional parameters and `prompt` will be added
        #    as user message content and other parameters as `custom_fields`)
        # 4. Create AsyncDial client (api_version is 2025-01-01-preview)
        # 5. Call chat completions with:
        #   - messages (here will be just user message. Optionally, in this class you can add system prompt `property`
        #     and if any deployment tool provides system prompt then we need to set it as first message (system prompt))
        #   - stream it
        #   - deployment_name
        #   - extra_body with `custom_fields` https://dialx.ai/dial_api#operation/sendChatCompletionRequest (last request param in documentation)
        #   - **self.tool_parameters (will load all tool parameters that were set up in deployment tools as params, like
        #     `top_p`, `temperature`, etc...)
        # 6. Collect content and it to stage, also, collect custom_content -> attachments and if they are present add
        #    them to stage as attachment as well
        # 7. Return Message with tool role, content, custom_content and tool_call_id
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments.get("prompt")
        if not prompt:
            return Message(
                role=Role.TOOL,
                name=tool_call_params.tool_call.function.name,
                tool_call_id=tool_call_params.tool_call.id,
                content="Error: prompt is required.",
            )

        custom_fields = dict(arguments)
        custom_fields.pop("prompt", None)

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version="2025-01-01-preview",
        )

        chunks = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            stream=True,
            messages=messages,
            extra_body=custom_fields or None,
            **self.tool_parameters,
        )

        content = ""
        attachments: list[Attachment] = []

        async for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                tool_call_params.stage.append_content(delta.content)
                content += delta.content
            if delta.custom_content and delta.custom_content.attachments:
                for raw_attachment in delta.custom_content.attachments:
                    attachment_data: dict[str, Any]
                    if hasattr(raw_attachment, "model_dump"):
                        attachment_data = raw_attachment.model_dump(exclude_none=True)
                    elif hasattr(raw_attachment, "dict"):
                        attachment_data = raw_attachment.dict(exclude_none=True)
                    elif isinstance(raw_attachment, dict):
                        attachment_data = {
                            key: value
                            for key, value in raw_attachment.items()
                            if value is not None
                        }
                    else:
                        attachment_data = {
                            "type": getattr(raw_attachment, "type", None),
                            "title": getattr(raw_attachment, "title", None),
                            "data": getattr(raw_attachment, "data", None),
                            "url": getattr(raw_attachment, "url", None),
                            "reference_type": getattr(raw_attachment, "reference_type", None),
                            "reference_url": getattr(raw_attachment, "reference_url", None),
                        }
                        attachment_data = {
                            key: value
                            for key, value in attachment_data.items()
                            if value is not None
                        }
                    try:
                        attachments.append(Attachment(**attachment_data))
                    except Exception:
                        continue

        if attachments:
            for attachment in attachments:
                tool_call_params.stage.add_attachment(attachment)

        custom_content = CustomContent(attachments=attachments) if attachments else None
        return Message(
            role=Role.TOOL,
            name=tool_call_params.tool_call.function.name,
            tool_call_id=tool_call_params.tool_call.id,
            content=content if content else None,
            custom_content=custom_content,
        )
