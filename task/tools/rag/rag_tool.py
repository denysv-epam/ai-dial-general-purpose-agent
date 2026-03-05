import json
from typing import Any

import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided document context.

You will receive:
- CONTEXT: Retrieved relevant excerpts from a document
- REQUEST: The user's question or search query

Instructions:
- Answer the request using only the information in the provided context
- If the context doesn't contain enough information to answer, clearly state that
- Be concise and direct in your response"""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(
        self, endpoint: str, deployment_name: str, document_cache: DocumentCache
    ):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.document_cache = document_cache
        self.model = SentenceTransformer(
            model_name_or_path="all-MiniLM-L6-v2",
            device="cpu",
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rag_search_tool"

    @property
    def description(self) -> str:
        return (
            "Performs semantic search on documents to find and answer questions based on relevant content. "
            "Supports: PDF, TXT, CSV, HTML. "
            "Use this tool when user asks questions about document content, needs specific information from large files, "
            "or wants to search for particular topics/keywords. "
            "Don't use it when: user wants to read entire document sequentially. "
            "HOW IT WORKS: Splits document into chunks, finds top 3 most relevant sections using semantic search, "
            "then generates answer based only on those sections."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document",
                },
                "file_url": {
                    "type": "string",
                    "description": "URL of the file to search in",
                },
            },
            "required": ["request", "file_url"],
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        request = arguments.get("request")
        file_url = arguments.get("file_url")

        stage = tool_call_params.stage
        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Request**: {request}\n\r")
        stage.append_content(f"**File URL**: {file_url}\n\r")

        if not request or not file_url:
            stage.append_content("## Response: \n")
            stage.append_content("Missing request or file_url.\n\r")
            return "Error: request and file_url are required."

        cache_document_key = f"{tool_call_params.conversation_id}:{file_url}"
        cached_data = self.document_cache.get(cache_document_key)

        if cached_data:
            embeddings_array, chunks = cached_data
        else:
            text_content = DialFileContentExtractor(
                endpoint=self.endpoint,
                api_key=tool_call_params.api_key,
            ).extract_text(file_url)
            if not text_content:
                stage.append_content("## Response: \n")
                stage.append_content("File content not found.\n\r")
                return "Error: File content not found."

            chunks = self.text_splitter.split_text(text_content)
            if not chunks:
                stage.append_content("## Response: \n")
                stage.append_content("File content is empty after splitting.\n\r")
                return "Error: File content is empty."

            embeddings = self.model.encode(chunks)
            embeddings_array = np.asarray(embeddings, dtype="float32")
            self.document_cache.set(cache_document_key, embeddings_array, chunks)

        k = min(3, len(chunks))
        if k == 0:
            stage.append_content("## Response: \n")
            stage.append_content("No chunks available for search.\n\r")
            return "Error: No chunks available for search."

        query_embedding = np.asarray(self.model.encode([request]), dtype="float32")
        query_vector = query_embedding[0]
        query_norm = np.linalg.norm(query_vector)
        doc_norms = np.linalg.norm(embeddings_array, axis=1)
        scores = embeddings_array @ query_vector
        denom = (doc_norms * query_norm) + 1e-8
        scores = scores / denom
        indices = np.argsort(scores)[-k:][::-1]

        retrieved_chunks: list[str] = []
        for idx in indices:
            if idx is None or idx < 0 or idx >= len(chunks):
                continue
            retrieved_chunks.append(chunks[idx])

        augmented_prompt = self.__augmentation(request, retrieved_chunks)
        stage.append_content("## RAG Request: \n")
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        stage.append_content("## Response: \n")

        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version="2025-01-01-preview",
        )
        chunks_stream = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            stream=True,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": augmented_prompt},
            ],
        )

        content = ""
        async for chunk in chunks_stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta is not None and delta.content:
                    stage.append_content(delta.content)
                    content += delta.content

        return content

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        context = "\n\n".join(
            [f"[Chunk {idx + 1}]\n{chunk}" for idx, chunk in enumerate(chunks)]
        )
        return (
            "Use the following context to answer the question. "
            "If the answer is not contained in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {request}\n"
            "Answer:"
        )
