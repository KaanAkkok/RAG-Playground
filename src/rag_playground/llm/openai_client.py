from typing import Any, override

from openai import AsyncOpenAI

from rag_playground.core.exceptions import EmbeddingClientError, LLMClientError
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.llm.llm_factory import LLMClient
from rag_playground.schemas.llm_client_schema import (
    ChatRequest,
    EmbeddingRequest,
    LLMResponse,
)

logger = get_logger(__name__)


class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.base_url = base_url if base_url is not None else settings.OPENAI_BASE_URL
        self.timeout = timeout if timeout is not None else settings.OPENAI_CHAT_TIMEOUT
        self.max_retries = (
            max_retries if max_retries is not None else settings.OPENAI_CHAT_MAX_RETRIES
        )
        self.client = None
        self.embedding_client = None

    @override
    def get_client(self, type: str | None = None) -> AsyncOpenAI:
        try:
            client_options: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout if type == "chat" else None,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                client_options["base_url"] = self.base_url
            client = AsyncOpenAI(**client_options)
            if type == "embedding":
                self.embedding_client = client
            else:
                self.client = client
            return client
        except Exception as exc:
            logger.exception("Error initializing OpenAI-compatible client")
            raise LLMClientError(
                f"Error initializing OpenAI-compatible client: {exc}"
            ) from exc

    @override
    def _build_chat_params(self, request: ChatRequest) -> dict[str, Any]:
        params = request.model_dump(
            include={
                "model",
                "messages",
                "temperature",
                "top_p",
                "max_tokens",
                "stop",
                "seed",
                "timeout",
            },
            exclude_none=True,
        )
        response_format = request.response_format
        if response_format.type == "json_object":
            params["response_format"] = {"type": "json_object"}
        elif response_format.type == "json_schema":
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.name,
                    "strict": response_format.strict,
                    "schema": response_format.json_schema,
                },
            }
        params.update(request.provider_options)
        return params

    @override
    def _build_embedding_params(self, request: EmbeddingRequest) -> dict[str, Any]:
        params = request.model_dump(
            include={"model", "input", "dimensions", "timeout"}, exclude_none=True
        )
        params.update(request.provider_options)
        return params

    @override
    async def generate_response(self, request: ChatRequest) -> LLMResponse:
        llm_client = self.client
        if llm_client is None:
            llm_client = self.get_client(type="chat")
            self.client = llm_client
        try:
            response = await llm_client.chat.completions.create(
                **self._build_chat_params(request)
            )
            if not response or not response.choices:
                raise LLMClientError(
                    "No response received from OpenAI-compatible provider"
                )
            return LLMResponse(text=response.choices[0].message.content, raw=response)
        except LLMClientError:
            raise
        except Exception as exc:
            logger.exception(
                "Error generating response from OpenAI-compatible provider"
            )
            raise LLMClientError(
                f"Error generating response from OpenAI-compatible provider: {exc}"
            ) from exc

    @override
    async def generate_embeddings(self, request: EmbeddingRequest) -> list[list[float]]:
        embedding_client = self.embedding_client
        if embedding_client is None:
            embedding_client = self.get_client(type="embedding")
            self.embedding_client = embedding_client
        try:
            response = await embedding_client.embeddings.create(
                **self._build_embedding_params(request)
            )
            if not response or not response.data:
                raise EmbeddingClientError(
                    "No embeddings received from OpenAI-compatible provider"
                )
            return [item.embedding for item in response.data]
        except EmbeddingClientError:
            raise
        except Exception as exc:
            logger.exception(
                "Error generating embeddings from OpenAI-compatible provider"
            )
            raise EmbeddingClientError(
                f"Error generating embeddings from OpenAI-compatible provider: {exc}"
            ) from exc
