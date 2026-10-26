from typing import Any, override

from openai import AsyncOpenAI

from rag_playground.core.exceptions import EmbeddingClientError, LLMClientError
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.llm.llm_factory import LLMClient

logger = get_logger(__name__)


class OpenAIClient(LLMClient):
    """Client for OpenAI and providers exposing an OpenAI-compatible API."""

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
    async def generate_response(self, **request_data: Any) -> Any:
        llm_client = self.client
        if llm_client is None:
            llm_client = self.get_client(type="chat")
        try:
            response = await llm_client.chat.completions.create(**request_data)
            if not response or not response.choices:
                raise LLMClientError(
                    "No response received from OpenAI-compatible provider"
                )
            return response
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
    async def generate_embeddings(self, **request_data: Any) -> list[list[float]]:
        embedding_client = self.embedding_client
        if embedding_client is None:
            embedding_client = self.get_client(type="embedding")
        try:
            response = await embedding_client.embeddings.create(**request_data)
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
