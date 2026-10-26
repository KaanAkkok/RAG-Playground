from typing import Any, override

from openai import AsyncAzureOpenAI

from rag_playground.core.exceptions import EmbeddingClientError, LLMClientError
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.llm.llm_factory import LLMClient

logger = get_logger(__name__)


class AzureOpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key if api_key is not None else settings.AZURE_OPENAI_API_KEY
        self.azure_endpoint = (
            azure_endpoint
            if azure_endpoint is not None
            else settings.AZURE_OPENAI_ENDPOINT
        )
        self.api_version = (
            api_version
            if api_version is not None
            else settings.AZURE_OPENAI_API_VERSION
        )
        self.timeout = (
            timeout if timeout is not None else settings.AZURE_OPENAI_CHAT_TIMEOUT
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else settings.AZURE_OPENAI_CHAT_MAX_RETRIES
        )

    @override
    def get_client(self, type: str | None = None) -> AsyncAzureOpenAI:
        try:
            client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                timeout=self.timeout if type == "chat" else None,
                max_retries=self.max_retries if self.max_retries is not None else 3,
            )
            if type == "embedding":
                self.embedding_client = client
            else:
                self.client = client
            return client
        except LLMClientError as e:
            logger.exception("Error initializing Azure OpenAI client")
            raise LLMClientError(f"Error initializing Azure OpenAI client: {e}") from e

    @override
    async def generate_response(self, **request_data: Any) -> Any:
        llm_client = self.client
        if llm_client is None:
            llm_client = self.get_client(type="chat")
        try:
            response = await llm_client.chat.completions.create(**request_data)
            if not response or not response.choices or len(response.choices) == 0:
                raise LLMClientError("No response received from Azure OpenAI")
            return response
        except LLMClientError:
            raise
        except Exception as e:
            logger.exception("Error generating response from Azure OpenAI")
            raise LLMClientError(
                f"Error generating response from Azure OpenAI: {e}"
            ) from e

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
