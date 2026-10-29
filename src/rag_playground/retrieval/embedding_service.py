from abc import ABC, abstractmethod
from typing import override

from rag_playground.core.exceptions import EmbeddingError
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.llm.llm_factory import LLMClient
from rag_playground.schemas.llm_client_schema import EmbeddingRequest

logger = get_logger(__name__)


class EmbeddingService(ABC):

    @abstractmethod
    async def embed_texts(self, texts: list[str] | str) -> list[list[float]]:
        pass


class LLMEmbeddingService(EmbeddingService):
    def __init__(
        self,
        client: LLMClient,
        deployment_name: str = settings.llm_models.embedding,
        embed_batch_size: int = settings.EMBED_BATCH_SIZE,
    ):
        self.deployment_name = deployment_name
        self.embed_batch_size = embed_batch_size
        self.validate_config()
        self.client = client

    def validate_config(self) -> None:
        if not self.deployment_name or not isinstance(self.deployment_name, str):
            raise ValueError(
                f"Embedding model for provider '{settings.LLM_PROVIDER}' "
                "cannot be None or empty. Please check your settings."
            )
        if self.embed_batch_size is None or self.embed_batch_size <= 0:
            raise ValueError("Embed batch size must be a positive integer.")

    async def _call_embedding_api(self, inputs: list[str]) -> list[list[float]]:
        if not inputs or len(inputs) == 0:
            raise ValueError("Input list for embedding cannot be empty.")

        try:
            return await self.client.generate_embeddings(
                EmbeddingRequest(model=self.deployment_name, input=inputs)
            )
        except Exception as e:
            logger.exception("Error generating embeddings")
            raise EmbeddingError(f"Error generating embeddings: {e}") from e

    @override
    async def embed_texts(self, texts: list[str] | str) -> list[list[float]]:
        if not texts or len(texts) == 0:
            raise ValueError("Input texts list cannot be empty.")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            raise TypeError("Input texts must be a list of strings.")
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.embed_batch_size):
            inputs = texts[i : i + self.embed_batch_size]
            batch_vectors = await self._call_embedding_api(inputs)
            vectors.extend(batch_vectors)
        return vectors
