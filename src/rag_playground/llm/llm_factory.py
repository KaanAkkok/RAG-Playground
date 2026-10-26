from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar


class LLMClient(ABC):
    def __init__(self) -> None:
        self.client: Any | None = None
        self.embedding_client: Any | None = None

    @abstractmethod
    def get_client(self, type: str | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def generate_response(self, **request_data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def generate_embeddings(self, **request_data: Any) -> list[list[float]]:
        raise NotImplementedError


class LLMFactory:
    _providers: ClassVar[dict[str, Callable[..., LLMClient]]] = {}
    _defaults_loaded: ClassVar[bool] = False

    @classmethod
    def register(cls, provider: str, client_class: Callable[..., LLMClient]) -> None:
        provider_name = provider.strip().lower()
        if not provider_name:
            raise ValueError("LLM provider name cannot be empty.")
        cls._providers[provider_name] = client_class

    @classmethod
    def create(cls, provider: str | None = None, **client_options: Any) -> LLMClient:
        cls._load_default_providers()
        from rag_playground.core.settings import settings

        provider_name = (provider or settings.LLM_PROVIDER).strip().lower()
        try:
            client_class = cls._providers[provider_name]
        except KeyError as exc:
            supported = ", ".join(sorted(cls._providers))
            raise ValueError(
                f"Unsupported LLM provider '{provider_name}'. "
                f"Supported providers: {supported}"
            ) from exc
        return client_class(**client_options)

    @classmethod
    def _load_default_providers(cls) -> None:
        if cls._defaults_loaded:
            return

        from rag_playground.llm.azure_client import AzureOpenAIClient
        from rag_playground.llm.openai_client import OpenAIClient

        cls.register("azure", AzureOpenAIClient)
        cls.register("azure_openai", AzureOpenAIClient)
        cls.register("openai", OpenAIClient)
        cls.register("openai_compatible", OpenAIClient)
        cls._defaults_loaded = True
