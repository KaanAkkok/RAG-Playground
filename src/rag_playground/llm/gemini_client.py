from typing import Any, override

from google import genai
from google.genai import types
from google.genai.client import AsyncClient

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


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.timeout = timeout if timeout is not None else settings.GEMINI_CHAT_TIMEOUT
        self.max_retries = (
            max_retries if max_retries is not None else settings.GEMINI_CHAT_MAX_RETRIES
        )
        self.client = None
        self.embedding_client = None

    @override
    def get_client(self, type: str | None = None) -> AsyncClient:
        try:
            client_options: dict[str, Any] = {
                "api_key": self.api_key,
                "http_options": types.HttpOptions(
                    # Gemini SDK timeout'u milisaniye bekler, settings saniye tutuyor
                    timeout=self.timeout * 1000 if type == "chat" else None,
                    retry_options=types.HttpRetryOptions(
                        attempts=self.max_retries + 1,
                    ),
                ),
            }
            client = genai.Client(**client_options)
            client = client.aio
            if type == "embedding":
                self.embedding_client = client
            else:
                self.client = client
            return client
        except Exception as exc:
            logger.exception("Error initializing Gemini-compatible client")
            raise LLMClientError(
                f"Error initializing Gemini-compatible client: {exc}"
            ) from exc

    @override
    def _build_chat_params(self, request: ChatRequest) -> dict[str, Any]:
        # Gemini system mesajlarını contents içinde değil, system_instruction olarak alır
        system_instruction = "\n\n".join(
            m.content for m in request.messages if m.role == "system"
        )
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in request.messages
            if m.role != "system"
        ]
        if not contents:
            raise LLMClientError("Gemini requires at least one user/assistant message")

        config_options: dict[str, Any] = {
            "system_instruction": system_instruction or None,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_output_tokens": request.max_tokens,
            "stop_sequences": request.stop,
            "seed": request.seed,
        }
        response_format = request.response_format
        if response_format.type != "text":
            config_options["response_mime_type"] = "application/json"
        if response_format.type == "json_schema":
            # Gemini'de strict bayrağı yok; şema verildiğinde çıktı her zaman ona zorlanır
            config_options["response_json_schema"] = response_format.json_schema
        if request.timeout is not None:
            config_options["http_options"] = types.HttpOptions(
                timeout=int(request.timeout * 1000)
            )
        config_options = {k: v for k, v in config_options.items() if v is not None}
        config_options.update(request.provider_options)

        return {
            "model": request.model,
            "contents": contents,
            "config": types.GenerateContentConfig(**config_options),
        }

    @override
    def _build_embedding_params(self, request: EmbeddingRequest) -> dict[str, Any]:
        config_options: dict[str, Any] = {
            "output_dimensionality": request.dimensions,
            "task_type": request.task_type,
            "http_options": types.HttpOptions(
                timeout=int(request.timeout * 1000) if request.timeout else None
            ),
        }
        config_options = {k: v for k, v in config_options.items() if v is not None}
        config_options.update(request.provider_options)
        raw_inputs = request.input
        contents = [
            types.Content(
                parts=[types.Part.from_text(text=text)]
            )
            for text in raw_inputs
        ]
        return {
            "model": request.model,
            "contents": contents,
            "config": types.EmbedContentConfig(**config_options),
        }

    @override
    async def generate_response(self, request: ChatRequest) -> LLMResponse:
        llm_client = self.client
        if llm_client is None:
            llm_client = self.get_client(type="chat")
            self.client = llm_client
        try:
            response = await llm_client.models.generate_content(
                **self._build_chat_params(request)
            )
            if not response or not response.candidates:
                raise LLMClientError(
                    "No response received from Gemini-compatible provider"
                )
            return LLMResponse(text=response.text, raw=response)
        except LLMClientError:
            raise
        except Exception as exc:
            logger.exception(
                "Error generating response from Gemini-compatible provider"
            )
            raise LLMClientError(
                f"Error generating response from Gemini-compatible provider: {exc}"
            ) from exc

    @override
    async def generate_embeddings(self, request: EmbeddingRequest) -> list[list[float]]:
        embedding_client = self.embedding_client
        if embedding_client is None:
            embedding_client = self.get_client(type="embedding")
            self.embedding_client = embedding_client
        try:
            response = await embedding_client.models.embed_content(
                **self._build_embedding_params(request)
            )
            if not response or not response.embeddings:
                raise EmbeddingClientError(
                    "No embeddings received from Gemini-compatible provider"
                )
            embeddings: list[list[float]] = []
            for item in response.embeddings:
                if item.values is None or len(item.values) == 0:
                    raise EmbeddingClientError(
                        "Gemini returned an embedding without valid values"
                    )
                embeddings.append(item.values)
            expected_count = (
                len(request.input) if isinstance(request.input, list) else 1
            )
            if len(embeddings) != expected_count:
                raise EmbeddingClientError(
                    "Embedding count mismatch: "
                    f"expected={expected_count}, got={len(embeddings)}"
                )
            return embeddings
        except EmbeddingClientError:
            raise
        except Exception as exc:
            logger.exception(
                "Error generating embeddings from Gemini-compatible provider"
            )
            raise EmbeddingClientError(
                f"Error generating embeddings from Gemini-compatible provider: {exc}"
            ) from exc
