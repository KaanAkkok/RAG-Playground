import asyncio
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, override

from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag_playground.core.exceptions import MemoryServiceError
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.llm.llm_factory import LLMClient
from rag_playground.models.vector_model import VectorRecord
from rag_playground.prompts.prompts import MEMORY_SYSTEM_PROMPT
from rag_playground.retrieval.vector_db import VectorDB
from rag_playground.schemas.llm_client_schema import (
    ChatRequest,
    LLMMessage,
    ResponseFormat,
)
from rag_playground.schemas.memory_schema import (
    LLM_MEMORY_RESPONSE_FORMAT,
    MemoryExtraction,
    MemoryResponse,
)

logger = get_logger(__name__)


class MemoryIService(ABC):

    @abstractmethod
    def reset_history(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_short_term_memory(self, role: str, content: str):
        raise NotImplementedError

    @abstractmethod
    def get_short_term_memories(self) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    async def extract_from_memories(
        self, messages: list[dict[str, str]]
    ) -> list[MemoryResponse]:
        raise NotImplementedError

    @abstractmethod
    async def store_memories(self, memories: list[MemoryResponse]) -> int:
        raise NotImplementedError

    @abstractmethod
    def schedule_consolidation(self, force: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_long_term_memories(
        self,
        query: str = "",
        filters: dict[str, Any] | None = None,
        limit: int = settings.LONG_TERM_MEMORY_SEARCH_LIMIT,
    ) -> list[MemoryResponse]:
        raise NotImplementedError


class MemoryService(MemoryIService):
    def __init__(
        self,
        vector_db: VectorDB,
        client: LLMClient,
        user_id: str = "default",
        max_history_tokens: int = settings.HISTORY_MAX_TOKENS,
        max_history_messages: int = settings.HISTORY_MAX_MESSAGES,
        long_term_batch_size: int = settings.LONG_TERM_MEMORY_BATCH_SIZE,
        long_term_confidence_threshold: float = settings.LONG_TERM_MEMORY_CONFIDENCE_THRESHOLD,
        long_term_min_score: float = settings.LONG_TERM_MEMORY_MIN_SCORE,
        collection_name: str = settings.QDRANT_MEMORY_COLLECTION_NAME,
        temperature: float = settings.ASSISTANT_CHAT_TEMPERATURE,
        memory_deployment: str = settings.llm_models.memory,
        system_prompt: str = MEMORY_SYSTEM_PROMPT,
        memory_response_format: dict = LLM_MEMORY_RESPONSE_FORMAT,
    ):
        self.user_id = user_id  # long-term memories are stored and searched per user
        self.max_history_tokens = max_history_tokens
        self.max_history_messages = max_history_messages
        self.long_term_batch_size = long_term_batch_size
        self.long_term_confidence_threshold = long_term_confidence_threshold
        self.long_term_min_score = long_term_min_score
        self.storage = vector_db
        self.collection_name = collection_name
        self.memory_extractor = client
        self.temperature = temperature
        self.memory_deployment = memory_deployment
        self.system_prompt = system_prompt
        self.memory_response_format = ResponseFormat(**memory_response_format)
        self.short_term_memory: list[dict[str, str]] = []
        self._pending_messages: list[dict[str, str]] = []
        self._background_tasks: set[asyncio.Task] = set()

    @override
    def reset_history(self) -> None:
        self.schedule_consolidation(force=True)
        self.short_term_memory = []

    def _trim_history(self) -> list[dict]:
        if self.max_history_messages <= 0:
            return []
        st_history = None
        if len(self.short_term_memory) <= self.max_history_messages * 2:
            st_history = self.short_term_memory
        else:
            st_history = list(self.short_term_memory or [])[
                -self.max_history_messages * 2 :
            ]
        if not st_history or len(st_history) == 0:
            return []
        total_tokens = sum(len(msg["content"].split()) for msg in st_history)
        while total_tokens > self.max_history_tokens and st_history:
            removed_msg = st_history.pop(0)
            total_tokens -= len(removed_msg["content"].split())
        return st_history

    @override
    def update_short_term_memory(self, role: str, content: str):
        message = {"role": role, "content": content}
        self.short_term_memory.append(message)
        self.short_term_memory = self._trim_history()
        self._pending_messages.append(message)

    @override
    def get_short_term_memories(self) -> list[dict[str, str]]:
        return self._trim_history()

    @override
    async def extract_from_memories(
        self, messages: list[dict[str, str]]
    ) -> list[MemoryResponse]:
        if not messages:
            return []
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        request = ChatRequest(
            model=self.memory_deployment,
            messages=[
                LLMMessage(role="system", content=self.system_prompt),
                LLMMessage(role="user", content=f"Konuşma:\n{transcript}"),
            ],
            temperature=self.temperature,
            response_format=self.memory_response_format,
        )
        try:
            response = await self.memory_extractor.generate_response(request)
        except Exception as e:
            logger.exception("Error during memory extraction")
            raise MemoryServiceError(f"Error during memory extraction: {e}") from e
        content = response.text
        if not content:
            return []
        try:
            return MemoryExtraction.model_validate_json(content).memories
        except Exception as e:
            logger.exception("Error parsing memory extraction response")
            raise MemoryServiceError(
                f"Error parsing memory extraction response: {e}"
            ) from e

    @override
    async def store_memories(self, memories: list[MemoryResponse]) -> int:
        created_at = datetime.now(UTC).isoformat()
        records = [
            VectorRecord(
                text=memory.content,
                payload={
                    "user_id": self.user_id,
                    "memory_type": memory.memory_type,
                    "inference": memory.inference,
                    "confidence": memory.confidence,
                    "created_at": created_at,
                },
                key=f"{self.user_id}|{memory.memory_type}|{memory.content.strip().lower()}",
            )
            for memory in memories
            if memory.memory_type not in ("no_extraction", "unknown")
            and memory.content.strip()
            and memory.confidence >= self.long_term_confidence_threshold
        ]
        if not records:
            return 0
        try:
            stored = await self.storage.store(
                records=records, collection_name=self.collection_name
            )
            return len(stored)
        except Exception as e:
            logger.exception("Error storing memory in vector database")
            raise MemoryServiceError(
                f"Error storing memory in vector database: {e}"
            ) from e

    async def _consolidate(self, messages: list[dict[str, str]]) -> int:
        try:
            memories = await self.extract_from_memories(messages)
            stored = await self.store_memories(memories)
            logger.info(
                f"Memory consolidation: {len(messages)} messages -> {len(memories)} extracted, {stored} stored"
            )
            return stored
        except Exception:
            logger.exception("Error consolidating long-term memory")
            return 0

    @override
    def schedule_consolidation(self, force: bool = False) -> None:
        if not self._pending_messages:
            return
        if not force and len(self._pending_messages) < self.long_term_batch_size:
            return
        messages, self._pending_messages = self._pending_messages, []
        try:
            task = asyncio.get_running_loop().create_task(self._consolidate(messages))
        except RuntimeError:
            logger.debug(
                "No running event loop, consolidating long-term memory asynchronously."
            )
            asyncio.run(self._consolidate(messages))
            return
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def wait_for_consolidation(self) -> None:
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

    async def count_memories(self, filters: dict[str, Any] | None = None) -> int:
        try:
            count = await self.storage.count(
                collection_name=self.collection_name,
                filters={"user_id": self.user_id, **(filters or {})},
            )
            return count
        except Exception as e:
            logger.exception("Error counting memories in vector database")
            raise MemoryServiceError(
                f"Error counting memories in vector database: {e}"
            ) from e

    @override
    async def get_long_term_memories(
        self,
        query: str = "",
        filters: dict[str, Any] | None = None,
        limit: int = settings.LONG_TERM_MEMORY_SEARCH_LIMIT,
    ) -> list[MemoryResponse]:
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            return []
        try:
            hits = await self.storage.dense_search(
                query=cleaned_query,
                collection_name=self.collection_name,
                limit=limit,
                query_filter=Filter(
                    must=[
                        FieldCondition(key=key, match=MatchValue(value=value))
                        for key, value in {
                            "user_id": self.user_id,
                            **(filters or {}),
                        }.items()
                    ]
                ),
            )
            memories = []
            for hit in hits:
                if hit.score < self.long_term_min_score:
                    continue
                memory = MemoryResponse(
                    memory_type=hit.payload.get("memory_type", "unknown"),
                    content=hit.payload.get("text", ""),
                    inference=hit.payload.get("inference"),
                    confidence=hit.payload.get("confidence", 0.0),
                )
                memories.append(memory)
            return memories
        except Exception as e:
            logger.exception("Error retrieving long-term memories from vector database")
            raise MemoryServiceError(
                f"Error retrieving long-term memories from vector database: {e}"
            ) from e
