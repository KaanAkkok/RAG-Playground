import json
from dataclasses import asdict, fields
from typing import Any

from rag_playground.core.exceptions import RetrievalServiceError
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.llm.llm_factory import LLMClient
from rag_playground.models.citation_model import Citation
from rag_playground.models.search_hit_model import SearchHit
from rag_playground.models.vector_model import VectorRecord
from rag_playground.prompts.prompts import RERANK_SYSTEM_PROMPT
from rag_playground.retrieval.vector_db import VectorDB
from rag_playground.schemas.llm_client_schema import (
    ChatRequest,
    LLMMessage,
    ResponseFormat,
)
from rag_playground.schemas.rerank_schema import (
    LLM_RERANK_RESPONSE_FORMAT,
    RerankResponse,
)

logger = get_logger(__name__)


class RetrievalService:
    def __init__(
        self,
        vector_db: VectorDB,
        client: LLMClient,
        reranker_active: bool = True,
        reranker: str = settings.llm_models.reranker,
        reranker_system_prompt: str = RERANK_SYSTEM_PROMPT,
        temperature: float = settings.RERANKER_TEMPERATURE,
        rerank_passage_chars: int = settings.RERANK_PASSAGE_CHARS,
        min_rerank_score: int = settings.MIN_RERANK_SCORE,
        response_format: dict = LLM_RERANK_RESPONSE_FORMAT,
        timeout: int = settings.LLM_RESPONSE_TIMEOUT,
    ):
        self.retriever = client
        self.vector_db = vector_db
        self.reranker_active = reranker_active
        self.reranker = reranker
        self.timeout = timeout
        self.reranker_system_prompt = reranker_system_prompt
        self.temperature = temperature
        self.rerank_passage_chars = rerank_passage_chars
        self.min_rerank_score = min_rerank_score
        self.response_format = ResponseFormat(**response_format)

    @staticmethod
    def _citation_key(c: Citation) -> str:
        return f"{c.doc_title}|{c.page}|{c.paragraph}|{c.sentences}"

    @staticmethod
    def _to_record(c: Citation) -> VectorRecord:
        payload = asdict(c)
        for runtime_field in ("point_id", "rrf_score", "rerank_score"):
            payload.pop(runtime_field, None)
        return VectorRecord(
            text=c.text, payload=payload, key=RetrievalService._citation_key(c)
        )

    @staticmethod
    def _to_citation(hit: SearchHit) -> Citation:
        citation_fields = {f.name for f in fields(Citation)}
        citation = Citation(
            **{k: v for k, v in hit.payload.items() if k in citation_fields}
        )
        citation.point_id = hit.id
        citation.rrf_score = hit.score
        return citation

    async def index_citations(
        self,
        citations: list[Citation],
        collection_name: str = settings.VECTORDB_KNOWLEDGE_COLLECTION_NAME,
    ) -> int:
        if not citations:
            return 0
        point_ids = await self.vector_db.store(
            [self._to_record(c) for c in citations],
            collection_name=collection_name,
        )
        for c, point_id in zip(citations, point_ids):
            c.point_id = point_id
        return len(point_ids)

    async def is_collection_exists(
        self, collection_name: str = settings.VECTORDB_KNOWLEDGE_COLLECTION_NAME
    ) -> bool:
        try:
            return await self.vector_db.is_collection_exists(collection_name)
        except RetrievalServiceError as e:
            logger.error(
                f"Error checking if collection '{collection_name}' exists: {e}"
            )
            raise RetrievalServiceError(
                f"Failed to check collection existence for '{collection_name}': {e}"
            )

    async def count(
        self,
        filter_dict: dict[str, Any],
        collection_name: str = settings.VECTORDB_KNOWLEDGE_COLLECTION_NAME,
    ) -> int:
        try:
            count = await self.vector_db.count(
                collection_name=collection_name,
                filters=filter_dict,
            )
            return count
        except RetrievalServiceError as e:
            raise RetrievalServiceError(f"Failed to count documents: {e}")

    async def _get_chunks(
        self,
        query: str,
        top_k: int = settings.TOP_K_CANDIDATES,
        limit: int = settings.FUSED_SEARCH_CANDIDATES,
        candidate_k: int = settings.SEARCH_CANDIDATES,
        collection_name: str = settings.VECTORDB_KNOWLEDGE_COLLECTION_NAME,
    ) -> list[Citation]:
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            return []
        try:
            hits = await self.vector_db.search(
                query=cleaned_query,
                collection_name=collection_name,
                limit=limit,
                candidate_k=candidate_k,
            )
            candidates = [self._to_citation(hit) for hit in hits]
        except Exception:
            logger.exception("Error during search")
            return []
        if not self.reranker_active or not candidates:
            return candidates[:top_k]
        return await self._llm_reranker(cleaned_query, candidates, top_k=top_k)

    async def _llm_reranker(
        self,
        query: str,
        candidates: list[Citation],
        top_k: int = settings.TOP_K_CANDIDATES,
    ) -> list[Citation]:
        if not candidates:
            return []
        passages = [
            {"id": index, "text": citation.text[: self.rerank_passage_chars]}
            for index, citation in enumerate(candidates)
        ]
        prompt = (
            f"Sorgu:\n{query}\n\n"
            "Pasajlar (JSON):\n"
            f"{json.dumps(passages, ensure_ascii=False)}"
        )
        messages = [
            LLMMessage(role="system", content=self.reranker_system_prompt),
            LLMMessage(role="user", content=prompt),
        ]
        request = ChatRequest(
            model=self.reranker,
            temperature=self.temperature,
            response_format=self.response_format,
            messages=messages,
            timeout=self.timeout,
        )
        try:
            response = await self.retriever.generate_response(request)
            raw = response.text or "{}"
            parsed = RerankResponse.model_validate_json(raw)
            score_map: dict[int, float] = {}
            for item in parsed.scores:
                index = item.id
                score = item.score
                if 0 <= index < len(candidates):
                    score_map[index] = score
            scored: list[Citation] = []
            for index, citation in enumerate(candidates):
                score = score_map.get(index, 0.0)
                citation.rerank_score = score
                if score >= self.min_rerank_score:
                    scored.append(citation)
            return sorted(
                scored,
                key=lambda citation: citation.rerank_score,
                reverse=True,
            )[:top_k]
        except Exception:
            logger.exception("Error during reranking")
            return candidates[:top_k]

    async def retrieve(
        self, query: str, top_k: int = settings.TOP_K_CANDIDATES
    ) -> list[Citation]:
        return await self._get_chunks(query=query, top_k=top_k)
