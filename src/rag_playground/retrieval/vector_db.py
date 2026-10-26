import hashlib
import uuid
from abc import ABC, abstractmethod
from typing import Any, override

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.models.search_hit_model import SearchHit
from rag_playground.models.vector_model import VectorRecord
from rag_playground.retrieval.bm25 import BM25
from rag_playground.retrieval.embedding_service import EmbeddingService
from rag_playground.retrieval.rrf import reciprocal_rank_fusion

logger = get_logger(__name__)


class VectorDB(ABC):

    @abstractmethod
    def _initialize_client(self) -> Any:
        pass

    @abstractmethod
    async def _ensure_collection(self, vector_size: int, collection_name: str) -> None:
        pass

    @abstractmethod
    async def is_collection_exists(
        self, collection_name: str = settings.QDRANT_KNOWLEDGE_COLLECTION_NAME
    ) -> bool:
        pass

    @abstractmethod
    async def store(
        self,
        records: list[VectorRecord],
        collection_name: str,
        vectors: list[list[float]] | None = None,
    ) -> list[str]:
        pass

    @abstractmethod
    async def count(
        self, collection_name: str, filters: dict[str, Any] | None = None
    ) -> int:
        pass

    @abstractmethod
    async def dense_search(
        self,
        query: str,
        limit: int,
        collection_name: str,
        query_filter: Filter | None = None,
    ) -> list[SearchHit]:
        pass

    @abstractmethod
    async def search(
        self, query: str, collection_name: str, limit: int, candidate_k: int
    ) -> list[SearchHit]:
        pass


class AsyncQdrantVectorDB(VectorDB):
    def __init__(
        self,
        embedder: EmbeddingService,
        host: str = settings.QDRANT_HOST,
        port: int = settings.QDRANT_PORT,
        api_key: str | None = settings.QDRANT_API_KEY,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        upsert_batch_size: int = settings.VECTORDB_UPSERT_BATCH_SIZE,
    ):
        self.host = host
        self.port = port
        self.api_key = api_key or None
        self.embedder = embedder
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.upsert_batch_size = upsert_batch_size
        self._bm25: dict[str, BM25] = {}
        self._bm25_loaded: set[str] = set()
        self.client = self._initialize_client()
        self._check_embedder()

    @override
    def _initialize_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(
            url=f"http://{self.host.strip()}:{self.port}", api_key=self.api_key
        )

    def _check_embedder(self) -> None:
        if not self.embedder:
            raise ValueError("Embedding service is not initialized.")

    @staticmethod
    def _validate_collection_name(collection_name: str) -> None:
        if collection_name is None or collection_name.strip() == "":
            raise ValueError("Collection name must be a non-empty string.")

    @staticmethod
    def _point_id(key: str) -> int:
        return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:15], 16)

    def _get_bm25(self, collection_name: str) -> BM25:
        if collection_name not in self._bm25:
            self._bm25[collection_name] = BM25(k1=self.bm25_k1, b=self.bm25_b)
        return self._bm25[collection_name]

    @override
    async def is_collection_exists(
        self, collection_name: str = settings.QDRANT_KNOWLEDGE_COLLECTION_NAME
    ) -> bool:
        self._validate_collection_name(collection_name)
        return await self.client.collection_exists(collection_name)

    @override
    async def _ensure_collection(self, vector_size: int, collection_name: str) -> None:
        if vector_size <= 0:
            raise ValueError("Vector size must be a positive integer.")
        self._validate_collection_name(collection_name)
        if not await self.client.collection_exists(collection_name):
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    @override
    async def store(
        self,
        records: list[VectorRecord],
        collection_name: str,
        vectors: list[list[float]] | None = None,
    ) -> list[str]:
        if not records:
            return []
        self._validate_collection_name(collection_name)
        bm25 = self._get_bm25(collection_name)
        total = len(records)
        log_progress = logger.info if total > self.upsert_batch_size else logger.debug
        added: list[str] = []
        for i in range(0, total, self.upsert_batch_size):
            batch_records = records[i : i + self.upsert_batch_size]
            if vectors is not None:
                batch_vectors = vectors[i : i + self.upsert_batch_size]
            else:
                batch_vectors = await self.embedder.embed_texts(
                    [r.text for r in batch_records]
                )
            if i == 0:
                await self._ensure_collection(
                    vector_size=len(batch_vectors[0]), collection_name=collection_name
                )
            batch = []
            for r, vector in zip(batch_records, batch_vectors):
                point_id = self._point_id(r.key or uuid.uuid4().hex)
                batch.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={**r.payload, "text": r.text},
                    )
                )
                bm25.add(str(point_id), r.text)
            try:
                await self.client.upsert(
                    collection_name=collection_name, points=batch, wait=True
                )
                added.extend(str(point.id) for point in batch)
            except Exception:
                logger.exception("Error upserting points to Qdrant")
                for point in batch:
                    bm25._remove(str(point.id))
            done = i + len(batch_records)
            log_progress(
                f"Storing into '{collection_name}': {done}/{total} ({done / total:.0%})"
            )
        return added

    @override
    async def count(
        self, collection_name: str, filters: dict[str, Any] | None = None
    ) -> int:
        self._validate_collection_name(collection_name)
        if not await self.client.collection_exists(collection_name):
            return 0
        count_filter = None
        if filters:
            count_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )
        try:
            result = await self.client.count(
                collection_name=collection_name,
                count_filter=count_filter,
                exact=True,
            )
            return result.count
        except Exception:
            logger.exception("Error counting points")
            return 0

    async def rebuild_bm25(self, collection_name: str) -> int:
        self._validate_collection_name(collection_name)
        if not await self.client.collection_exists(collection_name):
            return 0
        bm25 = self._get_bm25(collection_name)
        offset = None
        try:
            while True:
                points, offset = await self.client.scroll(
                    collection_name=collection_name,
                    offset=offset,
                    limit=self.upsert_batch_size * 2,
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    break
                for point in points:
                    bm25.add(str(point.id), (point.payload or {}).get("text", ""))
                if offset is None:
                    break
            self._bm25_loaded.add(collection_name)
            return len(bm25)
        except Exception:
            logger.exception("Error rebuilding BM25 index")
            return 0

    @override
    async def dense_search(
        self,
        query: str,
        limit: int,
        collection_name: str,
        query_filter: Filter | None = None,
    ) -> list[SearchHit]:
        self._validate_collection_name(collection_name)
        if not await self.client.collection_exists(collection_name):
            return []
        vector = (await self.embedder.embed_texts([query]))[0]
        try:
            results = await self.client.query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        except Exception:
            logger.exception("Error querying Qdrant")
            return []
        return [
            SearchHit(id=str(p.id), score=p.score, payload=p.payload or {})
            for p in results.points
        ]

    @override
    async def search(
        self,
        query: str,
        collection_name: str,
        limit: int = settings.FUSED_SEARCH_CANDIDATES,
        candidate_k: int = settings.SEARCH_CANDIDATES,
    ) -> list[SearchHit]:
        self._validate_collection_name(collection_name)
        if not await self.client.collection_exists(collection_name):
            return []
        if not query or not query.strip():
            return []
        if collection_name not in self._bm25_loaded:
            await self.rebuild_bm25(collection_name)

        try:
            dense_results = await self.dense_search(
                query, limit=candidate_k, collection_name=collection_name
            )
            payloads = {hit.id: hit.payload for hit in dense_results}
            dense_results = [(hit.id, hit.score) for hit in dense_results]
            sparse_results = self._get_bm25(collection_name).search(
                query, limit=candidate_k
            )
        except Exception:
            logger.exception("Error during search")
            return []

        fused_rankings = reciprocal_rank_fusion(sparse_results, dense_results, k=60)

        results: list[SearchHit] = []
        for point_id, score in fused_rankings[:limit]:
            payload = payloads.get(point_id) or await self._fetch_payload(
                point_id, collection_name
            )
            if payload is None:
                continue
            results.append(SearchHit(id=point_id, score=score, payload=payload))
        return results

    async def _fetch_payload(self, point_id: str, collection_name: str) -> dict | None:
        self._validate_collection_name(collection_name)
        found = await self.client.retrieve(
            collection_name=collection_name, ids=[int(point_id)], with_payload=True
        )
        return found[0].payload if found else None
