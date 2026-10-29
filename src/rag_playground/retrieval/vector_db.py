import hashlib
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, override

import aiofiles
import httpx2
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
    async def create_collection(self, vector_size: int, collection_name: str) -> None:
        pass

    @abstractmethod
    async def is_collection_exists(
        self, collection_name: str
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

    @abstractmethod
    async def create_snapshot(
        self, collection_name: str, snapshot_name: str | None = None
    ) -> str:
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        pass

    @abstractmethod
    async def recover_snapshot(self, collection_name: str, location: str) -> bool:
        pass

    @abstractmethod
    async def save_snapshot_to_file(
        self,
        collection_name: str,
        snapshot_name: str,
        output_dir: str,
        chunk_size: int = 1,
        api_key: str | None = None,
    ) -> str:
        pass

    @abstractmethod
    async def load_snapshot_from_file(
        self,
        collection_name: str,
        snapshot_file_path: str,
        chunk_size: int = 1,
        api_key: str | None = None,
    ) -> bool:
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
        self.http_client: httpx2.AsyncClient | None = None
        self.client = self._initialize_client()
        self._check_embedder()

    @override
    def _initialize_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(url=self._get_url(), api_key=self.api_key)

    def _get_url(self) -> str:
        return f"http://{self.host.strip()}:{self.port}"

    def _get_async_httpx_client(
        self, url: str, headers: dict[str, str]
    ) -> httpx2.AsyncClient:
        self._validate_str_input(url)
        if not isinstance(headers, dict):
            raise TypeError("Headers must be a dictionary.")
        if self.http_client and not self.http_client.is_closed:
            return self.http_client
        self.http_client = httpx2.AsyncClient(base_url=url, headers=headers)
        return self.http_client

    def _check_embedder(self) -> None:
        if not self.embedder:
            raise ValueError("Embedding service is not initialized.")

    @staticmethod
    def _validate_str_input(value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("Input must be a string.")
        if value is None or value.strip() == "":
            raise ValueError("Input must be a non-empty string.")

    @staticmethod
    def _point_id(key: str) -> int:
        return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:15], 16)

    def _get_bm25(self, collection_name: str) -> BM25:
        if collection_name not in self._bm25:
            self._bm25[collection_name] = BM25(k1=self.bm25_k1, b=self.bm25_b)
        return self._bm25[collection_name]

    @override
    async def is_collection_exists(
        self, collection_name: str
    ) -> bool:
        self._validate_str_input(collection_name)
        return await self.client.collection_exists(collection_name)

    @override
    async def create_collection(self, vector_size: int | None, collection_name: str) -> None:
        self._validate_str_input(collection_name)
        if not await self.client.collection_exists(collection_name):
            vector_config = None
            if vector_size is not None:
                if vector_size <= 0 or vector_size > 8192:
                    raise ValueError("Vector size must be a positive integer and not exceed 8192.")
                if not isinstance(vector_size, int):
                    raise TypeError("Vector size must be an integer.")
                vector_config = VectorParams(size=vector_size, distance=Distance.COSINE)
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_config,
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
        self._validate_str_input(collection_name)
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
                await self.create_collection(
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
        self._validate_str_input(collection_name)
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
        self._validate_str_input(collection_name)
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
        self._validate_str_input(collection_name)
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
        self._validate_str_input(collection_name)
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
        self._validate_str_input(collection_name)
        found = await self.client.retrieve(
            collection_name=collection_name, ids=[int(point_id)], with_payload=True
        )
        return found[0].payload if found else None

    @override
    async def create_snapshot(
        self, collection_name: str, snapshot_name: str | None = None
    ) -> str:
        self._validate_str_input(collection_name)
        if not await self.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        try:
            if snapshot_name is None or snapshot_name.strip() == "":
                snapshot_name = f"snapshot_{collection_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            snapshot_description = await self.client.create_snapshot(
                collection_name=collection_name, snapshot_name=snapshot_name
            )
            if (
                snapshot_description is None
                or snapshot_description.name != snapshot_name
            ):
                raise ValueError(f"Snapshot creation failed for '{snapshot_name}'.")
            if snapshot_description.checksum is None:
                raise ValueError(f"Snapshot '{snapshot_name}' has no checksum.")
            logger.info(
                f"Snapshot '{snapshot_name}' created successfully for collection '{collection_name}'."
            )
            return snapshot_name
        except Exception:
            logger.exception("Error creating snapshot")
            raise

    @override
    async def save_snapshot_to_file(
        self,
        collection_name: str,
        snapshot_name: str,
        output_dir: str,
        chunk_size: int = 1,
        api_key: str | None = None,
    ) -> str:
        self._validate_str_input(collection_name)
        self._validate_str_input(snapshot_name)
        self._validate_str_input(output_dir)
        if not isinstance(chunk_size, int):
            raise TypeError("Chunk size must be an integer.")
        if chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer.")
        if api_key is not None and api_key.strip() == "":
            raise ValueError("API key must be a non-empty string if provided.")
        output_dir_path = Path(output_dir)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True, exist_ok=True)
        file_path = output_dir_path / Path(snapshot_name).name
        url = (
            f"{self._get_url()}/collections/{collection_name}/snapshots/{snapshot_name}"
        )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            client = self._get_async_httpx_client(url, headers)
            timeout = httpx2.Timeout(
                connect=10.0, 
                read=60.0*chunk_size, 
                write=60.0*chunk_size, 
                pool=60.0
            )
            async with client.stream(method="GET", url=url, timeout=timeout) as response:
                response.raise_for_status()
                async with aiofiles.open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes(
                        chunk_size=1024 * 1024 * max(1, chunk_size)
                    ):
                        await f.write(chunk)
            logger.info(f"Snapshot '{snapshot_name}' saved to '{file_path}'.")
            return str(file_path)
        except Exception:
            logger.exception("Error saving snapshot to file")
            raise

    @override
    async def load_snapshot_from_file(
        self,
        collection_name: str,
        snapshot_file_path: str,
        chunk_size: int = 1,
        api_key: str | None = None,
    ) -> bool:
        self._validate_str_input(collection_name)
        self._validate_str_input(snapshot_file_path)
        if api_key is not None and api_key.strip() == "":
            raise ValueError("API key must be a non-empty string if provided.")
        url = (
            f"{self._get_url()}/collections/{collection_name}/snapshots/upload"
        )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            client = self._get_async_httpx_client(url, headers)
            timeout = httpx2.Timeout(
                connect=10.0, 
                read=60.0*chunk_size, 
                write=60.0*chunk_size, 
                pool=60.0
            )
            async with client.stream(method="POST", url=url, timeout=timeout) as response:
                async with aiofiles.open(snapshot_file_path, "rb") as f:
                    files = {"snapshot": (
                            Path(snapshot_file_path).name, 
                            await f.read(),
                            "application/octet-stream"
                        )
                    }
                    response = await client.post(url, files=files, timeout=timeout)
                    logger.info(
                        "Snapshot status: '%s', response text: '%s', collection: '%s'",
                        response.status_code,
                        response.text,
                        collection_name,
                    )
                    response.raise_for_status()
            logger.info(
                f"Snapshot from '{snapshot_file_path}' loaded into collection '{collection_name}'."
            )
            return True
        except Exception:
            logger.exception("Error loading snapshot from file")
            return False

    @override
    async def delete_collection(self, collection_name: str) -> bool:
        self._validate_str_input(collection_name)
        if not await self.client.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist.")
            return False
        try:
            await self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully.")
            return True
        except Exception:
            logger.exception(f"Error deleting collection '{collection_name}'")
            return False

    @override
    async def recover_snapshot(self, collection_name: str, location: str) -> bool:
        self._validate_str_input(collection_name)
        if not await self.client.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist.")
            return False
        location = location.strip()
        if not location or location == "":
            raise ValueError("Snapshot location must be a non-empty string.")
        try:
            await self.client.recover_snapshot(
                collection_name=collection_name, location=location
            )
            logger.info(
                f"Collection '{collection_name}' restored successfully from snapshot at '{location}'."
            )
            return True
        except Exception:
            logger.exception(
                f"Error restoring collection '{collection_name}' from snapshot at '{location}'"
            )
            return False
