import json
from pathlib import Path

import aiofiles

from rag_playground.core.logger import get_logger
from rag_playground.core.settings import PROJECT_ROOT, settings
from rag_playground.ingestion.chunker import DocumentChunker
from rag_playground.ingestion.document_io import DocumentIO
from rag_playground.ingestion.extractor import DocumentExtractor
from rag_playground.models.file_type_model import FileType
from rag_playground.retrieval.retrieval_service import RetrievalService


logger = get_logger(__name__)


def _resolve_document_path(document_path: Path) -> Path:
    if document_path.is_absolute():
        resolved_path = document_path
    elif document_path.is_file():
        resolved_path = document_path.resolve()
    else:
        resolved_path = (Path(PROJECT_ROOT) / document_path).resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Belge bulunamadı: {document_path}. "
            f"Çalışma dizini: {Path.cwd()}. "
            f"Aranan yol: {resolved_path}"
        )
    return resolved_path


async def save_citations_to_json(citations: list, output_path: Path):
    citations_dict = [citation.__dict__ for citation in citations]
    if output_path.exists():
        async with aiofiles.open(output_path, "r", encoding="utf-8") as f:
            content = await f.read()
            existing_citations = json.loads(content) if content.strip() else []
    else:
        existing_citations = []
    existing_citations.extend(citations_dict)
    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(existing_citations, ensure_ascii=False, indent=4))


async def init_vector_db(
    document_io: DocumentIO,
    extractor: DocumentExtractor,
    chunker: DocumentChunker,
    retriever: RetrievalService,
    document_paths: list[Path],
    collection_name: str = settings.QDRANT_KNOWLEDGE_COLLECTION_NAME,
    save_citations: bool = True,
    output_json_path: Path | None = None,
):
    logger.info("Indexleme işlemi başlatılıyor...")
    indexed_count = 0
    chunks_count = 0
    for document_path in document_paths:
        document_path = _resolve_document_path(document_path)
        document_name = f"{document_path.stem}.md"
        if (
            await retriever.count(
                filter_dict={"document": document_name},
                collection_name=collection_name,
            )
            > 0
        ):
            logger.info(f"Dosya zaten indexlenmiş: {document_name}")
            continue
        else:
            logger.info(f"Dosya indexleniyor: {document_name}")
        filename = document_path.stem
        content = document_path.read_bytes()
        saved_file_path = await document_io.save_document(
            content=content,
            filename=filename,
            save_dir=Path(settings.RAW_DOCUMENTS_DIR),
            file_type=FileType.PDF,
        )

        content = extractor.extract(file_path=saved_file_path)
        if not content or not content.strip():
            raise ValueError(f"Dosyadan metin çıkarılamadı: {saved_file_path}")

        saved_extracted_file_path = await document_io.save_document(
            content=content,
            filename=filename,
            save_dir=Path(settings.SAVED_DOCUMENTS_DIR),
            file_type=FileType.MD,
        )
        logger.info(f"Dosya kaydedildi: {saved_extracted_file_path}")

        chunks = await chunker.chunk(
            content=content, document_path=saved_extracted_file_path
        )
        if save_citations:
            if output_json_path is None:
                raise ValueError(
                    "output_json_path must be provided when save_citations is True"
                )
            await save_citations_to_json(chunks, output_json_path)
        logger.info(
            f"{saved_extracted_file_path.stem} için toplam parça sayısı: {len(chunks)}"
        )

        count = await retriever.index_citations(
            citations=chunks, collection_name=collection_name
        )
        indexed_count += count
        chunks_count += len(chunks)
        logger.info(
            f"{saved_extracted_file_path.stem} için toplam eklenen kayıt sayısı: {count}"
        )

    logger.info(f"total chunks: {chunks_count}")
    logger.info(f"total indexed: {indexed_count}")
    logger.info("Indexleme işlemi tamamlandı.")

# =============================================================================================

import asyncio

from rag_playground.core.settings import settings
from rag_playground.llm.llm_factory import LLMFactory
from rag_playground.retrieval.embedding_service import AzureEmbeddingService
from rag_playground.retrieval.vector_db import AsyncQdrantVectorDB

async def initilize_vector_db():
    client = LLMFactory.create()
    document_io = DocumentIO()
    extractor = DocumentExtractor()
    embedder = AzureEmbeddingService(client=client)
    chunker = DocumentChunker(embedder=embedder)
    vector_db = AsyncQdrantVectorDB(embedder=embedder)
    retriever = RetrievalService(vector_db=vector_db, client=client)

    document_paths = [
        Path("data/evaluation/AS19_Sayi-RAG-Sistemlerine-Giris.pdf"),
        Path("data/evaluation/TUBA-Akilli_Saglik_Sistemi_ve_Yapay_Zeka.pdf"),
    ]

    await init_vector_db(
        document_io=document_io,
        extractor=extractor,
        chunker=chunker,
        retriever=retriever,
        document_paths=document_paths,
        collection_name=settings.QDRANT_KNOWLEDGE_COLLECTION_NAME,
        save_citations=True,
        output_json_path=Path("data/citations.json"),
    )