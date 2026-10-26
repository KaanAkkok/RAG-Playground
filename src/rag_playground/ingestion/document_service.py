from pathlib import Path

from rag_playground.core.exceptions import (
    DocumentChunkingError,
    DocumentExtractionError,
    DocumentSavingError,
)
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.ingestion.chunker import DocumentChunker
from rag_playground.ingestion.document_io import DocumentIO
from rag_playground.ingestion.extractor import DocumentExtractor
from rag_playground.models.file_type_model import FileType

logger = get_logger(__name__)


class DocumentService:
    def __init__(
        self,
        extractor: DocumentExtractor,
        chunker: DocumentChunker,
        documentIO: DocumentIO,
        input_folder: Path = Path(settings.RAW_DOCUMENTS_DIR),
        output_folder: Path = Path(settings.SAVED_DOCUMENTS_DIR),
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.extractor = extractor
        self.chunker = chunker
        self.documentIO = documentIO
        self.index: dict[str, dict[str, str]] = {}

    def is_document_processed(self, document_path: Path, hash_value: str) -> bool:
        document_name = document_path.stem
        index_entry = self.index.get(document_name)
        if not index_entry:
            return False
        hash_in_index = index_entry.get("hash_value")
        if hash_in_index is None:
            return False
        return hash_in_index == hash_value

    async def process_document(self, document_path: list[Path] | Path) -> None:
        documents = []
        if isinstance(document_path, list):
            documents.extend(document_path)
        elif isinstance(document_path, Path):
            documents.append(document_path)
        else:
            raise TypeError("document_path must be a Path or a list of Paths.")
        logger.info(f"Processing {len(documents)} documents.")
        for doc_path in documents:
            if not doc_path.exists():
                raise FileNotFoundError(f"Document {doc_path} does not exist.")
            content, hash_value = self.documentIO.read_file(doc_path)
            if self.is_document_processed(doc_path, hash_value):
                logger.info(
                    f"Document {doc_path} has already been processed. Skipping."
                )
                continue
            try:
                content = self.extractor.extract(doc_path)
                logger.info(
                    f"Successfully extracted content from {doc_path}. Proceeding to chunking."
                )
            except DocumentExtractionError as e:
                logger.error(f"Failed to extract content from {doc_path}: {e}")
                continue
            try:
                await self.chunker.chunk(doc_path, content)
                logger.info(
                    f"Successfully chunked content from {doc_path}. Proceeding to save."
                )
            except DocumentChunkingError as e:
                logger.error(f"Failed to chunk content from {doc_path}: {e}")
                continue
            try:
                await self.documentIO.save_document(
                    content, doc_path.stem, self.output_folder, FileType.MD
                )
                logger.info(
                    f"Successfully saved content from {doc_path}. Updating index."
                )
            except DocumentSavingError as e:
                logger.error(f"Failed to save content from {doc_path}: {e}")
            self.index[doc_path.stem] = {"hash_value": hash_value}
            logger.info(f"Document {doc_path} processed and saved successfully.")
