class RAGError(Exception):
    """Base class for all errors raised by this package."""


class LLMClientError(RAGError):
    """An LLM client could not be created or used."""


class EmbeddingError(RAGError):
    """The embedding API call failed."""


class IntentClassificationError(RAGError):
    """The intent classification LLM call failed."""


class MemoryServiceError(RAGError):
    """Long-term memory extraction, storage or retrieval failed."""


class RetrievalServiceError(RAGError):
    """The retrieval service failed."""


class DocumentExtractionError(RAGError):
    """Document extraction failed."""


class DocumentChunkingError(RAGError):
    """Document chunking failed."""


class DocumentSavingError(RAGError):
    """Document saving failed."""


class InvalidGroundTruthSchemaError(RAGError):
    """The ground truth provided is invalid schema."""


class LLMGenerationError(RAGError):
    """The LLM generation call failed."""


class EmbeddingClientError(RAGError):
    """The embedding client call failed."""
