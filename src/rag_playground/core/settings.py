import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root() -> Path:
    configured_root = os.getenv("PROJECT_ROOT")
    if configured_root:
        return Path(configured_root).resolve()

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    return Path.cwd().resolve()


_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROJECT_ROOT = _find_project_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    LOG_LEVEL: str = "INFO"

    LLM_PROVIDER: str = "azure"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str | None = None
    OPENAI_CHAT_TIMEOUT: int = 60
    OPENAI_CHAT_MAX_RETRIES: int = 3

    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = ""
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = ""
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = ""
    AZURE_OPENAI_RERANKER_DEPLOYMENT: str = ""
    AZURE_OPENAI_CHAT_TIMEOUT: int = 60
    AZURE_OPENAI_INTENT_DEPLOYMENT: str = ""
    AZURE_OPENAI_INTENT_TIMEOUT: int = 10
    AZURE_OPENAI_CHAT_MAX_RETRIES: int = 3
    AZURE_OPENAI_RERANK_TIMEOUT: int = 30
    AZURE_OPENAI_MEMORY_DEPLOYMENT: str = ""
    AZURE_OPENAI_GENERATION_EVAL_DEPLOYMENT: str = ""
    AZURE_OPENAI_GUARDRAILS_DEPLOYMENT: str = ""

    ASSISTANT_CHAT_TEMPERATURE: float = 0.0
    ASSISTANT_INTENT_TEMPERATURE: float = 0.0
    EVALUATION_CHAT_TEMPERATURE: float = 0.0
    GUARDRAILS_TEMPERATURE: float = 0.0

    SUPPORTED_FILE_TYPES: list[str] = [".pdf"]
    RAW_DOCUMENTS_DIR: str = (
        PROJECT_ROOT / "data" / "documents" / "raw_documents"
    ).as_posix()
    SAVED_DOCUMENTS_DIR: str = (
        PROJECT_ROOT / "data" / "documents" / "saved_documents"
    ).as_posix()
    EVALUATION_FILE: str = (
        PROJECT_ROOT / "data" / "evaluation" / "ground_truth.json"
    ).as_posix()
    ANIMATIONS_DIR: str = (PROJECT_ROOT / "data" / "animations").as_posix()
    EMBED_BATCH_SIZE: int = 96
    MIN_CHUNK_CHARS: int = 260
    MAX_CHUNK_CHARS: int = 900
    SENTENCE_OVERLAP_RATIO: float = Field(default=0.2, ge=0.0, le=1.0)
    SEMANTIC_CHUNKING_BREAK_PERCENTAGE: float = Field(default=0.85, ge=0.0, le=1.0)
    SEMANTIC_CHUNKING_MIN_DIST: float = Field(default=0.2, ge=0.0, le=1.0)
    VECTORDB_UPSERT_BATCH_SIZE: int = 128
    SEARCH_CANDIDATES: int = 25
    FUSED_SEARCH_CANDIDATES: int = 10
    TOP_K_CANDIDATES: int = 5
    EVALUATION_LIMIT: int = 5
    MIN_RERANK_SCORE: int = Field(default=3, ge=0, le=10)
    RERANK_PASSAGE_CHARS: int = Field(default=900, ge=100, le=1000)

    INTENT_SYSTEM_PROMPT_PATH: str = str(_PROMPTS_DIR / "INTENT_PROMPT.md")
    ASSISTANT_SYSTEM_PROMPT_PATH: str = str(_PROMPTS_DIR / "ASSISTANT_PROMPT.md")
    RERANK_SYSTEM_PROMPT_PATH: str = str(_PROMPTS_DIR / "RERANKER_PROMPT.md")
    MEMORY_SYSTEM_PROMPT_PATH: str = str(_PROMPTS_DIR / "MEMORY_PROMPT.md")
    FAITHFULNESS_PROMPT_PATH: str = str(_PROMPTS_DIR / "FAITHFULNESS_PROMPT.md")
    ANSWER_RELEVANCY_PROMPT_PATH: str = str(_PROMPTS_DIR / "ANSWER_RELEVANCY_PROMPT.md")
    ANSWER_QUALITY_PROMPT_PATH: str = str(_PROMPTS_DIR / "ANSWER_QUALITY_PROMPT.md")
    LLM_AS_JUDGE_PROMPT_PATH: str = str(_PROMPTS_DIR / "LLM_AS_JUDGE_PROMPT.md")

    HISTORY_MAX_TOKENS: int = 2000
    HISTORY_MAX_MESSAGES: int = Field(default=6, ge=1, le=15)
    LONG_TERM_MEMORY_BATCH_SIZE: int = Field(default=10, ge=1, le=15)
    LONG_TERM_MEMORY_SEARCH_LIMIT: int = Field(default=10, ge=1, le=50)
    LONG_TERM_MEMORY_CONFIDENCE_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    LONG_TERM_MEMORY_MIN_SCORE: float = Field(default=0.5, ge=0.0, le=1.0)

    QDRANT_HOST: str = ""
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None
    QDRANT_KNOWLEDGE_COLLECTION_NAME: str = "Documents"
    QDRANT_MEMORY_COLLECTION_NAME: str = "Memories"


settings = Settings()
