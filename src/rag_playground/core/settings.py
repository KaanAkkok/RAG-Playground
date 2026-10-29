import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
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

# LLM_PROVIDER değeri -> o provider'ın env değişkenlerindeki prefix
_PROVIDER_SETTING_PREFIXES = {
    "azure": "AZURE_OPENAI",
    "azure_openai": "AZURE_OPENAI",
    "openai": "OPENAI",
    "openai_compatible": "OPENAI",
    "gemini": "GEMINI",
}


class LLMModels(BaseModel):
    chat: str
    intent: str
    reranker: str
    memory: str
    guardrails: str
    generation_eval: str
    embedding: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    LOG_LEVEL: str = "INFO"

    LLM_PROVIDER: str = "azure"

    # OpenAI-compatible settings
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str | None = None
    OPENAI_CHAT_TIMEOUT: int = 60
    OPENAI_CHAT_MAX_RETRIES: int = 3
    OPENAI_EMBEDDING_DEPLOYMENT: str = ""
    OPENAI_CHAT_DEPLOYMENT: str = ""
    OPENAI_RERANKER_DEPLOYMENT: str = ""
    OPENAI_INTENT_DEPLOYMENT: str = ""
    OPENAI_MEMORY_DEPLOYMENT: str = ""
    OPENAI_GENERATION_EVAL_DEPLOYMENT: str = ""
    OPENAI_GUARDRAILS_DEPLOYMENT: str = ""

    # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = ""
    AZURE_OPENAI_CHAT_TIMEOUT: int = 60
    AZURE_OPENAI_CHAT_MAX_RETRIES: int = 3
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = ""
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = ""
    AZURE_OPENAI_RERANKER_DEPLOYMENT: str = ""
    AZURE_OPENAI_INTENT_DEPLOYMENT: str = ""
    AZURE_OPENAI_MEMORY_DEPLOYMENT: str = ""
    AZURE_OPENAI_GENERATION_EVAL_DEPLOYMENT: str = ""
    AZURE_OPENAI_GUARDRAILS_DEPLOYMENT: str = ""

    # Gemini-compatible settings
    GEMINI_API_KEY: str = ""
    GEMINI_CHAT_TIMEOUT: int = 60
    GEMINI_CHAT_MAX_RETRIES: int = 3
    GEMINI_EMBEDDING_DEPLOYMENT: str = ""
    GEMINI_CHAT_DEPLOYMENT: str = ""
    GEMINI_RERANKER_DEPLOYMENT: str = ""
    GEMINI_INTENT_DEPLOYMENT: str = ""
    GEMINI_MEMORY_DEPLOYMENT: str = ""
    GEMINI_GENERATION_EVAL_DEPLOYMENT: str = ""
    GEMINI_GUARDRAILS_DEPLOYMENT: str = ""

    LLM_RESPONSE_TIMEOUT: int = 30
    EMBEDDING_TIMEOUT: int = 60

    CHAT_TEMPERATURE: float = Field(default=0.4, ge=0.0, le=2.0)
    INTENT_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    RERANKER_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    MEMORY_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    EVALUATION_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    GUARDRAILS_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)

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
    ANIMATIONS_DIR: str = (
        PROJECT_ROOT / "data" / "animations"
    ).as_posix()
    EVAL_SNAPSHOT_DIR: str = (
        PROJECT_ROOT / "data" / "evaluation" / "snapshots"
    ).as_posix()
    RETRIEVAL_EVALUATION_LEVEL: Literal[1, 2] = Field(
        default=2, description="1: passage-level, 2: document-level"
    )
    EVAL_SNAPSHOT_PATH: str = Path(f"{EVAL_SNAPSHOT_DIR}/EVALUATION_SNAPSHOT.snapshot").as_posix()
    EMBED_BATCH_SIZE: int = 96
    MIN_CHUNK_CHARS: int = 260
    MAX_CHUNK_CHARS: int = 700
    SENTENCE_OVERLAP_RATIO: float = Field(default=0.2, ge=0.0, le=1.0)
    SEMANTIC_CHUNKING_BREAK_PERCENTAGE: float = Field(default=0.85, ge=0.0, le=1.0)
    SEMANTIC_CHUNKING_MIN_DIST: float = Field(default=0.2, ge=0.0, le=1.0)
    SEMANTIC_CHUNKING_BUFFER_SIZE: int = Field(default=1, ge=0, le=5)
    SEMANTIC_CHUNKING_MIN_SENTENCES: int = Field(default=5, ge=2, le=20)
    VECTORDB_UPSERT_BATCH_SIZE: int = 128
    SEARCH_CANDIDATES: int = 25
    FUSED_SEARCH_CANDIDATES: int = 10
    TOP_K_CANDIDATES: int = 5
    EVALUATION_LIMIT: int = 5
    MIN_RERANK_SCORE: int = Field(default=3, ge=0, le=10)
    RERANK_PASSAGE_CHARS: int = Field(default=700, ge=100, le=1000)

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
    VECTORDB_KNOWLEDGE_COLLECTION_NAME: str = "Documents"
    VECTORDB_MEMORY_COLLECTION_NAME: str = "Memories"
    VECTORDB_EVALUATION_COLLECTION_NAME: str = "Evaluation"


    @model_validator(mode="after")
    def be_sure_dirs(self):
        os.makedirs(self.RAW_DOCUMENTS_DIR, exist_ok=True)
        os.makedirs(self.SAVED_DOCUMENTS_DIR, exist_ok=True)
        os.makedirs(self.ANIMATIONS_DIR, exist_ok=True)
        os.makedirs(self.EVAL_SNAPSHOT_DIR, exist_ok=True)
        return self

    @property
    def llm_models(self) -> LLMModels:
        """Aktif LLM_PROVIDER'ın görev bazlı model/deployment isimleri."""
        provider = self.LLM_PROVIDER.strip().lower()
        try:
            prefix = _PROVIDER_SETTING_PREFIXES[provider]
        except KeyError as exc:
            supported = ", ".join(sorted(_PROVIDER_SETTING_PREFIXES))
            raise ValueError(
                f"Unsupported LLM provider '{provider}'. Supported providers: {supported}"
            ) from exc
        return LLMModels(
            **{
                task: getattr(self, f"{prefix}_{task.upper()}_DEPLOYMENT")
                for task in LLMModels.model_fields
            }
        )


settings = Settings()
