from pathlib import Path

from rag_playground.core.settings import settings

INTENT_SYSTEM_PROMPT = Path(settings.INTENT_SYSTEM_PROMPT_PATH).read_text(
    encoding="utf-8"
)
ASSISTANT_SYSTEM_PROMPT = Path(settings.ASSISTANT_SYSTEM_PROMPT_PATH).read_text(
    encoding="utf-8"
)
RERANK_SYSTEM_PROMPT = Path(settings.RERANK_SYSTEM_PROMPT_PATH).read_text(
    encoding="utf-8"
)
MEMORY_SYSTEM_PROMPT = Path(settings.MEMORY_SYSTEM_PROMPT_PATH).read_text(
    encoding="utf-8"
)
FAITHFULNESS_PROMPT = Path(settings.FAITHFULNESS_PROMPT_PATH).read_text(
    encoding="utf-8"
)
ANSWER_RELEVANCY_PROMPT = Path(settings.ANSWER_RELEVANCY_PROMPT_PATH).read_text(
    encoding="utf-8"
)
ANSWER_QUALITY_PROMPT = Path(settings.ANSWER_QUALITY_PROMPT_PATH).read_text(
    encoding="utf-8"
)
LLM_AS_JUDGE_PROMPT = Path(settings.LLM_AS_JUDGE_PROMPT_PATH).read_text(encoding="utf-8")
