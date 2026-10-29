from rag_playground.core.exceptions import IntentClassificationError
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.llm.llm_factory import LLMClient
from rag_playground.prompts.prompts import INTENT_SYSTEM_PROMPT
from rag_playground.schemas.intent_schema import LLM_INTENT_FORMAT, Intent
from rag_playground.schemas.llm_client_schema import (
    ChatRequest,
    LLMMessage,
    ResponseFormat,
)

logger = get_logger(__name__)


class IntentClassifier:
    def __init__(
        self,
        client: LLMClient,
        system_prompt: str = INTENT_SYSTEM_PROMPT,
        classifier_model: str = settings.llm_models.intent,
        temperature: float = settings.INTENT_TEMPERATURE,
        response_format: dict = LLM_INTENT_FORMAT,
        timeout: int = settings.LLM_RESPONSE_TIMEOUT,
    ):
        self.system_prompt = system_prompt
        self.classifier_model = classifier_model
        self.temperature = temperature
        self.response_format = ResponseFormat(**response_format)
        self.timeout = timeout
        self.classifier = client

    async def classify_intent(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> Intent:
        cleaned_query = query.strip() if query else ""
        if not cleaned_query or not cleaned_query.strip():
            return Intent(
                intent="out_of_scope", confidence=0.0, resolved_query=cleaned_query
            )
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": cleaned_query})
        messages = [LLMMessage(**msg) for msg in messages]
        request = ChatRequest(
            model=self.classifier_model,
            messages=messages,
            temperature=self.temperature,
            response_format=self.response_format,
            timeout=self.timeout,
        )
        try:
            response = await self.classifier.generate_response(request)
        except IntentClassificationError as e:
            logger.error(f"Error during intent classification: {e}")
            raise IntentClassificationError(
                f"Error during intent classification: {e}"
            ) from e
        content = response.text
        if not content:
            return Intent(
                intent="needs_clarification",
                confidence=0.0,
                resolved_query=cleaned_query,
            )
        return Intent.model_validate_json(content)
