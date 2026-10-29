import re

from rag_playground.assistant.guardrails import GuardrailsService
from rag_playground.assistant.intent import IntentClassifier
from rag_playground.core.exceptions import (
    IntentClassificationError,
    LLMClientError,
    MemoryServiceError,
    RetrievalServiceError,
)
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.llm.llm_factory import LLMClient
from rag_playground.memory.memory_service import MemoryIService
from rag_playground.models.vector_model import VectorRecord
from rag_playground.prompts.prompts import ASSISTANT_SYSTEM_PROMPT
from rag_playground.retrieval.retrieval_service import RetrievalService
from rag_playground.schemas.assistant_schema import AssistantResponse
from rag_playground.schemas.intent_schema import Intent
from rag_playground.schemas.llm_client_schema import ChatRequest, LLMMessage

logger = get_logger(__name__)


NOT_FOUND_ANSWER = "Bu bilgiyi yüklenen dokümanlarda bulamadım."
ERROR_ANSWER = "Bir hata oluştu. Lütfen tekrar deneyin."
TASKS = {
    "knowledge_question": "Görev: knowledge_question",
    "memory_question": "Görev: memory_question",
    "chitchat": "Görev: chitchat",
    "needs_clarification": "Görev: needs_clarification",
    "out_of_scope": "Görev: out_of_scope",
}
CITATION_RE = re.compile(r"\[(\d+)\]")


class AssistantService:
    def __init__(
        self,
        client: LLMClient,
        intent_classifier: IntentClassifier,
        retriever: RetrievalService,
        guardrails: GuardrailsService,
        memory_service: MemoryIService | None,
        guardrails_enabled: bool = True,
        memory_enabled: bool = True,
        model: str = settings.llm_models.chat,
        temperature: float = settings.ASSISTANT_CHAT_TEMPERATURE,
        system_prompt: str = ASSISTANT_SYSTEM_PROMPT,
    ):
        self.guardrails_enabled = guardrails_enabled
        self.guardrails = guardrails
        self.intent_classifier = intent_classifier
        self.memory_enabled = memory_enabled
        self.memory_service = memory_service if memory_enabled else None
        self.retriever = retriever
        self.client = client
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.validate_configuration()

    def validate_configuration(self):
        if not self.intent_classifier:
            raise ValueError("Intent classifier is not configured.")
        if not self.retriever:
            raise ValueError("Retriever is not configured.")
        if not self.client:
            raise ValueError("LLM client is not configured.")
        if not self.model:
            raise ValueError("Model deployment name is not configured.")
        if not self.system_prompt:
            raise ValueError("System prompt is not configured.")
        if not self.memory_service and self.memory_enabled:
            raise ValueError("Memory service is not configured.")
        if not self.guardrails:
            raise ValueError("Guardrails service is not configured.")

    def _input_guardrails_check(self, user_input: str) -> tuple[bool, str]:
        if self.guardrails_enabled and self.guardrails:
            is_not_safe, message = self.guardrails.input_guardrail(user_input)
            return is_not_safe, message
        return False, user_input

    async def get_intent(
        self, user_input: str, memories: list[dict] | None = None
    ) -> tuple[Intent, str | None]:
        is_not_safe, message = self._input_guardrails_check(user_input)
        if is_not_safe:
            return (
                Intent(
                    intent="out_of_scope", confidence=1.0, resolved_query=user_input
                ),
                message,
            )
        try:
            st_history = None
            if memories:
                st_history = memories
            elif self.memory_enabled and self.memory_service:
                st_history = self.memory_service.get_short_term_memories()
            intent = await self.intent_classifier.classify_intent(
                query=message, history=st_history
            )
            logger.info(
                f"Intent classified: {intent.intent} (confidence={intent.confidence:.2f})"
            )
            return intent, None
        except IntentClassificationError:
            logger.exception("Error occurred while classifying intent")
            return (
                Intent(
                    intent="out_of_scope", confidence=1.0, resolved_query=user_input
                ),
                ERROR_ANSWER,
            )

    async def retrieve(self, query: str) -> list[VectorRecord]:
        try:
            citations = await self.retriever.retrieve(query)
            vector_records = []
            for c in citations:
                rrf_score = c.rrf_score if hasattr(c, "rrf_score") else 0.0
                rerank_score = c.rerank_score if hasattr(c, "rerank_score") else 0.0
                vector_record = self.retriever._to_record(c)
                vector_record.rrf_score = rrf_score
                vector_record.rerank_score = rerank_score
                vector_records.append(vector_record)
            return vector_records
        except RetrievalServiceError:
            logger.exception("Error occurred while retrieving documents")
            return []

    @staticmethod
    def _build_context(
        citations: list[VectorRecord],
        doc_title_field: str = "doc_title",
        page_field: str = "page",
    ) -> str:
        return "\n\n".join(
            f"[{i}] ({c.payload.get(doc_title_field)}, s.{c.payload.get(page_field)})\n{c.text}"
            for i, c in enumerate(citations, start=1)
        )

    @staticmethod
    def _used_citations(
        answer: str, citations: list[VectorRecord]
    ) -> list[VectorRecord]:
        used = sorted({int(n) for n in CITATION_RE.findall(answer)})
        return [citations[n - 1] for n in used if 1 <= n <= len(citations)]

    @staticmethod
    def _format_memory_context(memory_citations: list[VectorRecord]) -> str:
        return "\n".join(
            f"- {c.text}"
            + (f" ({c.payload.get('inference')})" if c.payload.get("inference") else "")
            for c in memory_citations
        )

    async def _get_memory_context(self, query: str) -> str:
        memory_citations = await self._get_memory_citations(query)
        return self._format_memory_context(memory_citations)

    async def _get_memory_citations(self, query: str) -> list[VectorRecord]:
        if not self.memory_enabled or not self.memory_service:
            return []
        try:
            memories = await self.memory_service.get_long_term_memories(query=query)
        except MemoryServiceError:
            logger.exception("Error occurred while retrieving long-term memories")
            return []
        logger.info(f"Long-term memories used: {len(memories)}")
        return [
            VectorRecord(
                text=m.content,
                payload={
                    "doc_title": "Memory",
                    "page": 0,
                    "paragraph": 0,
                    "inference": m.inference or "",
                },
            )
            for m in memories
        ]

    def _update_conversation_memory(self, user_message: str, answer: str) -> None:
        if self.memory_enabled and self.memory_service:
            self.memory_service.update_short_term_memory(
                "user", self.guardrails.redact_text(user_message)
            )
            self.memory_service.update_short_term_memory("assistant", answer)
            self.memory_service.schedule_consolidation()

    async def _generate_response(
        self,
        task: str,
        user_input: str,
        context: str = "",
        memories: list[dict] | None = None,
        memory_context: str = "",
    ) -> str:
        is_not_safe, message = self._input_guardrails_check(user_input)
        if is_not_safe:
            return message
        messages = [{"role": "system", "content": self.system_prompt}]
        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Kullanıcı hakkında hatırlananlar:\n{memory_context}",
                }
            )
        st_history = None
        if memories:
            st_history = memories
        elif self.memory_enabled and self.memory_service:
            st_history = self.memory_service.get_short_term_memories()
        if st_history:
            messages.extend(st_history)
        system_task = f"{task}\n\nKaynaklar:\n{context}" if context else task
        messages.append({"role": "system", "content": system_task})
        messages.append({"role": "user", "content": message})
        messages = [LLMMessage.model_validate(message) for message in messages]
        request = ChatRequest(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        try:
            response = await self.client.generate_response(request)
        except LLMClientError:
            logger.exception("Error occurred while generating response")
            raise
        answer = response.text
        if not answer:
            answer = ERROR_ANSWER
        return answer.strip()

    async def _generate_checked_answer(
        self,
        question: str,
        task: str,
        context: str = "",
        memories: list[dict] | None = None,
        citations: list[VectorRecord] | None = None,
        memory_context: str = "",
        max_retries: int = 3,
    ) -> str:
        if not self.guardrails_enabled or not self.guardrails:
            answer = await self._generate_response(
                task, question, context, memories, memory_context=memory_context
            )
            self._update_conversation_memory(question, answer)
            return answer
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        user_input = question
        answer = ""
        for attempt in range(max_retries + 1):
            answer = await self._generate_response(
                task, user_input, context, memories, memory_context=memory_context
            )
            answer = self.guardrails.output_guardrail(answer)
            is_blocked, feedback, sanitized_answer = await self.guardrails.llm_as_judge(
                question, answer, citations, memory=memories
            )
            if is_blocked:
                logger.warning(f"Guardrail triggered: {feedback}")
                return sanitized_answer
            if not feedback.strip():
                self._update_conversation_memory(question, sanitized_answer)
                return sanitized_answer
            if attempt == max_retries:
                logger.error(
                    "Max retries reached after LLM-as-a-judge feedback. Returning the last generated answer."
                )
                self._update_conversation_memory(question, sanitized_answer)
                return sanitized_answer
            logger.warning(f"LLM as Judge feedback: {feedback}")
            user_input = (
                f"{question}\n\n"
                "Lütfen cevabını geliştir ve aşağıdaki geri bildirimi dikkate al:\n"
                f"{feedback}"
            )
        raise RuntimeError("Answer generation loop exited unexpectedly.")

    async def answer(
        self,
        user_input: str,
        reset_history: bool = False,
        memories: list[dict] | None = None,
        citations: list[VectorRecord] | None = None,
    ) -> AssistantResponse:
        if reset_history and self.memory_enabled and self.memory_service:
            self.memory_service.reset_history()
        intent, msg = await self.get_intent(user_input, memories)
        if msg is not None:
            return AssistantResponse(answer=msg, intent=intent)
        if intent.intent == "memory_question":
            query = user_input
            memory_citations: list[VectorRecord] = []
            judge_citations: list[VectorRecord] = []
            memory_context = ""
            if memories:
                for m in memories:
                    if isinstance(m, dict) and m.get("content"):
                        record = VectorRecord(
                            text=str(m["content"]),
                            payload={"doc_title": "Memory", "page": 0, "paragraph": 0},
                        )
                        memory_citations.append(record)
                        judge_citations.append(record)
            elif self.memory_enabled and self.memory_service:
                memory_citations = await self._get_memory_citations(query)
                memory_context = self._format_memory_context(memory_citations)
                judge_citations.extend(memory_citations)
                st_msgs = self.memory_service.get_short_term_memories()
                if st_msgs:
                    st_text = "\n".join(
                        f"{m.get('role', '')}: {m.get('content', '')}"
                        for m in st_msgs
                        if isinstance(m, dict) and m.get("content")
                    )
                    if st_text:
                        judge_citations.append(
                            VectorRecord(
                                text=st_text,
                                payload={
                                    "doc_title": "ShortTermMemory",
                                    "page": 0,
                                    "paragraph": 0,
                                },
                            )
                        )
            logger.info(f"Memory citations retrieved: {len(memory_citations)}")
            answer = await self._generate_checked_answer(
                query,
                TASKS["memory_question"],
                memories=memories,
                citations=judge_citations or None,
                memory_context=memory_context,
            )
            return AssistantResponse(
                answer=answer, intent=intent, citations=memory_citations
            )
        if intent.intent == "knowledge_question":
            query = intent.resolved_query or user_input
            memory_context = (
                await self._get_memory_context(query) if not memories else ""
            )
            if citations is not None:
                context = self._build_context(citations)
                answer = await self._generate_checked_answer(
                    user_input,
                    TASKS[intent.intent],
                    context,
                    memories,
                    citations=citations,
                    memory_context=memory_context,
                )
                return AssistantResponse(
                    answer=answer, intent=intent, citations=citations
                )
            citations = await self.retrieve(query)
            logger.info(f"Citations retrieved: {len(citations)}")
            if not citations:
                return AssistantResponse(answer=NOT_FOUND_ANSWER, intent=intent)
            context = self._build_context(citations)
            answer = await self._generate_checked_answer(
                query,
                TASKS["knowledge_question"],
                context,
                memories=memories,
                citations=citations,
                memory_context=memory_context,
            )
            return AssistantResponse(
                answer=answer,
                intent=intent,
                citations=citations,
            )
        answer = await self._generate_checked_answer(
            user_input,
            TASKS[intent.intent],
            memories=memories,
        )
        return AssistantResponse(answer=answer, intent=intent)

    @staticmethod
    def show_citations(
        citations: list[VectorRecord],
        doc_title_field: str = "doc_title",
        page_field: str = "page",
        paragraph_field: str = "paragraph",
        score_field: str = "rerank_score",
    ) -> str:
        result = ""
        for i, c in enumerate(citations, start=1):
            preview = c.text.replace("\n", " ")[:160]
            score = getattr(c, score_field, c.payload.get(score_field, 0.0))
            result += (
                "\n"
                + f"[{i}] {c.payload.get(doc_title_field)} s.{c.payload.get(page_field)} P.{c.payload.get(paragraph_field)} | {score_field}={score:.3f}\n{preview}…"
            )
        return result
