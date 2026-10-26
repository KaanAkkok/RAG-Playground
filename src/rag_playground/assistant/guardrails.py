import re
import json
import unicodedata
from typing import Literal

from rag_playground.core.exceptions import LLMGenerationError
from rag_playground.core.logger import get_logger
from rag_playground.core.security import (
    _INVISIBLE_CHARS,
    _MASKED_ADDRESS_PLACEHOLDER,
    _MASKED_BIRTH_DATE_PLACEHOLDER,
    _MASKED_CARD_PLACEHOLDER,
    _MASKED_EMAIL_PLACEHOLDER,
    _MASKED_IBAN_PLACEHOLDER,
    _MASKED_IP_PLACEHOLDER,
    _MASKED_NAME_PLACEHOLDER,
    _MASKED_PHONE_PLACEHOLDER,
    _MASKED_PLACEHOLDER,
    _MASKED_PLATE_PLACEHOLDER,
    _MASKED_TCKN_PLACEHOLDER,
    _PROMP_INJECTION_ANSWER,
    _PROMPT_INJECTION_RE,
    _TEXT_ADDRESS_RE,
    _TEXT_BIRTH_DATE_RE,
    _TEXT_CARD_RE,
    _TEXT_EMAIL_RE,
    _TEXT_IBAN_RE,
    _TEXT_IP_RE,
    _TEXT_NAME_RE,
    _TEXT_PHONE_RE,
    _TEXT_PLATE_RE,
    _TEXT_TC_RE,
    MASK_PATTERN_MAP,
    is_iban_valid,
    is_luhn_valid,
    is_tckn_valid,
)
from rag_playground.core.settings import settings
from rag_playground.llm.llm_factory import LLMClient
from rag_playground.models.vector_model import VectorRecord
from rag_playground.prompts.prompts import LLM_AS_JUDGE_PROMPT
from rag_playground.schemas.guardrails_schema import (
    LLM_GUARDRAILS_FORMAT,
    GuardrailsResponseSchema,
)

logger = get_logger(__name__)


class GuardrailsService:
    def __init__(
        self,
        mode: Literal["strict", "lenient", "loose"] = "strict",
        add_pattern: re.Pattern | None = None,
        proprietary_dictionary: set[str] | None = None,
        normalization_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC",
        llm_as_judge_enabled: bool = True,
        client: LLMClient | None = None,
        temperature: float | None = settings.GUARDRAILS_TEMPERATURE,
        guardrails_deployment: str | None = settings.AZURE_OPENAI_GUARDRAILS_DEPLOYMENT,
        llm_as_judge_prompt: str | None = LLM_AS_JUDGE_PROMPT,
        response_format: dict | None = LLM_GUARDRAILS_FORMAT,
    ):
        self.mode = mode
        self.add_re = add_pattern
        self.proprietary_dictionary = proprietary_dictionary or set()
        self.llm_as_judge_prompt = llm_as_judge_prompt
        self.guardrails_deployment = guardrails_deployment
        self.guardrails_client = client
        self.temperature = temperature
        self.response_format = response_format
        self.llm_as_judge_enabled = llm_as_judge_enabled
        self.normalization_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = (
            normalization_form
        )
        self.masked_map: dict[str, str] = {}
        self.strict = [
            _TEXT_TC_RE,
            _TEXT_EMAIL_RE,
            _TEXT_PHONE_RE,
            _TEXT_PLATE_RE,
            _TEXT_IBAN_RE,
            _TEXT_CARD_RE,
            _TEXT_IP_RE,
            _TEXT_BIRTH_DATE_RE,
            _TEXT_NAME_RE,
            _TEXT_ADDRESS_RE,
        ]
        self.lenient = [
            _TEXT_TC_RE,
            _TEXT_EMAIL_RE,
            _TEXT_PHONE_RE,
            _TEXT_PLATE_RE,
            _TEXT_IBAN_RE,
            _TEXT_CARD_RE,
            _TEXT_IP_RE,
        ]
        self.loose = [
            _TEXT_TC_RE,
            _TEXT_EMAIL_RE,
            _TEXT_PHONE_RE,
            _TEXT_CARD_RE,
        ]
        if self.add_re is not None:
            if self.mode == "strict":
                self.strict.append(self.add_re)
            elif self.mode == "lenient":
                self.lenient.append(self.add_re)
            elif self.mode == "loose":
                self.loose.append(self.add_re)
            else:
                raise ValueError(
                    f"Invalid mode: {self.mode}. Must be 'strict', 'lenient', or 'loose'."
                )
        self._validate_judge_configuration()

    def _validate_judge_configuration(self):
        if self.llm_as_judge_enabled:
            if self.guardrails_client is None:
                raise ValueError(
                    "Guardrails client must be provided when llm_as_judge_enabled is True."
                )
            if not self.guardrails_deployment:
                raise ValueError(
                    "Guardrails deployment must be provided when llm_as_judge_enabled is True."
                )
            if not self.response_format:
                raise ValueError(
                    "Response format must be provided when llm_as_judge_enabled is True."
                )

    @staticmethod
    def redact_text(value: str | None) -> str:
        text = (value or "").translate(_INVISIBLE_CHARS)
        text = _TEXT_ADDRESS_RE.sub(_MASKED_ADDRESS_PLACEHOLDER, text)
        text = _TEXT_NAME_RE.sub(_MASKED_NAME_PLACEHOLDER, text)
        text = _TEXT_BIRTH_DATE_RE.sub(_MASKED_BIRTH_DATE_PLACEHOLDER, text)
        text = _TEXT_EMAIL_RE.sub(_MASKED_EMAIL_PLACEHOLDER, text)
        text = _TEXT_PHONE_RE.sub(_MASKED_PHONE_PLACEHOLDER, text)
        candidates = _TEXT_TC_RE.findall(text)
        for candidate in candidates:
            if is_tckn_valid(candidate):
                text = text.replace(candidate, _MASKED_TCKN_PLACEHOLDER)
        text = _TEXT_PLATE_RE.sub(_MASKED_PLATE_PLACEHOLDER, text)
        candidates = _TEXT_IBAN_RE.findall(text)
        for candidate in candidates:
            if is_iban_valid(candidate):
                text = text.replace(candidate, _MASKED_IBAN_PLACEHOLDER)
        candidates = _TEXT_CARD_RE.findall(text)
        for candidate in candidates:
            if is_luhn_valid(candidate):
                text = text.replace(candidate, _MASKED_CARD_PLACEHOLDER)
        text = _TEXT_IP_RE.sub(_MASKED_IP_PLACEHOLDER, text)
        return text

    def _selected_mode_patterns(self) -> list[re.Pattern]:
        if self.mode == "strict":
            return self.strict
        elif self.mode == "lenient":
            return self.lenient
        elif self.mode == "loose":
            return self.loose
        else:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'strict', 'lenient', or 'loose'."
            )

    def strip_nul_chars(self, value: str | None) -> str:
        return (value or "").translate(_INVISIBLE_CHARS)

    def _contains_pii(self, text: str) -> tuple[str, int, int]:
        if not text:
            return text, 0, 0
        pattern_found = 0
        unmasked_count = 0
        masked_count = 0
        for pattern in self._selected_mode_patterns():
            matches = list(pattern.finditer(text))
            for match in reversed(matches):
                item = match.group(0)
                pattern_found += 1
                if pattern == _TEXT_CARD_RE and not is_luhn_valid(item):
                    unmasked_count += 1
                    continue
                if pattern == _TEXT_TC_RE and not is_tckn_valid(item):
                    unmasked_count += 1
                    continue
                if pattern == _TEXT_IBAN_RE and not is_iban_valid(item):
                    unmasked_count += 1
                    continue
                self.masked_map.setdefault(
                    item, MASK_PATTERN_MAP.get(pattern, _MASKED_PLACEHOLDER)
                )
                text = (
                    text[: match.start()]
                    + MASK_PATTERN_MAP.get(pattern, _MASKED_PLACEHOLDER)
                    + text[match.end() :]
                )
                masked_count += 1
        return text, pattern_found, masked_count

    def _proprietary_dictionary_check(self, text: str) -> tuple[str, int]:
        if not text or not self.proprietary_dictionary:
            return text, 0
        count_proprietary = 0
        for word in sorted(self.proprietary_dictionary, key=len, reverse=True):
            if not word:
                continue
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            matches = list(pattern.finditer(text))
            for match in reversed(matches):
                item = match.group(0)
                self.masked_map.setdefault(item, _MASKED_PLACEHOLDER)
                text = text[: match.start()] + _MASKED_PLACEHOLDER + text[match.end() :]
                count_proprietary += 1
        return text, count_proprietary

    def _prompt_injection_check(self, text: str) -> bool:
        if not text or len(text) == 0:
            return False
        return bool((text or "").strip() and _PROMPT_INJECTION_RE.search(text))

    def input_guardrail(self, text: str) -> tuple[bool, str]:
        self.masked_map.clear()
        text = self.strip_nul_chars(text)
        text = unicodedata.normalize(self.normalization_form, text)
        if self._prompt_injection_check(text):
            logger.warning(
                "Guardrail triggered: prompt injection detected, request blocked."
            )
            return True, _PROMP_INJECTION_ANSWER
        text, pattern_found, masked_count = self._contains_pii(text)
        text, count_proprietary = self._proprietary_dictionary_check(text)
        if pattern_found or count_proprietary:
            logger.warning(
                "Guardrail triggered: "
                f"{pattern_found} PII matches, {masked_count} masked, "
                f"and {count_proprietary} proprietary matches masked."
            )
        return False, text

    async def _llm_call(
        self, messages: list[dict], response_format: dict
    ) -> GuardrailsResponseSchema:
        if self.guardrails_client is None:
            raise ValueError("Guardrails client is not initialized.")
        if messages is None or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list of dictionaries.")
        if response_format is None or not isinstance(response_format, dict):
            raise ValueError("Response format must be a non-empty dictionary.")
        request = {
            "model": self.guardrails_deployment,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": response_format,
        }
        try:
            response = await self.guardrails_client.generate_response(**request)
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Judge model boş yanıt döndürdü.")
            data = json.loads(content)
            return GuardrailsResponseSchema(**data)
        except LLMGenerationError:
            logger.exception("Error occurred while evaluating question.")
            raise

    def output_guardrail(self, answer: str) -> str:
        if not answer or len(answer) == 0:
            return answer
        self.masked_map.clear()
        answer = self.strip_nul_chars(answer)
        answer = unicodedata.normalize(self.normalization_form, answer)
        answer, pattern_found, masked_count = self._contains_pii(answer)
        answer, count_proprietary = self._proprietary_dictionary_check(answer)
        if pattern_found or count_proprietary:
            logger.warning(
                "Guardrail triggered: "
                f"{pattern_found} PII matches, {masked_count} masked, "
                f"and {count_proprietary} proprietary matches masked."
            )
            return answer
        return answer

    async def llm_as_judge(
        self, question: str, answer: str, citations: list[VectorRecord] | None, memory: list[dict[str, str]] | None = None
    ) -> tuple[bool, str, str]:
        """
        Evaluates the answer for guardrail violations based on the provided question and citations.
        Uses the LLM to check for sensitive information, NSFW content, and other guardrail violations.

        Args:
            question (str): The user's question.
            answer (str): The assistant's answer to the question.
            citations (list[VectorRecord] | None): Optional list of citations related to the answer.
            memory (list[dict[str, str]] | None): Optional list of memory records related to the conversation.

        Returns:
            tuple[bool, str, str]: A tuple containing whether the answer is
            blocked, feedback for regeneration, and the sanitized answer.
        """
        if not self.llm_as_judge_enabled:
            return (False, "", answer)
        if not answer or len(answer) == 0:
            return (False, "", answer)
        if question is None or len(question) == 0:
            logger.warning("Guardrail triggered: empty question provided.")
            return (False, "", answer)
        citations_text = ""
        if citations is not None:
            citations_text = "\n".join([str(citation.text) for citation in citations])
        if self.response_format is None:
            raise ValueError(
                "Response format must be provided for output guardrail evaluation."
            )
        messages = [
            {"role": "system", "content": self.llm_as_judge_prompt},
            {
                "role": "user",
                "content": f"Kullanıcı sorusu: {question}\nAsistanın cevabı: {answer}\n\nMemory bilgileri: {memory if memory else 'Yok'}\nKaynaklar: {citations_text if citations_text else 'Yok'}",
            },
        ]
        response = await self._llm_call(messages, self.response_format)
        if response.contain_sensitive_info:
            logger.warning(
                "Guardrail triggered: sensitive information detected in the answer."
            )
            return (True, "", "Hassas bilgi tespit edildi. Lütfen tekrar deneyin.")
        if response.nfsw:
            logger.warning("Guardrail triggered: NSFW content detected in the answer.")
            return (True, "", "NSFW içerik tespit edildi. Lütfen tekrar deneyin.")
        if response.feedback is None:
            response.feedback = ""
        return (False, response.feedback, answer)
