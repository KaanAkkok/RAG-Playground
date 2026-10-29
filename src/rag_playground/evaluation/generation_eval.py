"""
Generation evaluation module for RAG Playground.
- faithfullness
- Answer relevancy
- Answer correctness
- completeness
- answer accuracy
- noise sensitivity
"""

import asyncio
import json
import math
from pathlib import Path

from rag_playground.assistant.assistant import AssistantService
from rag_playground.core.exceptions import LLMGenerationError
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.evaluation.evaluation_service import EvaluationService
from rag_playground.llm.llm_factory import LLMClient
from rag_playground.models.vector_model import VectorRecord
from rag_playground.prompts.prompts import (
    ANSWER_QUALITY_PROMPT,
    ANSWER_RELEVANCY_PROMPT,
    FAITHFULNESS_PROMPT,
)
from rag_playground.retrieval.embedding_service import EmbeddingService
from rag_playground.retrieval.similarity import cosine_similarity
from rag_playground.schemas.assistant_schema import AssistantResponse
from rag_playground.schemas.generation_eval_schemas import (
    LLM_ANSWER_QUALITY_FORMAT,
    LLM_ANSWER_RELEVANCY_FORMAT,
    LLM_FAITHFULNESS_FORMAT,
    AnswerQualityEvalResponse,
    AnswerQualityScores,
    AnswerRelevancyEvalResponse,
    FaithfulnessEvalResponse,
)
from rag_playground.schemas.llm_client_schema import (
    ChatRequest,
    LLMMessage,
    ResponseFormat,
)

logger = get_logger(__name__)

GENERATION_EVAL_RESULTS_SCHEMA: dict[str, dict[str, float]] = {
    "Generation Evaluation Metrics": {
        "faithfulness_score": 0.0,
        "answer_relevancy_score": 0.0,
        "answer_correctness_score": 0.0,
        "completeness_score": 0.0,
        "answer_accuracy_score": 0.0,
        "noise_sensitivity_score": 0.0,
    },
}
NOISE_CHUNKS = [
    VectorRecord(
        text="Antik Roma tarihi, imparatorların yükselişi ve imparatorluğun zaman içinde geçirdiği değişimleri kapsar."
    ),
    VectorRecord(
        text="Fotosentez, bitkilerin güneş ışığını kimyasal enerjiye dönüştürdüğü biyolojik bir süreçtir."
    ),
    VectorRecord(
        text="Python programlama dili, nesne yönelimli ve fonksiyonel programlama gibi farklı yaklaşımları destekler."
    ),
    VectorRecord(text="Pasifik Okyanusu, dünyadaki en büyük ve en derin okyanustur."),
    VectorRecord(
        text="Türkiye'nin farklı bölgelerinde iklim koşulları ve bitki örtüsü önemli ölçüde değişiklik gösterir."
    ),
    VectorRecord(
        text="Güneş panelleri, güneş ışığını elektrik enerjisine dönüştürerek enerji üretir."
    ),
    VectorRecord(
        text="İnsan iskeleti, vücuda yapısal destek sağlar ve iç organların korunmasına yardımcı olur."
    ),
    VectorRecord(
        text="Geleneksel İtalyan mutfağında makarna, risotto ve lazanya gibi yemekler önemli bir yere sahiptir."
    ),
    VectorRecord(
        text="Veritabanı normalizasyonu, veri tekrarını azaltmak ve veri tutarlılığını artırmak için kullanılır."
    ),
    VectorRecord(
        text="Su döngüsü; buharlaşma, yoğunlaşma ve yağış gibi temel aşamalardan oluşur."
    ),
    VectorRecord(
        text="Basketbol, iki takımın topu rakip takımın potasından geçirerek sayı kazanmaya çalıştığı bir spordur."
    ),
    VectorRecord(
        text="Eyfel Kulesi, 1889 yılında Paris'te düzenlenen Dünya Fuarı için inşa edilmiştir."
    ),
    VectorRecord(
        text="Düzenli fiziksel aktivite, kas gücünün gelişmesine ve kalp sağlığının korunmasına katkıda bulunabilir."
    ),
    VectorRecord(
        text="Bulut bilişim, internet üzerinden sunucu, depolama ve işlem gücü gibi kaynaklara erişim sağlar."
    ),
    VectorRecord(
        text="Osmanlı döneminde kullanılan mimari teknikler, günümüzdeki birçok tarihî yapıda görülebilir."
    ),
    VectorRecord(
        text="Kahve çekirdekleri, farklı kavurma seviyelerine göre aroma ve tat bakımından değişiklik gösterir."
    ),
    VectorRecord(
        text="Dünya'nın kendi ekseni etrafında dönmesi, gece ve gündüzün oluşmasına neden olur."
    ),
    VectorRecord(
        text="Makine öğrenmesi modelleri, verilerdeki örüntüleri öğrenerek tahminler veya sınıflandırmalar yapabilir."
    ),
    VectorRecord(
        text="Bahçecilikte toprağın nem oranı, bitkilerin sağlıklı büyümesini etkileyen önemli faktörlerden biridir."
    ),
    VectorRecord(
        text="Elektrikli araçların yaygınlaşması, ulaşım sektöründeki enerji tüketimi alışkanlıklarını değiştirmektedir."
    ),
]


class GenerationEvaluator(EvaluationService):
    def __init__(
        self,
        client: LLMClient,
        assistant: AssistantService,
        embedder: EmbeddingService,
        temperature: float = settings.EVALUATION_CHAT_TEMPERATURE,
        eval_deployment: str = settings.llm_models.generation_eval,
        ground_truth_path: Path | str = settings.EVALUATION_FILE,
        noise_chunks: list[VectorRecord] = NOISE_CHUNKS,
        faithfulness_prompt: str = FAITHFULNESS_PROMPT,
        answer_relevancy_prompt: str = ANSWER_RELEVANCY_PROMPT,
        answer_quality_prompt: str = ANSWER_QUALITY_PROMPT,
        faithfulness_response_format: dict = LLM_FAITHFULNESS_FORMAT,
        answer_relevancy_response_format: dict = LLM_ANSWER_RELEVANCY_FORMAT,
        answer_quality_response_format: dict = LLM_ANSWER_QUALITY_FORMAT,
    ):
        self.evaluator = client
        self.assistant = assistant
        self.embedder = embedder
        super().__init__(ground_truth_path)
        self.eval_deployment = eval_deployment
        self.temperature = temperature
        self.noise_chunks = noise_chunks
        self.faithfulness_prompt = faithfulness_prompt
        self.answer_relevancy_prompt = answer_relevancy_prompt
        self.answer_quality_prompt = answer_quality_prompt
        self.faithfulness_response_format = ResponseFormat(
            **faithfulness_response_format
        )
        self.answer_relevancy_response_format = ResponseFormat(
            **answer_relevancy_response_format
        )
        self.answer_quality_response_format = ResponseFormat(
            **answer_quality_response_format
        )
        self._validate_params()

    def _validate_params(self):
        if self.evaluator is None or not isinstance(self.evaluator, LLMClient):
            raise ValueError(
                "LLM client must be provided and be an instance of LLMClient."
            )
        if self.assistant is None or not isinstance(self.assistant, AssistantService):
            raise ValueError(
                "Assistant service must be provided and be an instance of AssistantService."
            )
        if self.embedder is None or not isinstance(self.embedder, EmbeddingService):
            raise ValueError(
                "Embedder service must be provided and be an instance of EmbeddingService."
            )

    async def _llm_call(
        self, messages: list[LLMMessage], response_format: ResponseFormat
    ) -> dict:
        if messages is None or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list of LLMMessage objects.")
        if response_format is None or not isinstance(response_format, ResponseFormat):
            raise ValueError(
                "Response format must be a non-empty ResponseFormat object."
            )
        request = ChatRequest(
            model=self.eval_deployment,
            messages=messages,
            temperature=self.temperature,
            response_format=response_format,
        )
        try:
            response = await self.evaluator.generate_response(request)
            content = response.text
            if not content or not isinstance(content, str):
                raise ValueError("LLM response content is empty or not a string.")
            parsed_response = json.loads(content)
            if not isinstance(parsed_response, dict):
                raise TypeError("LLM response content must be a JSON object.")
            return parsed_response
        except LLMGenerationError:
            logger.exception("Error occurred while evaluating question.")
            raise

    async def faithfulness(
        self,
        answer: str,
        retrieved_chunks: list[str],
        faithfulness_prompt: str,
        response_format: ResponseFormat,
    ) -> float:
        if not answer or not isinstance(answer, str):
            raise ValueError("Answer must be a non-empty string.")
        if not retrieved_chunks or not isinstance(retrieved_chunks, list):
            raise ValueError("Retrieved chunks must be a non-empty list of strings.")
        if not faithfulness_prompt or not isinstance(faithfulness_prompt, str):
            raise ValueError("Faithfulness prompt must be a non-empty string.")
        if not response_format or not isinstance(response_format, ResponseFormat):
            raise ValueError(
                "Response format must be a non-empty ResponseFormat object."
            )
        score = 0.0
        messages = [
            LLMMessage(role="system", content=faithfulness_prompt),
            LLMMessage(
                role="user",
                content=f"Asistanın cevabı: {answer}\nAsistanın alıntıladığı metinler: {retrieved_chunks}",
            ),
        ]
        response = await self._llm_call(messages, response_format)
        if (
            not response
            or response == {}
            or response is None
            or not isinstance(response, dict)
        ):
            raise ValueError("LLM response is empty or not a dictionary.")
        claims = FaithfulnessEvalResponse(**response)
        if not claims.claims:
            return score
        for claim in claims.claims:
            if claim.claim_type == "supported" and claim.classification == "TP":
                score += 1.0
            elif claim.claim_type == "unsupported" and claim.classification == "FP":
                score += 0.0
            else:
                raise ValueError(
                    "Faithfulness claim_type and classification are inconsistent."
                )
        return score / len(claims.claims)

    async def answer_relevancy(
        self,
        query: str,
        answer: str,
        answer_relevancy_prompt: str,
        response_format: ResponseFormat,
    ) -> float:
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")
        if not answer or not isinstance(answer, str):
            raise ValueError("Answer must be a non-empty string.")
        if not answer_relevancy_prompt or not isinstance(answer_relevancy_prompt, str):
            raise ValueError("Answer relevancy prompt must be a non-empty string.")
        if not response_format or not isinstance(response_format, ResponseFormat):
            raise ValueError(
                "Response format must be a non-empty ResponseFormat object."
            )
        score = 0.0
        messages = [
            LLMMessage(role="system", content=answer_relevancy_prompt),
            LLMMessage(
                role="user",
                content=f"Kullanıcı sorusu: {query}\nDeğerlendirilecek asistanın cevap: {answer}",
            ),
        ]
        response = await self._llm_call(messages, response_format)
        if (
            not response
            or response == {}
            or response is None
            or not isinstance(response, dict)
        ):
            raise ValueError("LLM response is empty or not a dictionary.")
        response = AnswerRelevancyEvalResponse(**response)
        if len(response.generated_questions) == 0:
            return score
        generated_questions = response.generated_questions
        generated_questions_vectors = await self.embedder.embed_texts(
            [q.question for q in generated_questions]
        )
        question_vector = await self.embedder.embed_texts([query])
        if (
            len(generated_questions_vectors) != len(generated_questions)
            or len(question_vector) != 1
        ):
            raise ValueError("Embedder returned an unexpected number of vectors.")
        for generated_question, generated_question_vector in zip(
            generated_questions, generated_questions_vectors
        ):
            similarity = float(
                cosine_similarity(question_vector[0], generated_question_vector)
            )
            if not math.isfinite(similarity):
                raise ValueError("Question similarity must be finite.")
            similarity = max(0.0, min(1.0, similarity))
            if generated_question.is_relevant:
                score += similarity
        return score / len(response.generated_questions)

    async def answer_quality(
        self,
        query: str,
        answer: str,
        expected_answer: str,
        answer_correctness_prompt: str,
        response_format: ResponseFormat,
        factual_weight: float = 0.75,
        semantic_weight: float = 0.25,
    ) -> AnswerQualityScores:
        required_strings = {
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "answer_correctness_prompt": answer_correctness_prompt,
        }
        for field, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string.")
        if not isinstance(response_format, dict) or not response_format:
            raise ValueError("response_format must be a non-empty dictionary.")
        weights = {
            "factual_weight": factual_weight,
            "semantic_weight": semantic_weight,
        }
        for name, weight in weights.items():
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or not 0 <= weight <= 1
            ):
                raise ValueError(f"{name} must be a finite number between 0 and 1.")
        total_weight = factual_weight + semantic_weight
        if total_weight <= 0:
            raise ValueError("At least one weight must be greater than zero.")
        factual_weight /= total_weight
        semantic_weight /= total_weight
        messages = [
            LLMMessage(role="system", content=answer_correctness_prompt),
            LLMMessage(
                role="user",
                content=f"Kullanıcı sorusu: {query}\nAsistanın cevabı: {answer}\nBeklenen cevap: {expected_answer}",
            ),
        ]
        response = await self._llm_call(messages, response_format)
        if not isinstance(response, dict) or not response:
            raise ValueError("LLM response must be a non-empty dictionary.")
        evaluation = AnswerQualityEvalResponse(**response)
        claims = evaluation.claims or []
        valid_classifications = {"TP", "FP", "FN"}
        for claim in claims:
            if claim.classification not in valid_classifications:
                raise ValueError(
                    f"Invalid claim classification: {claim.classification!r}"
                )
        tp = sum(c.classification == "TP" for c in claims)
        fp = sum(c.classification == "FP" for c in claims)
        fn = sum(c.classification == "FN" for c in claims)
        precision_denominator = tp + fp
        recall_denominator = tp + fn
        precision = tp / precision_denominator if precision_denominator else 0.0
        recall = tp / recall_denominator if recall_denominator else 0.0
        f1_denominator = precision + recall
        factual_f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0
        vectors = await self.embedder.embed_texts([answer, expected_answer])
        if vectors is None or len(vectors) != 2:
            raise ValueError("Embedder must return two vectors.")
        semantic_score = float(cosine_similarity(vectors[0], vectors[1]))
        if not math.isfinite(semantic_score):
            raise ValueError("Semantic similarity must be finite.")
        semantic_score = max(0.0, min(1.0, semantic_score))
        score = float(factual_weight * factual_f1 + semantic_weight * semantic_score)
        return AnswerQualityScores(
            correctness=score,
            completeness=recall,
            accuracy=factual_f1,
        )

    async def get_answer(
        self, query: str, citations: list[VectorRecord] | None = None
    ) -> tuple[str, list[VectorRecord]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        try:
            await asyncio.sleep(0.5)
            response = await self.assistant.answer(
                query, citations=citations, reset_history=True
            )
        except LLMGenerationError as e:
            raise LLMGenerationError(f"Error occurred while fetching answer: {e}")
        if not isinstance(response, AssistantResponse):
            raise TypeError(
                "Assistant response must be an instance of AssistantResponse."
            )
        return response.answer, response.citations

    async def noise_sensitivity(
        self, clean_quality: AnswerQualityScores, noisy_quality: AnswerQualityScores
    ) -> float:
        clean_score = (
            clean_quality.correctness
            + clean_quality.completeness
            + clean_quality.accuracy
        ) / 3
        noisy_score = (
            noisy_quality.correctness
            + noisy_quality.completeness
            + noisy_quality.accuracy
        ) / 3
        if clean_score == 0:
            return 0.0
        sensitivity = (clean_score - noisy_score) / clean_score
        return max(0.0, min(1.0, float(sensitivity)))

    async def evaluate(self) -> dict[str, dict[str, float]]:
        questions = list(self.ground_truth.keys())
        if not questions:
            raise ValueError("No questions found in ground truth for evaluation.")
        faithfulness_scores = []
        answer_relevancy_scores = []
        answer_correctness_scores = []
        completeness_scores = []
        answer_accuracy_scores = []
        noise_sensitivity_scores = []
        for question in questions:
            answer_tuple = await self.get_answer(question)
            noisly_answer_tuple = await self.get_answer(
                question, citations=self.noise_chunks + answer_tuple[1]
            )
            answer = answer_tuple[0]
            retrieved_chunks = [
                f"[Citation {i+1}] {c.text}" for i, c in enumerate(answer_tuple[1])
            ]
            expected_answer = self.ground_truth[question].answer
            noisly_answer = noisly_answer_tuple[0]
            if retrieved_chunks:
                faithfulness_score = await self.faithfulness(
                    answer,
                    retrieved_chunks,
                    self.faithfulness_prompt,
                    self.faithfulness_response_format,
                )
            else:
                logger.warning(
                    "No citations were returned for question; "
                    "assigning a faithfulness score of 0.0: %s",
                    question,
                )
                faithfulness_score = 0.0
            answer_relevancy_score = await self.answer_relevancy(
                question,
                answer,
                self.answer_relevancy_prompt,
                self.answer_relevancy_response_format,
            )
            answer_quality_scores = await self.answer_quality(
                question,
                answer,
                expected_answer,
                self.answer_quality_prompt,
                self.answer_quality_response_format,
            )
            noisely_answer_quality_scores = await self.answer_quality(
                question,
                noisly_answer,
                expected_answer,
                self.answer_quality_prompt,
                self.answer_quality_response_format,
            )
            noise_sensitivity_score = await self.noise_sensitivity(
                answer_quality_scores, noisely_answer_quality_scores
            )
            faithfulness_scores.append(faithfulness_score)
            answer_relevancy_scores.append(answer_relevancy_score)
            answer_correctness_scores.append(answer_quality_scores.correctness)
            completeness_scores.append(answer_quality_scores.completeness)
            answer_accuracy_scores.append(answer_quality_scores.accuracy)
            noise_sensitivity_scores.append(noise_sensitivity_score)
        num_questions = len(questions)
        if num_questions > 0:
            GENERATION_EVAL_RESULTS_SCHEMA["Generation Evaluation Metrics"][
                "faithfulness_score"
            ] = (sum(faithfulness_scores) / num_questions)
            GENERATION_EVAL_RESULTS_SCHEMA["Generation Evaluation Metrics"][
                "answer_relevancy_score"
            ] = (sum(answer_relevancy_scores) / num_questions)
            GENERATION_EVAL_RESULTS_SCHEMA["Generation Evaluation Metrics"][
                "answer_correctness_score"
            ] = (sum(answer_correctness_scores) / num_questions)
            GENERATION_EVAL_RESULTS_SCHEMA["Generation Evaluation Metrics"][
                "completeness_score"
            ] = (sum(completeness_scores) / num_questions)
            GENERATION_EVAL_RESULTS_SCHEMA["Generation Evaluation Metrics"][
                "answer_accuracy_score"
            ] = (sum(answer_accuracy_scores) / num_questions)
            GENERATION_EVAL_RESULTS_SCHEMA["Generation Evaluation Metrics"][
                "noise_sensitivity_score"
            ] = (sum(noise_sensitivity_scores) / num_questions)
        logger.info(f"Generation Evaluation Metrics: {GENERATION_EVAL_RESULTS_SCHEMA}")
        return GENERATION_EVAL_RESULTS_SCHEMA
