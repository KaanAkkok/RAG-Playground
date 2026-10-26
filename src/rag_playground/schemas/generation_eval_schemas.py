from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# =========================== CLAIM SCHEMA ===========================


class ClaimSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(..., description="The claim made in the answer.")
    claim_type: Literal["supported", "unsupported"] = Field(
        ..., description="The type of the claim."
    )
    classification: Literal["TP", "FP", "FN"] = Field(
        description=(
            "TP: İddia hem cevapta hem referansta mevcut."
            "FP: İddia cevapta mevcut ama referansta yok."
            "FN: İddia referansta mevcut ama cevapta yok."
        )
    )

    @field_validator("claim", mode="before")
    def strip_and_lower(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("claim_type", mode="before")
    def validate_claim_type(cls, value):
        if value not in ["supported", "unsupported"]:
            raise ValueError(
                f"Invalid claim type: {value}. Must be one of 'supported' or 'unsupported'."
            )
        return value

    @field_validator("classification", mode="before")
    def validate_classification(cls, value):
        if value not in ["TP", "FP", "FN"]:
            raise ValueError(
                f"Invalid classification: {value}. Must be one of 'TP', 'FP', or 'FN'."
            )
        return value


# =========================== FAITHFULNESS EVALUATION SCHEMAS ===========================


class FaithfulnessEvalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[ClaimSchema] = Field(..., description="The claims made in the answer.")


_faithfulness_schema = FaithfulnessEvalResponse.model_json_schema()
_faithfulness_schema["additionalProperties"] = False

LLM_FAITHFULNESS_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "faithfulness",
        "strict": True,
        "schema": _faithfulness_schema,
    },
}


# =========================== ANSWER RELEVANCY EVALUATION SCHEMAS ===========================


class GeneratedQuestionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., description="The generated question.")
    is_relevant: bool = Field(
        ..., description="Whether the generated question is relevant to the answer."
    )

    @field_validator("question", mode="before")
    def strip_and_lower(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value


class AnswerRelevancyEvalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generated_questions: list[GeneratedQuestionSchema] = Field(
        ..., description="The questions generated from the answer."
    )

    @field_validator("generated_questions", mode="before")
    def strip_and_lower(cls, value):
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("Exactly three generated questions are required.")
        if not all(q.strip() for q in value if isinstance(q, str)):
            raise ValueError(
                "Generated questions cannot contain empty or whitespace-only strings"
            )
        return value


_answer_relevancy_schema = AnswerRelevancyEvalResponse.model_json_schema()
_answer_relevancy_schema["additionalProperties"] = False

LLM_ANSWER_RELEVANCY_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_relevancy",
        "strict": True,
        "schema": _answer_relevancy_schema,
    },
}


# =========================== ANSWER QUALITY EVALUATION SCHEMAS ===========================


class AnswerQualityScores(BaseModel):
    correctness: float
    completeness: float
    accuracy: float


class AnswerQualityEvalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[ClaimSchema] = Field(..., description="The claims made in the answer.")
    is_correct: bool = Field(
        ..., description="Whether the answer is correct based on the expected answer."
    )


_answer_quality_schema = AnswerQualityEvalResponse.model_json_schema()
_answer_quality_schema["additionalProperties"] = False

LLM_ANSWER_QUALITY_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_quality",
        "strict": True,
        "schema": _answer_quality_schema,
    },
}


# =========================== GENERATION EVALUATION SCHEMAS ===========================


class GenerationEvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(..., description="The model used for evaluation.")
    messages: list[dict] = Field(..., description="The messages used for evaluation.")
    temperature: float = Field(..., description="The temperature used for evaluation.")
    response_format: dict = Field(
        ..., description="The response format used for evaluation."
    )

    @field_validator(
        "model", "messages", "temperature", "response_format", mode="before"
    )
    def validate_fields(cls, value):
        if value is None:
            raise ValueError("Field cannot be None")
        if isinstance(value, float) and (value < 0 or value > 1):
            raise ValueError("Temperature must be between 0 and 1")
        if isinstance(value, str):
            return value.strip()
        elif isinstance(value, list):
            return [v for v in value if v is not None]
        elif isinstance(value, dict):
            return {k: v for k, v in value.items() if v is not None}
        return value
