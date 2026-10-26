from pydantic import BaseModel, ConfigDict, Field


class GuardrailsResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    faithfulness: bool = Field(
        ..., description="Whether the answer is faithful to the provided context."
    )
    relevancy: bool = Field(
        ..., description="Whether the answer is relevant to the question."
    )
    hallucination: bool = Field(
        ..., description="Whether the answer contains hallucinated information."
    )
    nfsw: bool = Field(..., description="Whether the answer contains NSFW content.")
    contain_sensitive_info: bool = Field(
        ..., description="Whether the answer contains sensitive information."
    )
    feedback: str | None = Field(
        ...,
        description="Feedback or comments regarding the answer; use null when there is no feedback.",
    )


_guardrails_schema = GuardrailsResponseSchema.model_json_schema()
_guardrails_schema["additionalProperties"] = False


LLM_GUARDRAILS_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "guardrails",
        "strict": True,
        "schema": _guardrails_schema,
    },
}
