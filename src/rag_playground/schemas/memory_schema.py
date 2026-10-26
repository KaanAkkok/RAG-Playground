from typing import Literal

from pydantic import BaseModel, Field


class MemoryResponse(BaseModel):
    model_config = {"extra": "forbid"}
    memory_type: Literal[
        "fact", "event", "preference", "goal", "inference", "no_extraction", "unknown"
    ] = Field(
        ...,
        description="The type of memory, e.g., 'fact', 'event', 'preference', 'goal', 'inference', or 'no_extraction'",
    )
    content: str = Field(..., description="The summary of the memory")
    inference: str | None = Field(
        ..., description="Inference or conclusion drawn from the memory"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level of the inference"
    )


class MemoryExtraction(BaseModel):
    model_config = {"extra": "forbid"}
    memories: list[MemoryResponse] = Field(
        ...,
        description="Long-term memories extracted from the conversation; empty if nothing worth remembering",
    )


_memory_schema = MemoryExtraction.model_json_schema()
_memory_schema["additionalProperties"] = False


LLM_MEMORY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_extraction",
        "strict": True,
        "schema": _memory_schema,
    },
}
