import builtins
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LLMMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"] | str
    content: str

    @field_validator("role", mode="before")
    @classmethod
    def validate_role(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.lower()
            if value in {"system", "user", "assistant"}:
                return value
        raise ValueError("role must be one of 'system', 'user', or 'assistant'")


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "json_object", "json_schema"] = "text"
    name: str | None = None
    json_schema: dict[str, Any] | None = None
    strict: bool = True

    @model_validator(mode="before")
    @classmethod
    def flatten_openai_format(cls, data: Any) -> Any:
        if isinstance(data, dict):
            nested = data.get("json_schema")
            if isinstance(nested, dict) and "schema" in nested:
                return {
                    "type": data.get("type", "json_schema"),
                    "name": nested.get("name"),
                    "json_schema": nested["schema"],
                    "strict": nested.get("strict", True),
                }
        return data

    @model_validator(mode="after")
    def validate_schema(self) -> Self:
        if self.type == "json_schema" and (not self.name or not self.json_schema):
            raise ValueError("json_schema formatı için name ve json_schema gereklidir")
        return self

    @classmethod
    def from_model(
        cls, model: builtins.type[BaseModel], name: str, strict: bool = True
    ) -> "ResponseFormat":
        schema = model.model_json_schema()
        schema["additionalProperties"] = False
        return cls(type="json_schema", name=name, json_schema=schema, strict=strict)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[LLMMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    stop: list[str] | None = None
    seed: int | None = None
    timeout: float | None = Field(default=None, gt=0)
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    provider_options: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str | None
    raw: Any = Field(default=None, exclude=True, repr=False)


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input: list[str] = Field(min_length=1)
    dimensions: int | None = Field(default=None, gt=0)
    timeout: float | None = Field(default=None, gt=0)
    task_type: str | None = (
        None  # Sadece Gemini kullanır (örn. "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY")
    )
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input", mode="before")
    @classmethod
    def wrap_single_input(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value
