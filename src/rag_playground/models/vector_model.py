from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorRecord:
    text: str
    payload: dict[str, Any] = field(default_factory=dict)
    key: str | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
