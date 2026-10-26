from dataclasses import dataclass
from typing import Any


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]
