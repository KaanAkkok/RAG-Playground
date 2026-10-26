from dataclasses import dataclass


@dataclass
class Memory:
    point_id: str = ""
    memory_type: str = ""
    content: str = ""
    inference: str = ""
    confidence: float = 0.0
    created_at: str = ""
