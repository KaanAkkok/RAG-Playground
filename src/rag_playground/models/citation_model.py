from dataclasses import dataclass


@dataclass
class Citation:
    point_id: str = ""
    document_id: str = ""
    chunk_id: str = ""
    document: str = ""
    page: int = 0
    paragraph: int = 0
    sentences: int = 0
    text_length: int = 0
    breadcrumb: str = ""
    text: str = ""
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    doc_type: str = ""
    doc_title: str = ""
    doc_source: str = ""
