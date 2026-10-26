import math
import re

from rag_playground.core.settings import settings


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_terms: dict[str, list[str]] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._doc_terms)

    def _tokenize(self, text: str) -> list[str]:
        for src in ("İ", "I", "ı"):
            text = text.replace(src, "i")
        return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._doc_terms:
            self._remove(doc_id)
        terms = self._tokenize(text)
        self._doc_terms[doc_id] = terms
        self._total_len += len(terms)
        for term in terms:
            postings = self._postings.setdefault(term, {})
            postings[doc_id] = postings.get(doc_id, 0) + 1

    def _remove(self, doc_id: str) -> None:
        if doc_id not in self._doc_terms:
            return
        terms = self._doc_terms[doc_id]
        self._total_len -= len(terms)
        for term in terms:
            if term in self._postings and doc_id in self._postings[term]:
                del self._postings[term][doc_id]
                if not self._postings[term]:
                    del self._postings[term]
        del self._doc_terms[doc_id]

    def search(
        self,
        query: str,
        limit: int = settings.SEARCH_CANDIDATES,
        contents: list[tuple[str, str]] | None = None,
    ) -> list[tuple[str, float]]:
        if query is None or not query.strip():
            return []
        if contents is not None:
            for point_id, content in contents:
                self.add(point_id, content)
        n_docs = len(self._doc_terms)
        if not n_docs or n_docs == 0:
            return []
        avgdl = self._total_len / n_docs
        scores: dict[str, float] = {}
        query_terms = self._tokenize(query)
        for term in query_terms:
            if term not in self._postings:
                continue
            postings = self._postings[term]
            df = len(postings)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in postings.items():
                dl = len(self._doc_terms[doc_id])
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (
                    tf * (self.k1 + 1) / denominator
                )
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
