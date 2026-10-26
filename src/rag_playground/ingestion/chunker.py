import hashlib
import re
import unicodedata
from hashlib import sha256
from pathlib import Path

from rag_playground.core.exceptions import EmbeddingError
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings
from rag_playground.ingestion.extractor import PAGE_BREAK_PLACEHOLDER
from rag_playground.models.citation_model import Citation
from rag_playground.retrieval.embedding_service import EmbeddingService
from rag_playground.retrieval.similarity import cosine_similarity

logger = get_logger(__name__)


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
IMAGE_TAG_RE = re.compile(r"<!-- image -->")
HYPHEN_BREAK_RE = re.compile(r"(?<=\w)-\n(?=[a-zçğıöşü])")


def clean_markdown(md: str) -> str:
    text = unicodedata.normalize("NFC", md)
    text = text.replace("\r", "")
    text = INVISIBLE_RE.sub("", text)
    text = HYPHEN_BREAK_RE.sub("", text)
    text = IMAGE_TAG_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


class DocumentChunker:
    def __init__(
        self,
        embedder: EmbeddingService,
        max_chunk_chars: int = settings.MAX_CHUNK_CHARS,
        min_chunk_chars: int = settings.MIN_CHUNK_CHARS,
        sentence_overlap_ratio: float = settings.SENTENCE_OVERLAP_RATIO,
        semantic_chunking_break_percentage: float = settings.SEMANTIC_CHUNKING_BREAK_PERCENTAGE,
        semantic_chunking_min_dist: float = settings.SEMANTIC_CHUNKING_MIN_DIST,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.sentence_overlap_ratio = sentence_overlap_ratio
        self.semantic_chunking_break_percentage = semantic_chunking_break_percentage
        self.semantic_chunking_min_dist = semantic_chunking_min_dist
        self.embedder = embedder
        self.currend_document_name_hash = ""
        self.current_chunk_id = 1

    def _parse_md_blocks(self, content: str) -> list[dict]:
        blocks: list[dict] = []
        para: list[str] = []
        table: list[str] = []

        def flush_para():
            if para:
                blocks.append(
                    {
                        "type": "text",
                        "content": "\n".join(para).strip(),
                    }
                )
                para.clear()

        def flush_table():
            if table:
                blocks.append(
                    {
                        "type": "table",
                        "content": "\n".join(table).strip(),
                    }
                )
                table.clear()

        for line in content.splitlines():
            heading_match = HEADING_RE.match(line)
            table_row_match = TABLE_ROW_RE.match(line)
            if heading_match:
                flush_para()
                flush_table()
                blocks.append(
                    {
                        "type": "heading",
                        "level": len(heading_match.group(1)),
                        "content": heading_match.group(2).strip(),
                    }
                )
            elif table_row_match:
                flush_para()
                table.append(line)
            elif not line.strip():
                flush_para()
                flush_table()
            else:
                flush_table()
                para.append(line)

        flush_para()
        flush_table()
        return [b for b in blocks if b["content"]]

    def _split_sentence(self, text: str) -> list[str]:
        raw = re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])", text.strip())
        sentences: list[str] = []
        for sentence in raw:
            while len(sentence) > self.max_chunk_chars:
                cut = sentence.rfind(" ", 0, self.max_chunk_chars)
                cut = cut if cut != -1 else len(sentence)
                sentences.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                sentences.append(sentence)
        return sentences

    # for improvement use cell based splitting for tables
    def _split_table(self, table: str) -> list[str]:
        lines = table.splitlines()
        if len(lines) <= 2 or len(table) <= self.max_chunk_chars:
            return [table.strip()]
        header = "\n".join(lines[:2])
        parts: list[str] = []
        current_part: list[str] = [header]
        for line in lines[2:]:
            if len("\n".join(current_part + [line])) > self.max_chunk_chars:
                parts.append("\n".join(current_part).strip())
                current_part = [header, line]
            else:
                current_part.append(line)
        if current_part:
            parts.append("\n".join(current_part).strip())
        return parts

    def _pack_by_size(self, sentences: list[str]) -> list[list[str]]:
        chunks: list[list[str]] = []
        current: list[str] = []
        for sentence in sentences:
            if (
                current
                and len(" ".join(current)) + 1 + len(sentence) > self.max_chunk_chars
            ):
                chunks.append(current)
                current = []
            current.append(sentence)
        if current:
            chunks.append(current)
        return chunks

    def _add_overlap(self, groups: list[list[str]]) -> list[list[str]]:
        for i in range(1, len(groups)):
            overlap_count = max(
                1, int(len(groups[i - 1]) * self.sentence_overlap_ratio)
            )
            overlap = groups[i - 1][-overlap_count:]
            candidate = overlap + groups[i]
            if len(" ".join(candidate)) <= self.max_chunk_chars:
                groups[i] = candidate
        return groups

    async def _semantic_split(self, sentences: list[str]) -> list[list[str]]:
        if not sentences:
            return []
        if len(" ".join(sentences)) <= self.max_chunk_chars:
            return [sentences]
        try:
            embeddings = await self.embedder.embed_texts(sentences)
        except EmbeddingError:
            logger.exception(
                "Error embedding sentences, falling back to size-based chunking"
            )
            return self._pack_by_size(sentences)

        distances = [
            (1 - cosine_similarity(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]
        ordered = sorted(distances)
        threshold = ordered[
            min(
                int(len(ordered) * self.semantic_chunking_break_percentage),
                len(ordered) - 1,
            )
        ]

        groups: list[list[str]] = [[sentences[0]]]
        for i, distance in enumerate(distances):
            if distance >= threshold and distance >= self.semantic_chunking_min_dist:
                groups.append([])
            groups[-1].append(sentences[i + 1])

        merged_groups: list[list[str]] = []
        for group in groups:
            if (
                merged_groups
                and len(" ".join(merged_groups[-1] + group)) <= self.max_chunk_chars
            ):
                merged_groups[-1].extend(group)
            else:
                merged_groups.append(group)

        chunks: list[list[str]] = []
        for group in merged_groups:
            if len(" ".join(group)) <= self.max_chunk_chars:
                chunks.append(group)
            else:
                packed = self._pack_by_size(group)
                chunks.extend(packed)

        return self._add_overlap(self._merge_small_groups(chunks))

    def _merge_small_groups(self, groups: list[list[str]]) -> list[list[str]]:
        if len(groups) < 2 or self.min_chunk_chars <= 0:
            return groups
        merged: list[list[str]] = []
        for group in groups:
            if (
                merged
                and len(" ".join(group)) < self.min_chunk_chars
                and len(" ".join(merged[-1] + group)) <= self.max_chunk_chars
            ):
                merged[-1].extend(group)
            else:
                merged.append(group)
        if (
            len(merged) > 1
            and len(" ".join(merged[-1])) < self.min_chunk_chars
            and len(" ".join(merged[-2] + merged[-1])) <= self.max_chunk_chars
        ):
            merged[-2].extend(merged[-1])
            merged.pop()
        return merged

    def catch_page_breaks(self, text: str) -> list[str]:
        if PAGE_BREAK_PLACEHOLDER in text:
            return [t.strip() for t in text.split(PAGE_BREAK_PLACEHOLDER) if t.strip()]
        return [text]

    async def chunk(
        self,
        content: str,
        document_path: Path | str,
    ) -> list[Citation]:
        if not content and not document_path:
            raise ValueError("Either content or document_path must be provided.")
        if isinstance(document_path, str):
            document_path = Path(document_path)
        chunks: list[Citation] = []
        path: list[str] = []
        level_path_list: list[str] = []
        pending: list[str] = []

        level_path_list.append(document_path.stem)

        def hash_string(s: str) -> str:
            return hashlib.sha256(s.encode("utf-8")).hexdigest()

        def emit(body: str, level: str, sentence_count: int = 1, page: int = 1) -> None:
            old_hash = self.currend_document_name_hash
            new_hash = hash_string(str(document_path.stem))
            if old_hash != new_hash:
                self.currend_document_name_hash = new_hash
                self.current_chunk_id = 1
            else:
                self.current_chunk_id += 1
            breadcrumb = " > ".join(p for p in path if p)
            text = f"{breadcrumb}\n\n{body}" if breadcrumb else body
            chunks.append(
                Citation(
                    document_id=sha256(
                        str(document_path.stem).encode("utf-8")
                    ).hexdigest(),
                    chunk_id=f"{document_path.stem}_chunk_{self.current_chunk_id}",
                    document=document_path.name,
                    page=page,
                    paragraph=len(chunks) + 1,
                    sentences=sentence_count,
                    text_length=len(text),
                    level=level,
                    breadcrumb=breadcrumb,
                    text=text,
                    doc_type=document_path.suffix[1:],
                    doc_title=document_path.stem,
                    doc_source=str(document_path),
                )
            )

        async def flush(page: int, level: str) -> None:
            if not pending:
                return
            grouped = await self._semantic_split(list(pending))
            for group in grouped:
                emit(" ".join(group), level, len(group), page)
            pending.clear()

        page_contents = self.catch_page_breaks(content)
        for page_number, page_content in enumerate(page_contents, start=1):
            blocks = self._parse_md_blocks(clean_markdown(page_content))
            total_lines = max(len(page_content.splitlines()), 1)
            processed_lines = 0
            last_logged_percentage = 0
            level_path = " > ".join(level_path_list)
            for block in blocks:
                level_path = " > ".join(path for path in level_path_list)
                if block["type"] == "heading":
                    await flush(page=page_number, level=level_path)
                    level = block["level"]
                    level_path_list.append(block["content"])
                    del path[level - 1 :]
                    path.extend([""] * (level - 1 - len(path)))
                    path.append(block["content"])
                elif block["type"] == "table":
                    await flush(page=page_number, level=level_path)
                    for table_chunk in self._split_table(block["content"]):
                        emit(
                            table_chunk,
                            level_path,
                            len(self._split_sentence(table_chunk)),
                            page_number,
                        )
                else:
                    pending.extend(self._split_sentence(block["content"]))

                processed_lines += len(block["content"].splitlines())
                percentage = min(processed_lines / total_lines * 100, 100)

                milestone = int(percentage // 20) * 20
                if milestone >= last_logged_percentage + 20:
                    last_logged_percentage = milestone
                    logger.info(
                        f"Chunking progress for {document_path.name} (page {page_number}): "
                        f"{percentage:.1f}% of page processed, "
                        f"{len(chunks)} chunks emitted"
                    )

            await flush(page=page_number, level=level_path)

        logger.info(
            f"Chunking completed for {document_path.name}: "
            f"100% processed, {len(chunks)} chunks emitted"
        )
        return chunks
