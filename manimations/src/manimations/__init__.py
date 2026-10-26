from __future__ import annotations

from manimations.base import (
    DEFAULT_BG_COLOR,
    RAGScene,
    get_default_bg_color,
    set_default_bg_color,
)
from manimations.chunking_anim import ChunkingScene
from manimations.dense_search_anim import DenseSearchScene
from manimations.embedding_anim import EmbeddingScene
from manimations.evaluation_anim import (
    AnswerCorrectnessScene,
    AnswerQualityFactualScene,
    AnswerRelevancyScene,
    CalcScene,
    ContextPrecisionScene,
    ContextRecallScene,
    F1ScoreScene,
    FaithfulnessScene,
    HitRateScene,
    MRRScene,
    NDCGScene,
    NoiseSensitivityScene,
)
from manimations.generation_anim import GenerationScene
from manimations.guardrails_anim import GuardrailsScene
from manimations.hybrid_search_anim import HybridSearchScene
from manimations.intent_classification_anim import IntentClassificationScene
from manimations.memory_extraction_anim import MemoryExtractionScene
from manimations.panel import (
    DEFAULT_PANEL_TITLE,
    DEFAULT_PANEL_WIDTH,
    SidePanelManager,
    build_video_html,
    compute_render_hash,
    get_prebuilt_video_path,
    render_scene_live,
    reopen_panel,
    set_panel_config,
    show_animation,
)
from manimations.reranker_anim import RerankerScene
from manimations.rrf_anim import RRFScene
from manimations.sparse_search_anim import SparseSearchScene
from manimations.text_extraction_anim import TextExtractionScene

__all__ = [
    "DEFAULT_BG_COLOR",
    "DEFAULT_PANEL_TITLE",
    "DEFAULT_PANEL_WIDTH",
    "AnswerCorrectnessScene",
    "AnswerQualityFactualScene",
    "AnswerRelevancyScene",
    "CalcScene",
    "ChunkingScene",
    "ContextPrecisionScene",
    "ContextRecallScene",
    "DenseSearchScene",
    "EmbeddingScene",
    "F1ScoreScene",
    "FaithfulnessScene",
    "GenerationScene",
    "GuardrailsScene",
    "HitRateScene",
    "HybridSearchScene",
    "IntentClassificationScene",
    "MRRScene",
    "MemoryExtractionScene",
    "NDCGScene",
    "NoiseSensitivityScene",
    "RAGScene",
    "RRFScene",
    "RerankerScene",
    "SidePanelManager",
    "SparseSearchScene",
    "TextExtractionScene",
    "build_video_html",
    "compute_render_hash",
    "get_default_bg_color",
    "get_prebuilt_video_path",
    "render_scene_live",
    "reopen_panel",
    "set_default_bg_color",
    "set_panel_config",
    "show_animation",
]
