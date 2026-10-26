from __future__ import annotations

import shutil
from typing import Any

from manim import BOLD, WHITE, ManimColor, MathTex, Scene, Text

from rag_playground.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BG_COLOR: str = "#000000"


def set_default_bg_color(color: str) -> None:
    """Tüm sahneler için varsayılan arka plan rengini değiştirir."""
    global DEFAULT_BG_COLOR
    DEFAULT_BG_COLOR = color


def get_default_bg_color() -> str:
    """Mevcut varsayılan arka plan rengini döndürür."""
    return DEFAULT_BG_COLOR


class RAGScene(Scene):
    """
    Tüm RAG eğitim animasyonları için generic taban sınıf (Base Scene).

    Özellikler:
    - Notebook hücresinden gönderilen tüm dinamik verileri (`data` sözlüğü veya `**kwargs`)
      `self.data` üzerinden sahneye aktarır.
    - `self.get_param("anahtar", varsayilan)` ile güvenli veri okuma sağlar.
    - Varsayılan olarak siyah (`DEFAULT_BG_COLOR`) arka plan kullanır;
      istenirse sahne bazında veya global olarak değiştirilebilir.
    """

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        bg_color: str | None = None,
        **scene_kwargs: Any,
    ) -> None:
        self.data: dict[str, Any] = dict(data) if data else {}
        self._custom_bg_color: str = bg_color or DEFAULT_BG_COLOR
        super().__init__(**scene_kwargs)
        self.camera.background_color = ManimColor(self._custom_bg_color)

    def get_param(self, key: str, default: Any = None) -> Any:
        """Notebook'tan gönderilen canlı veriden bir parametreyi okur."""
        return self.data.get(key, default)

    @staticmethod
    def create_math(
        latex_str: str,
        fallback_text: str,
        font_size: int = 24,
        color: Any = WHITE,
    ) -> MathTex | Text:
        """LaTeX kuruluysa MathTex, değilse Text döndürür."""
        if shutil.which("latex"):
            try:
                return MathTex(latex_str, font_size=font_size, color=color)
            except Exception:
                logger.warning(
                    "Error occurred while rendering MathLaTeX. Returning fallback text instead."
                )
        return Text(fallback_text, font_size=font_size, color=color, weight=BOLD)
