from __future__ import annotations

import base64
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from IPython.display import HTML, display
from manim import Scene, tempconfig

from manimations.base import RAGScene, get_default_bg_color
from rag_playground.core.logger import get_logger
from rag_playground.core.settings import settings

logger = get_logger(__name__)

# Sağ panelin varsayılan genişliği (px veya vw olarak ayarlanabilir)
DEFAULT_PANEL_WIDTH: int = 680
DEFAULT_PANEL_TITLE: str = "Animation"


def _serialize_for_hash(obj: Any) -> Any:
    """
    Herhangi bir Python nesnesini (dict, list, str, float, Pydantic BaseModel,
    dataclass, set, tuple vb.) deterministik JSON serileştirmeye uygun hale getirir.
    Böylece hangi animasyonda hangi veri tipi değişirse değişsin önbellek (cache)
    anahtarı doğru hesaplanır.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_serialize_for_hash(item) for item in obj]
    if isinstance(obj, dict):
        return {
            str(k): _serialize_for_hash(v)
            for k, v in sorted(obj.items(), key=lambda x: str(x[0]))
        }
    # Pydantic v2 BaseModel
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return _serialize_for_hash(obj.model_dump())
    # Pydantic v1 BaseModel
    if hasattr(obj, "dict") and callable(obj.dict):
        return _serialize_for_hash(obj.dict())
    # Python dataclass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _serialize_for_hash(dataclasses.asdict(obj))
    # NumPy array / Torch tensor benzeri nesneler
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return _serialize_for_hash(tolist())
    # Genel nesneler (__dict__ varsa)
    if hasattr(obj, "__dict__"):
        return _serialize_for_hash(vars(obj))
    return repr(obj)


def compute_render_hash(
    scene_cls: type[Scene],
    data: dict[str, Any],
    quality: str,
    bg_color: str,
) -> str:
    """Sahne kaynak kodu + canlı girdi verileri + kalite + arka plan rengi için tekil hash üretir."""
    try:
        source_code = inspect.getsource(scene_cls)
    except (OSError, TypeError):
        source_code = scene_cls.__name__

    payload = {
        "scene_name": scene_cls.__name__,
        "source_code": source_code,
        "data": _serialize_for_hash(data),
        "quality": quality,
        "bg_color": bg_color,
    }
    raw_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:16]


def get_cache_dir() -> Path:
    """Render edilen videoların tutulacağı önbellek dizinini döndürür."""
    cache_dir = Path(settings.ANIMATIONS_DIR) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_prebuilt_video_path(scene_cls: type[Scene]) -> Path:
    """Önceden render edilmiş (pre-rendered) sabit sahne videolarının yolunu döndürür."""
    prebuilt_dir = Path(settings.ANIMATIONS_DIR)
    prebuilt_dir.mkdir(parents=True, exist_ok=True)
    return prebuilt_dir / f"{scene_cls.__name__}.mp4"


def render_scene_live(
    scene_cls: type[Scene],
    data: dict[str, Any] | None = None,
    quality: str = "low_quality",
    bg_color: str | None = None,
    force_render: bool = False,
) -> Path:
    """
    Verilen Manim sahnesini çalıştırır:
    - Eğer özel `data` verilmemişse ve `ANIMATIONS_DIR/<SceneName>.mp4` altında
      önceden render edilmiş (pre-rendered) video varsa hiç beklemeden anında onu döndürür.
    - Aynı girdi verisiyle daha önce render edilmişse önbellekten (cache) anında döndürür.
    - Yoksa canlı render alır.
    """
    resolved_data = dict(data) if data else {}
    resolved_bg = bg_color or get_default_bg_color()

    if not resolved_data and not force_render:
        prebuilt = get_prebuilt_video_path(scene_cls)
        if prebuilt.exists():
            return prebuilt

    render_hash = compute_render_hash(scene_cls, resolved_data, quality, resolved_bg)

    cache_dir = get_cache_dir()
    output_filename = f"{scene_cls.__name__}_{render_hash}"
    target_video = cache_dir / f"{output_filename}.mp4"

    if target_video.exists() and not force_render:
        return target_video

    if not resolved_data and not force_render:
        existing_cached = sorted(
            cache_dir.glob(f"{scene_cls.__name__}_*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if existing_cached:
            return existing_cached[0]

    media_dir = cache_dir / "_manim_work"
    media_dir.mkdir(parents=True, exist_ok=True)

    config_overrides: dict[str, Any] = {
        "quality": quality,
        "background_color": resolved_bg,
        "media_dir": str(media_dir),
        "output_file": output_filename,
        "disable_caching": True,
        "flush_cache": False,
        "write_to_movie": True,
        "format": "mp4",
        "verbosity": "WARNING",
        "progress_bar": "none",
    }

    with tempconfig(config_overrides):
        if issubclass(scene_cls, RAGScene):
            scene = scene_cls(data=resolved_data, bg_color=resolved_bg)
        else:
            scene = scene_cls()
            scene.data = resolved_data  # type: ignore[attr-defined]
        scene.render()
        rendered_path = Path(scene.renderer.file_writer.movie_file_path)

    if rendered_path.exists() and rendered_path.resolve() != target_video.resolve():
        target_video.write_bytes(rendered_path.read_bytes())
        return target_video

    return rendered_path


def _build_panel_resize_html(width: int, bg_color: str = "#000000") -> str:
    """
    JupyterLab sağ Sidecar panelinin çok dar açılmasını engelleyen ve
    panelin tamamının arka planını (kenar boşlukları dahil) siyah yapan CSS/JS.
    """
    return f"""
    <style>
        /* JupyterLab Sidecar sağ panelini önceden tanımlı genişlikte ve tam siyah arka planda tut */
        .jp-Sidecar,
        .jp-Sidecar .lm-Widget,
        .jp-Sidecar .p-Widget,
        .jp-Sidecar .jp-OutputArea,
        .jp-Sidecar .jp-OutputArea-child,
        .jp-Sidecar .jp-OutputArea-output,
        .jp-Sidecar .jp-RenderedHTMLCommon,
        .lm-DockPanel-widget:has(.manim-side-panel-root),
        .p-DockPanel-widget:has(.manim-side-panel-root) {{
            background-color: {bg_color} !important;
            background: {bg_color} !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .jp-Sidecar,
        .lm-DockPanel-widget:has(.manim-side-panel-root),
        .p-DockPanel-widget:has(.manim-side-panel-root) {{
            min-width: {width}px !important;
        }}
        .jp-Sidecar .jp-OutputArea-prompt {{
            display: none !important;
        }}
        .jp-Sidecar .jp-OutputArea-output,
        .jp-Sidecar .jp-OutputArea-child,
        .jp-Sidecar .lm-Widget {{
            height: 100%;
        }}
    </style>
    <script>
        (function() {{
            const targetWidth = {width};
            const bgColor = "{bg_color}";
            const root = document.querySelector('.manim-side-panel-root');
            if (!root) return;
            const sidecar = root.closest('.jp-Sidecar') || root.closest('.lm-DockPanel-widget');
            if (sidecar) {{
                sidecar.style.minWidth = targetWidth + 'px';
                sidecar.style.width = targetWidth + 'px';
                sidecar.style.backgroundColor = bgColor;
                window.dispatchEvent(new Event('resize'));
            }}
        }})();
    </script>
    """


def build_video_html(
    video_path: Path,
    panel_width: int = DEFAULT_PANEL_WIDTH,
    bg_color: str = "#000000",
) -> HTML:
    """
    Şatafatsız, sadece videoyu sağ panelin tam ortasında ve siyah arka planla
    gösteren sade HTML oynatıcı.
    """
    raw_bytes = video_path.read_bytes()
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    resize_snippet = _build_panel_resize_html(panel_width, bg_color=bg_color)

    html_code = f"""
    {resize_snippet}
    <div class="manim-side-panel-root" style="
        width: 100%;
        height: 100%;
        min-height: calc(100vh - 40px);
        background-color: {bg_color};
        display: flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
        padding: 16px;
        margin: 0;
    ">
        <video controls autoplay loop muted playsinline style="
            width: 100%;
            max-width: 100%;
            max-height: calc(100vh - 120px);
            background-color: {bg_color};
            display: block;
            outline: none;
        ">
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            Tarayıcınız video etiketini desteklemiyor.
        </video>
    </div>
    """
    return HTML(html_code)


def _build_loading_html(
    panel_width: int = DEFAULT_PANEL_WIDTH,
    bg_color: str = "#000000",
) -> HTML:
    """Canlı render sürerken sağ panelin ortasında gösterilecek sade durum ekranı."""
    resize_snippet = _build_panel_resize_html(panel_width, bg_color=bg_color)
    return HTML(f"""
        {resize_snippet}
        <div class="manim-side-panel-root" style="
            width: 100%;
            height: 100%;
            min-height: calc(100vh - 40px);
            background-color: {bg_color};
            color: #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 15px;
            box-sizing: border-box;
            margin: 0;
        ">
            <span>Animasyon canlı render ediliyor...</span>
        </div>
        """)


class SidePanelManager:
    """
    JupyterLab sağ panelini (Sidecar) tekil (singleton) olarak yönetir.

    - Hücre altında (inline output) kesinlikle kopya çıktı oluşturmaz.
    - İlk çağrıda sağ paneli açar, sonraki tüm hücrelerde aynı sağ panelin içeriğini günceller.
    - Sağ panelin genişliğini otomatik olarak `panel_width` değerine ayarlar.
    """

    def __init__(self) -> None:
        self._sidecar: Any = None
        self.panel_width: int = DEFAULT_PANEL_WIDTH
        self.panel_title: str = DEFAULT_PANEL_TITLE
        self._last_content: Any = None
        self._last_title: str | None = None

    def configure(
        self,
        *,
        panel_width: int | None = None,
        panel_title: str | None = None,
    ) -> None:
        if panel_width is not None:
            self.panel_width = panel_width
        if panel_title is not None:
            self.panel_title = panel_title
            if self._sidecar is not None:
                self._sidecar.title = panel_title

    def _ensure_sidecar(self, title: str | None = None) -> Any:
        from sidecar import Sidecar

        if self._sidecar is None:
            self._sidecar = Sidecar(
                title=title or self.panel_title,
                anchor="right",
            )
        elif title is not None:
            self._sidecar.title = title
        return self._sidecar

    def display_in_panel(
        self,
        widget_or_html: Any,
        title: str | None = None,
        *,
        remember: bool = True,
    ) -> None:
        """
        Verilen içeriği sağ Sidecar panelinde gösterir.
        """
        if remember:
            self._last_content = widget_or_html
            self._last_title = title or self.panel_title
        sc = self._ensure_sidecar(title=title)
        sc.clear_output(wait=True)
        with sc:
            display(widget_or_html)

    def reset(self) -> None:
        """Paneli sıfırlamak gerekirse mevcut Sidecar referansını kapatır."""
        if self._sidecar is not None:
            try:
                self._sidecar.close()
            except Exception:
                logger.warning(
                    "Sidecar paneli kapatılırken bir hata oluştu. Bu, JupyterLab'ın bazı sürümlerinde normaldir."
                )
            self._sidecar = None

    def reopen(
        self,
        widget_or_html: Any | None = None,
        title: str | None = None,
    ) -> None:
        """Kapatılmış olan sağ paneli sıfırlayıp yeniden açar ve içeriği yükler."""
        content = widget_or_html if widget_or_html is not None else self._last_content
        resolved_title = (
            title if title is not None else (self._last_title or self.panel_title)
        )
        self.reset()
        if content is not None:
            self.display_in_panel(content, title=resolved_title)


# Tekil (Singleton) panel yöneticisi
_panel_manager = SidePanelManager()


def reopen_panel() -> None:
    """Son gösterilen animasyonla sağ paneli yeniden açar."""
    _panel_manager.reopen()


def _display_reopen_button(video_html: HTML, title: str | None = None) -> None:
    """
    Hücre altında küçük bir 'Animasyon Panelini Aç' butonu oluşturur.
    Kullanıcı sağdaki Sidecar sekmesini kapatsa bile bu butona tıklayarak
    yeniden render beklemeden paneli anında geri açabilir.
    """
    import ipywidgets as widgets

    btn = widgets.Button(
        description="Animasyon Paneli",
        tooltip="Sağdaki animasyon panelini yeniden açar",
        layout=widgets.Layout(width="150px", height="34px", margin="4px 0 0 0"),
    )
    btn.style.text_color = "#000000"
    btn.style.font_weight = "bold"

    def _on_click(_: Any) -> None:
        _panel_manager.reopen(video_html, title=title)

    try:
        btn.on_click(_on_click)
    except Exception:
        logger.warning(
            "Animasyon paneli butonu oluşturulamadı. Bu, JupyterLab'ın bazı sürümlerinde normaldir."
        )
        return
    display(btn)


def set_panel_config(
    *,
    width: int | None = None,
    title: str | None = None,
    bg_color: str | None = None,
) -> None:
    """
    Sağ panel genişliğini, başlığını veya varsayılan arka plan rengini
    notebook'un başında tek seferde ayarlamak için yardımcı fonksiyon.
    """
    _panel_manager.configure(panel_width=width, panel_title=title)
    if bg_color is not None:
        from manimations.base import set_default_bg_color

        set_default_bg_color(bg_color)


def show_animation(
    scene_or_video: type[Scene] | str | Path,
    data: dict[str, Any] | None = None,
    *,
    title: str | None = None,
    width: int | None = None,
    quality: str = "high_quality",
    bg_color: str | None = None,
    force_render: bool = False,
    show_button: bool = True,
    **extra_data: Any,
) -> None:
    """
    Notebook hücresinden çağrılan ana fonksiyon.

    Kullanım 1 (Canlı Manim Sahnesi + Dinamik Veri):
        show_animation(ChunkingScene, text=girilen_metin, chunks=chunk_listesi)
        # veya
        show_animation(BM25Scene, data={"query": sorgu, "scores": skorlar})

    Kullanım 2 (Hazır Video Dosyası):
        show_animation("path/to/video.mp4")

    Davranış:
    - Sağdaki Sidecar panelinde açılır.
    - Hücre altında küçük bir "🎬 Animasyon Panelini Aç" butonu gösterir;
      böylece sağ panel kapatılsa bile tek tıkla (yeniden render etmeden) geri açılır.
    """
    panel_width = width or _panel_manager.panel_width
    resolved_bg = bg_color or get_default_bg_color()

    merged_data: dict[str, Any] = {}
    if data:
        merged_data.update(data)
    if extra_data:
        merged_data.update(extra_data)

    # Eğer doğrudan bir video dosya yolu verildiyse:
    if isinstance(scene_or_video, (str, Path)):
        video_path = Path(scene_or_video).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Animasyon videosu bulunamadı: {video_path}")
    elif isinstance(scene_or_video, type) and issubclass(scene_or_video, Scene):
        # Canlı render başlamadan önce sağ panelde sade "render ediliyor" mesajı göster
        _panel_manager.display_in_panel(
            _build_loading_html(panel_width=panel_width, bg_color=resolved_bg),
            title=title,
            remember=False,
        )
        video_path = render_scene_live(
            scene_cls=scene_or_video,
            data=merged_data,
            quality=quality,
            bg_color=resolved_bg,
            force_render=force_render,
        )
    else:
        raise TypeError(
            "scene_or_video bir Manim Scene sınıfı (örn. RAGScene alt sınıfı) "
            "veya bir video dosya yolu (str | Path) olmalıdır."
        )

    video_html = build_video_html(
        video_path=video_path,
        panel_width=panel_width,
        bg_color=resolved_bg,
    )
    _panel_manager.display_in_panel(video_html, title=title)
    if show_button:
        _display_reopen_button(video_html, title=title)
