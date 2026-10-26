from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from manim import *

# Hız sabiti: Tüm animasyon ve bekleme süreleri bu çarpanla ayarlanır.
SPEED: float = 1.0

DEFAULT_TEXT = "Kedi köpek araba otomobil elma armut"

# 3 anlam kümesi:
# - Hayvanlar (Kedi, köpek)
# - Taşıtlar (araba, otomobil)
# - Meyveler (elma, armut)
DEFAULT_VECTORS_3D: list[tuple[float, float, float]] = [
    (1.1, 3.3, 2.5),  # Kedi
    (1.5, 3.0, 2.2),  # köpek
    (3.4, 1.1, 2.4),  # araba
    (3.1, 1.4, 2.7),  # otomobil
    (1.5, 1.2, 0.7),  # elma
    (1.9, 0.9, 0.9),  # armut
]

WORD_PALETTE: list[ManimColor] = [
    BLUE_B,
    TEAL_B,
    ORANGE,
    GOLD_B,
    GREEN_B,
    MAROON_B,
    PURPLE_B,
    PINK,
    YELLOW_B,
    RED_B,
]


class EmbeddingScene(ThreeDScene):
    """
    Bir metindeki kelimelerin embedding ile vektöre çevrilip 3B uzayda nokta
    olarak nasıl temsil edildiğini anlatan 3 adımlı generic ThreeDScene.
    """

    def __init__(
        self,
        text: str | None = None,
        vectors_3d: list[tuple[float, float, float]] | None = None,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_text = text
        self.input_vectors_3d = vectors_3d
        self.data: dict[str, Any] = data or {}

    def rt(self, seconds: float) -> float:
        return max(0.08, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.08, seconds / SPEED))

    def _resolve_inputs(
        self,
    ) -> tuple[str, list[str], list[tuple[float, float, float]]]:
        """Metni ve kelime başına 3B koordinatları hazırlar."""
        raw_text = self.input_text
        if raw_text is None and isinstance(self.data, dict):
            raw_text = self.data.get("text")
        if not raw_text or not str(raw_text).strip():
            raw_text = DEFAULT_TEXT
        raw_text = str(raw_text).strip()

        words = raw_text.split()
        if not words:
            words = DEFAULT_TEXT.split()
            raw_text = DEFAULT_TEXT

        raw_vecs = self.input_vectors_3d
        if raw_vecs is None and isinstance(self.data, dict):
            raw_vecs = self.data.get("vectors_3d")

        if raw_vecs is None and raw_text == DEFAULT_TEXT:
            raw_vecs = DEFAULT_VECTORS_3D

        resolved_vecs: list[tuple[float, float, float]] = []
        if raw_vecs is not None and len(raw_vecs) > 0:
            arr = np.array(raw_vecs, dtype=float)
            # Koordinatları [0.6, 3.6] aralığına normalize et (zaten o aralıktaysa koru)
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            if np.any(mins < 0.3) or np.any(maxs > 3.9):
                span = np.where((maxs - mins) < 1e-6, 1.0, maxs - mins)
                arr = 0.7 + 2.8 * (arr - mins) / span
            for i in range(len(words)):
                if i < len(arr):
                    v = arr[i]
                    resolved_vecs.append((float(v[0]), float(v[1]), float(v[2])))
                else:
                    resolved_vecs.append(self._fallback_vec(words[i], i))
        else:
            for i, w in enumerate(words):
                resolved_vecs.append(self._fallback_vec(w, i))

        return raw_text, words, resolved_vecs

    @staticmethod
    def _fallback_vec(word: str, index: int) -> tuple[float, float, float]:
        """Verilmeyen koordinatlar için deterministik 3B koordinat üretir."""
        digest = hashlib.sha256(f"{word.lower()}_{index}".encode()).digest()
        x = 0.7 + (digest[0] / 255.0) * 2.8
        y = 0.7 + (digest[1] / 255.0) * 2.8
        z = 0.5 + (digest[2] / 255.0) * 2.3
        return (round(x, 2), round(y, 2), round(z, 2))

    def construct(self) -> None:
        self.camera.background_color = BLACK

        # =====================================================================
        # SABİT BAŞLIK (Tüm video boyunca sol üstte sabit kalır)
        # =====================================================================
        title = Text("1. Ingestion Pipeline", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text("Embedding", font_size=18, color=RED_B)
        title_group = (
            VGroup(title, subtitle)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .to_corner(UL, buff=0.5)
        )
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(FadeIn(title_group, shift=DOWN * 0.2), run_time=1.2)
        self.wait(0.3)

        raw_text, words, vectors_3d = self._resolve_inputs()
        colors = [WORD_PALETTE[i % len(WORD_PALETTE)] for i in range(len(words))]

        # 3 Adımı sırayla çalıştır
        chips = self.step_1_text_to_words(raw_text, words, colors)
        axes, arrows, tip_points = self.step_2_words_to_vectors(
            words, colors, vectors_3d, chips
        )
        self.step_3_vectors_to_points(
            words, colors, vectors_3d, axes, arrows, tip_points
        )

    # =========================================================================
    # ADIM 1: METİNDEN KELİMELERE (~8 sn)
    # =========================================================================
    def step_1_text_to_words(
        self,
        raw_text: str,
        words: list[str],
        colors: list[ManimColor],
    ) -> VGroup:
        n_words = len(words)

        # Metin uzunluğuna göre dinamik yazı boyutu ve satır sarma
        base_font = 28 if n_words <= 8 else max(14, int(26 - (n_words - 8) * 0.45))

        # Tam metni satırlara bölerek oluştur (ekrandan taşmasın)
        max_chars_per_line = 52 if n_words <= 10 else 65
        lines_str: list[str] = []
        current_line: list[str] = []
        current_len = 0
        for w in words:
            if current_line and current_len + len(w) + 1 > max_chars_per_line:
                lines_str.append(" ".join(current_line))
                current_line = [w]
                current_len = len(w)
            else:
                current_line.append(w)
                current_len += len(w) + 1
        if current_line:
            lines_str.append(" ".join(current_line))

        text_lines = VGroup(
            *[Text(line, font_size=base_font, color=GREY_A) for line in lines_str]
        ).arrange(DOWN, buff=0.14)
        if text_lines.width > 10.8:
            text_lines.scale_to_fit_width(10.8)
        if text_lines.height > 2.2:
            text_lines.scale_to_fit_height(2.2)

        text_box = RoundedRectangle(
            corner_radius=0.14,
            width=text_lines.width + 0.8,
            height=text_lines.height + 0.55,
            fill_color="#121620",
            fill_opacity=0.95,
            stroke_color=GREY_B,
            stroke_width=1.8,
        )
        text_group = VGroup(text_box, text_lines).move_to(UP * 1.15)
        text_lines.move_to(text_box.get_center())

        self.play(FadeIn(text_group, shift=UP * 0.2), run_time=self.rt(1.4))
        self.pause(1.2)

        # Kelime çiplerini oluştur (kelime sayısına göre otomatik küçülür)
        chip_font = 22 if n_words <= 8 else max(11, int(22 * (8 / n_words) ** 0.4))
        chip_h = 0.56 if n_words <= 8 else max(0.32, 0.56 * (8 / n_words) ** 0.35)
        chip_pad = 0.42 if n_words <= 8 else max(0.22, 0.42 * (8 / n_words) ** 0.35)

        chips = VGroup()
        for w, col in zip(words, colors):
            lbl = Text(w, font_size=chip_font, color=WHITE, weight=BOLD)
            bg = RoundedRectangle(
                corner_radius=chip_h * 0.28,
                width=max(lbl.width + chip_pad, chip_h * 1.3),
                height=chip_h,
                fill_color=col,
                fill_opacity=0.25,
                stroke_color=col,
                stroke_width=2.2,
            )
            lbl.move_to(bg.get_center())
            chips.add(VGroup(bg, lbl))

        # Çipleri satırlar halinde diz (taşma olmasın)
        max_row_width = 11.2
        rows: list[VGroup] = []
        curr_row: list[VGroup] = []
        curr_w = 0.0
        h_buff = 0.22 if n_words <= 8 else 0.14
        v_buff = 0.22 if n_words <= 8 else 0.14

        for chip in chips:
            if curr_row and (curr_w + chip.width + h_buff > max_row_width):
                rows.append(VGroup(*curr_row).arrange(RIGHT, buff=h_buff))
                curr_row = [chip]
                curr_w = chip.width
            else:
                curr_row.append(chip)
                curr_w += chip.width + h_buff
        if curr_row:
            rows.append(VGroup(*curr_row).arrange(RIGHT, buff=h_buff))

        chips_grid = VGroup(*rows).arrange(DOWN, buff=v_buff)
        if chips_grid.width > 11.4:
            chips_grid.scale_to_fit_width(11.4)
        if chips_grid.height > 2.8:
            chips_grid.scale_to_fit_height(2.8)
        chips_grid.move_to(DOWN * 0.95)

        self.play(
            LaggedStart(
                *[TransformFromCopy(text_lines, chip) for chip in chips],
                lag_ratio=min(0.15, 1.2 / max(1, n_words)),
            ),
            run_time=self.rt(2.2),
        )
        self.pause(1.4)

        # Üstteki ham metin kutusunu kaldır
        self.play(FadeOut(text_group, shift=UP * 0.2), run_time=self.rt(0.8))
        return chips

    # =========================================================================
    # ADIM 2: KELİMELERDEN VEKTÖRLERE (~16 sn)
    # =========================================================================
    def step_2_words_to_vectors(
        self,
        words: list[str],
        colors: list[ManimColor],
        vectors_3d: list[tuple[float, float, float]],
        chips: VGroup,
    ) -> tuple[ThreeDAxes, list[Arrow3D], list[np.ndarray]]:
        n_words = len(words)

        # Çipleri sol tarafa düzenli bir listeye/ızgaraya yerleştir
        cols_count = 1 if n_words <= 7 else (2 if n_words <= 16 else 3)
        target_chips = chips.copy()
        chip_scale = 0.82 if n_words <= 7 else max(0.45, 0.82 * (7 / n_words) ** 0.4)
        for c in target_chips:
            c.scale(chip_scale)

        col_groups: list[VGroup] = []
        per_col = int(np.ceil(n_words / cols_count))
        for ci in range(cols_count):
            slice_chips = target_chips[ci * per_col : (ci + 1) * per_col]
            if len(slice_chips) > 0:
                col_groups.append(
                    VGroup(*slice_chips).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
                )
        left_panel = VGroup(*col_groups).arrange(RIGHT, buff=0.18, aligned_edge=UP)
        if left_panel.height > 4.2:
            left_panel.scale_to_fit_height(4.2)
        if left_panel.width > 3.4:
            left_panel.scale_to_fit_width(3.4)
        left_panel.to_edge(LEFT, buff=0.5).shift(DOWN * 0.55)

        # Önce 2B düzlemde çipleri sol panele taşı
        self.play(
            *[
                chips[i].animate.move_to(target_chips[i].get_center()).scale(chip_scale)
                for i in range(n_words)
            ],
            run_time=self.rt(1.0),
        )

        # embed_texts() kutusu (sol panelin sağında, sabit HUD)
        fn_box = RoundedRectangle(
            corner_radius=0.12,
            width=2.35,
            height=0.62,
            fill_color="#1A2130",
            fill_opacity=0.95,
            stroke_color=YELLOW_B,
            stroke_width=2.0,
        )
        fn_txt = Text("embed_texts()", font_size=18, color=YELLOW_B, weight=BOLD)
        fn_txt.move_to(fn_box.get_center())
        fn_badge = VGroup(fn_box, fn_txt)
        fn_badge.move_to(np.array([left_panel.get_right()[0] + 1.45, -0.35, 0.0]))

        # Alt bilgi notu (sabit HUD)
        dim_note = Text(
            "Gerçekte yüzlerce/binlerce boyut, burada 3 boyut",
            font_size=15,
            color=GREY_B,
        ).to_edge(DOWN, buff=0.32)

        # 3B Eksenleri hazırla (sağ-orta alanda)
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        axes = ThreeDAxes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            z_range=[0, 3.2, 1],
            x_length=5.0,
            y_length=5.0,
            z_length=3.6,
            axis_config={"color": GREY_C, "include_tip": True},
        ).move_to(RIGHT * 2.1 + DOWN * 0.35)

        self.add_fixed_in_frame_mobjects(*chips, fn_badge, dim_note)

        # embed_texts() ve 3B ekseni göster
        self.play(
            FadeIn(fn_badge, shift=RIGHT * 0.2),
            Create(axes),
            FadeIn(dim_note, shift=UP * 0.15),
            run_time=self.rt(1.4),
        )
        self.pause(0.5)

        origin_pt = axes.c2p(0, 0, 0)
        arrow_thickness = (
            0.022 if n_words <= 8 else max(0.01, 0.022 * (8 / n_words) ** 0.5)
        )

        arrows: list[Arrow3D] = []
        tip_points: list[np.ndarray] = []

        # Kelime sayısına göre tekil animasyon hızını otomatik ölçekle
        step_scale = 1.0 if n_words <= 6 else max(0.25, 6.0 / n_words)

        for idx, (word, col, vec_coords) in enumerate(zip(words, colors, vectors_3d)):
            chip = chips[idx]
            tip_pt = axes.c2p(*vec_coords)
            tip_points.append(tip_pt)
            arrow_3d = Arrow3D(
                origin_pt,
                tip_pt,
                color=col,
                thickness=arrow_thickness,
            )
            arrows.append(arrow_3d)

            # Çipin kopyası embed_texts() kutusuna girer
            runner = chip.copy()
            self.add_fixed_in_frame_mobjects(runner)

            self.play(
                Indicate(chip, color=col, scale_factor=1.08),
                runner.animate.move_to(fn_badge.get_center()).scale(0.65),
                Indicate(fn_badge, color=col, scale_factor=1.05),
                run_time=self.rt(0.65 * step_scale),
            )

            if idx == 0:
                # İlk kelime için kısa sayı dizisi gösterip 3B vektör okuna dönüştür
                v_x, _v_y, v_z = vec_coords
                num_str = f"[{v_x:.2f}, -0.80, {v_z:.2f}, …]"
                num_badge_txt = Text(num_str, font_size=16, color=col, weight=BOLD)
                num_badge_bg = RoundedRectangle(
                    corner_radius=0.08,
                    width=num_badge_txt.width + 0.3,
                    height=0.42,
                    fill_color="#121620",
                    fill_opacity=0.95,
                    stroke_color=col,
                    stroke_width=1.6,
                )
                num_badge = VGroup(num_badge_bg, num_badge_txt).next_to(
                    fn_badge, UP, buff=0.28
                )
                num_badge_txt.move_to(num_badge_bg.get_center())
                self.add_fixed_in_frame_mobjects(num_badge)

                self.play(
                    FadeOut(runner),
                    FadeIn(num_badge, shift=UP * 0.15),
                    run_time=self.rt(0.8),
                )
                self.pause(0.8)
                self.play(
                    FadeOut(num_badge, shift=RIGHT * 0.6),
                    Create(arrow_3d),
                    run_time=self.rt(1.1),
                )
            else:
                self.play(
                    FadeOut(runner, shift=RIGHT * 0.3),
                    Create(arrow_3d),
                    run_time=self.rt(0.65 * step_scale),
                )

        self.pause(1.2)

        # Adım 3'e geçerken embed_texts() rozetini, çipleri ve alt notu kaldır
        self.play(
            FadeOut(fn_badge),
            *[FadeOut(c) for c in chips],
            FadeOut(dim_note),
            run_time=self.rt(0.9),
        )
        return axes, arrows, tip_points

    # =========================================================================
    # ADIM 3: OKLARDAN NOKTALARA VE KÜMELER (~12 sn)
    # =========================================================================
    def step_3_vectors_to_points(
        self,
        words: list[str],
        colors: list[ManimColor],
        vectors_3d: list[tuple[float, float, float]],
        axes: ThreeDAxes,
        arrows: list[Arrow3D],
        tip_points: list[np.ndarray],
    ) -> None:
        n_words = len(words)

        # Ekseni merkeze yaklaştırırken okları da beraberinde kaydır
        shift_vec = LEFT * 1.85
        axes_group = VGroup(axes, *arrows)
        self.play(axes_group.animate.shift(shift_vec), run_time=self.rt(1.1))

        # Güncel uç noktaları hesapla
        updated_tips = [pt + shift_vec for pt in tip_points]

        dot_radius = 0.09 if n_words <= 8 else max(0.045, 0.09 * (8 / n_words) ** 0.45)
        lbl_font = 18 if n_words <= 8 else max(11, int(18 * (8 / n_words) ** 0.4))

        dots_3d: list[Dot3D] = []
        labels_3d: list[Text] = []

        for idx, (word, col, tip_pt) in enumerate(zip(words, colors, updated_tips)):
            dot = Dot3D(point=tip_pt, radius=dot_radius, color=col)
            dots_3d.append(dot)

            # Çakışmayı önlemek için tek/çift indekslere hafif farklı ofset
            z_off = 0.26 if idx % 2 == 0 else -0.24
            xy_off = 0.22 if idx % 2 == 0 else -0.18
            lbl = Text(word, font_size=lbl_font, color=col, weight=BOLD).move_to(
                tip_pt + np.array([xy_off, xy_off, z_off])
            )
            labels_3d.append(lbl)

        self.add_fixed_orientation_mobjects(*labels_3d)

        # Oklar silinirken uçlarında aynı renkte noktalar ve kelime etiketleri belirir
        self.play(
            *[FadeOut(a) for a in arrows],
            *[FadeIn(d, scale=1.4) for d in dots_3d],
            *[FadeIn(lbl) for lbl in labels_3d],
            run_time=self.rt(1.6),
        )

        # Yakın nokta çiftleri arasına ince kesikli bağ çizgileri (küme algısını güçlendirir)
        cluster_lines: list[Line3D] = []
        coords_arr = np.array(vectors_3d, dtype=float)
        for i in range(n_words):
            for j in range(i + 1, n_words):
                dist = float(np.linalg.norm(coords_arr[i] - coords_arr[j]))
                if dist < 0.85:
                    cline = Line3D(
                        updated_tips[i],
                        updated_tips[j],
                        color=colors[i],
                        thickness=0.012,
                    )
                    cluster_lines.append(cline)

        if cluster_lines:
            self.play(
                *[Create(cl) for cl in cluster_lines],
                run_time=self.rt(1.0),
            )

        # Kamera yavaşça döner, benzer kelimelerin 3B uzayda kümelendiği görülür
        self.move_camera(
            phi=58 * DEGREES,
            theta=-15 * DEGREES,
            run_time=self.rt(3.8),
        )
        self.move_camera(
            phi=66 * DEGREES,
            theta=20 * DEGREES,
            run_time=self.rt(3.8),
        )
        self.pause(1.2)
