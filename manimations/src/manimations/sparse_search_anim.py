from __future__ import annotations

from typing import Any

import numpy as np
from manim import *

# Hız sabiti: Tüm animasyon ve bekleme süreleri bu çarpanla ayarlanır.
SPEED: float = 1.0


class SparseSearchScene(ThreeDScene):
    """
    Retrieval Pipeline aşamasında Sparse Search (BM25) mekanizmasını
    adım adım anlatan Manim sahnesi.
    """

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data: dict[str, Any] = data or {}

    def rt(self, seconds: float) -> float:
        return max(0.1, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.1, seconds / SPEED))

    def construct(self) -> None:
        self.camera.background_color = BLACK

        # =====================================================================
        # SABİT BAŞLIK (Aynen kullanılacak)
        # =====================================================================
        title = Text("3. Retrieval Pipeline", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text("Sparse Search (BM25)", font_size=18, color=RED_B)
        title_group = (
            VGroup(title, subtitle)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .to_corner(UL, buff=0.5)
        )
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(FadeIn(title_group, shift=DOWN * 0.2), run_time=1.2)
        self.wait(0.3)

        # =====================================================================
        # 1. KULLANICI SORGUSU (GİRDİ)
        # =====================================================================
        q_bg = RoundedRectangle(
            corner_radius=0.14,
            width=8.2,
            height=0.82,
            fill_color="#121824",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=2.0,
        )
        q_tag = Text("Sorgu:", font_size=16, color=BLUE_B, weight=BOLD)

        # Sorgu parçalarını ayrı tutarak 2. adımda vurgulama yapacağız
        q_part_rag = Text("RAG", font_size=18, color=WHITE, weight=BOLD)
        q_part_mid1 = Text("sistemi", font_size=18, color=WHITE)
        q_part_docs = Text("ilgili dokümanları", font_size=18, color=WHITE, weight=BOLD)
        q_part_mid2 = Text("nasıl", font_size=18, color=WHITE)
        q_part_find = Text("bulur?", font_size=18, color=WHITE, weight=BOLD)

        q_sentence = VGroup(
            q_part_rag, q_part_mid1, q_part_docs, q_part_mid2, q_part_find
        ).arrange(RIGHT, buff=0.12)

        q_content = VGroup(q_tag, q_sentence).arrange(RIGHT, buff=0.22)
        if q_content.width > 7.7:
            q_content.scale_to_fit_width(7.7)

        # Sol üst başlıkla çakışmaması için hafif sağ-üst merkeze hizala
        query_card = VGroup(q_bg, q_content).move_to(RIGHT * 1.45 + UP * 2.55)
        q_content.move_to(q_bg.get_center())

        self.play(FadeIn(query_card, shift=DOWN * 0.2), run_time=self.rt(1.1))
        self.pause(0.6)

        # =====================================================================
        # 2. TERİM ÇIKARIMI (TOKENIZE & VURGULAMA)
        # =====================================================================
        hl_rag = SurroundingRectangle(
            q_part_rag,
            color=YELLOW_B,
            fill_color=YELLOW_B,
            fill_opacity=0.2,
            buff=0.06,
            corner_radius=0.06,
        )
        hl_docs = SurroundingRectangle(
            q_part_docs,
            color=GREEN_B,
            fill_color=GREEN_B,
            fill_opacity=0.2,
            buff=0.06,
            corner_radius=0.06,
        )
        hl_find = SurroundingRectangle(
            q_part_find,
            color=TEAL_B,
            fill_color=TEAL_B,
            fill_opacity=0.2,
            buff=0.06,
            corner_radius=0.06,
        )

        def make_term_chip(term_str: str, col: ManimColor) -> VGroup:
            lbl = Text(term_str, font_size=15, color=col, weight=BOLD)
            box = RoundedRectangle(
                corner_radius=0.1,
                width=lbl.width + 0.42,
                height=0.44,
                fill_color="#141C24",
                fill_opacity=0.95,
                stroke_color=col,
                stroke_width=2.0,
            )
            lbl.move_to(box.get_center())
            return VGroup(box, lbl)

        chip_t1 = make_term_chip('"RAG"', YELLOW_B)
        chip_t2 = make_term_chip('"ilgili dokümanlar"', GREEN_B)
        chip_t3 = make_term_chip('"bulur/arama"', TEAL_B)

        terms_label = Text("Anahtar Terimler:", font_size=14, color=GREY_B)

        self.play(
            Create(hl_rag),
            Create(hl_docs),
            Create(hl_find),
            q_part_rag.animate.set_color(YELLOW_B),
            q_part_docs.animate.set_color(GREEN_B),
            q_part_find.animate.set_color(TEAL_B),
            run_time=self.rt(0.9),
        )
        self.play(
            FadeIn(terms_label),
            TransformFromCopy(q_part_rag, chip_t1),
            TransformFromCopy(q_part_docs, chip_t2),
            TransformFromCopy(q_part_find, chip_t3),
            run_time=self.rt(1.1),
        )
        self.pause(0.5)

        # =====================================================================
        # 3. DOKÜMAN PARÇALARI (İNDEKS HAVUZU)
        # =====================================================================
        # 4 adet Chunk kartı (y-koordinatları yuva olarak sabitlenir)
        slot_y = [0.65, -0.45, -1.55, -2.65]

        def build_chunk_card(
            chunk_id: str,
            line1_parts: list[tuple[str, bool]],
            line2_parts: list[tuple[str, bool]],
            y_pos: float,
        ) -> tuple[VGroup, RoundedRectangle, VGroup]:
            bg = RoundedRectangle(
                corner_radius=0.12,
                width=12.4,
                height=0.92,
                fill_color="#11161F",
                fill_opacity=0.95,
                stroke_color=GREY_C,
                stroke_width=1.8,
            )
            badge_bg = RoundedRectangle(
                corner_radius=0.08,
                width=1.45,
                height=0.54,
                fill_color="#1B2434",
                fill_opacity=0.95,
                stroke_color=BLUE_C,
                stroke_width=1.5,
            )
            badge_txt = Text(chunk_id, font_size=15, color=BLUE_A, weight=BOLD)
            badge_txt.move_to(badge_bg.get_center())
            badge = VGroup(badge_bg, badge_txt).move_to(bg.get_left() + RIGHT * 0.95)

            matched_mobs = VGroup()

            def build_line(parts: list[tuple[str, bool]]) -> VGroup:
                row = VGroup()
                for text_seg, is_kw in parts:
                    t_mob = Text(text_seg, font_size=14, color=GREY_A)
                    row.add(t_mob)
                    if is_kw:
                        matched_mobs.add(t_mob)
                row.arrange(RIGHT, buff=0.09)
                return row

            r1 = build_line(line1_parts)
            r2 = build_line(line2_parts)
            text_block = VGroup(r1, r2).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            if text_block.width > 8.1:
                text_block.scale_to_fit_width(8.1)
            text_block.move_to(
                bg.get_left() + RIGHT * 1.85 + RIGHT * (text_block.width / 2)
            )

            card = VGroup(bg, badge, text_block).move_to(np.array([0.0, y_pos, 0.0]))
            return card, bg, matched_mobs

        # Chunk A: "RAG sistemi, kullanıcı sorusuyla ilişkili doküman parçalarını retrieval aşamasında bulur."
        card_a, bg_a, kw_a = build_chunk_card(
            "Chunk A",
            [
                ("RAG", True),
                ("sistemi, kullanıcı sorusuyla", False),
                ("ilişkili doküman", True),
            ],
            [
                ("parçalarını", False),
                ("retrieval", True),
                ("aşamasında", False),
                ("bulur.", True),
            ],
            slot_y[0],
        )

        # Chunk B: "Embedding modelleri, metinleri anlamsal vektör temsillerine dönüştürür."
        card_b, bg_b, _kw_b = build_chunk_card(
            "Chunk B",
            [("Embedding modelleri, metinleri anlamsal", False)],
            [("vektör temsillerine dönüştürür.", False)],
            slot_y[1],
        )

        # Chunk C: "Sparse search, sorgu terimleri ve indekslenmiş metinler arasındaki eşleşmeleri kullanarak ilgili sonuçları bulur."
        card_c, bg_c, kw_c = build_chunk_card(
            "Chunk C",
            [
                (
                    "Sparse search, sorgu terimleri ve indekslenmiş metinler arasındaki",
                    False,
                )
            ],
            [
                ("eşleşmeleri kullanarak", False),
                ("ilgili", True),
                ("sonuçları", False),
                ("bulur.", True),
            ],
            slot_y[2],
        )

        # Chunk D: "Retrieved context, dil modelinin kullanıcı sorusuna daha doğru ve kaynaklara dayalı yanıt üretmesine yardımcı olur."
        card_d, bg_d, kw_d = build_chunk_card(
            "Chunk D",
            [
                ("Retrieved context, dil modelinin", False),
                ("kullanıcı sorusuna", True),
                ("daha doğru", False),
            ],
            [
                ("ve", False),
                ("kaynaklara", True),
                ("dayalı yanıt üretmesine yardımcı olur.", False),
            ],
            slot_y[3],
        )

        all_cards = [card_a, card_b, card_c, card_d]

        self.play(
            LaggedStart(
                *[FadeIn(c, shift=UP * 0.18) for c in all_cards],
                lag_ratio=0.15,
            ),
            run_time=self.rt(1.5),
        )
        self.pause(0.6)

        # =====================================================================
        # 4. EŞLEŞME (KEYWORD MATCHING)
        # =====================================================================
        # Tarama çizgisi (anahtar terimlerden aşağıya doğru chunk'ları tarar)
        scan_line = Line(
            LEFT * 6.2,
            RIGHT * 6.2,
            color=YELLOW_B,
            stroke_width=3.5,
        ).move_to(np.array([0.0, 1.15, 0.0]))

        self.play(FadeIn(scan_line), run_time=self.rt(0.25))
        self.play(
            scan_line.animate.move_to(np.array([0.0, -3.15, 0.0])),
            run_time=self.rt(1.2),
        )
        self.play(FadeOut(scan_line), run_time=self.rt(0.25))

        # Eşleşen kelimeler yeşil/sarı çerçevelerle parlar; Chunk B soluklaşır
        hl_boxes_a = VGroup(
            *[
                SurroundingRectangle(
                    m,
                    color=GREEN_B,
                    fill_color=GREEN_B,
                    fill_opacity=0.22,
                    buff=0.05,
                    corner_radius=0.05,
                )
                for m in kw_a
            ]
        )
        hl_boxes_c = VGroup(
            *[
                SurroundingRectangle(
                    m,
                    color=YELLOW_B,
                    fill_color=YELLOW_B,
                    fill_opacity=0.22,
                    buff=0.05,
                    corner_radius=0.05,
                )
                for m in kw_c
            ]
        )
        hl_boxes_d = VGroup(
            *[
                SurroundingRectangle(
                    m,
                    color=YELLOW_B,
                    fill_color=YELLOW_B,
                    fill_opacity=0.18,
                    buff=0.05,
                    corner_radius=0.05,
                )
                for m in kw_d
            ]
        )

        # Vurgu kutularını ilgili kart gruplarına ekle ki sıralamada birlikte hareket etsinler
        card_a.add(hl_boxes_a)
        card_c.add(hl_boxes_c)
        card_d.add(hl_boxes_d)

        self.play(
            FadeIn(hl_boxes_a, scale=1.08),
            FadeIn(hl_boxes_c, scale=1.08),
            FadeIn(hl_boxes_d, scale=1.08),
            *[m.animate.set_color(GREEN_A) for m in kw_a],
            *[m.animate.set_color(YELLOW_A) for m in kw_c],
            *[m.animate.set_color(YELLOW_A) for m in kw_d],
            bg_a.animate.set_stroke(color=GREEN_B, width=2.4),
            bg_c.animate.set_stroke(color=TEAL_B, width=2.2),
            bg_d.animate.set_stroke(color=YELLOW_B, width=2.0),
            card_b.animate.set_opacity(0.28),
            run_time=self.rt(1.3),
        )
        self.pause(0.8)

        # =====================================================================
        # 5. SKORLAMA VE YENİDEN SIRALAMA (BM25 RANKING)
        # =====================================================================
        def attach_score_badge(
            card: VGroup,
            bg_rect: RoundedRectangle,
            score_text: str,
            sub_label: str,
            col: ManimColor,
            dimmed: bool = False,
        ) -> VGroup:
            s_box = RoundedRectangle(
                corner_radius=0.09,
                width=2.05,
                height=0.62,
                fill_color="#142220" if not dimmed else "#1A1A1A",
                fill_opacity=0.95 if not dimmed else 0.4,
                stroke_color=col,
                stroke_width=2.0 if not dimmed else 1.2,
            )
            s_val = Text(f"BM25: {score_text}", font_size=14, color=col, weight=BOLD)
            s_sub = Text(sub_label, font_size=11, color=GREY_B if dimmed else col)
            s_grp = VGroup(s_val, s_sub).arrange(DOWN, buff=0.04)
            s_grp.move_to(s_box.get_center())
            badge = VGroup(s_box, s_grp).move_to(bg_rect.get_right() + LEFT * 1.2)
            return badge

        score_a = attach_score_badge(card_a, bg_a, "0.89", "En yüksek alaka", GREEN_B)
        score_b = attach_score_badge(
            card_b, bg_b, "0.12", "Filtrelenir", GREY_C, dimmed=True
        )
        score_c = attach_score_badge(card_c, bg_c, "0.74", "Yüksek alaka", TEAL_B)
        score_d = attach_score_badge(card_d, bg_d, "0.52", "Orta alaka", YELLOW_B)

        self.play(
            FadeIn(score_a, shift=LEFT * 0.2),
            FadeIn(score_b, shift=LEFT * 0.2),
            FadeIn(score_c, shift=LEFT * 0.2),
            FadeIn(score_d, shift=LEFT * 0.2),
            run_time=self.rt(1.1),
        )
        card_a.add(score_a)
        card_b.add(score_b)
        card_c.add(score_c)
        card_d.add(score_d)
        self.pause(0.7)

        # Kartlar BM25 skorlarına göre yukarıdan aşağıya yeniden dizilir:
        # 1. Chunk A (0.89) -> slot_y[0]
        # 2. Chunk C (0.74) -> slot_y[1]
        # 3. Chunk D (0.52) -> slot_y[2]
        # 4. Chunk B (0.12) -> slot_y[3]
        self.play(
            card_a.animate.move_to(np.array([0.0, slot_y[0], 0.0])),
            card_c.animate.move_to(np.array([0.0, slot_y[1], 0.0])),
            card_d.animate.move_to(np.array([0.0, slot_y[2], 0.0])),
            card_b.animate.move_to(np.array([0.0, slot_y[3], 0.0])),
            run_time=self.rt(1.6),
        )

        # En yüksek skorlu Chunk A'yı vurgula
        self.play(
            Circumscribe(card_a, color=GREEN_B, time_width=1.2, run_time=self.rt(1.1)),
            run_time=self.rt(1.1),
        )
        self.pause(1.8)
