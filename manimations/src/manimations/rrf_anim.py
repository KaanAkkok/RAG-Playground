from __future__ import annotations

from typing import Any

import numpy as np
from manim import *

from manimations.base import RAGScene

# Hiz sabiti: Tum animasyon ve bekleme sureleri bu carpanla ayarlanir.
SPEED: float = 1.0


class RRFScene(RAGScene):
    """
    Retrieval Pipeline asamasinda Reciprocal Rank Fusion (RRF) algoritmasini
    LaTeX (MathTex) formulu ve adim adim kesirli hesaplamalariyla anlatan,
    ikon veya emoji icermeyen Manim sahnesi.
    """

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(data=data, **kwargs)

    def add_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        """RAGScene (2D Scene) uzerinde baslik sabitleme cagrisiyla tam uyumluluk saglar."""
        super_fn = getattr(super(), "add_fixed_in_frame_mobjects", None)
        if callable(super_fn):
            super_fn(*mobjects)

    def rt(self, seconds: float) -> float:
        return max(0.1, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.1, seconds / SPEED))

    def construct(self) -> None:
        self.camera.background_color = BLACK

        # =====================================================================
        # SABIT BASLIK BILESENI (Aynen kullanilacak)
        # =====================================================================
        title = Text("3. Retrieval Pipeline", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text("Reciprocal Rank Fusion (RRF)", font_size=18, color=RED_B)
        title_group = (
            VGroup(title, subtitle)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .to_corner(UL, buff=0.5)
        )
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(FadeIn(title_group, shift=DOWN * 0.2), run_time=1.2)
        self.wait(0.3)

        # Chunk renk haritasi
        chunk_colors: dict[str, ManimColor] = {
            "Chunk A": GREEN_B,
            "Chunk C": TEAL_B,
            "Chunk D": ORANGE,
            "Chunk B": PURPLE_B,
        }

        # =====================================================================
        # 1. IKI BAGIMSIZ ARAMA LISTESI (GIRDI)
        # =====================================================================
        bm25_items = [
            ("Chunk A", "1. Chunk A (Rank: 1)"),
            ("Chunk C", "2. Chunk C (Rank: 2)"),
            ("Chunk B", "3. Chunk B (Rank: 3)"),
            ("Chunk D", "4. Chunk D (Rank: 4)"),
        ]
        dense_items = [
            ("Chunk C", "1. Chunk C (Rank: 1)"),
            ("Chunk A", "2. Chunk A (Rank: 2)"),
            ("Chunk D", "3. Chunk D (Rank: 3)"),
            ("Chunk B", "4. Chunk B (Rank: 4)"),
        ]

        def make_rank_column(
            header_str: str,
            header_col: ManimColor,
            items: list[tuple[str, str]],
            x_pos: float,
        ) -> tuple[VGroup, dict[str, VGroup]]:
            col_header_bg = RoundedRectangle(
                corner_radius=0.1,
                width=3.15,
                height=0.5,
                fill_color="#162030",
                fill_opacity=0.95,
                stroke_color=header_col,
                stroke_width=2.0,
            )
            col_header_txt = Text(
                header_str, font_size=14, color=header_col, weight=BOLD
            )
            if col_header_txt.width > 2.9:
                col_header_txt.scale_to_fit_width(2.9)
            col_header_txt.move_to(col_header_bg.get_center())
            col_header = VGroup(col_header_bg, col_header_txt)

            cards_group = VGroup()
            card_map: dict[str, VGroup] = {}

            for cid, label_str in items:
                c_col = chunk_colors[cid]
                box = RoundedRectangle(
                    corner_radius=0.09,
                    width=3.15,
                    height=0.68,
                    fill_color="#111722",
                    fill_opacity=0.95,
                    stroke_color=c_col,
                    stroke_width=1.8,
                )
                txt = Text(label_str, font_size=14, color=WHITE, weight=BOLD)
                if txt.width > 2.85:
                    txt.scale_to_fit_width(2.85)
                txt.move_to(box.get_center())
                card = VGroup(box, txt)
                cards_group.add(card)
                card_map[cid] = card

            cards_group.arrange(DOWN, buff=0.22)
            full_col = (
                VGroup(col_header, cards_group)
                .arrange(DOWN, buff=0.24)
                .move_to(np.array([x_pos, -0.95, 0.0]))
            )
            return full_col, card_map

        left_col, bm25_map = make_rank_column(
            "BM25 (Sparse) Siralamasi", BLUE_B, bm25_items, x_pos=-5.25
        )
        right_col, dense_map = make_rank_column(
            "Dense Search Siralamasi", GOLD_B, dense_items, x_pos=5.25
        )

        self.play(
            FadeIn(left_col, shift=RIGHT * 0.25),
            FadeIn(right_col, shift=LEFT * 0.25),
            run_time=self.rt(1.3),
        )
        self.pause(0.5)

        # =====================================================================
        # 2. RRF MATEMATIKSEL FORMULU (LATEX - UST-ORTA)
        # =====================================================================
        formula_eq = MathTex(
            r"\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}",
            font_size=30,
            color=WHITE,
        )
        formula_note = Text(
            "k = 60 sabiti, r = listedeki sira numarasi",
            font_size=13,
            color=GREY_B,
        )
        formula_content = VGroup(formula_eq, formula_note).arrange(DOWN, buff=0.14)
        if formula_content.width > 5.8:
            formula_content.scale_to_fit_width(5.8)

        formula_box = RoundedRectangle(
            corner_radius=0.14,
            width=formula_content.width + 0.55,
            height=formula_content.height + 0.35,
            fill_color="#121824",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=1.8,
        )
        formula_content.move_to(formula_box.get_center())

        self.play(
            FadeIn(formula_box, shift=DOWN * 0.15),
            Write(formula_eq),
            FadeIn(formula_note, shift=UP * 0.1),
            run_time=self.rt(1.5),
        )
        self.pause(0.6)

        # =====================================================================
        # 3. SKORLAMA VE BIRLESTIRME ADIMI (LATEX ILE ADIM ADIM HESAPLAMA)
        # =====================================================================
        # Her dokuman icin:
        # 1) Yerine koyma: 1/(60 + r_bm25) + 1/(60 + r_dense)
        # 2) Payda toplami ve kesir degerleri: 1/61 + 1/62 = 0.01639 + 0.01613
        # 3) Toplam RRF skoru: = 0.03251
        calc_specs = [
            (
                "Chunk A",
                r"\frac{1}{60+1} + \frac{1}{60+2}",
                r"= \frac{1}{61} + \frac{1}{62}",
                r"= 0.03251",
            ),
            (
                "Chunk C",
                r"\frac{1}{60+2} + \frac{1}{60+1}",
                r"= \frac{1}{62} + \frac{1}{61}",
                r"= 0.03251",
            ),
            (
                "Chunk D",
                r"\frac{1}{60+4} + \frac{1}{60+3}",
                r"= \frac{1}{64} + \frac{1}{63}",
                r"= 0.03149",
            ),
            (
                "Chunk B",
                r"\frac{1}{60+3} + \frac{1}{60+4}",
                r"= \frac{1}{63} + \frac{1}{64}",
                r"= 0.03149",
            ),
        ]

        calc_header = Text(
            "RRF Skor Hesaplama (k = 60)", font_size=15, color=YELLOW_B, weight=BOLD
        )
        calc_rows = VGroup()
        calc_parts_list: list[tuple[VGroup, MathTex, MathTex, MathTex]] = []

        for cid, step1_tex, step2_tex, step3_tex in calc_specs:
            c_col = chunk_colors[cid]
            r_box = RoundedRectangle(
                corner_radius=0.09,
                width=5.45,
                height=0.68,
                fill_color="#141B26",
                fill_opacity=0.95,
                stroke_color=c_col,
                stroke_width=1.9,
            )
            lbl_txt = Text(f"{cid}:", font_size=14, color=c_col, weight=BOLD)
            m_step1 = MathTex(step1_tex, font_size=22, color=WHITE)
            m_step2 = MathTex(step2_tex, font_size=22, color=GREY_A)
            m_step3 = MathTex(step3_tex, font_size=24, color=YELLOW_B)

            eq_row = VGroup(lbl_txt, m_step1, m_step2, m_step3).arrange(
                RIGHT, buff=0.14
            )
            if eq_row.width > 5.15:
                eq_row.scale_to_fit_width(5.15)
            eq_row.move_to(r_box.get_center())

            row_grp = VGroup(r_box, eq_row)
            calc_rows.add(row_grp)
            calc_parts_list.append((row_grp, m_step1, m_step2, m_step3))

        calc_rows.arrange(DOWN, buff=0.22)

        self.play(FadeIn(calc_header, shift=DOWN * 0.15), run_time=self.rt(0.5))

        all_arrows = VGroup()
        for idx, (cid, _, _, _) in enumerate(calc_specs):
            c_col = chunk_colors[cid]
            row_grp, m_step1, m_step2, m_step3 = calc_parts_list[idx]
            r_box = row_grp[0]
            lbl_txt = row_grp[1][0]

            left_src = bm25_map[cid]
            right_src = dense_map[cid]

            arr_l = Arrow(
                left_src.get_right(),
                r_box.get_left(),
                buff=0.07,
                color=c_col,
                stroke_width=2.0,
                max_tip_length_to_length_ratio=0.14,
            )
            arr_r = Arrow(
                right_src.get_left(),
                r_box.get_right(),
                buff=0.07,
                color=c_col,
                stroke_width=2.0,
                max_tip_length_to_length_ratio=0.14,
            )
            all_arrows.add(arr_l, arr_r)

            # Adim 3a: Sol ve sag rank kutularindan oklar gelir, kesirde yerine konur
            self.play(
                GrowArrow(arr_l),
                GrowArrow(arr_r),
                FadeIn(r_box),
                FadeIn(lbl_txt),
                Write(m_step1),
                run_time=self.rt(0.85),
            )
            # Adim 3b: Kesir paydalari toplanir ve RRF skoru hesaplanir
            self.play(
                FadeIn(m_step2, shift=RIGHT * 0.1),
                Write(m_step3),
                run_time=self.rt(0.75),
            )

        self.pause(1.0)

        # =====================================================================
        # 4. NIHAI BIRLESIK SIRALAMA (FINAL RRF LIST)
        # =====================================================================
        self.play(
            FadeOut(left_col, shift=LEFT * 0.25),
            FadeOut(right_col, shift=RIGHT * 0.25),
            FadeOut(all_arrows),
            FadeOut(calc_header),
            run_time=self.rt(0.9),
        )

        final_header = Text(
            "Nihai Birlesik Siralama (Final RRF List)",
            font_size=18,
            color=BLUE_A,
            weight=BOLD,
        )

        final_specs = [
            ("Chunk A", "Sira 1: Chunk A (Skor: 0.0325)", True),
            ("Chunk C", "Sira 2: Chunk C (Skor: 0.0325)", True),
            ("Chunk D", "Sira 3: Chunk D (Skor: 0.0315)", False),
            ("Chunk B", "Sira 4: Chunk B (Skor: 0.0315)", False),
        ]

        final_cards = VGroup()
        for cid, final_str, is_topk in final_specs:
            c_col = chunk_colors[cid]
            f_box = RoundedRectangle(
                corner_radius=0.11,
                width=5.8,
                height=0.68,
                fill_color="#12241E" if is_topk else "#121720",
                fill_opacity=0.95 if is_topk else 0.78,
                stroke_color=c_col,
                stroke_width=2.2 if is_topk else 1.5,
            )
            f_txt = Text(
                final_str,
                font_size=16,
                color=WHITE if is_topk else GREY_A,
                weight=BOLD,
            )
            f_txt.move_to(f_box.get_center())
            final_cards.add(VGroup(f_box, f_txt))

        final_cards.arrange(DOWN, buff=0.2)

        self.play(
            FadeIn(final_header, shift=DOWN * 0.15),
            *[
                ReplacementTransform(calc_rows[i], final_cards[i])
                for i in range(len(final_specs))
            ],
            run_time=self.rt(1.4),
        )

        # Ilk iki siradaki kartlari yesil cerceve ile vurgula ve "Secilen Parcalar (Top-K)" etiketi ver
        topk_group = VGroup(final_cards[0], final_cards[1])
        topk_frame = RoundedRectangle(
            corner_radius=0.16,
            width=topk_group.width + 0.36,
            height=topk_group.height + 0.28,
            fill_opacity=0.0,
            stroke_color=GREEN_B,
            stroke_width=3.2,
        ).move_to(topk_group.get_center())

        topk_badge_bg = RoundedRectangle(
            corner_radius=0.09,
            width=3.1,
            height=0.48,
            fill_color="#132A1F",
            fill_opacity=0.95,
            stroke_color=GREEN_B,
            stroke_width=2.0,
        )
        topk_badge_txt = Text(
            "Secilen Parcalar (Top-K)", font_size=14, color=GREEN_A, weight=BOLD
        )
        topk_badge_txt.move_to(topk_badge_bg.get_center())
        topk_badge = VGroup(topk_badge_bg, topk_badge_txt).next_to(
            topk_frame, RIGHT, buff=0.35
        )

        topk_arrow = Arrow(
            topk_badge.get_left(),
            topk_frame.get_right(),
            buff=0.08,
            color=GREEN_B,
            stroke_width=2.4,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(
            Create(topk_frame),
            FadeIn(topk_badge, shift=LEFT * 0.2),
            GrowArrow(topk_arrow),
            final_cards[2].animate.set_opacity(0.55),
            final_cards[3].animate.set_opacity(0.55),
            run_time=self.rt(1.3),
        )
        self.pause(2.0)
