from __future__ import annotations

from typing import Any

from manim import *

# Hiz sabiti: Tum animasyon ve bekleme sureleri bu carpanla ayarlanir.
SPEED: float = 0.78


class RerankerScene(ThreeDScene):
    """
    Retrieval Pipeline asamasinda Cross-Encoder ile Reranking (Yeniden Siralama &
    Self-Attention) mekanizmasini ikon veya emoji kullanmadan, adim adim geometrik
    gorsellestirmeyle anlatan 3B Manim sahnesi.
    """

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data: dict[str, Any] = data or {}

    def rt(self, seconds: float) -> float:
        return max(0.12, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.12, seconds / SPEED))

    def construct(self) -> None:
        self.camera.background_color = BLACK

        # =====================================================================
        # SABIT BASLIK BILESENI (Aynen kullanilacak)
        # =====================================================================
        title = Text("3. Retrieval Pipeline", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text("Reranking", font_size=18, color=RED_B)
        title_group = (
            VGroup(title, subtitle)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .to_corner(UL, buff=0.5)
        )
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(FadeIn(title_group, shift=DOWN * 0.2), run_time=1.2)
        self.wait(0.3)

        # 6 Asamayi sirayla calistir
        self.stage_1_hybrid_candidates()
        q_tokens, c_tokens, sep_tokens = self.stage_2_cross_encoder_input()
        attn_group, stage3_note = self.stage_3_self_attention_3d(
            q_tokens, c_tokens, sep_tokens
        )
        self.stage_4_scoring_head(
            q_tokens, c_tokens, sep_tokens, attn_group, stage3_note
        )
        ranked_cards = self.stage_5_score_all_and_rerank()
        self.stage_6_topk_to_llm(ranked_cards)

    # =========================================================================
    # 1. HYBRID SEARCH ADAYLARI (GIRDI HAVUZU)
    # =========================================================================
    def stage_1_hybrid_candidates(self) -> None:
        # Ust alanda kullanici sorgusu (sol ust baslikla cakismayacak sekilde)
        q_bg = RoundedRectangle(
            corner_radius=0.12,
            width=7.8,
            height=0.72,
            fill_color="#121824",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=2.0,
        )
        q_txt = Text(
            'Sorgu: "RAG sisteminde dokumanlar nasil aranir?"',
            font_size=16,
            color=WHITE,
            weight=BOLD,
        )
        if q_txt.width > 7.3:
            q_txt.scale_to_fit_width(7.3)
        q_txt.move_to(q_bg.get_center())
        query_card = VGroup(q_bg, q_txt).move_to(RIGHT * 1.25 + UP * 2.45)

        # Iki arama kaynagi: Sparse (A, B, D) ve Dense (B, C, E)
        sp_box = RoundedRectangle(
            corner_radius=0.1,
            width=4.2,
            height=0.68,
            fill_color="#14202E",
            fill_opacity=0.95,
            stroke_color=YELLOW_B,
            stroke_width=1.8,
        )
        sp_lbl = Text(
            "Sparse Adaylari: [A] [B] [D]",
            font_size=14,
            color=YELLOW_B,
            weight=BOLD,
        ).move_to(sp_box.get_center())
        sparse_pool = VGroup(sp_box, sp_lbl).move_to(LEFT * 2.75 + UP * 1.15)

        dn_box = RoundedRectangle(
            corner_radius=0.1,
            width=4.2,
            height=0.68,
            fill_color="#1B1528",
            fill_opacity=0.95,
            stroke_color=PURPLE_A,
            stroke_width=1.8,
        )
        dn_lbl = Text(
            "Dense Adaylari: [B] [C] [E]",
            font_size=14,
            color=PURPLE_A,
            weight=BOLD,
        ).move_to(dn_box.get_center())
        dense_pool = VGroup(dn_box, dn_lbl).move_to(RIGHT * 2.75 + UP * 1.15)

        # Ortada tekillestirilen 5 aday kart: [A] [B] [C] [D] [E]
        cand_ids = ["A", "B", "C", "D", "E"]
        cand_colors = [TEAL_B, ORANGE, BLUE_B, PURPLE_B, GREEN_B]
        cand_cards = VGroup()
        for cid, col in zip(cand_ids, cand_colors):
            c_box = RoundedRectangle(
                corner_radius=0.12,
                width=1.75,
                height=1.1,
                fill_color="#141B24",
                fill_opacity=0.95,
                stroke_color=col,
                stroke_width=2.2,
            )
            c_title = Text(f"[{cid}]", font_size=22, color=col, weight=BOLD)
            c_sub = Text(f"Chunk {cid}", font_size=12, color=GREY_B)
            c_grp = (
                VGroup(c_title, c_sub)
                .arrange(DOWN, buff=0.1)
                .move_to(c_box.get_center())
            )
            cand_cards.add(VGroup(c_box, c_grp))

        cand_cards.arrange(RIGHT, buff=0.35).move_to(DOWN * 0.65)

        arr_sp = Arrow(
            sparse_pool.get_bottom(),
            cand_cards.get_top() + LEFT * 1.2,
            buff=0.12,
            color=YELLOW_B,
            stroke_width=2.2,
            max_tip_length_to_length_ratio=0.16,
        )
        arr_dn = Arrow(
            dense_pool.get_bottom(),
            cand_cards.get_top() + RIGHT * 1.2,
            buff=0.12,
            color=PURPLE_A,
            stroke_width=2.2,
            max_tip_length_to_length_ratio=0.16,
        )

        note_1 = Text(
            "Hybrid Search adaylari bulur, ancak gercek alakalilik henuz kesin degildir.",
            font_size=15,
            color=GREY_A,
        ).to_edge(DOWN, buff=0.55)

        self.play(FadeIn(query_card, shift=DOWN * 0.15), run_time=self.rt(0.9))
        self.play(
            FadeIn(sparse_pool, shift=RIGHT * 0.2),
            FadeIn(dense_pool, shift=LEFT * 0.2),
            run_time=self.rt(0.9),
        )
        self.play(
            GrowArrow(arr_sp),
            GrowArrow(arr_dn),
            LaggedStart(
                *[FadeIn(c, shift=UP * 0.15) for c in cand_cards],
                lag_ratio=0.12,
            ),
            FadeIn(note_1, shift=UP * 0.1),
            run_time=self.rt(1.3),
        )
        self.pause(1.1)

        self.play(
            FadeOut(query_card),
            FadeOut(sparse_pool),
            FadeOut(dense_pool),
            FadeOut(arr_sp),
            FadeOut(arr_dn),
            FadeOut(cand_cards),
            FadeOut(note_1),
            run_time=self.rt(0.75),
        )

    # =========================================================================
    # 2. CROSS-ENCODER PRENSIBI ([QUERY] + [CHUNK B])
    # =========================================================================
    def stage_2_cross_encoder_input(
        self,
    ) -> tuple[VGroup, VGroup, VGroup]:
        banner = Text(
            "Cross-Encoder: Iki ayri vektor yerine tek birlesik girdi",
            font_size=16,
            color=GOLD_B,
            weight=BOLD,
        ).move_to(RIGHT * 1.15 + UP * 2.35)

        # Sol ve sagda baslayan [QUERY] ve [CHUNK B] bloklari
        q_block_bg = RoundedRectangle(
            corner_radius=0.12,
            width=4.4,
            height=1.1,
            fill_color="#102033",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=2.2,
        )
        q_block_lbl = (
            VGroup(
                Text("[QUERY]", font_size=16, color=BLUE_B, weight=BOLD),
                Text(
                    "RAG sisteminde dokumanlar nasil aranir?", font_size=12, color=WHITE
                ),
            )
            .arrange(DOWN, buff=0.1)
            .move_to(q_block_bg.get_center())
        )
        q_block = VGroup(q_block_bg, q_block_lbl).move_to(LEFT * 3.1 + UP * 0.6)

        c_block_bg = RoundedRectangle(
            corner_radius=0.12,
            width=4.8,
            height=1.1,
            fill_color="#2A1B10",
            fill_opacity=0.95,
            stroke_color=ORANGE,
            stroke_width=2.2,
        )
        c_block_lbl = (
            VGroup(
                Text("[CHUNK B]", font_size=16, color=ORANGE, weight=BOLD),
                Text(
                    "dokuman embedding'leri ... retrieval icerikleri getirir",
                    font_size=12,
                    color=WHITE,
                ),
            )
            .arrange(DOWN, buff=0.1)
            .move_to(c_block_bg.get_center())
        )
        c_block = VGroup(c_block_bg, c_block_lbl).move_to(RIGHT * 3.1 + UP * 0.6)

        # Birlesik yatay girdi kutusu: [CLS] QUERY [SEP] CHUNK B [SEP]
        merged_bg = RoundedRectangle(
            corner_radius=0.14,
            width=10.8,
            height=0.95,
            fill_color="#141A24",
            fill_opacity=0.95,
            stroke_color=YELLOW_B,
            stroke_width=2.2,
        ).move_to(UP * 0.6)

        def make_seq_segment(txt_str: str, col: ManimColor, w: float) -> VGroup:
            b = RoundedRectangle(
                corner_radius=0.08,
                width=w,
                height=0.62,
                fill_color=col,
                fill_opacity=0.22,
                stroke_color=col,
                stroke_width=2.0,
            )
            t = Text(txt_str, font_size=15, color=col, weight=BOLD).move_to(
                b.get_center()
            )
            return VGroup(b, t)

        seg_cls = make_seq_segment("[CLS]", GREY_A, 1.25)
        seg_q = make_seq_segment("QUERY", BLUE_B, 2.8)
        seg_sep1 = make_seq_segment("[SEP]", GREY_A, 1.25)
        seg_c = make_seq_segment("CHUNK B", ORANGE, 3.1)
        seg_sep2 = make_seq_segment("[SEP]", GREY_A, 1.25)

        merged_segments = (
            VGroup(seg_cls, seg_q, seg_sep1, seg_c, seg_sep2)
            .arrange(RIGHT, buff=0.18)
            .move_to(merged_bg.get_center())
        )
        merged_box = VGroup(merged_bg, merged_segments)

        self.play(
            FadeIn(banner, shift=DOWN * 0.15),
            FadeIn(q_block, shift=RIGHT * 0.25),
            FadeIn(c_block, shift=LEFT * 0.25),
            run_time=self.rt(1.0),
        )
        self.pause(0.5)

        self.play(
            ReplacementTransform(VGroup(q_block, c_block), merged_box),
            run_time=self.rt(1.1),
        )
        self.pause(0.5)

        # Simdi token kutucuklarina ayristir (Query mavi, Chunk B turuncu)
        def make_token_box(tok_str: str, col: ManimColor) -> VGroup:
            lbl = Text(tok_str, font_size=14, color=WHITE, weight=BOLD)
            box = RoundedRectangle(
                corner_radius=0.08,
                width=max(1.35, lbl.width + 0.36),
                height=0.58,
                fill_color=(
                    "#102236"
                    if col == BLUE_B
                    else ("#2E1C10" if col == ORANGE else "#1E222A")
                ),
                fill_opacity=0.95,
                stroke_color=col,
                stroke_width=2.2,
            )
            lbl.move_to(box.get_center())
            return VGroup(box, lbl)

        q_tok_strs = ["RAG", "sisteminde", "dokumanlar", "nasil aranir"]
        c_tok_strs = [
            "dokuman embedding'leri",
            "ve",
            "retrieval",
            "icerikleri getirir",
        ]

        q_tokens = VGroup(*[make_token_box(s, BLUE_B) for s in q_tok_strs]).arrange(
            RIGHT, buff=0.25
        )
        c_tokens = VGroup(*[make_token_box(s, ORANGE) for s in c_tok_strs]).arrange(
            RIGHT, buff=0.25
        )
        sep_cls = make_token_box("[CLS]", GREY_B)
        sep_mid = make_token_box("[SEP]", GREY_B)
        sep_end = make_token_box("[SEP]", GREY_B)

        # Ust sira: [CLS] + Query token'lari, Alt sira: [SEP] + Chunk B token'lari + [SEP]
        VGroup(sep_cls, *q_tokens).arrange(RIGHT, buff=0.22).move_to(UP * 1.25)
        bot_row_tmp = (
            VGroup(sep_mid, *c_tokens, sep_end)
            .arrange(RIGHT, buff=0.2)
            .move_to(DOWN * 1.35)
        )
        if bot_row_tmp.width > 12.2:
            bot_row_tmp.scale_to_fit_width(12.2)

        sep_tokens = VGroup(sep_cls, sep_mid, sep_end)

        self.play(
            FadeOut(banner),
            FadeOut(merged_box, scale=0.95),
            FadeIn(q_tokens, shift=UP * 0.15),
            FadeIn(c_tokens, shift=DOWN * 0.15),
            FadeIn(sep_tokens),
            run_time=self.rt(1.2),
        )
        self.pause(0.4)
        return q_tokens, c_tokens, sep_tokens

    # =========================================================================
    # 3. SELF-ATTENTION VE CAPRAZ TOKEN ETKILESIMI (3D PERSPEKTIF)
    # =========================================================================
    def stage_3_self_attention_3d(
        self,
        q_tokens: VGroup,
        c_tokens: VGroup,
        sep_tokens: VGroup,
    ) -> tuple[VGroup, Text]:
        # Token satirlarina 3B derinlik (Z ekseni farki) kazandirarak kamera acisinda
        # tum capraz baglanti cizgilerinin net ve ayrik gorunmesini sagla
        self.move_camera(
            phi=45 * DEGREES,
            theta=-60 * DEGREES,
            added_anims=[
                q_tokens.animate.shift(OUT * 0.55),
                sep_tokens[0].animate.shift(OUT * 0.55),
                c_tokens.animate.shift(IN * 0.35),
                sep_tokens[1].animate.shift(IN * 0.35),
                sep_tokens[2].animate.shift(IN * 0.35),
            ],
            run_time=self.rt(1.4),
        )

        # Her token kutusunun baglanti noktasina parlak dugum (node) ekle
        port_dots = VGroup()
        for qt in q_tokens:
            port_dots.add(Dot(point=qt.get_bottom(), radius=0.065, color=BLUE_A))
        for ct in c_tokens:
            port_dots.add(Dot(point=ct.get_top(), radius=0.065, color=ORANGE))

        # Tum token ciftleri arasinda belirgin capraz baglanti cizgileri (Self-Attention agi)
        bg_lines = VGroup()
        mesh_colors = [BLUE_B, TEAL_B, PURPLE_A, GOLD_B]
        for i, qt in enumerate(q_tokens):
            for j, ct in enumerate(c_tokens):
                col_ij = mesh_colors[(i + j) % len(mesh_colors)]
                ln = Line(
                    qt.get_bottom(),
                    ct.get_top(),
                    color=col_ij,
                    stroke_width=3.2,
                    stroke_opacity=0.85,
                )
                bg_lines.add(ln)

        # Guclu capraz etkilesim cizgileri (Kalın ve parlayan):
        # 1) "dokumanlar" (q_tokens[2]) <-> "dokuman embedding'leri" (c_tokens[0])
        hl_line_1 = Line(
            q_tokens[2].get_bottom(),
            c_tokens[0].get_top(),
            color=YELLOW_B,
            stroke_width=6.5,
        )
        hl_glow_1 = Line(
            q_tokens[2].get_bottom(),
            c_tokens[0].get_top(),
            color=YELLOW_A,
            stroke_width=13.0,
            stroke_opacity=0.45,
        )

        # 2) "nasil aranir" (q_tokens[3]) <-> "retrieval" (c_tokens[2]) & "icerikleri getirir" (c_tokens[3])
        hl_line_2a = Line(
            q_tokens[3].get_bottom(),
            c_tokens[2].get_top(),
            color=GREEN_B,
            stroke_width=6.0,
        )
        hl_glow_2a = Line(
            q_tokens[3].get_bottom(),
            c_tokens[2].get_top(),
            color=GREEN_A,
            stroke_width=12.0,
            stroke_opacity=0.45,
        )
        hl_line_2b = Line(
            q_tokens[3].get_bottom(),
            c_tokens[3].get_top(),
            color=GREEN_B,
            stroke_width=6.0,
        )
        hl_glow_2b = Line(
            q_tokens[3].get_bottom(),
            c_tokens[3].get_top(),
            color=GREEN_A,
            stroke_width=12.0,
            stroke_opacity=0.45,
        )

        # 3) "RAG" (q_tokens[0]) <-> "retrieval" (c_tokens[2])
        hl_line_3 = Line(
            q_tokens[0].get_bottom(),
            c_tokens[2].get_top(),
            color=YELLOW_B,
            stroke_width=5.5,
        )
        hl_glow_3 = Line(
            q_tokens[0].get_bottom(),
            c_tokens[2].get_top(),
            color=YELLOW_A,
            stroke_width=11.0,
            stroke_opacity=0.4,
        )

        strong_lines = VGroup(
            hl_glow_1,
            hl_line_1,
            hl_glow_2a,
            hl_line_2a,
            hl_glow_2b,
            hl_line_2b,
            hl_glow_3,
            hl_line_3,
            port_dots,
        )

        stage3_note = Text(
            "Model tum kelimeleri ayni anda birlikte degerlendirir.",
            font_size=16,
            color=YELLOW_B,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.42)

        # Tum capraz baglanti agini ve port dugumlerini ciz
        self.play(
            FadeIn(port_dots),
            LaggedStart(*[Create(l) for l in bg_lines], lag_ratio=0.04),
            run_time=self.rt(1.3),
        )

        # Kamera hafifce cepheye dogru donerken (cizgilerin ust uste binmesini onler)
        # guclu semantik eslesme hatlarini ve sinyal akisini parlat
        self.add_fixed_in_frame_mobjects(stage3_note)
        self.move_camera(
            phi=38 * DEGREES,
            theta=-78 * DEGREES,
            added_anims=[
                Create(hl_glow_1),
                Create(hl_line_1),
                Create(hl_glow_2a),
                Create(hl_line_2a),
                Create(hl_glow_2b),
                Create(hl_line_2b),
                Create(hl_glow_3),
                Create(hl_line_3),
                q_tokens[0][0].animate.set_stroke(YELLOW_B, width=3.2),
                q_tokens[2][0].animate.set_stroke(YELLOW_B, width=3.5),
                c_tokens[0][0].animate.set_stroke(YELLOW_B, width=3.5),
                q_tokens[3][0].animate.set_stroke(GREEN_B, width=3.5),
                c_tokens[2][0].animate.set_stroke(GREEN_B, width=3.5),
                c_tokens[3][0].animate.set_stroke(GREEN_B, width=3.5),
                FadeIn(stage3_note, shift=UP * 0.15),
            ],
            run_time=self.rt(1.8),
        )

        # Tum capraz cizgiler uzerinde dikkat (attention) sinyali akisi
        pulse_copies = VGroup(
            *[
                l.copy().set_stroke(color=WHITE, width=4.5, opacity=0.95)
                for l in bg_lines
            ]
        )
        self.play(
            LaggedStart(
                *[ShowPassingFlash(p, time_width=0.55) for p in pulse_copies],
                lag_ratio=0.03,
            ),
            run_time=self.rt(1.4),
        )
        self.pause(0.8)

        attn_group = VGroup(bg_lines, strong_lines)
        return attn_group, stage3_note

    # =========================================================================
    # 4. SKOR URETIMI (SCORING HEAD)
    # =========================================================================
    def stage_4_scoring_head(
        self,
        q_tokens: VGroup,
        c_tokens: VGroup,
        sep_tokens: VGroup,
        attn_group: VGroup,
        stage3_note: Text,
    ) -> None:
        # Kamera 2D duzleme doner
        self.play(FadeOut(stage3_note), run_time=self.rt(0.4))
        self.move_camera(
            phi=0 * DEGREES,
            theta=-90 * DEGREES,
            run_time=self.rt(1.2),
        )

        # Sol tarafa kucultulmus Transformer cikisi, ortada Scoring Head, sagda 0.91 Skor Karti
        head_bg = RoundedRectangle(
            corner_radius=0.14,
            width=2.8,
            height=1.35,
            fill_color="#1B1528",
            fill_opacity=0.95,
            stroke_color=PURPLE_A,
            stroke_width=2.4,
        )
        head_lbl = (
            VGroup(
                Text("Scoring Head", font_size=18, color=PURPLE_A, weight=BOLD),
                Text("Linear + Sigmoid", font_size=12, color=GREY_B),
            )
            .arrange(DOWN, buff=0.1)
            .move_to(head_bg.get_center())
        )
        scoring_head = VGroup(head_bg, head_lbl).move_to(DOWN * 0.35)

        score_card_bg = RoundedRectangle(
            corner_radius=0.14,
            width=4.1,
            height=1.65,
            fill_color="#10241C",
            fill_opacity=0.95,
            stroke_color=GREEN_B,
            stroke_width=2.4,
        )
        score_title = Text(
            "Query + Chunk B -> 0.91", font_size=17, color=GREEN_A, weight=BOLD
        )
        bar_bg = RoundedRectangle(
            corner_radius=0.06,
            width=3.3,
            height=0.28,
            fill_color="#1A2622",
            fill_opacity=0.95,
            stroke_color=GREY_C,
            stroke_width=1.2,
        )
        bar_fill = RoundedRectangle(
            corner_radius=0.06,
            width=3.3 * 0.91,
            height=0.28,
            fill_color=GREEN_B,
            fill_opacity=0.9,
            stroke_width=0.0,
        )
        bar_fill.align_to(bar_bg, LEFT)
        bar_grp = VGroup(bar_bg, bar_fill)

        score_inner = (
            VGroup(score_title, bar_grp)
            .arrange(DOWN, buff=0.22)
            .move_to(score_card_bg.get_center())
        )
        score_card = VGroup(score_card_bg, score_inner).move_to(
            RIGHT * 4.15 + DOWN * 0.35
        )

        self.play(
            q_tokens.animate.scale(0.42).move_to(LEFT * 4.35 + UP * 0.15),
            c_tokens.animate.scale(0.42).move_to(LEFT * 4.35 + DOWN * 0.75),
            sep_tokens.animate.scale(0.42).move_to(LEFT * 4.35 + DOWN * 0.35),
            attn_group.animate.scale(0.42).move_to(LEFT * 4.35 + DOWN * 0.3),
            FadeIn(scoring_head, scale=0.92),
            run_time=self.rt(1.1),
        )

        arr_to_head = Arrow(
            LEFT * 1.95 + DOWN * 0.35,
            scoring_head.get_left(),
            buff=0.14,
            color=PURPLE_A,
            stroke_width=2.6,
            max_tip_length_to_length_ratio=0.18,
        )
        arr_to_score = Arrow(
            scoring_head.get_right(),
            score_card.get_left(),
            buff=0.14,
            color=GREEN_B,
            stroke_width=2.6,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(
            GrowArrow(arr_to_head),
            Indicate(scoring_head, color=YELLOW_B, scale_factor=1.05),
            run_time=self.rt(0.8),
        )
        self.play(
            GrowArrow(arr_to_score),
            FadeIn(score_card, shift=RIGHT * 0.2),
            run_time=self.rt(1.0),
        )
        self.pause(1.0)

        self.play(
            FadeOut(q_tokens),
            FadeOut(c_tokens),
            FadeOut(sep_tokens),
            *[FadeOut(m) for m in attn_group[0]],
            *[FadeOut(m) for m in attn_group[1]],
            FadeOut(scoring_head),
            FadeOut(arr_to_head),
            FadeOut(arr_to_score),
            FadeOut(score_card),
            run_time=self.rt(0.75),
        )

    # =========================================================================
    # 5. TUM ADAYLARIN PUANLANMASI VE RERANKING (YENIDEN SIRALAMA)
    # =========================================================================
    def stage_5_score_all_and_rerank(self) -> list[VGroup]:
        stage5_hdr = Text(
            "Cross-Encoder Skorlari ve Yeniden Siralama (Reranking)",
            font_size=17,
            color=BLUE_A,
            weight=BOLD,
        ).move_to(RIGHT * 0.8 + UP * 2.15)

        # Baslangicta 5 aday yan yana: [A: 0.74] [B: 0.91] [C: 0.66] [D: 0.59] [E: 0.71]
        initial_specs = [
            ("A", "0.74", TEAL_B, 1),  # Hedef sira: 2 (index 1)
            ("B", "0.91", GREEN_B, 0),  # Hedef sira: 1 (index 0 - En Yuksek)
            ("C", "0.66", BLUE_B, 3),  # Hedef sira: 4 (index 3)
            ("D", "0.59", PURPLE_B, 4),  # Hedef sira: 5 (index 4)
            ("E", "0.71", ORANGE, 2),  # Hedef sira: 3 (index 2)
        ]

        cards_by_initial: list[VGroup] = []
        for cid, sc_str, col, _ in initial_specs:
            box = RoundedRectangle(
                corner_radius=0.12,
                width=2.15,
                height=1.25,
                fill_color="#121A24",
                fill_opacity=0.95,
                stroke_color=col,
                stroke_width=2.2,
            )
            lbl_top = Text(f"[{cid}: {sc_str}]", font_size=17, color=WHITE, weight=BOLD)
            lbl_sub = Text(
                "En Yuksek" if cid == "B" else f"Skor: {sc_str}",
                font_size=12,
                color=col,
                weight=BOLD,
            )
            grp_txt = (
                VGroup(lbl_top, lbl_sub)
                .arrange(DOWN, buff=0.14)
                .move_to(box.get_center())
            )
            cards_by_initial.append(VGroup(box, grp_txt))

        slot_positions = [c.get_center().copy() for c in cards_by_initial]

        self.play(
            FadeIn(stage5_hdr, shift=DOWN * 0.15),
            LaggedStart(
                *[FadeIn(c, shift=UP * 0.18) for c in cards_by_initial],
                lag_ratio=0.12,
            ),
            run_time=self.rt(1.3),
        )
        self.pause(0.8)

        # Kartlar yuksek puandan dusuge dogru yer degistirir:
        # 1. [B: 0.91] -> slot 0
        # 2. [A: 0.74] -> slot 1
        # 3. [E: 0.71] -> slot 2
        # 4. [C: 0.66] -> slot 3
        # 5. [D: 0.59] -> slot 4
        move_anims = []
        ranked_cards: list[VGroup] = [None] * 5  # type: ignore[list-item]
        for idx, (_, _, _, target_slot) in enumerate(initial_specs):
            card = cards_by_initial[idx]
            ranked_cards[target_slot] = card
            arc_angle = PI / 3 if idx != target_slot else 0.0
            move_anims.append(
                card.animate(path_arc=arc_angle).move_to(slot_positions[target_slot])
            )

        self.play(*move_anims, run_time=self.rt(1.6))
        self.pause(0.8)

        self.play(FadeOut(stage5_hdr), run_time=self.rt(0.4))
        return ranked_cards

    # =========================================================================
    # 6. TOP-K SECIMI VE LLM CONTEXT'E AKTARIM
    # =========================================================================
    def stage_6_topk_to_llm(self, ranked_cards: list[VGroup]) -> None:
        # Ilk 3 kart ([B: 0.91], [A: 0.74], [E: 0.71]) parlak yesil cerceve icine alinir
        top3_bounds = VGroup(ranked_cards[0], ranked_cards[1], ranked_cards[2])

        topk_frame = RoundedRectangle(
            corner_radius=0.16,
            width=top3_bounds.width + 0.32,
            height=top3_bounds.height + 0.32,
            fill_opacity=0.0,
            stroke_color=GREEN_B,
            stroke_width=3.2,
        ).move_to(top3_bounds.get_center())

        topk_lbl = Text(
            "Top-K Secimi (K=3)", font_size=15, color=GREEN_A, weight=BOLD
        ).next_to(topk_frame, UP, buff=0.18)

        self.play(
            Create(topk_frame),
            FadeIn(topk_lbl, shift=UP * 0.1),
            ranked_cards[3].animate.set_opacity(0.28),
            ranked_cards[4].animate.set_opacity(0.28),
            run_time=self.rt(1.1),
        )
        self.pause(0.6)

        # Bu 3 kart birleserek "Retrieved Context (Top-K)" kutusuna donusur ve sagdaki "LLM" bloguna akar
        ctx_box = RoundedRectangle(
            corner_radius=0.14,
            width=4.3,
            height=1.45,
            fill_color="#11261E",
            fill_opacity=0.95,
            stroke_color=GREEN_B,
            stroke_width=2.4,
        )
        ctx_txt = (
            VGroup(
                Text(
                    "Retrieved Context (Top-K)",
                    font_size=16,
                    color=GREEN_A,
                    weight=BOLD,
                ),
                Text(
                    "[B: 0.91]  +  [A: 0.74]  +  [E: 0.71]", font_size=13, color=WHITE
                ),
            )
            .arrange(DOWN, buff=0.14)
            .move_to(ctx_box.get_center())
        )
        ctx_block = VGroup(ctx_box, ctx_txt).move_to(LEFT * 2.1 + DOWN * 0.15)

        llm_box = RoundedRectangle(
            corner_radius=0.16,
            width=2.8,
            height=1.45,
            fill_color="#1A1426",
            fill_opacity=0.95,
            stroke_color=GOLD_B,
            stroke_width=2.5,
        )
        llm_txt = (
            VGroup(
                Text("LLM", font_size=22, color=GOLD_A, weight=BOLD),
                Text("Yanit Uretimi", font_size=13, color=GREY_A),
            )
            .arrange(DOWN, buff=0.12)
            .move_to(llm_box.get_center())
        )
        llm_block = VGroup(llm_box, llm_txt).move_to(RIGHT * 3.25 + DOWN * 0.15)

        arr_ctx_to_llm = Arrow(
            ctx_block.get_right(),
            llm_block.get_left(),
            buff=0.16,
            color=GREEN_B,
            stroke_width=3.0,
            max_tip_length_to_length_ratio=0.18,
        )

        # Altta ozet akis semasi:
        # Query -> Hybrid Search -> Cross-Encoder -> Reranking -> Top-K Context -> LLM
        pipeline_steps = [
            ("Query", BLUE_B),
            ("Hybrid Search", TEAL_B),
            ("Cross-Encoder", ORANGE),
            ("Reranking", PURPLE_A),
            ("Top-K Context", GREEN_B),
            ("LLM", GOLD_B),
        ]
        pipe_mobs = VGroup()
        for idx, (st_name, st_col) in enumerate(pipeline_steps):
            p_bg = RoundedRectangle(
                corner_radius=0.07,
                width=1.68,
                height=0.44,
                fill_color="#141A22",
                fill_opacity=0.95,
                stroke_color=st_col,
                stroke_width=1.6,
            )
            p_t = Text(st_name, font_size=11, color=st_col, weight=BOLD).move_to(
                p_bg.get_center()
            )
            pipe_mobs.add(VGroup(p_bg, p_t))
            if idx < len(pipeline_steps) - 1:
                p_arr = Arrow(
                    ORIGIN,
                    RIGHT * 0.35,
                    buff=0.02,
                    color=GREY_B,
                    stroke_width=1.8,
                    max_tip_length_to_length_ratio=0.25,
                )
                pipe_mobs.add(p_arr)

        pipe_mobs.arrange(RIGHT, buff=0.08).to_edge(DOWN, buff=0.48)
        if pipe_mobs.width > 12.6:
            pipe_mobs.scale_to_fit_width(12.6)

        self.play(
            FadeOut(ranked_cards[3]),
            FadeOut(ranked_cards[4]),
            FadeOut(topk_lbl),
            FadeOut(topk_frame),
            FadeOut(ranked_cards[0], target_position=ctx_block.get_center()),
            FadeOut(ranked_cards[1], target_position=ctx_block.get_center()),
            FadeOut(ranked_cards[2], target_position=ctx_block.get_center()),
            FadeIn(ctx_block, scale=0.92),
            FadeIn(llm_block, shift=LEFT * 0.2),
            run_time=self.rt(1.2),
        )
        self.play(
            GrowArrow(arr_ctx_to_llm),
            Indicate(llm_block, color=GOLD_A, scale_factor=1.05),
            FadeIn(pipe_mobs, shift=UP * 0.15),
            run_time=self.rt(1.3),
        )
        self.pause(2.0)
