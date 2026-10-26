from __future__ import annotations

import shutil
from typing import Any

import numpy as np
from manim import *

# Hız sabiti: Tüm animasyon ve bekleme süreleri bu çarpanla ayarlanır.
SPEED: float = 1.0


class DenseSearchScene(ThreeDScene):
    """
    Retrieval Pipeline aşamasında Dense Search (Vektörel Arama & Cosine Similarity)
    mekanizmasını 5 aşamada anlatan 3B Manim sahnesi.
    """

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data: dict[str, Any] = data or {}

    def rt(self, seconds: float) -> float:
        return max(0.1, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.1, seconds / SPEED))

    @staticmethod
    def _build_cosine_formula() -> VGroup:
        """
        Sistemde LaTeX kuruluysa MathTex, değilse kesir çizgili vektörel matematik
        grubu kullanarak Cosine Similarity formülünü oluşturur:
        Cosine Similarity = cos(θ) = (u · v) / (||u|| ||v||)
        """
        if shutil.which("latex"):
            try:
                tex = MathTex(
                    r"\text{Cosine Similarity} = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}",
                    font_size=28,
                    color=WHITE,
                )
                return VGroup(tex)
            except Exception:
                logger.warning("Error occurred while rendering MathLaTeX.")

        lhs = Text(
            "Cosine Similarity = cos(θ) =", font_size=18, color=WHITE, weight=BOLD
        )
        num = Text("u · v", font_size=18, color=YELLOW_B, weight=BOLD)
        den = Text("‖u‖ ‖v‖", font_size=17, color=BLUE_B, weight=BOLD)
        frac_line = Line(LEFT * 0.75, RIGHT * 0.75, color=GREY_A, stroke_width=2.2)
        frac = VGroup(num, frac_line, den).arrange(DOWN, buff=0.08)
        return VGroup(lhs, frac).arrange(RIGHT, buff=0.2)

    def construct(self) -> None:
        self.camera.background_color = BLACK

        # =====================================================================
        # SABİT BAŞLIK (Aynen kullanılacak)
        # =====================================================================
        title = Text("3. Retrieval Pipeline", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text("Dense Search", font_size=18, color=RED_B)
        title_group = (
            VGroup(title, subtitle)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .to_corner(UL, buff=0.5)
        )
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(FadeIn(title_group, shift=DOWN * 0.2), run_time=1.2)
        self.wait(0.3)

        # 5 aşamayı sırayla çalıştır
        encoder_block, vdb_block, vdb_rows = self.stage_1_chunks_and_vectordb()
        self.stage_2_query_encoding(encoder_block, vdb_block, vdb_rows)
        self.stage_3_2d_cosine_similarity()
        self.stage_4_and_5_3d_space_and_results()

    # =========================================================================
    # 1. DOKÜMAN PARÇALARI VE VEKTÖR DB (HAZIRLIK)
    # =========================================================================
    def stage_1_chunks_and_vectordb(self) -> tuple[VGroup, VGroup, VGroup]:
        chunk_specs = [
            (
                "Chunk A",
                "RAG sistemi, kullanıcı sorusuyla ilişkili doküman\nparçalarını retrieval aşamasında bulur.",
                GREEN_B,
                "[0.82, 0.51, 0.26, …]",
            ),
            (
                "Chunk B",
                "Embedding modelleri, metinleri anlamsal\nvektör temsillerine dönüştürür.",
                PURPLE_B,
                "[-0.35, 0.74, 0.12, …]",
            ),
            (
                "Chunk C",
                "Sparse search, sorgu terimleri ve indekslenmiş\nmetinler arasındaki eşleşmeleri kullanır.",
                TEAL_B,
                "[0.58, 0.44, -0.19, …]",
            ),
            (
                "Chunk D",
                "Retrieved context, dil modelinin kullanıcı sorusuna\ndoğru ve kaynaklara dayalı yanıt üretmesini sağlar.",
                ORANGE,
                "[0.73, 0.49, 0.31, …]",
            ),
        ]

        chunk_cards = VGroup()
        for cid, ctext, col, _ in chunk_specs:
            bg = RoundedRectangle(
                corner_radius=0.1,
                width=4.7,
                height=0.88,
                fill_color="#111722",
                fill_opacity=0.95,
                stroke_color=col,
                stroke_width=1.8,
            )
            tag = Text(cid, font_size=14, color=col, weight=BOLD)
            body = Text(ctext, font_size=11, color=GREY_A, line_spacing=0.9)
            if body.width > 3.45:
                body.scale_to_fit_width(3.45)
            tag.move_to(bg.get_left() + RIGHT * 0.62)
            body.next_to(bg.get_left() + RIGHT * 1.18, RIGHT, buff=0.06)
            chunk_cards.add(VGroup(bg, tag, body))

        chunk_cards.arrange(DOWN, buff=0.18).move_to(LEFT * 4.35 + DOWN * 0.55)

        # Ortada Encoder / Embedding Model bloğu
        enc_bg = RoundedRectangle(
            corner_radius=0.16,
            width=2.55,
            height=1.75,
            fill_color="#1B1528",
            fill_opacity=0.95,
            stroke_color=GOLD_B,
            stroke_width=2.4,
        )
        enc_title = Text("Encoder", font_size=20, color=GOLD_A, weight=BOLD)
        enc_sub = Text("Embedding Model", font_size=13, color=GREY_A)
        enc_lbl = (
            VGroup(enc_title, enc_sub)
            .arrange(DOWN, buff=0.12)
            .move_to(enc_bg.get_center())
        )
        encoder_block = VGroup(enc_bg, enc_lbl).move_to(ORIGIN + DOWN * 0.55)

        # Sağda Vektör Veritabanı (Vector DB) bloğu
        vdb_bg = RoundedRectangle(
            corner_radius=0.16,
            width=4.1,
            height=3.95,
            fill_color="#0F1C1E",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=2.2,
        )
        vdb_header = Text(
            "Vektör Veritabanı (Vector DB)", font_size=15, color=BLUE_A, weight=BOLD
        )
        vdb_header.next_to(vdb_bg.get_top(), DOWN, buff=0.22)

        vdb_rows = VGroup()
        for cid, _, col, vec_str in chunk_specs:
            r_bg = RoundedRectangle(
                corner_radius=0.08,
                width=3.65,
                height=0.62,
                fill_color="#152629",
                fill_opacity=0.9,
                stroke_color=col,
                stroke_width=1.5,
            )
            r_txt = Text(f"{cid}: {vec_str}", font_size=13, color=col, weight=BOLD)
            if r_txt.width > 3.35:
                r_txt.scale_to_fit_width(3.35)
            r_txt.move_to(r_bg.get_center())
            vdb_rows.add(VGroup(r_bg, r_txt))

        vdb_rows.arrange(DOWN, buff=0.16).next_to(vdb_header, DOWN, buff=0.26)
        vdb_block = VGroup(vdb_bg, vdb_header).move_to(RIGHT * 4.45 + DOWN * 0.55)
        vdb_rows.move_to(vdb_bg.get_center() + DOWN * 0.25)

        arr_to_enc = Arrow(
            chunk_cards.get_right(),
            encoder_block.get_left(),
            buff=0.14,
            color=GOLD_B,
            stroke_width=2.8,
            max_tip_length_to_length_ratio=0.18,
        )
        arr_to_vdb = Arrow(
            encoder_block.get_right(),
            vdb_bg.get_left(),
            buff=0.14,
            color=BLUE_B,
            stroke_width=2.8,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(
            LaggedStart(
                *[FadeIn(c, shift=RIGHT * 0.2) for c in chunk_cards], lag_ratio=0.12
            ),
            FadeIn(encoder_block, scale=0.92),
            FadeIn(vdb_block, shift=LEFT * 0.2),
            run_time=self.rt(1.4),
        )
        self.play(
            GrowArrow(arr_to_enc),
            GrowArrow(arr_to_vdb),
            Indicate(encoder_block, color=GOLD_A, scale_factor=1.05),
            LaggedStart(
                *[TransformFromCopy(chunk_cards[i], vdb_rows[i]) for i in range(4)],
                lag_ratio=0.15,
            ),
            run_time=self.rt(1.8),
        )
        self.pause(0.7)

        # Sol taraftaki chunk kartlarını ve ilk oku kaldırarak sorguya yer aç
        self.play(
            FadeOut(chunk_cards, shift=LEFT * 0.2),
            FadeOut(arr_to_enc),
            FadeOut(arr_to_vdb),
            run_time=self.rt(0.7),
        )
        return encoder_block, vdb_block, vdb_rows

    # =========================================================================
    # 2. KULLANICI SORGUSUNUN VEKTÖRLEŞTİRİLMESİ
    # =========================================================================
    def stage_2_query_encoding(
        self,
        encoder_block: VGroup,
        vdb_block: VGroup,
        vdb_rows: VGroup,
    ) -> None:
        q_bg = RoundedRectangle(
            corner_radius=0.14,
            width=4.5,
            height=1.65,
            fill_color="#181424",
            fill_opacity=0.95,
            stroke_color=YELLOW_B,
            stroke_width=2.2,
        )
        q_title = Text("Kullanıcı Sorgusu", font_size=15, color=YELLOW_B, weight=BOLD)
        q_text = Text(
            '"RAG sistemi ilgili\ndokümanları nasıl bulur?"',
            font_size=15,
            color=WHITE,
            line_spacing=1.0,
        )
        q_grp = (
            VGroup(q_title, q_text).arrange(DOWN, buff=0.18).move_to(q_bg.get_center())
        )
        query_card = VGroup(q_bg, q_grp).move_to(LEFT * 4.35 + DOWN * 0.55)

        q_arrow = Arrow(
            query_card.get_right(),
            encoder_block.get_left(),
            buff=0.15,
            color=YELLOW_B,
            stroke_width=3.0,
            max_tip_length_to_length_ratio=0.18,
        )

        vq_box = RoundedRectangle(
            corner_radius=0.1,
            width=2.7,
            height=0.65,
            fill_color="#241E12",
            fill_opacity=0.95,
            stroke_color=YELLOW_B,
            stroke_width=2.2,
        )
        vq_txt = Text(
            "v_q = [0.80, 0.53, 0.28, …]", font_size=14, color=YELLOW_B, weight=BOLD
        )
        if vq_txt.width > 2.45:
            vq_txt.scale_to_fit_width(2.45)
        vq_txt.move_to(vq_box.get_center())
        vq_badge = VGroup(vq_box, vq_txt).next_to(encoder_block, UP, buff=0.35)

        self.play(
            FadeIn(query_card, shift=RIGHT * 0.25),
            GrowArrow(q_arrow),
            run_time=self.rt(1.0),
        )
        self.play(
            Indicate(encoder_block, color=YELLOW_B, scale_factor=1.06),
            TransformFromCopy(query_card, vq_badge),
            run_time=self.rt(1.2),
        )
        self.pause(0.8)

        self.play(
            FadeOut(query_card),
            FadeOut(q_arrow),
            FadeOut(vq_badge),
            FadeOut(encoder_block),
            FadeOut(vdb_block),
            FadeOut(vdb_rows),
            run_time=self.rt(0.8),
        )

    # =========================================================================
    # 3. 2D DÜZLEMDE COSINE SIMILARITY VE FORMÜL
    # =========================================================================
    def stage_3_2d_cosine_similarity(self) -> None:
        axes_2d = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 3.5, 1],
            x_length=5.6,
            y_length=4.6,
            axis_config={"color": GREY_C, "include_tip": True},
        ).move_to(LEFT * 2.35 + DOWN * 0.55)

        x_lbl = Text("X", font_size=15, color=GREY_B).next_to(
            axes_2d.x_axis.get_end(), RIGHT, buff=0.12
        )
        y_lbl = Text("Y", font_size=15, color=GREY_B).next_to(
            axes_2d.y_axis.get_end(), UP, buff=0.12
        )

        # Sağ üstte Cosine Similarity formül kartı
        formula_mob = self._build_cosine_formula()
        if formula_mob.width > 5.2:
            formula_mob.scale_to_fit_width(5.2)
        f_box = RoundedRectangle(
            corner_radius=0.14,
            width=formula_mob.width + 0.6,
            height=1.35,
            fill_color="#121824",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=2.0,
        )
        formula_card = VGroup(f_box, formula_mob).move_to(RIGHT * 3.55 + UP * 1.35)
        formula_mob.move_to(f_box.get_center())

        origin_2d = axes_2d.c2p(0, 0)

        # Vektör uçları (Sorgu v_q'ya en yakın açı Chunk A'da: θ -> 0°)
        vec_specs_2d = [
            ("Sorgu (v_q)", (2.9, 2.2), YELLOW_B, 4.2),
            ("Chunk A", (3.2, 2.15), GREEN_B, 3.8),
            ("Chunk D", (2.6, 2.75), ORANGE, 3.2),
            ("Chunk C", (3.3, 1.25), TEAL_B, 3.2),
            ("Chunk B", (0.85, 3.0), PURPLE_B, 3.0),
        ]

        arrows_2d = VGroup()
        labels_2d = VGroup()
        label_offsets = [
            np.array([-0.15, 0.26, 0.0]),  # Sorgu (v_q)
            np.array([0.55, -0.05, 0.0]),  # Chunk A
            np.array([-0.1, 0.25, 0.0]),  # Chunk D
            np.array([0.55, 0.0, 0.0]),  # Chunk C
            np.array([0.0, 0.25, 0.0]),  # Chunk B
        ]

        for (v_name, coords, col, sw), off in zip(vec_specs_2d, label_offsets):
            tip = axes_2d.c2p(*coords)
            arr = Arrow(
                origin_2d,
                tip,
                buff=0,
                color=col,
                stroke_width=sw,
                max_tip_length_to_length_ratio=0.14,
            )
            lbl = Text(v_name, font_size=14, color=col, weight=BOLD).move_to(tip + off)
            arrows_2d.add(arr)
            labels_2d.add(lbl)

        self.play(
            Create(axes_2d),
            FadeIn(x_lbl),
            FadeIn(y_lbl),
            FadeIn(formula_card, shift=LEFT * 0.2),
            run_time=self.rt(1.2),
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows_2d], lag_ratio=0.12),
            LaggedStart(*[FadeIn(l) for l in labels_2d], lag_ratio=0.12),
            run_time=self.rt(1.6),
        )

        # Sorgu vektörü (arrows_2d[0]) ile Chunk A (arrows_2d[1]) arasındaki küçük θ açısını vurgula
        angle_arc = Angle(
            arrows_2d[1],
            arrows_2d[0],
            radius=1.35,
            color=YELLOW_B,
            stroke_width=3.5,
        )
        theta_lbl = Text(
            "θ ≈ 4° (Çok yakın!)", font_size=14, color=YELLOW_B, weight=BOLD
        ).next_to(angle_arc, RIGHT, buff=0.18)

        sim_highlight_box = RoundedRectangle(
            corner_radius=0.12,
            width=4.8,
            height=1.3,
            fill_color="#11241C",
            fill_opacity=0.95,
            stroke_color=GREEN_B,
            stroke_width=2.2,
        )
        sim_line1 = Text(
            "En Küçük Açı (θ → 0°): Chunk A", font_size=15, color=GREEN_B, weight=BOLD
        )
        sim_line2 = Text(
            "cos(θ) = 0.91  (Yüksek Benzerlik)",
            font_size=15,
            color=YELLOW_B,
            weight=BOLD,
        )
        sim_txt = (
            VGroup(sim_line1, sim_line2)
            .arrange(DOWN, buff=0.14)
            .move_to(sim_highlight_box.get_center())
        )
        sim_card = VGroup(sim_highlight_box, sim_txt).move_to(
            RIGHT * 3.55 + DOWN * 1.15
        )

        self.play(
            Create(angle_arc),
            FadeIn(theta_lbl),
            Indicate(arrows_2d[0], color=YELLOW_A, scale_factor=1.04),
            Indicate(arrows_2d[1], color=GREEN_A, scale_factor=1.04),
            FadeIn(sim_card, shift=UP * 0.2),
            run_time=self.rt(1.4),
        )
        self.pause(1.2)

        self.play(
            FadeOut(axes_2d),
            FadeOut(x_lbl),
            FadeOut(y_lbl),
            FadeOut(formula_card),
            FadeOut(arrows_2d),
            FadeOut(labels_2d),
            FadeOut(angle_arc),
            FadeOut(theta_lbl),
            FadeOut(sim_card),
            run_time=self.rt(0.8),
        )

    # =========================================================================
    # 4 & 5. 3D VEKTÖR UZAYINA GEÇİŞ, ARAMA YARIÇAPI VE SIRALI SONUÇLAR
    # =========================================================================
    def stage_4_and_5_3d_space_and_results(self) -> None:
        axes_3d = ThreeDAxes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            z_range=[0, 3.2, 1],
            x_length=5.2,
            y_length=5.2,
            z_length=3.6,
            axis_config={"color": GREY_C, "include_tip": True},
        ).move_to(LEFT * 2.1 + DOWN * 0.5)

        origin_3d = axes_3d.c2p(0, 0, 0)

        # 3B Koordinatlar (Sorgu v_q'ya en yakın Chunk A, sonra D, sonra C, en uzak B)
        pts_spec = [
            ("Sorgu (v_q)", (2.7, 2.5, 2.2), YELLOW_B),
            ("Chunk A", (2.9, 2.4, 2.15), GREEN_B),
            ("Chunk D", (2.3, 2.8, 1.85), ORANGE),
            ("Chunk C", (3.1, 1.6, 1.5), TEAL_B),
            ("Chunk B", (0.9, 3.2, 0.7), PURPLE_B),
        ]

        lines_3d: list[Line3D] = []
        dots_3d: list[Dot3D] = []
        labels_3d: list[Text] = []
        coords_3d: list[np.ndarray] = []

        offsets_3d = [
            np.array([-0.2, -0.25, 0.32]),
            np.array([0.32, 0.2, 0.28]),
            np.array([-0.25, 0.3, 0.25]),
            np.array([0.3, -0.2, 0.24]),
            np.array([0.25, 0.25, 0.25]),
        ]

        for (name, c_xyz, col), off in zip(pts_spec, offsets_3d):
            p_world = axes_3d.c2p(*c_xyz)
            coords_3d.append(p_world)
            line_3d = Line3D(origin_3d, p_world, color=col, thickness=0.02)
            dot_3d = Dot3D(point=p_world, radius=0.095, color=col)
            lbl_3d = Text(name, font_size=16, color=col, weight=BOLD).move_to(
                p_world + off
            )
            lines_3d.append(line_3d)
            dots_3d.append(dot_3d)
            labels_3d.append(lbl_3d)

        self.add_fixed_orientation_mobjects(*labels_3d)

        self.play(
            Create(axes_3d),
            *[Create(l) for l in lines_3d],
            *[FadeIn(d, scale=1.3) for d in dots_3d],
            *[FadeIn(lbl) for lbl in labels_3d],
            run_time=self.rt(1.4),
        )

        # Kamera açısı yumuşak bir hareketle 3B uzaya geçer:
        # move_camera(phi=65*DEGREES, theta=-45*DEGREES)
        self.move_camera(
            phi=65 * DEGREES,
            theta=-45 * DEGREES,
            run_time=self.rt(2.4),
        )

        # Sorgu noktası etrafında arama yarıçapı (3B küresel dalga & en yakın komşu bağlantıları)
        q_pt = coords_3d[0]
        search_ring_1 = Circle(radius=0.55, color=YELLOW_B, stroke_width=2.8).move_to(
            q_pt
        )
        search_ring_2 = Circle(
            radius=1.15, color=GREEN_B, stroke_width=2.0, stroke_opacity=0.75
        ).move_to(q_pt)
        self.add_fixed_orientation_mobjects(search_ring_1, search_ring_2)

        sim_link_a = Line3D(q_pt, coords_3d[1], color=GREEN_B, thickness=0.025)
        sim_link_d = Line3D(q_pt, coords_3d[2], color=ORANGE, thickness=0.018)
        sim_link_c = Line3D(q_pt, coords_3d[3], color=TEAL_B, thickness=0.015)

        self.play(
            Create(search_ring_1),
            Create(search_ring_2),
            Create(sim_link_a),
            Create(sim_link_d),
            Create(sim_link_c),
            run_time=self.rt(1.4),
        )

        # =====================================================================
        # 5. ARAMA SONUÇLARI (RANKED SEARCH RESULTS - SAĞ PANELDE SABİT HUD)
        # =====================================================================
        panel_bg = RoundedRectangle(
            corner_radius=0.16,
            width=5.4,
            height=4.35,
            fill_color="#0E151D",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=2.2,
        )
        panel_title = Text(
            "Retrieved Chunks (Cosine Ranking)", font_size=16, color=BLUE_A, weight=BOLD
        )
        panel_title.next_to(panel_bg.get_top(), DOWN, buff=0.22)

        ranked_data = [
            ("1. Chunk A", "Benzerlik: 0.91 — [En İlgili Sonuç]", GREEN_B, True),
            ("2. Chunk D", "Benzerlik: 0.78", ORANGE, False),
            ("3. Chunk C", "Benzerlik: 0.65", TEAL_B, False),
            ("4. Chunk B", "Benzerlik: 0.42 (Uzak açı)", GREY_B, False),
        ]

        ranked_rows = VGroup()
        for rank_title, sim_str, col, is_top in ranked_data:
            r_box = RoundedRectangle(
                corner_radius=0.1,
                width=4.95,
                height=0.74,
                fill_color="#142921" if is_top else "#151C26",
                fill_opacity=0.95 if is_top else 0.85,
                stroke_color=col,
                stroke_width=2.5 if is_top else 1.5,
            )
            t1 = Text(rank_title, font_size=14, color=col, weight=BOLD)
            t2 = Text(sim_str, font_size=12, color=WHITE if is_top else GREY_A)
            txt_col = VGroup(t1, t2).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
            if txt_col.width > 4.55:
                txt_col.scale_to_fit_width(4.55)
            txt_col.move_to(
                r_box.get_left() + RIGHT * 0.25 + RIGHT * (txt_col.width / 2)
            )
            ranked_rows.add(VGroup(r_box, txt_col))

        ranked_rows.arrange(DOWN, buff=0.15).next_to(panel_title, DOWN, buff=0.24)
        results_panel = VGroup(panel_bg, panel_title, ranked_rows).move_to(
            RIGHT * 3.85 + DOWN * 0.45
        )

        # Sağdaki sonuç panelini 3B kameradan etkilenmemesi için ekrana sabitle
        self.add_fixed_in_frame_mobjects(results_panel)

        self.play(
            FadeIn(panel_bg, shift=LEFT * 0.2),
            FadeIn(panel_title, shift=LEFT * 0.2),
            LaggedStart(
                *[FadeIn(r, shift=LEFT * 0.2) for r in ranked_rows],
                lag_ratio=0.18,
            ),
            run_time=self.rt(1.8),
        )

        # Kamera hafifçe dönerken en yüksek benzerlikli Chunk A öne çıkar
        self.move_camera(
            phi=62 * DEGREES,
            theta=-25 * DEGREES,
            added_anims=[Indicate(ranked_rows[0], color=GREEN_A, scale_factor=1.04)],
            run_time=self.rt(2.2),
        )
        self.pause(1.5)
