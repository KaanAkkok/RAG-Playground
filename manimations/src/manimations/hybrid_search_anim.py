from __future__ import annotations

from typing import Any

import numpy as np
from manim import *

# Hiz sabiti: Tum animasyon ve bekleme sureleri bu carpanla ayarlanir.
# Daha rahat takip edilebilmesi icin yavaslatilmis deger (0.65).
SPEED: float = 0.65


class HybridSearchScene(ThreeDScene):
    """
    Retrieval Pipeline asamasinda Hybrid Search (Sparse BM25 + Dense Vektorel Arama + RRF)
    mekanizmasini ikon veya emoji kullanmadan, gorsel ve geometrik odakli olarak
    anlatan 3B Manim sahnesi.
    """

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data: dict[str, Any] = data or {}

    def rt(self, seconds: float) -> float:
        return max(0.15, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.15, seconds / SPEED))

    def construct(self) -> None:
        self.camera.background_color = BLACK

        # =====================================================================
        # SABIT BASLIK BILESENI (Aynen kullanilacak)
        # =====================================================================
        title = Text("3. Retrieval Pipeline", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text("Hybrid Search", font_size=18, color=RED_B)
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
            "Chunk B": PURPLE_B,
            "Chunk D": ORANGE,
        }

        # =====================================================================
        # 1. SORGU GIRISI VE PARALEL CATALLANMA (FORKING)
        # =====================================================================
        q_box = RoundedRectangle(
            corner_radius=0.14,
            width=2.0,
            height=1.15,
            fill_color="#121824",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=2.2,
        )
        q_lbl = Text("Query", font_size=20, color=BLUE_A, weight=BOLD)
        q_sub = Text("Kullanici Sorgusu", font_size=12, color=GREY_B)
        q_txt = VGroup(q_lbl, q_sub).arrange(DOWN, buff=0.1).move_to(q_box.get_center())
        query_node = VGroup(q_box, q_txt).move_to(LEFT * 5.65 + DOWN * 0.45)

        # Ust Kol: Sparse (BM25) - Kelime eslesmesi
        sp_box = RoundedRectangle(
            corner_radius=0.12,
            width=2.55,
            height=0.95,
            fill_color="#14202E",
            fill_opacity=0.95,
            stroke_color=YELLOW_B,
            stroke_width=2.0,
        )
        sp_title = Text("Sparse (BM25)", font_size=16, color=YELLOW_B, weight=BOLD)
        sp_sub = Text("Kelime eslesmesi", font_size=12, color=GREY_A)
        sp_txt = (
            VGroup(sp_title, sp_sub)
            .arrange(DOWN, buff=0.08)
            .move_to(sp_box.get_center())
        )
        sparse_branch = VGroup(sp_box, sp_txt).move_to(LEFT * 2.45 + UP * 1.1)

        # Alt Kol: Dense (Vectors) - Anlamsal yakinlik
        dn_box = RoundedRectangle(
            corner_radius=0.12,
            width=2.55,
            height=0.95,
            fill_color="#1B1528",
            fill_opacity=0.95,
            stroke_color=PURPLE_A,
            stroke_width=2.0,
        )
        dn_title = Text("Dense (Vectors)", font_size=16, color=PURPLE_A, weight=BOLD)
        dn_sub = Text("Anlamsal yakinlik", font_size=12, color=GREY_A)
        dn_txt = (
            VGroup(dn_title, dn_sub)
            .arrange(DOWN, buff=0.08)
            .move_to(dn_box.get_center())
        )
        dense_branch = VGroup(dn_box, dn_txt).move_to(LEFT * 2.45 + DOWN * 1.95)

        fork_up = Arrow(
            query_node.get_right(),
            sparse_branch.get_left(),
            buff=0.1,
            color=YELLOW_B,
            stroke_width=2.6,
            max_tip_length_to_length_ratio=0.16,
        )
        fork_down = Arrow(
            query_node.get_right(),
            dense_branch.get_left(),
            buff=0.1,
            color=PURPLE_A,
            stroke_width=2.6,
            max_tip_length_to_length_ratio=0.16,
        )

        # ONEMLI: Her nesneyi YALNIZCA sahneye girdigi anda add_fixed_in_frame_mobjects ile ekle
        self.add_fixed_in_frame_mobjects(query_node)
        self.play(FadeIn(query_node, shift=RIGHT * 0.2), run_time=self.rt(1.1))
        self.pause(0.5)

        self.add_fixed_in_frame_mobjects(
            fork_up, fork_down, sparse_branch, dense_branch
        )
        self.play(
            GrowArrow(fork_up),
            GrowArrow(fork_down),
            FadeIn(sparse_branch, shift=RIGHT * 0.15),
            FadeIn(dense_branch, shift=RIGHT * 0.15),
            run_time=self.rt(1.3),
        )
        self.pause(0.6)

        # =====================================================================
        # 2A. UST KOL: SPARSE MATRISI VE SIRALI LISTE
        # =====================================================================
        n_rows, n_cols = 3, 4
        cell_w, cell_h = 0.42, 0.34
        matrix_cells = VGroup()
        active_coords = {(0, 0), (0, 2), (0, 3), (1, 0), (1, 2), (2, 2)}
        highlight_cells = VGroup()

        for r in range(n_rows):
            row_grp = VGroup()
            for c in range(n_cols):
                cell = RoundedRectangle(
                    corner_radius=0.04,
                    width=cell_w,
                    height=cell_h,
                    fill_color="#162230",
                    fill_opacity=0.9,
                    stroke_color=GREY_C,
                    stroke_width=1.2,
                )
                row_grp.add(cell)
                if (r, c) in active_coords:
                    highlight_cells.add(cell)
            row_grp.arrange(RIGHT, buff=0.06)
            matrix_cells.add(row_grp)
        matrix_cells.arrange(DOWN, buff=0.06)

        mat_frame = RoundedRectangle(
            corner_radius=0.08,
            width=matrix_cells.width + 0.24,
            height=matrix_cells.height + 0.24,
            fill_color="#0E151F",
            fill_opacity=0.95,
            stroke_color=BLUE_C,
            stroke_width=1.6,
        )
        mat_lbl = Text("Terim Matrisi", font_size=11, color=GREY_B)
        sparse_matrix = VGroup(mat_frame, matrix_cells).move_to(RIGHT * 1.15 + UP * 1.1)
        matrix_cells.move_to(mat_frame.get_center())
        mat_lbl.next_to(mat_frame, UP, buff=0.08)

        arr_sp_to_mat = Arrow(
            sparse_branch.get_right(),
            mat_frame.get_left(),
            buff=0.1,
            color=YELLOW_B,
            stroke_width=2.2,
            max_tip_length_to_length_ratio=0.18,
        )

        scan_beam = Line(
            mat_frame.get_top() + LEFT * (mat_frame.width * 0.42),
            mat_frame.get_bottom() + LEFT * (mat_frame.width * 0.42),
            color=YELLOW_B,
            stroke_width=4.0,
        )

        def make_mini_list(
            title_str: str,
            border_col: ManimColor,
            ranks: list[tuple[str, str]],
            pos: np.ndarray,
        ) -> VGroup:
            hdr = Text(title_str, font_size=13, color=border_col, weight=BOLD)
            rows = VGroup()
            for cid, label_str in ranks:
                c_col = chunk_colors[cid]
                r_bg = RoundedRectangle(
                    corner_radius=0.07,
                    width=2.65,
                    height=0.42,
                    fill_color="#121A24",
                    fill_opacity=0.95,
                    stroke_color=c_col,
                    stroke_width=1.7,
                )
                r_txt = Text(label_str, font_size=12, color=WHITE, weight=BOLD)
                r_txt.move_to(r_bg.get_center())
                rows.add(VGroup(r_bg, r_txt))
            rows.arrange(DOWN, buff=0.1)
            grp = VGroup(hdr, rows).arrange(DOWN, buff=0.12).move_to(pos)
            return grp

        sparse_list = make_mini_list(
            "Sparse Liste",
            YELLOW_B,
            [
                ("Chunk A", "Rank 1: Chunk A"),
                ("Chunk C", "Rank 2: Chunk C"),
                ("Chunk B", "Rank 3: Chunk B"),
            ],
            RIGHT * 4.9 + UP * 1.1,
        )
        arr_mat_to_list = Arrow(
            mat_frame.get_right(),
            sparse_list[1].get_left(),
            buff=0.12,
            color=YELLOW_B,
            stroke_width=2.2,
            max_tip_length_to_length_ratio=0.18,
        )

        # Sadece matris ve giris okunu ekle ve oynat
        self.add_fixed_in_frame_mobjects(arr_sp_to_mat, sparse_matrix, mat_lbl)
        self.play(
            GrowArrow(arr_sp_to_mat),
            FadeIn(sparse_matrix),
            FadeIn(mat_lbl),
            run_time=self.rt(1.0),
        )

        # Tarama isigini simdi ekle ve oynat
        self.add_fixed_in_frame_mobjects(scan_beam)
        self.play(FadeIn(scan_beam), run_time=self.rt(0.25))
        self.play(
            scan_beam.animate.shift(RIGHT * (mat_frame.width * 0.84)),
            *[
                c.animate.set_fill(YELLOW_B, opacity=0.65).set_stroke(
                    YELLOW_A, width=1.8
                )
                for c in highlight_cells
            ],
            run_time=self.rt(1.2),
        )

        # Sparse listesini YALNIZCA simdi ekle ve oynat (erken gorunup silinme hatasini onler)
        self.add_fixed_in_frame_mobjects(arr_mat_to_list, sparse_list)
        self.play(
            FadeOut(scan_beam),
            GrowArrow(arr_mat_to_list),
            FadeIn(sparse_list, shift=RIGHT * 0.15),
            run_time=self.rt(1.2),
        )
        self.pause(0.6)

        # =====================================================================
        # 2B. ALT KOL: DENSE 3D VEKTOR UZAYI VE SIRALI LISTE
        # =====================================================================
        # ONEMLI: Kamera phi=60, theta=-45 acisina egildiginde 3B nesnenin ekranda
        # tam (X_screen = +1.15, Y_screen = -1.95) noktasina dusmesi icin
        # rotasyon matrisinin tersiyle (R.T @ screen_pos) 3B dunya koordinatini hesapliyoruz.
        phi_val = 60 * DEGREES
        theta_val = -45 * DEGREES
        rot_mat = np.array(
            [
                [np.cos(theta_val + PI / 2), np.sin(theta_val + PI / 2), 0.0],
                [
                    -np.cos(phi_val) * np.sin(theta_val + PI / 2),
                    np.cos(phi_val) * np.cos(theta_val + PI / 2),
                    np.sin(phi_val),
                ],
                [
                    np.sin(phi_val) * np.sin(theta_val + PI / 2),
                    -np.sin(phi_val) * np.cos(theta_val + PI / 2),
                    np.cos(phi_val),
                ],
            ]
        )
        desired_screen_center = np.array([1.15, -1.95, 0.0])
        world_center_3d = rot_mat.T @ desired_screen_center

        axes_3d = ThreeDAxes(
            x_range=[0, 3, 1],
            y_range=[0, 3, 1],
            z_range=[0, 2.5, 1],
            x_length=2.1,
            y_length=2.1,
            z_length=1.6,
            axis_config={"color": GREY_C, "include_tip": True},
        ).move_to(world_center_3d)

        origin_3d = axes_3d.c2p(0, 0, 0)
        q_pt_3d = axes_3d.c2p(2.1, 1.9, 1.8)
        c_pt_3d = axes_3d.c2p(2.3, 1.8, 1.75)  # Rank 1: Chunk C
        a_pt_3d = axes_3d.c2p(1.8, 2.2, 1.6)  # Rank 2: Chunk A
        d_pt_3d = axes_3d.c2p(2.4, 1.2, 1.1)  # Rank 3: Chunk D

        q_vec_line = Line3D(origin_3d, q_pt_3d, color=RED_B, thickness=0.02)
        q_dot = Dot3D(point=q_pt_3d, radius=0.075, color=RED_B)
        dot_c = Dot3D(point=c_pt_3d, radius=0.07, color=TEAL_B)
        dot_a = Dot3D(point=a_pt_3d, radius=0.07, color=GREEN_B)
        dot_d = Dot3D(point=d_pt_3d, radius=0.07, color=ORANGE)

        dist_q_c = Line3D(q_pt_3d, c_pt_3d, color=TEAL_B, thickness=0.018)
        dist_q_a = Line3D(q_pt_3d, a_pt_3d, color=GREEN_B, thickness=0.018)
        dist_q_d = Line3D(q_pt_3d, d_pt_3d, color=ORANGE, thickness=0.015)

        arr_dn_to_3d = Arrow(
            dense_branch.get_right(),
            np.array([0.05, -1.95, 0.0]),
            buff=0.1,
            color=PURPLE_A,
            stroke_width=2.2,
            max_tip_length_to_length_ratio=0.18,
        )

        # Once kamerayi 3B aciya eg, ardindan 3B eksen ve noktalari tam ortada olustur
        self.add_fixed_in_frame_mobjects(arr_dn_to_3d)
        self.move_camera(
            phi=60 * DEGREES,
            theta=-45 * DEGREES,
            added_anims=[GrowArrow(arr_dn_to_3d)],
            run_time=self.rt(1.0),
        )
        self.play(
            Create(axes_3d),
            Create(q_vec_line),
            FadeIn(q_dot),
            FadeIn(dot_c),
            FadeIn(dot_a),
            FadeIn(dot_d),
            run_time=self.rt(1.3),
        )

        dense_list = make_mini_list(
            "Dense Liste",
            PURPLE_A,
            [
                ("Chunk C", "Rank 1: Chunk C"),
                ("Chunk A", "Rank 2: Chunk A"),
                ("Chunk D", "Rank 3: Chunk D"),
            ],
            RIGHT * 4.9 + DOWN * 1.95,
        )
        arr_3d_to_list = Arrow(
            np.array([2.25, -1.95, 0.0]),
            dense_list[1].get_left(),
            buff=0.12,
            color=PURPLE_A,
            stroke_width=2.2,
            max_tip_length_to_length_ratio=0.18,
        )

        # Dense listesini YALNIZCA simdi ekle ve oynat (erken gorunup silinme hatasini onler)
        self.add_fixed_in_frame_mobjects(arr_3d_to_list, dense_list)
        self.play(
            Create(dist_q_c),
            Create(dist_q_a),
            Create(dist_q_d),
            GrowArrow(arr_3d_to_list),
            FadeIn(dense_list, shift=RIGHT * 0.15),
            run_time=self.rt(1.4),
        )
        self.pause(1.0)

        # =====================================================================
        # 3. RECIPROCAL RANK FUSION (RRF BIRLESIMI)
        # =====================================================================
        self.play(
            FadeOut(query_node),
            FadeOut(fork_up),
            FadeOut(fork_down),
            FadeOut(sparse_branch),
            FadeOut(dense_branch),
            FadeOut(arr_sp_to_mat),
            FadeOut(sparse_matrix),
            FadeOut(mat_lbl),
            FadeOut(arr_mat_to_list),
            FadeOut(arr_dn_to_3d),
            FadeOut(axes_3d),
            FadeOut(q_vec_line),
            FadeOut(q_dot),
            FadeOut(dot_c),
            FadeOut(dot_a),
            FadeOut(dot_d),
            FadeOut(dist_q_c),
            FadeOut(dist_q_a),
            FadeOut(dist_q_d),
            FadeOut(arr_3d_to_list),
            sparse_list.animate.move_to(LEFT * 5.15 + DOWN * 0.85),
            dense_list.animate.move_to(RIGHT * 5.15 + DOWN * 0.85),
            run_time=self.rt(1.6),
        )
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.pause(0.4)

        # Merkez ustte RRF LaTeX formulu kutusu acilir
        rrf_tex = MathTex(
            r"\text{RRF Score}(d) = \sum_{m \in \{\text{Sparse, Dense}\}} \frac{1}{k + r_m(d)}",
            font_size=28,
            color=WHITE,
        )
        rrf_param = Text("k = 60, r = Rank", font_size=13, color=GREY_B)
        rrf_inner = VGroup(rrf_tex, rrf_param).arrange(DOWN, buff=0.12)
        if rrf_inner.width > 5.8:
            rrf_inner.scale_to_fit_width(5.8)

        rrf_box = RoundedRectangle(
            corner_radius=0.14,
            width=rrf_inner.width + 0.5,
            height=rrf_inner.height + 0.34,
            fill_color="#121824",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=1.9,
        )
        rrf_card = VGroup(rrf_box, rrf_inner).move_to(RIGHT * 0.85 + UP * 2.25)
        rrf_inner.move_to(rrf_box.get_center())

        self.add_fixed_in_frame_mobjects(rrf_card)
        self.play(
            FadeIn(rrf_box, shift=DOWN * 0.15),
            Write(rrf_tex),
            FadeIn(rrf_param),
            run_time=self.rt(1.4),
        )
        self.pause(0.5)

        # Iki ayri siralama listesinden r degerleri cekilerek merkezde toplanir
        fusion_specs = [
            (
                "Chunk A",
                r"\text{Chunk A: } \frac{1}{60+1} + \frac{1}{60+2} = 0.03251",
                GREEN_B,
                sparse_list[1][0],
                dense_list[1][1],
            ),
            (
                "Chunk C",
                r"\text{Chunk C: } \frac{1}{60+2} + \frac{1}{60+1} = 0.03251",
                TEAL_B,
                sparse_list[1][1],
                dense_list[1][0],
            ),
            (
                "Chunk D / B",
                r"\text{Chunk D / B: } \frac{1}{60+3} \text{ (Tek kanal)} = 0.01587",
                ORANGE,
                sparse_list[1][2],
                dense_list[1][2],
            ),
        ]

        fusion_rows = VGroup()
        for _, tex_str, col, _, _ in fusion_specs:
            f_bg = RoundedRectangle(
                corner_radius=0.09,
                width=5.4,
                height=0.66,
                fill_color="#141C26",
                fill_opacity=0.95,
                stroke_color=col,
                stroke_width=1.9,
            )
            f_math = MathTex(tex_str, font_size=22, color=WHITE)
            if f_math.width > 5.05:
                f_math.scale_to_fit_width(5.05)
            f_math.move_to(f_bg.get_center())
            fusion_rows.add(VGroup(f_bg, f_math))

        fusion_rows.arrange(DOWN, buff=0.22).move_to(DOWN * 0.85)

        fusion_arrows = VGroup()
        for idx, (_, _, col, sp_row, dn_row) in enumerate(fusion_specs):
            target_row = fusion_rows[idx]
            arr_l = Arrow(
                sp_row.get_right(),
                target_row.get_left(),
                buff=0.08,
                color=col,
                stroke_width=2.0,
                max_tip_length_to_length_ratio=0.14,
            )
            arr_r = Arrow(
                dn_row.get_left(),
                target_row.get_right(),
                buff=0.08,
                color=col,
                stroke_width=2.0,
                max_tip_length_to_length_ratio=0.14,
            )
            fusion_arrows.add(arr_l, arr_r)

            # Yalnizca o satirin sırası geldiginde sahneye sabitle ve oynat
            self.add_fixed_in_frame_mobjects(arr_l, arr_r, target_row)
            self.play(
                GrowArrow(arr_l),
                GrowArrow(arr_r),
                FadeIn(target_row, scale=0.96),
                run_time=self.rt(1.0),
            )

        self.pause(1.1)

        # =====================================================================
        # 4. NIHAI HIBRIT SIRALAMA (FINAL RRF LIST)
        # =====================================================================
        self.play(
            FadeOut(sparse_list, shift=LEFT * 0.25),
            FadeOut(dense_list, shift=RIGHT * 0.25),
            FadeOut(fusion_arrows),
            run_time=self.rt(0.9),
        )

        final_title = Text(
            "Final RRF List (Nihai Hibrit Siralama)",
            font_size=17,
            color=BLUE_A,
            weight=BOLD,
        )

        final_data = [
            (
                "1. Chunk A  (RRF Skoru en yuksek - her iki kanaldan guclu sinyal)",
                GREEN_B,
                True,
            ),
            (
                "2. Chunk C  (RRF Skoru yuksek - her iki kanaldan guclu sinyal)",
                TEAL_B,
                True,
            ),
            (
                "3. Chunk D / Chunk B  (Dusuk sirali veya tek tarafli sinyal)",
                ORANGE,
                False,
            ),
        ]

        final_cards = VGroup()
        for label_str, col, is_top in final_data:
            c_bg = RoundedRectangle(
                corner_radius=0.1,
                width=7.4,
                height=0.68,
                fill_color="#12261F" if is_top else "#141820",
                fill_opacity=0.95 if is_top else 0.72,
                stroke_color=col,
                stroke_width=2.2 if is_top else 1.4,
            )
            c_txt = Text(
                label_str,
                font_size=14,
                color=WHITE if is_top else GREY_A,
                weight=BOLD,
            )
            if c_txt.width > 7.0:
                c_txt.scale_to_fit_width(7.0)
            c_txt.move_to(c_bg.get_center())
            final_cards.add(VGroup(c_bg, c_txt))

        final_cards.arrange(DOWN, buff=0.2)

        self.add_fixed_in_frame_mobjects(final_title, final_cards)
        self.play(
            FadeIn(final_title, shift=DOWN * 0.15),
            *[ReplacementTransform(fusion_rows[i], final_cards[i]) for i in range(3)],
            run_time=self.rt(1.4),
        )

        # En tepedeki 2 kart yesil cerceveyle parlayarak "Selected Top-K Context" olur
        topk_pair = VGroup(final_cards[0], final_cards[1])
        topk_border = RoundedRectangle(
            corner_radius=0.15,
            width=topk_pair.width + 0.32,
            height=topk_pair.height + 0.26,
            fill_opacity=0.0,
            stroke_color=GREEN_B,
            stroke_width=3.2,
        ).move_to(topk_pair.get_center())

        topk_tag_bg = RoundedRectangle(
            corner_radius=0.09,
            width=2.75,
            height=0.46,
            fill_color="#132A1F",
            fill_opacity=0.95,
            stroke_color=GREEN_B,
            stroke_width=2.0,
        )
        topk_tag_txt = Text(
            "Selected Top-K Context", font_size=13, color=GREEN_A, weight=BOLD
        )
        topk_tag_txt.move_to(topk_tag_bg.get_center())
        topk_tag = VGroup(topk_tag_bg, topk_tag_txt).next_to(
            topk_border, RIGHT, buff=0.25
        )

        self.add_fixed_in_frame_mobjects(topk_border, topk_tag)
        self.play(
            Create(topk_border),
            FadeIn(topk_tag, shift=LEFT * 0.15),
            final_cards[2].animate.set_opacity(0.5),
            run_time=self.rt(1.3),
        )
        self.pause(2.0)
