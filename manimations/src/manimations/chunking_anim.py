from __future__ import annotations

import numpy as np
from manim import *

# Hız sabiti: Tüm animasyon ve bekleme süreleri bu çarpanla ayarlanır.
SPEED: float = 1.0


class ChunkingScene(ThreeDScene):
    """
    Bir Markdown dokümanın RAG için nasıl chunk'lara bölündüğünü
    az yazıyla, hareket, renk ve şekillerle anlatan 4 sahnelik ThreeDScene.
    """

    def rt(self, seconds: float) -> float:
        return max(0.1, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.1, seconds / SPEED))

    def construct(self) -> None:
        self.camera.background_color = BLACK

        # =====================================================================
        # SABİT BAŞLIK (Tüm video boyunca sol üstte sabit kalır)
        # =====================================================================
        title = Text("1. Ingestion Pipeline", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text("Chunking", font_size=18, color=RED_B)
        title_group = (
            VGroup(title, subtitle)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .to_corner(UL, buff=0.5)
        )
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(FadeIn(title_group, shift=DOWN * 0.2), run_time=1.2)
        self.wait(0.3)

        # 4 sahneyi sırayla çalıştır
        blocks = self.scene_1_structure()
        self.scene_2_sentences(blocks)
        self.scene_3_semantic()
        self.scene_4_metadata()

    # =========================================================================
    # SAHNE 1: YAPIYA GÖRE PARÇALAMA (~14 sn)
    # =========================================================================
    def scene_1_structure(self) -> dict[str, VGroup]:
        # Sol tarafta örnek Markdown doküman kartı
        doc_bg = RoundedRectangle(
            corner_radius=0.15,
            width=5.2,
            height=4.8,
            fill_color="#121620",
            fill_opacity=0.95,
            stroke_color=GREY_B,
            stroke_width=2,
        )
        doc_header = Text("Doküman", font_size=18, color=GREY_A, weight=BOLD)

        md_h1 = Text("# RAG", font_size=18, color=BLUE_B, weight=BOLD)
        md_h2 = Text("## Retrieval", font_size=16, color=BLUE_B, weight=BOLD)

        para_lines = VGroup(
            Text("Sorgu vektöre çevrilir. Vektör DB'de", font_size=13, color=GREEN_B),
            Text(
                "arama yapılır. En yakın parçalar seçilir.", font_size=13, color=GREEN_B
            ),
            Text("Bulunan bağlam prompt'a eklenir.", font_size=13, color=GREEN_B),
            Text("LLM yalnızca bağlamdan yanıt üretir.", font_size=13, color=GREEN_B),
        ).arrange(DOWN, buff=0.06, aligned_edge=LEFT)

        tbl_lines = VGroup(
            Text("| Bileşen   | Görev        |", font_size=13, color=ORANGE),
            Text("|-----------|--------------|", font_size=13, color=ORANGE),
            Text("| Chunker   | Metni böler  |", font_size=13, color=ORANGE),
            Text("| Embedder  | Vektör yapar |", font_size=13, color=ORANGE),
            Text("| Retriever | Parça bulur  |", font_size=13, color=ORANGE),
        ).arrange(DOWN, buff=0.05, aligned_edge=LEFT)

        doc_body = VGroup(md_h1, md_h2, para_lines, tbl_lines).arrange(
            DOWN, buff=0.22, aligned_edge=LEFT
        )
        doc_header.next_to(doc_bg.get_top(), DOWN, buff=0.22)
        doc_body.next_to(doc_header, DOWN, buff=0.25).align_to(
            doc_bg.get_left() + RIGHT * 0.35, LEFT
        )

        doc_card = VGroup(doc_bg, doc_header, doc_body).move_to(
            LEFT * 3.6 + DOWN * 0.45
        )

        self.play(FadeIn(doc_card, shift=RIGHT * 0.25), run_time=self.rt(1.4))
        self.pause(1.2)

        # Sağ tarafta yapısal bloklar: Heading (mavi), Paragraf (yeşil), Tablo satırı (turuncu)
        def make_struct_block(
            tag_str: str, content_str: str, col: ManimColor, h: float = 0.65
        ) -> VGroup:
            box = RoundedRectangle(
                corner_radius=0.1,
                width=5.6,
                height=h,
                fill_color=col,
                fill_opacity=0.22,
                stroke_color=col,
                stroke_width=2.2,
            )
            tag = Text(tag_str, font_size=15, color=col, weight=BOLD)
            txt = Text(content_str, font_size=16, color=WHITE)
            if txt.width > 3.5:
                txt.scale_to_fit_width(3.5)
            tag.move_to(box.get_left() + RIGHT * 0.95)
            txt.next_to(box.get_left() + RIGHT * 1.85, RIGHT, buff=0.08)
            return VGroup(box, tag, txt)

        b_head = make_struct_block("Heading", "# RAG > ## Retrieval", BLUE_B, h=0.62)
        b_para = make_struct_block(
            "Paragraf", "5 cümle (Retrieval + Generation)", GREEN_B, h=0.82
        )
        b_tbl1 = make_struct_block(
            "Tablo satırı", "Bileşen: Chunker | Görev: Metni böler", ORANGE, h=0.58
        )
        b_tbl2 = make_struct_block(
            "Tablo satırı", "Bileşen: Embedder | Görev: Vektör yapar", ORANGE, h=0.58
        )
        b_tbl3 = make_struct_block(
            "Tablo satırı", "Bileşen: Retriever | Görev: Parça bulur", ORANGE, h=0.58
        )

        right_blocks = (
            VGroup(b_head, b_para, b_tbl1, b_tbl2, b_tbl3)
            .arrange(DOWN, buff=0.18)
            .move_to(RIGHT * 2.9 + DOWN * 0.45)
        )

        arrows = VGroup(
            Arrow(
                md_h2.get_right(),
                b_head.get_left(),
                buff=0.15,
                color=BLUE_B,
                stroke_width=2.5,
            ),
            Arrow(
                para_lines.get_right(),
                b_para.get_left(),
                buff=0.15,
                color=GREEN_B,
                stroke_width=2.5,
            ),
            Arrow(
                tbl_lines.get_right(),
                b_tbl2.get_left(),
                buff=0.15,
                color=ORANGE,
                stroke_width=2.5,
            ),
        )

        self.play(
            Create(arrows[0]),
            TransformFromCopy(VGroup(md_h1, md_h2), b_head),
            run_time=self.rt(1.4),
        )
        self.play(
            Create(arrows[1]),
            TransformFromCopy(para_lines, b_para),
            run_time=self.rt(1.4),
        )
        self.play(
            Create(arrows[2]),
            TransformFromCopy(tbl_lines, VGroup(b_tbl1, b_tbl2, b_tbl3)),
            run_time=self.rt(1.6),
        )
        self.pause(2.5)

        self.play(FadeOut(doc_card), FadeOut(arrows), run_time=self.rt(0.8))
        return {
            "all": right_blocks,
            "heading": b_head,
            "paragraph": b_para,
            "tables": VGroup(b_tbl1, b_tbl2, b_tbl3),
        }

    # =========================================================================
    # SAHNE 2: CÜMLELERE BÖLME (~14 sn)
    # =========================================================================
    def scene_2_sentences(self, blocks: dict[str, VGroup]) -> None:
        all_blocks = blocks["all"]
        b_head = blocks["heading"]
        b_para = blocks["paragraph"]
        b_tables = blocks["tables"]

        # Blokları sola taşı
        self.play(
            all_blocks.animate.move_to(LEFT * 3.4 + DOWN * 0.45), run_time=self.rt(1.0)
        )

        # Heading ve Tablo satırları bölünmez (kilit/tek parça etiketi)
        keep_tag1 = Text("Tek parça kalır", font_size=15, color=BLUE_B).next_to(
            b_head, RIGHT, buff=0.35
        )
        keep_tag2 = Text("Tek parça kalır", font_size=15, color=ORANGE).next_to(
            b_tables, RIGHT, buff=0.35
        )

        self.play(
            FadeIn(keep_tag1, shift=RIGHT * 0.15),
            FadeIn(keep_tag2, shift=RIGHT * 0.15),
            b_head[0].animate.set_fill(opacity=0.10).set_stroke(opacity=0.45),
            b_tables[0][0].animate.set_fill(opacity=0.10).set_stroke(opacity=0.45),
            b_tables[1][0].animate.set_fill(opacity=0.10).set_stroke(opacity=0.45),
            b_tables[2][0].animate.set_fill(opacity=0.10).set_stroke(opacity=0.45),
            run_time=self.rt(1.2),
        )
        self.pause(1.2)

        # Paragraf bloğu vurgulanır ve içinden S1..S5 cümle çipleri çıkar
        sent_data = [
            ("S1", "Sorgu vektöre çevrilir."),
            ("S2", "Vektör DB'de arama yapılır."),
            ("S3", "En yakın parçalar seçilir."),
            ("S4", "Bulunan bağlam prompt'a eklenir."),
            ("S5", "LLM yalnızca bağlamdan yanıt üretir."),
        ]

        chips = VGroup()
        for sid, stext in sent_data:
            chip_bg = RoundedRectangle(
                corner_radius=0.18,
                width=5.2,
                height=0.58,
                fill_color="#143326",
                fill_opacity=0.95,
                stroke_color=GREEN_B,
                stroke_width=2,
            )
            badge = RoundedRectangle(
                corner_radius=0.1,
                width=0.65,
                height=0.38,
                fill_color=GREEN_B,
                fill_opacity=0.9,
                stroke_width=0,
            ).move_to(chip_bg.get_left() + RIGHT * 0.48)
            badge_txt = Text(sid, font_size=16, color=BLACK, weight=BOLD).move_to(
                badge.get_center()
            )
            txt = Text(stext, font_size=16, color=WHITE).next_to(badge, RIGHT, buff=0.2)
            if txt.width > 3.9:
                txt.scale_to_fit_width(3.9)
            chips.add(VGroup(chip_bg, badge, badge_txt, txt))

        chips.arrange(DOWN, buff=0.15).move_to(RIGHT * 3.1 + DOWN * 0.4)

        fan_arrows = VGroup(
            *[
                Arrow(
                    b_para.get_right(),
                    chip.get_left(),
                    buff=0.12,
                    color=GREEN_B,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.18,
                )
                for chip in chips
            ]
        )

        self.play(
            Indicate(b_para, color=GREEN_A, scale_factor=1.04),
            FadeOut(keep_tag1),
            FadeOut(keep_tag2),
            run_time=self.rt(0.9),
        )
        self.play(
            LaggedStart(*[Create(a) for a in fan_arrows], lag_ratio=0.1),
            LaggedStart(*[TransformFromCopy(b_para, c) for c in chips], lag_ratio=0.12),
            run_time=self.rt(2.4),
        )
        self.pause(3.0)

        self.play(
            FadeOut(all_blocks),
            FadeOut(fan_arrows),
            FadeOut(chips),
            run_time=self.rt(0.9),
        )

    # =========================================================================
    # SAHNE 3: ANLAMSAL CHUNK'LAMA (3D KONSEPT + 2D ÇUBUKLAR & OVERLAP) (~28 sn)
    # =========================================================================
    def scene_3_semantic(self) -> None:
        # ---------------------------------------------------------------------
        # 3A: 3D Vektör Uzayında Anlamsal Mesafe ve Eşik Konsepti
        # ---------------------------------------------------------------------
        axes = ThreeDAxes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            z_range=[0, 3, 1],
            x_length=5.0,
            y_length=5.0,
            z_length=3.5,
            axis_config={"color": GREY_C, "include_tip": True},
        ).move_to(DOWN * 0.45)

        # 3 örnek cümle kartı (sağ üstte sabit HUD olarak)
        s_legend = (
            VGroup(
                VGroup(
                    Dot(color=BLUE_B, radius=0.08),
                    Text(
                        "C1: RAG, yapay zekanın veriyle doğru yanıt vermesini sağlar.",
                        font_size=14,
                        color=BLUE_B,
                    ),
                ).arrange(RIGHT, buff=0.12),
                VGroup(
                    Dot(color=TEAL_B, radius=0.08),
                    Text(
                        "C2: Yapay zeka, RAG sayesinde kaynakları tarayarak doğru cevaplar üretir.",
                        font_size=14,
                        color=TEAL_B,
                    ),
                ).arrange(RIGHT, buff=0.12),
                VGroup(
                    Dot(color=ORANGE, radius=0.08),
                    Text(
                        "C3: Yaz aylarında karpuz peynir yemek çok serinleticidir.",
                        font_size=14,
                        color=ORANGE,
                    ),
                ).arrange(RIGHT, buff=0.12),
            )
            .arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            .to_corner(UR, buff=0.45)
        )

        for row in s_legend:
            if row.width > 7.4:
                row.scale_to_fit_width(7.4)

        # Eşik referans çubuğu (sol altta sabit HUD)
        ref_line = Line(ORIGIN, RIGHT * 1.5, color=RED_B, stroke_width=4)
        ref_lbl = Text("Referans Eşik (Threshold)", font_size=15, color=RED_B)
        ref_group = (
            VGroup(ref_lbl, ref_line).arrange(RIGHT, buff=0.2).to_corner(DL, buff=0.55)
        )

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.add_fixed_in_frame_mobjects(s_legend, ref_group)

        origin_pt = axes.c2p(0, 0, 0)
        p1 = axes.c2p(1.2, 1.4, 2.3)  # C1 (RAG)
        p2 = axes.c2p(1.6, 1.9, 2.0)  # C2 (RAG - C1'e çok yakın)
        p3 = axes.c2p(3.5, 0.8, 0.5)  # C3 (Karpuz peynir - çok uzakta)

        vec1 = Arrow3D(origin_pt, p1, color=BLUE_B, thickness=0.02)
        vec2 = Arrow3D(origin_pt, p2, color=TEAL_B, thickness=0.02)
        vec3 = Arrow3D(origin_pt, p3, color=ORANGE, thickness=0.02)

        tag1 = Text("C1", font_size=20, color=BLUE_B, weight=BOLD).move_to(
            p1 + np.array([0, -0.25, 0.38])
        )
        tag2 = Text("C2", font_size=20, color=TEAL_B, weight=BOLD).move_to(
            p2 + np.array([0, 0.35, 0.35])
        )
        tag3 = Text("C3", font_size=20, color=ORANGE, weight=BOLD).move_to(
            p3 + np.array([0.45, -0.35, 0.15])
        )
        self.add_fixed_orientation_mobjects(tag1, tag2, tag3)

        self.play(
            Create(axes),
            FadeIn(s_legend),
            FadeIn(ref_group),
            Create(vec1),
            Create(vec2),
            Create(vec3),
            FadeIn(tag1),
            FadeIn(tag2),
            FadeIn(tag3),
            run_time=self.rt(2.4),
        )

        # Kamera hafifçe dönerken komşu mesafeler gösterilir
        dist_12 = Line3D(p1, p2, color=GREEN_B, thickness=0.025)
        dist_23 = Line3D(p2, p3, color=YELLOW_B, thickness=0.025)

        self.move_camera(
            phi=60 * DEGREES,
            theta=-25 * DEGREES,
            added_anims=[Create(dist_12), Create(dist_23)],
            run_time=self.rt(2.8),
        )

        # C2-C3 mesafesi eşikten büyük -> Kırmızıya döner ve "✂ Böl" çıkar
        dist_23_red = Line3D(p2, p3, color=RED_B, thickness=0.035)
        mid_23 = (p2 + p3) / 2
        split_badge = Text("✂ Böl", font_size=24, color=RED_B, weight=BOLD).move_to(
            mid_23 + np.array([0.55, 0.55, 0.25])
        )
        self.add_fixed_orientation_mobjects(split_badge)

        self.play(
            ReplacementTransform(dist_23, dist_23_red),
            FadeIn(split_badge, scale=1.3),
            run_time=self.rt(1.5),
        )
        self.move_camera(phi=62 * DEGREES, theta=-10 * DEGREES, run_time=self.rt(2.4))
        self.pause(1.5)

        # 3D sahneyi temizle ve kamerayı 2D karşı cepheye döndür
        self.play(
            FadeOut(axes),
            FadeOut(vec1),
            FadeOut(vec2),
            FadeOut(vec3),
            FadeOut(tag1),
            FadeOut(tag2),
            FadeOut(tag3),
            FadeOut(dist_12),
            FadeOut(dist_23_red),
            FadeOut(split_badge),
            FadeOut(s_legend),
            FadeOut(ref_group),
            run_time=self.rt(0.9),
        )
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        # ---------------------------------------------------------------------
        # 3B: 2D Cümle Zinciri, Mesafe Çubukları (2 Kırılma -> 3 Chunk) ve Overlap
        # ---------------------------------------------------------------------
        sample_note = Text(
            "Mesafeler: örnek değerler", font_size=15, color=GREY_B
        ).to_corner(UR, buff=0.55)

        # 6 cümle çipi (2 kırılma ile 3 chunk oluşturmak için S1..S6)
        s_labels = ["S1", "S2", "S3", "S4", "S5", "S6"]
        nodes = VGroup()
        for s_lbl in s_labels:
            b = RoundedRectangle(
                corner_radius=0.12,
                width=1.35,
                height=0.68,
                fill_color="#1E293B",
                fill_opacity=0.95,
                stroke_color=WHITE,
                stroke_width=2,
            )
            t = Text(s_lbl, font_size=24, color=WHITE, weight=BOLD).move_to(
                b.get_center()
            )
            nodes.add(VGroup(b, t))

        nodes.arrange(RIGHT, buff=0.65).move_to(UP * 0.85)

        # Komşu cümleler arası 5 mesafe çubuğu (2 tanesi eşiği aşar)
        d_vals = [0.15, 0.48, 0.18, 0.52, 0.14]  # d(S2-S3) ve d(S4-S5) eşiği aşar
        base_y = -1.25
        scale_h = 2.6

        base_line = Line(
            LEFT * 5.2 + UP * base_y,
            RIGHT * 5.2 + UP * base_y,
            color=GREY_B,
            stroke_width=2,
        )
        bars = VGroup()
        bar_txts = VGroup()
        for i, dv in enumerate(d_vals):
            mx = (nodes[i].get_center()[0] + nodes[i + 1].get_center()[0]) / 2
            bh = dv * scale_h
            bar = Rectangle(
                width=0.55,
                height=bh,
                fill_color=GREY_A,
                fill_opacity=0.85,
                stroke_color=WHITE,
                stroke_width=1.2,
            ).move_to([mx, base_y + bh / 2, 0])
            bt = Text(f"{dv:.2f}", font_size=16, color=WHITE).next_to(
                bar, UP, buff=0.08
            )
            bars.add(bar)
            bar_txts.add(bt)

        thresh_y = base_y + 0.30 * scale_h
        thresh_line = DashedLine(
            [-5.2, thresh_y, 0], [5.2, thresh_y, 0], color=RED_B, stroke_width=3
        )
        thresh_lbl = Text("Eşik", font_size=18, color=RED_B, weight=BOLD).next_to(
            thresh_line, LEFT, buff=0.18
        )

        self.play(
            FadeIn(sample_note),
            LaggedStart(*[FadeIn(n, shift=UP * 0.12) for n in nodes], lag_ratio=0.07),
            Create(base_line),
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.08),
            FadeIn(bar_txts),
            Create(thresh_line),
            FadeIn(thresh_lbl),
            run_time=self.rt(2.0),
        )
        self.pause(1.2)

        # Eşiği aşan 2 noktada (index 1 ve index 3) dikey kırmızı kesik ve 3 chunk rengi
        cut1_x = bars[1].get_center()[0]
        cut2_x = bars[3].get_center()[0]
        cut1 = DashedLine(
            [cut1_x, base_y, 0], [cut1_x, 1.35, 0], color=RED_B, stroke_width=4
        )
        cut2 = DashedLine(
            [cut2_x, base_y, 0], [cut2_x, 1.35, 0], color=RED_B, stroke_width=4
        )

        self.play(
            bars[1].animate.set_fill(RED_B, opacity=0.95),
            bars[3].animate.set_fill(RED_B, opacity=0.95),
            Create(cut1),
            Create(cut2),
            # 3 ayrı chunk tonu
            nodes[0][0]
            .animate.set_fill(BLUE_D, opacity=0.9)
            .set_stroke(BLUE_B, width=2.5),
            nodes[1][0]
            .animate.set_fill(BLUE_D, opacity=0.9)
            .set_stroke(BLUE_B, width=2.5),
            nodes[2][0]
            .animate.set_fill(TEAL_E, opacity=0.9)
            .set_stroke(TEAL_B, width=2.5),
            nodes[3][0]
            .animate.set_fill(TEAL_E, opacity=0.9)
            .set_stroke(TEAL_B, width=2.5),
            nodes[4][0]
            .animate.set_fill(PURPLE_E, opacity=0.9)
            .set_stroke(PURPLE_B, width=2.5),
            nodes[5][0]
            .animate.set_fill(PURPLE_E, opacity=0.9)
            .set_stroke(PURPLE_B, width=2.5),
            run_time=self.rt(1.8),
        )
        self.pause(1.4)

        # Çubukları kaldır, 3 chunk grubunu ve Overlap kopyalarını göster
        self.play(
            FadeOut(base_line),
            FadeOut(bars),
            FadeOut(bar_txts),
            FadeOut(thresh_line),
            FadeOut(thresh_lbl),
            FadeOut(cut1),
            FadeOut(cut2),
            run_time=self.rt(0.7),
        )

        # Grupları aşağıda 3 ayrı küme olarak hizala ve soluk overlap çiplerini başlarına ekle
        def make_ghost_chip(lbl: str, stroke_col: ManimColor) -> VGroup:
            gb = RoundedRectangle(
                corner_radius=0.12,
                width=1.25,
                height=0.68,
                fill_color=LIGHT_PINK,
                fill_opacity=0.22,
                stroke_color=stroke_col,
                stroke_width=2,
            )
            gt = Text(lbl, font_size=22, color=LIGHT_PINK, weight=BOLD).move_to(
                gb.get_center()
            )
            return VGroup(gb, gt)

        overlap_s2 = make_ghost_chip("S2*", LIGHT_PINK)
        overlap_s4 = make_ghost_chip("S4*", LIGHT_PINK)

        # Yeni pozisyonlar:
        # Chunk 1: [S1, S2] (sol)
        # Chunk 2: [S2*, S3, S4] (orta)
        # Chunk 3: [S4*, S5, S6] (sağ)
        c1_target = VGroup(nodes[0].copy(), nodes[1].copy()).arrange(RIGHT, buff=0.15)
        c2_target = VGroup(overlap_s2, nodes[2].copy(), nodes[3].copy()).arrange(
            RIGHT, buff=0.15
        )
        c3_target = VGroup(overlap_s4, nodes[4].copy(), nodes[5].copy()).arrange(
            RIGHT, buff=0.15
        )

        # Önce mevcut düğümleri yerlerine indir
        self.play(
            nodes[0].animate.move_to(c1_target[0].get_center()),
            nodes[1].animate.move_to(c1_target[1].get_center()),
            nodes[2].animate.move_to(c2_target[1].get_center()),
            nodes[3].animate.move_to(c2_target[2].get_center()),
            nodes[4].animate.move_to(c3_target[1].get_center()),
            nodes[5].animate.move_to(c3_target[2].get_center()),
            run_time=self.rt(1.2),
        )

        # Overlap: önceki grubun son cümlesi sonraki grubun başına soluk kopya olarak uçar
        ov_lbl = Text("Overlap", font_size=20, color=LIGHT_PINK, weight=BOLD).move_to(
            UP * 0.4
        )
        arc1 = CurvedArrow(
            nodes[1].get_top(),
            overlap_s2.get_top(),
            angle=-0.55,
            color=LIGHT_PINK,
            stroke_width=2.8,
        )
        arc2 = CurvedArrow(
            nodes[3].get_top(),
            overlap_s4.get_top(),
            angle=-0.55,
            color=LIGHT_PINK,
            stroke_width=2.8,
        )

        self.play(
            FadeIn(ov_lbl),
            Create(arc1),
            TransformFromCopy(nodes[1], overlap_s2),
            Create(arc2),
            TransformFromCopy(nodes[3], overlap_s4),
            run_time=self.rt(1.8),
        )

        # Her chunk etrafına çerçeve
        box1 = SurroundingRectangle(
            VGroup(nodes[0], nodes[1]), buff=0.16, corner_radius=0.14, color=BLUE_B
        )
        box2 = SurroundingRectangle(
            VGroup(overlap_s2, nodes[2], nodes[3]),
            buff=0.16,
            corner_radius=0.14,
            color=TEAL_B,
        )
        box3 = SurroundingRectangle(
            VGroup(overlap_s4, nodes[4], nodes[5]),
            buff=0.16,
            corner_radius=0.14,
            color=PURPLE_B,
        )

        lbl_c1 = Text("Chunk 1", font_size=18, color=BLUE_B, weight=BOLD).next_to(
            box1, DOWN, buff=0.15
        )
        lbl_c2 = Text("Chunk 2", font_size=18, color=TEAL_B, weight=BOLD).next_to(
            box2, DOWN, buff=0.15
        )
        lbl_c3 = Text("Chunk 3", font_size=18, color=PURPLE_B, weight=BOLD).next_to(
            box3, DOWN, buff=0.15
        )

        self.play(
            Create(box1),
            Create(box2),
            Create(box3),
            FadeIn(lbl_c1),
            FadeIn(lbl_c2),
            FadeIn(lbl_c3),
            run_time=self.rt(1.2),
        )
        self.pause(2.2)

        self.play(
            FadeOut(sample_note),
            FadeOut(ov_lbl),
            FadeOut(arc1),
            FadeOut(arc2),
            FadeOut(nodes),
            FadeOut(overlap_s2),
            FadeOut(overlap_s4),
            FadeOut(box1),
            FadeOut(box2),
            FadeOut(box3),
            FadeOut(lbl_c1),
            FadeOut(lbl_c2),
            FadeOut(lbl_c3),
            run_time=self.rt(0.8),
        )

    # =========================================================================
    # SAHNE 4: METADATA (~14 sn)
    # =========================================================================
    def scene_4_metadata(self) -> None:
        # Önce örnek bir Chunk kartına 4 metadata çipinin birer birer takılması
        focus_card_bg = RoundedRectangle(
            corner_radius=0.15,
            width=4.2,
            height=2.4,
            fill_color="#162032",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=2.5,
        ).move_to(LEFT * 2.4 + DOWN * 0.2)

        focus_title = Text("Chunk 1", font_size=26, color=BLUE_B, weight=BOLD).next_to(
            focus_card_bg.get_top(), DOWN, buff=0.3
        )
        focus_body = Text(
            "[S1, S2]\nSorgu vektöre çevrilir…", font_size=18, color=WHITE
        ).move_to(focus_card_bg.get_center() + DOWN * 0.2)
        focus_card = VGroup(focus_card_bg, focus_title, focus_body)

        self.play(FadeIn(focus_card, shift=UP * 0.2), run_time=self.rt(1.0))

        meta_specs = [
            ("document", "rag_notlari.md", TEAL_B),
            ("page", "1", YELLOW_B),
            ("paragraph", "1", GREEN_B),
            ("breadcrumb", "RAG > Retrieval", PURPLE_A),
        ]

        meta_chips = VGroup()
        for m_key, m_val, m_col in meta_specs:
            chip_box = RoundedRectangle(
                corner_radius=0.12,
                width=4.5,
                height=0.52,
                fill_color=m_col,
                fill_opacity=0.18,
                stroke_color=m_col,
                stroke_width=2,
            )
            k_txt = Text(m_key, font_size=16, color=m_col, weight=BOLD).move_to(
                chip_box.get_left() + RIGHT * 0.95
            )
            v_txt = Text(m_val, font_size=16, color=WHITE).next_to(
                chip_box.get_left() + RIGHT * 1.9, RIGHT, buff=0.1
            )
            meta_chips.add(VGroup(chip_box, k_txt, v_txt))

        meta_chips.arrange(DOWN, buff=0.15).next_to(focus_card, RIGHT, buff=0.7)

        connectors = VGroup()
        for mc in meta_chips:
            conn = Line(
                focus_card_bg.get_right(),
                mc.get_left(),
                color=GREY_B,
                stroke_width=2,
            )
            connectors.add(conn)

        # Birer birer metadata çipleri takılır
        for conn, mc in zip(connectors, meta_chips):
            self.play(
                Create(conn),
                FadeIn(mc, shift=LEFT * 0.25),
                run_time=self.rt(0.65),
            )
        self.pause(1.4)

        # Sonunda tüm chunk kartları yan yana kısa bir özet olarak görünür
        summary_specs = [
            ("Chunk 1", "[S1, S2]", "paragraph: 1", "RAG > Retrieval", BLUE_B),
            ("Chunk 2", "[S2*, S3, S4]", "paragraph: 2", "RAG > Retrieval", TEAL_B),
            ("Chunk 3", "[S4*, S5, S6]", "paragraph: 3", "RAG > Retrieval", PURPLE_B),
            ("Chunk 4", "Tablo Satırları", "paragraph: 4", "RAG > Tablo", ORANGE),
        ]

        summary_cards = VGroup()
        for c_name, c_txt, c_par, c_bc, c_col in summary_specs:
            sc_bg = RoundedRectangle(
                corner_radius=0.12,
                width=2.95,
                height=2.8,
                fill_color="#141B29",
                fill_opacity=0.95,
                stroke_color=c_col,
                stroke_width=2.2,
            )
            t1 = Text(c_name, font_size=22, color=c_col, weight=BOLD)
            t2 = Text(c_txt, font_size=16, color=WHITE)
            m1 = Text("doc: rag_notlari.md", font_size=13, color=GREY_A)
            m2 = Text(f"page: 1 · {c_par}", font_size=13, color=YELLOW_B)
            m3 = Text(c_bc, font_size=13, color=GREEN_B)
            stack = (
                VGroup(t1, t2, m1, m2, m3)
                .arrange(DOWN, buff=0.14)
                .move_to(sc_bg.get_center())
            )
            summary_cards.add(VGroup(sc_bg, stack))

        summary_cards.arrange(RIGHT, buff=0.3).move_to(DOWN * 0.35)

        self.play(
            ReplacementTransform(
                VGroup(focus_card, connectors, meta_chips), summary_cards
            ),
            run_time=self.rt(1.5),
        )
        self.pause(3.0)
