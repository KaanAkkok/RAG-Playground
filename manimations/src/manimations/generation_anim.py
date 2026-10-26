from __future__ import annotations

import shutil
from typing import Any

import numpy as np
from manim import *

# Hız sabiti: Tüm animasyon ve bekleme süreleri bu çarpanla ayarlanır.
SPEED: float = 0.85


class GenerationScene(ThreeDScene):
    """
    RAG Pipeline 4. Aşama: LLM Üretimi & Temperature Dinamikleri.
    - Sayfa 1: RAG Bağlamı Toplama ve LLM'e Giriş
    - Sayfa 2: LLM Mimarisi İçi (2.5D İstiflenmiş Sinir Ağı Katmanları & Ağırlık Parlaması)
    - Sayfa 3: Temperature Ölçekleme & Olasılık Dağılımı (T=0.2 vs T=1.0)
    """

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data: dict[str, Any] = data or {}

    def rt(self, seconds: float) -> float:
        return max(0.12, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.12, seconds / SPEED))

    @staticmethod
    def _make_sub_token(
        base_char: str, sub_char: str, font_size: int = 18, color: Any = WHITE
    ) -> VGroup:
        """
        Özel Unicode alt-indis karakterleri (ᵢ, ⱼ) bazı fontlarda beyaz kutucuk (tofu)
        olarak göründüğü için standart ASCII harflerle temiz alt-indis oluşturur.
        """
        base = Text(base_char, font_size=font_size, color=color, weight=BOLD)
        sub = Text(
            sub_char, font_size=max(10, int(font_size * 0.65)), color=color, weight=BOLD
        )
        sub.next_to(base, DR, buff=0.02).shift(UP * 0.06)
        return VGroup(base, sub)

    @classmethod
    def _build_header_with_symbol(
        cls,
        label_str: str,
        latex_sym: str,
        base_char: str,
        sub_char: str,
        font_size: int = 15,
        color: Any = YELLOW_B,
    ) -> VGroup:
        """Başlık metni yanına LaTeX varsa MathTex, yoksa ASCII alt-indis ekler."""
        txt = Text(label_str, font_size=font_size, color=color, weight=BOLD)
        if shutil.which("latex"):
            try:
                sym = MathTex(latex_sym, font_size=font_size + 7, color=color)
                return VGroup(txt, sym).arrange(RIGHT, buff=0.12)
            except Exception:
                logger.warning(
                    "Error occurred while rendering MathLaTeX. Returning fallback text instead."
                )
        lp = Text("(", font_size=font_size, color=color, weight=BOLD)
        sub_tok = cls._make_sub_token(
            base_char, sub_char, font_size=font_size, color=color
        )
        rp = Text(")", font_size=font_size, color=color, weight=BOLD)
        sym_fallback = VGroup(lp, sub_tok, rp).arrange(RIGHT, buff=0.03)
        return VGroup(txt, sym_fallback).arrange(RIGHT, buff=0.10)

    @classmethod
    def _build_softmax_formula(cls) -> tuple[VGroup, VGroup]:
        """
        LaTeX kuruluysa (örn. turkcell_rag Docker ortamı) gerçek MathTex ile,
        değilse beyaz kutucuk üretmeyen saf ASCII alt-indisli vektörel grupla
        T parametresi sarı (YELLOW) vurgulu Softmax formülünü oluşturur:
        P_i = exp(z_i / T) / sum_j exp(z_j / T)
        """
        if shutil.which("latex"):
            try:
                tex = MathTex(
                    r"P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}",
                    substrings_to_isolate=["T"],
                    font_size=33,
                    color=WHITE,
                )
                tex.set_color_by_tex("T", YELLOW)
                t_parts = tex.get_parts_by_tex("T")
                return VGroup(tex), t_parts
            except Exception:
                logger.warning(
                    "Error occurred while rendering MathLaTeX. Returning fallback text instead."
                )

        # Beyaz kutucuk (tofu) oluşturmayan saf ASCII fallback formül
        pi_tok = cls._make_sub_token("P", "i", font_size=21, color=WHITE)
        eq_tok = Text("=", font_size=21, color=WHITE, weight=BOLD)
        lhs = VGroup(pi_tok, eq_tok).arrange(RIGHT, buff=0.14)

        # Pay: exp(z_i / T)
        num_exp = Text("exp(", font_size=18, color=WHITE, weight=BOLD)
        num_zi = cls._make_sub_token("z", "i", font_size=18, color=WHITE)
        num_div = Text("/", font_size=18, color=WHITE, weight=BOLD)
        num_t = Text("T", font_size=21, color=YELLOW, weight=BOLD)
        num_close = Text(")", font_size=18, color=WHITE, weight=BOLD)
        num_grp = VGroup(num_exp, num_zi, num_div, num_t, num_close).arrange(
            RIGHT, buff=0.06
        )

        frac_line = Line(LEFT * 1.45, RIGHT * 1.45, stroke_width=2.4, color=GREY_A)

        # Payda: sum_j exp(z_j / T)
        den_sum = cls._make_sub_token("Σ", "j", font_size=18, color=WHITE)
        den_exp = Text("exp(", font_size=17, color=WHITE, weight=BOLD)
        den_zj = cls._make_sub_token("z", "j", font_size=17, color=WHITE)
        den_div = Text("/", font_size=17, color=WHITE, weight=BOLD)
        den_t = Text("T", font_size=20, color=YELLOW, weight=BOLD)
        den_close = Text(")", font_size=17, color=WHITE, weight=BOLD)
        den_grp = VGroup(den_sum, den_exp, den_zj, den_div, den_t, den_close).arrange(
            RIGHT, buff=0.05
        )

        frac_grp = VGroup(num_grp, frac_line, den_grp).arrange(DOWN, buff=0.10)
        formula = VGroup(lhs, frac_grp).arrange(RIGHT, buff=0.18)
        t_targets = VGroup(num_t, den_t)
        return formula, t_targets

    @staticmethod
    def _create_25d_nn_slab(
        center: np.ndarray,
        col: ManimColor,
        fill_hex: str,
        title_str: str,
        sub_str: str,
    ) -> dict[str, Any]:
        """
        2.5D izometrik perspektifte tek bir sinir ağı katman dikdörtgeni (prizma plakası)
        ve içindeki nöron + ağırlık ağını oluşturur.
        """
        wx = 0.56  # Perspektif yatay yarı-genişlik
        hy = 1.36  # Dikey yarı-yükseklik
        sy = 0.36  # İzometrik eğim (yukarı-sağa)
        dx_3d = -0.13  # 3B kalınlık (sol yan yüzey)
        dy_3d = -0.09  # 3B kalınlık (alt yüzey)

        # Ön yüzey köşe noktaları (Sol-Alt, Sağ-Alt, Sağ-Üst, Sol-Üst)
        p_bl = center + np.array([-wx, -hy - sy, 0.0])
        p_br = center + np.array([wx, -hy + sy, 0.0])
        p_tr = center + np.array([wx, hy + sy, 0.0])
        p_tl = center + np.array([-wx, hy - sy, 0.0])

        # Arka/Yan kalınlık köşe noktaları (2.5D hacim hissi)
        shift_3d = np.array([dx_3d, dy_3d, 0.0])
        p_bl_b = p_bl + shift_3d
        p_br_b = p_br + shift_3d
        p_tl_b = p_tl + shift_3d

        # Sol yan kalınlık yüzeyi
        side_face = Polygon(
            p_tl_b,
            p_tl,
            p_bl,
            p_bl_b,
            fill_color=fill_hex,
            fill_opacity=0.95,
            stroke_color=col,
            stroke_width=1.4,
            stroke_opacity=0.65,
        )
        # Alt kalınlık yüzeyi
        bottom_face = Polygon(
            p_bl_b,
            p_bl,
            p_br,
            p_br_b,
            fill_color=fill_hex,
            fill_opacity=0.95,
            stroke_color=col,
            stroke_width=1.4,
            stroke_opacity=0.65,
        )
        # Ana 2.5D dikdörtgen plaka yüzeyi
        front_face = Polygon(
            p_bl,
            p_br,
            p_tr,
            p_tl,
            fill_color=fill_hex,
            fill_opacity=0.88,
            stroke_color=col,
            stroke_width=2.4,
        )

        # Katman içi 3 sütunlu sinir ağı düğümleri (3 -> 4 -> 3)
        def slab_pt(u: float, v: float) -> np.ndarray:
            return center + np.array([u * wx, v * hy + u * sy, 0.0])

        u_cols = [-0.62, 0.0, 0.62]
        v_rows_3 = [0.58, 0.0, -0.58]
        v_rows_4 = [0.68, 0.23, -0.23, -0.68]

        col1_dots = [
            Dot(slab_pt(u_cols[0], v), radius=0.055, color=GREY_B) for v in v_rows_3
        ]
        col2_dots = [
            Dot(slab_pt(u_cols[1], v), radius=0.055, color=GREY_B) for v in v_rows_4
        ]
        col3_dots = [
            Dot(slab_pt(u_cols[2], v), radius=0.055, color=GREY_B) for v in v_rows_3
        ]

        # Katman içi ağırlık bağlantıları (Weights)
        weights_12: list[tuple[int, int, Line]] = []
        weights_23: list[tuple[int, int, Line]] = []
        weights_grp = VGroup()

        for i, d1 in enumerate(col1_dots):
            for j, d2 in enumerate(col2_dots):
                ln = Line(
                    d1.get_center(),
                    d2.get_center(),
                    stroke_color=GREY_C,
                    stroke_width=1.1,
                    stroke_opacity=0.28,
                )
                weights_12.append((i, j, ln))
                weights_grp.add(ln)

        for j, d2 in enumerate(col2_dots):
            for k, d3 in enumerate(col3_dots):
                ln = Line(
                    d2.get_center(),
                    d3.get_center(),
                    stroke_color=GREY_C,
                    stroke_width=1.1,
                    stroke_opacity=0.28,
                )
                weights_23.append((j, k, ln))
                weights_grp.add(ln)

        nodes_grp = VGroup(*col1_dots, *col2_dots, *col3_dots)

        # Üst etiket (Düz 2B okunaklı başlık)
        top_anchor = (p_tl + p_tr) / 2.0 + UP * 0.34
        lbl_title = Text(title_str, font_size=13, color=col, weight=BOLD)
        lbl_sub = Text(sub_str, font_size=11, color=GREY_B)
        lbl_grp = (
            VGroup(lbl_title, lbl_sub).arrange(DOWN, buff=0.05).move_to(top_anchor)
        )

        full_slab = VGroup(
            side_face,
            bottom_face,
            front_face,
            weights_grp,
            nodes_grp,
            lbl_grp,
        )

        return {
            "group": full_slab,
            "front_face": front_face,
            "col1": col1_dots,
            "col2": col2_dots,
            "col3": col3_dots,
            "weights_12": weights_12,
            "weights_23": weights_23,
            "label": lbl_grp,
            "color": col,
        }

    def construct(self) -> None:
        self.camera.background_color = BLACK

        # =====================================================================
        # SABİT BAŞLIK (Prompt kuralı - Aynen korundu)
        # =====================================================================
        title = Text("4. Generation", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text("LLM Generation", font_size=18, color=RED_B)
        title_group = (
            VGroup(title, subtitle)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .to_corner(UL, buff=0.5)
        )
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(FadeIn(title_group, shift=DOWN * 0.2), run_time=1.2)
        self.wait(0.3)

        # 3 Sayfayı sırayla çalıştır
        self.stage_1_rag_context_collection()
        logit_column = self.stage_2_llm_transformer_architecture()
        self.stage_3_temperature_and_distribution(logit_column)

    # =========================================================================
    # SAYFA 1: RAG Bağlamı Toplama ve LLM'e Giriş
    # =========================================================================
    def stage_1_rag_context_collection(self) -> None:
        # Sol Taraf: Dikey sıralı 4 Girdi Kartı
        card_specs = [
            ("Sistem", '"Belgelere göre yanıtla"', TEAL_B, "#082828"),
            ("Sorgu", '"RAG nedir?"', BLUE_B, "#0C2238"),
            ("Hafıza", '"Önceki sohbet"', PURPLE_B, "#231238"),
            ("Parçalar", '"[Metin-1, Metin-2]"', YELLOW_B, "#2E2208"),
        ]

        cards_group = VGroup()
        port_dots = VGroup()

        for tag_str, val_str, col, fill_hex in card_specs:
            card_bg = RoundedRectangle(
                corner_radius=0.12,
                width=4.05,
                height=0.78,
                fill_color="#111722",
                fill_opacity=0.95,
                stroke_color=col,
                stroke_width=2.2,
            )
            badge_bg = RoundedRectangle(
                corner_radius=0.08,
                width=1.18,
                height=0.48,
                fill_color=fill_hex,
                fill_opacity=0.95,
                stroke_color=col,
                stroke_width=1.4,
            )
            badge_txt = Text(tag_str, font_size=14, color=col, weight=BOLD)
            badge_txt.move_to(badge_bg.get_center())
            badge = VGroup(badge_bg, badge_txt).move_to(
                card_bg.get_left() + RIGHT * 0.74
            )

            val_txt = Text(val_str, font_size=14, color=WHITE, weight=BOLD)
            if val_txt.width > 2.45:
                val_txt.scale_to_fit_width(2.45)
            val_txt.next_to(badge, RIGHT, buff=0.18)

            port = Dot(card_bg.get_right(), radius=0.065, color=col)
            card_item = VGroup(card_bg, badge, val_txt, port)
            cards_group.add(card_item)
            port_dots.add(port)

        cards_group.arrange(DOWN, buff=0.22).move_to(LEFT * 4.35 + DOWN * 0.42)

        # Sağ Taraf: Büyük, temiz hatlı, parlayan mavi kenarlıklı "LLM" Kutusu
        llm_w, llm_h = 3.25, 3.85
        llm_glow_outer = RoundedRectangle(
            corner_radius=0.26,
            width=llm_w + 0.28,
            height=llm_h + 0.28,
            fill_opacity=0.0,
            stroke_color=BLUE_A,
            stroke_width=2.0,
            stroke_opacity=0.28,
        )
        llm_glow_inner = RoundedRectangle(
            corner_radius=0.24,
            width=llm_w + 0.12,
            height=llm_h + 0.12,
            fill_opacity=0.0,
            stroke_color=BLUE_B,
            stroke_width=3.2,
            stroke_opacity=0.45,
        )
        llm_box = RoundedRectangle(
            corner_radius=0.22,
            width=llm_w,
            height=llm_h,
            fill_color="#0D1828",
            fill_opacity=0.96,
            stroke_color=BLUE_B,
            stroke_width=3.2,
        )
        llm_label = Text("LLM", font_size=44, color=WHITE, weight=BOLD).move_to(
            llm_box.get_center()
        )
        llm_group = VGroup(llm_glow_outer, llm_glow_inner, llm_box, llm_label).move_to(
            RIGHT * 4.15 + DOWN * 0.42
        )

        # 1. Girdi kartları soldan içeri kayarak belirir
        self.play(
            LaggedStart(
                *[FadeIn(card, shift=RIGHT * 0.45) for card in cards_group],
                lag_ratio=0.14,
            ),
            FadeIn(llm_group, shift=LEFT * 0.35),
            run_time=self.rt(1.3),
        )
        self.pause(0.35)

        # 2. Kartlardan çıkan akış çizgileri ve parçacıklar merkezde birleşir
        merge_pt = np.array([-0.65, -0.42, 0.0])
        merge_hub = Dot(merge_pt, radius=0.11, color=BLUE_A)
        merge_hub_glow = Circle(
            radius=0.22,
            stroke_color=BLUE_B,
            stroke_width=2.0,
            fill_color=BLUE_E,
            fill_opacity=0.4,
        ).move_to(merge_pt)

        flow_curves = VGroup()
        moving_packets = VGroup()
        for card, (_, _, col, _) in zip(cards_group, card_specs):
            start_pt = card[0].get_right()
            ctrl1 = start_pt + RIGHT * 0.95
            ctrl2 = merge_pt + LEFT * 0.85
            curve = CubicBezier(
                start_pt,
                ctrl1,
                ctrl2,
                merge_pt,
                stroke_color=col,
                stroke_width=2.4,
                stroke_opacity=0.75,
            )
            flow_curves.add(curve)
            pkt = Dot(start_pt, radius=0.085, color=col)
            moving_packets.add(pkt)

        self.play(
            LaggedStart(*[Create(c) for c in flow_curves], lag_ratio=0.08),
            FadeIn(merge_hub_glow, scale=0.6),
            FadeIn(merge_hub, scale=0.6),
            run_time=self.rt(0.85),
        )
        self.play(
            *[
                MoveAlongPath(pkt, curve, rate_func=smooth)
                for pkt, curve in zip(moving_packets, flow_curves)
            ],
            run_time=self.rt(0.9),
        )

        # Merkezde birleşen akış tek bir kalın bilgi akış okuna dönüşür
        context_arrow_glow = Line(
            merge_pt,
            llm_box.get_left() + LEFT * 0.08,
            stroke_color=BLUE_A,
            stroke_width=14.0,
            stroke_opacity=0.32,
        )
        context_arrow = Arrow(
            merge_pt,
            llm_box.get_left(),
            buff=0.06,
            color=BLUE_B,
            stroke_width=7.5,
            max_tip_length_to_length_ratio=0.18,
        )
        context_tag = Text("bağlam", font_size=15, color=BLUE_A, weight=BOLD).next_to(
            context_arrow, UP, buff=0.14
        )
        unified_flow = VGroup(context_arrow_glow, context_arrow, context_tag)

        self.play(
            ReplacementTransform(moving_packets, unified_flow),
            Flash(merge_pt, color=BLUE_A, flash_radius=0.45, num_lines=10),
            run_time=self.rt(0.8),
        )

        # 3. Birleşik bağlam oku doğrudan LLM kutusuna girer; kutu büyür/titrer ve veriyi çeker
        stream_pulse = context_arrow.copy().set_stroke(color=WHITE, width=9.0)
        self.play(
            ShowPassingFlash(stream_pulse, time_width=0.6),
            Indicate(llm_group, color=BLUE_A, scale_factor=1.07),
            Flash(llm_box.get_center(), color=BLUE_A, flash_radius=1.45, num_lines=14),
            run_time=self.rt(1.0),
        )
        self.pause(0.35)

        # 4 & 5. Girdi kartları solar, LLM kutusu merkeze odaklanarak iç mimari için açılır
        expanded_frame = RoundedRectangle(
            corner_radius=0.24,
            width=12.2,
            height=5.2,
            fill_color="#070B14",
            fill_opacity=0.0,
            stroke_color=BLUE_B,
            stroke_width=1.8,
            stroke_opacity=0.0,
        ).move_to(DOWN * 0.35)

        self.play(
            FadeOut(cards_group, shift=LEFT * 0.4),
            FadeOut(flow_curves),
            FadeOut(merge_hub_glow),
            FadeOut(merge_hub),
            FadeOut(unified_flow, shift=RIGHT * 0.4),
            ReplacementTransform(llm_group, expanded_frame),
            run_time=self.rt(1.0),
        )
        self.remove(expanded_frame)
        self.pause(0.2)

    # =========================================================================
    # SAYFA 2: LLM Mimarisi İçi (2.5D İstiflenmiş Sinir Ağı Katmanları)
    # =========================================================================
    def stage_2_llm_transformer_architecture(self) -> VGroup:
        # 1. Sol Taraf: Token Akışı ["vektör", "arama", "bağlam"]
        token_specs = [
            ("vektör", TEAL_B, "#0D2628"),
            ("arama", BLUE_B, "#0E2238"),
            ("bağlam", PURPLE_B, "#221436"),
        ]
        token_boxes = VGroup()
        token_ports = []

        for tok_str, col, fill_hex in token_specs:
            t_bg = RoundedRectangle(
                corner_radius=0.1,
                width=1.65,
                height=0.58,
                fill_color=fill_hex,
                fill_opacity=0.95,
                stroke_color=col,
                stroke_width=2.2,
            )
            t_lbl = Text(tok_str, font_size=15, color=WHITE, weight=BOLD).move_to(
                t_bg.get_center()
            )
            t_port = Dot(t_bg.get_right(), radius=0.055, color=col)
            token_boxes.add(VGroup(t_bg, t_lbl, t_port))
            token_ports.append(t_port)

        token_boxes.arrange(DOWN, buff=0.38).move_to(LEFT * 5.35 + DOWN * 0.42)
        tok_hdr = Text("Token Akışı", font_size=14, color=BLUE_A, weight=BOLD).next_to(
            token_boxes, UP, buff=0.26
        )

        # 2. Orta Alan: 2.5D İstiflenmiş 4 Sinir Ağı Dikdörtgen Katmanı
        slab_configs = [
            (
                np.array([-2.65, -0.58, 0.0]),
                TEAL_B,
                "#0A1B24",
                "Attention",
                "Katman 1",
            ),
            (
                np.array([-1.08, -0.38, 0.0]),
                BLUE_B,
                "#0B182B",
                "Gizli Ağ 1",
                "Katman 2",
            ),
            (
                np.array([0.48, -0.18, 0.0]),
                PURPLE_B,
                "#15122B",
                "Gizli Ağ 2",
                "Katman 3",
            ),
            (
                np.array([2.05, 0.02, 0.0]),
                GOLD_B,
                "#241A0B",
                "FFN Bloğu",
                "Katman N",
            ),
        ]

        slabs_data: list[dict[str, Any]] = []
        slabs_vgroup = VGroup()
        for center_pt, col, f_hex, t_str, s_str in slab_configs:
            s_info = self._create_25d_nn_slab(center_pt, col, f_hex, t_str, s_str)
            slabs_data.append(s_info)
            slabs_vgroup.add(s_info["group"])

        # Katmanlar arası (Inter-layer) sinaptik bağlantı çizgileri
        inter_layer_groups: list[VGroup] = []
        for idx in range(len(slabs_data) - 1):
            src_dots = slabs_data[idx]["col3"]
            dst_dots = slabs_data[idx + 1]["col1"]
            bridge_grp = VGroup()
            for d_src in src_dots:
                for d_dst in dst_dots:
                    b_ln = Line(
                        d_src.get_center(),
                        d_dst.get_center(),
                        stroke_color=GREY_C,
                        stroke_width=1.1,
                        stroke_opacity=0.22,
                    )
                    bridge_grp.add(b_ln)
            inter_layer_groups.append(bridge_grp)

        all_inter_bridges = VGroup(*inter_layer_groups)

        # Mimariyi sahneye getir
        self.play(
            LaggedStart(
                *[
                    FadeIn(s["group"], shift=UP * 0.2 + RIGHT * 0.15)
                    for s in slabs_data
                ],
                lag_ratio=0.14,
            ),
            FadeIn(all_inter_bridges),
            run_time=self.rt(1.3),
        )

        # 1. ["vektör", "arama", "bağlam"] kelimeleri mimariye soldan adım adım girer
        self.play(
            FadeIn(tok_hdr, shift=DOWN * 0.15),
            LaggedStart(
                *[FadeIn(tb, shift=RIGHT * 0.35) for tb in token_boxes],
                lag_ratio=0.22,
            ),
            run_time=self.rt(1.2),
        )
        self.pause(0.25)

        # 2. Multi-Head Attention: Token kutucukları arasında çapraz parlayan bağlantı çizgileri
        first_slab_inputs = slabs_data[0]["col1"]
        attn_mesh = VGroup()
        attn_highlights = VGroup()
        attn_colors = [TEAL_B, BLUE_B, PURPLE_A, YELLOW_B]

        for i, tb in enumerate(token_boxes):
            p_start = tb[2].get_center()
            for j, d_in in enumerate(first_slab_inputs):
                p_end = d_in.get_center()
                c_ij = attn_colors[(i + j) % len(attn_colors)]
                ln = Line(
                    p_start,
                    p_end,
                    stroke_color=c_ij,
                    stroke_width=1.8,
                    stroke_opacity=0.45,
                )
                attn_mesh.add(ln)
                if (i, j) in [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)]:
                    glow = Line(
                        p_start,
                        p_end,
                        stroke_color=YELLOW_A if (i + j) % 2 == 0 else TEAL_A,
                        stroke_width=5.5,
                        stroke_opacity=0.38,
                    )
                    core = Line(
                        p_start,
                        p_end,
                        stroke_color=YELLOW_B if (i + j) % 2 == 0 else TEAL_B,
                        stroke_width=2.8,
                        stroke_opacity=0.95,
                    )
                    attn_highlights.add(VGroup(glow, core))

        self.play(
            LaggedStart(*[Create(l) for l in attn_mesh], lag_ratio=0.04),
            run_time=self.rt(0.8),
        )
        self.play(
            LaggedStart(*[FadeIn(h) for h in attn_highlights], lag_ratio=0.06),
            *[tb[0].animate.set_stroke(width=3.0) for tb in token_boxes],
            run_time=self.rt(0.75),
        )

        # 3. Her bir 2.5D dikdörtgen katman içinde belirli ağırlıkların ve nöronların parlaması
        active_specs = [
            ([0, 2], [0, 2, 3], [0, 1], TEAL_A),
            ([0, 1], [1, 2, 3], [1, 2], BLUE_A),
            ([1, 2], [0, 1, 3], [0, 2], PURPLE_A),
            ([0, 2], [0, 1, 2, 3], [0, 1, 2], YELLOW_B),
        ]

        active_overlays = VGroup()

        for idx, (act_c1, act_c2, act_c3, glow_col) in enumerate(active_specs):
            s_info = slabs_data[idx]
            c1_dots = s_info["col1"]
            c2_dots = s_info["col2"]
            c3_dots = s_info["col3"]

            layer_glow_lines = VGroup()
            pulse_lines = VGroup()

            for i, j, ln in s_info["weights_12"]:
                if i in act_c1 and j in act_c2:
                    g_ln = Line(
                        ln.get_start(),
                        ln.get_end(),
                        stroke_color=glow_col,
                        stroke_width=5.5,
                        stroke_opacity=0.35,
                    )
                    c_ln = Line(
                        ln.get_start(),
                        ln.get_end(),
                        stroke_color=glow_col,
                        stroke_width=2.5,
                        stroke_opacity=0.95,
                    )
                    layer_glow_lines.add(g_ln, c_ln)
                    pulse_lines.add(
                        Line(
                            ln.get_start(),
                            ln.get_end(),
                            stroke_color=WHITE,
                            stroke_width=3.6,
                        )
                    )

            for j, k, ln in s_info["weights_23"]:
                if j in act_c2 and k in act_c3:
                    g_ln = Line(
                        ln.get_start(),
                        ln.get_end(),
                        stroke_color=glow_col,
                        stroke_width=5.5,
                        stroke_opacity=0.35,
                    )
                    c_ln = Line(
                        ln.get_start(),
                        ln.get_end(),
                        stroke_color=glow_col,
                        stroke_width=2.5,
                        stroke_opacity=0.95,
                    )
                    layer_glow_lines.add(g_ln, c_ln)
                    pulse_lines.add(
                        Line(
                            ln.get_start(),
                            ln.get_end(),
                            stroke_color=WHITE,
                            stroke_width=3.6,
                        )
                    )

            active_overlays.add(layer_glow_lines)

            node_anims = (
                [c1_dots[i].animate.set_color(glow_col).scale(1.35) for i in act_c1]
                + [c2_dots[j].animate.set_color(glow_col).scale(1.35) for j in act_c2]
                + [c3_dots[k].animate.set_color(glow_col).scale(1.35) for k in act_c3]
            )

            self.play(
                s_info["front_face"].animate.set_stroke(color=glow_col, width=3.2),
                FadeIn(layer_glow_lines),
                *node_anims,
                LaggedStart(
                    *[ShowPassingFlash(p, time_width=0.55) for p in pulse_lines],
                    lag_ratio=0.03,
                ),
                run_time=self.rt(0.68),
            )

            if idx < len(slabs_data) - 1:
                bridge_pulses = VGroup()
                for k_out in act_c3:
                    for next_in in active_specs[idx + 1][0]:
                        p1 = c3_dots[k_out].get_center()
                        p2 = slabs_data[idx + 1]["col1"][next_in].get_center()
                        b_core = Line(
                            p1,
                            p2,
                            stroke_color=glow_col,
                            stroke_width=2.2,
                            stroke_opacity=0.85,
                        )
                        active_overlays.add(b_core)
                        bridge_pulses.add(
                            Line(p1, p2, stroke_color=WHITE, stroke_width=3.5)
                        )
                self.play(
                    LaggedStart(
                        *[ShowPassingFlash(bp, time_width=0.5) for bp in bridge_pulses],
                        lag_ratio=0.04,
                    ),
                    run_time=self.rt(0.38),
                )

        # 4. Sağ Taraf: Dikey Logit Sütunu (z_i)
        logit_specs = [
            ("retrieval", "+4.2", GREEN_B, "#0D281E"),
            ("üretim", "+1.8", BLUE_B, "#0E2238"),
            ("hafıza", "+0.6", PURPLE_B, "#221436"),
            ("halüsinasyon", "-1.2", RED_B, "#2D1216"),
        ]

        logit_rows = VGroup()
        for word_str, z_str, col, f_hex in logit_specs:
            cell_bg = RoundedRectangle(
                corner_radius=0.09,
                width=2.55,
                height=0.58,
                fill_color=f_hex,
                fill_opacity=0.95,
                stroke_color=col,
                stroke_width=2.0,
            )
            w_txt = Text(word_str, font_size=13, color=WHITE, weight=BOLD)
            if w_txt.width > 1.35:
                w_txt.scale_to_fit_width(1.35)
            w_txt.move_to(cell_bg.get_left() + RIGHT * 0.82)

            z_badge_bg = RoundedRectangle(
                corner_radius=0.06,
                width=0.72,
                height=0.38,
                fill_color="#090D14",
                fill_opacity=0.9,
                stroke_color=col,
                stroke_width=1.2,
            ).move_to(cell_bg.get_right() + LEFT * 0.48)
            z_txt = Text(z_str, font_size=13, color=col, weight=BOLD).move_to(
                z_badge_bg.get_center()
            )
            logit_rows.add(VGroup(cell_bg, w_txt, z_badge_bg, z_txt))

        logit_rows.arrange(DOWN, buff=0.14)
        logit_hdr = self._build_header_with_symbol(
            "Logit Sütunu", r"(z_i)", "z", "i", font_size=15, color=YELLOW_B
        )
        logit_sub = Text("Ham Skorlar", font_size=11, color=GREY_B)
        logit_top = VGroup(logit_hdr, logit_sub).arrange(DOWN, buff=0.06)

        logit_column = (
            VGroup(logit_top, logit_rows)
            .arrange(DOWN, buff=0.18)
            .move_to(RIGHT * 4.95 + DOWN * 0.15)
        )

        # FFN çıkış düğümlerinden Logit hücrelerine sinyal akışı
        ffn_out_dots = slabs_data[-1]["col3"]
        to_logit_lines = VGroup()
        for d_out in ffn_out_dots:
            for l_row in logit_rows:
                to_logit_lines.add(
                    Line(
                        d_out.get_center(),
                        l_row[0].get_left(),
                        stroke_color=GOLD_B,
                        stroke_width=1.8,
                        stroke_opacity=0.65,
                    )
                )

        self.play(
            LaggedStart(*[Create(l) for l in to_logit_lines], lag_ratio=0.04),
            FadeIn(logit_column, shift=RIGHT * 0.35),
            run_time=self.rt(1.0),
        )
        self.pause(0.5)

        # 5. Mimari blokları solar, Logit sütunu ekranın sol tarafına kayar
        self.play(
            FadeOut(tok_hdr),
            FadeOut(token_boxes, shift=LEFT * 0.3),
            FadeOut(attn_mesh),
            FadeOut(attn_highlights),
            FadeOut(slabs_vgroup),
            FadeOut(all_inter_bridges),
            FadeOut(active_overlays),
            FadeOut(to_logit_lines),
            logit_column.animate.move_to(LEFT * 5.05 + DOWN * 0.25),
            run_time=self.rt(1.1),
        )
        self.pause(0.2)
        return logit_column

    # =========================================================================
    # SAYFA 3: Temperature Ölçekleme & Olasılık Dağılımı (Çekirdek Mekanizma)
    # =========================================================================
    def stage_3_temperature_and_distribution(self, logit_column: VGroup) -> None:
        # 1. Sol / Orta Alan: Softmax Formülü ve Sarı T Vurgusu
        formula_mob, t_targets = self._build_softmax_formula()
        if formula_mob.width > 3.45:
            formula_mob.scale_to_fit_width(3.45)

        formula_card_bg = RoundedRectangle(
            corner_radius=0.14,
            width=3.85,
            height=1.82,
            fill_color="#101826",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=2.0,
        )
        formula_title = Text(
            "Softmax & Temperature", font_size=13, color=BLUE_A, weight=BOLD
        ).next_to(formula_card_bg.get_top(), DOWN, buff=0.15)
        formula_mob.next_to(formula_title, DOWN, buff=0.16)
        formula_card = VGroup(formula_card_bg, formula_title, formula_mob).move_to(
            LEFT * 1.35 + UP * 1.12
        )

        arr_logit_to_soft = Arrow(
            logit_column.get_right() + UP * 0.4,
            formula_card_bg.get_left(),
            buff=0.12,
            color=YELLOW_B,
            stroke_width=2.8,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(
            GrowArrow(arr_logit_to_soft),
            FadeIn(formula_card, shift=DOWN * 0.2),
            run_time=self.rt(0.95),
        )

        # T parametresi sarı renkle parlayarak dikkat çeker
        self.play(
            *[
                Indicate(t_part, color=YELLOW, scale_factor=1.45)
                for t_part in t_targets
            ],
            *[
                Flash(t_part.get_center(), color=YELLOW, flash_radius=0.35, num_lines=8)
                for t_part in t_targets
            ],
            run_time=self.rt(0.85),
        )

        # Hemen altında Dinamik T Göstergesi ve Mini Rozet Kutu
        t_panel_bg = RoundedRectangle(
            corner_radius=0.14,
            width=3.85,
            height=1.75,
            fill_color="#0D1F1D",
            fill_opacity=0.95,
            stroke_color=GREEN_B,
            stroke_width=2.2,
        ).move_to(LEFT * 1.35 + DOWN * 0.92)

        t_val_low = Text("T = 0.2", font_size=28, color=YELLOW, weight=BOLD)
        badge_low_bg = RoundedRectangle(
            corner_radius=0.08,
            width=3.45,
            height=0.46,
            fill_color="#09352A",
            fill_opacity=0.95,
            stroke_color=GREEN_B,
            stroke_width=1.5,
        )
        badge_low_txt = Text(
            "Düşük T: Tutarlı & Bilgiye Dayalı",
            font_size=12,
            color=GREEN_A,
            weight=BOLD,
        )
        if badge_low_txt.width > 3.2:
            badge_low_txt.scale_to_fit_width(3.2)
        badge_low_txt.move_to(badge_low_bg.get_center())
        badge_low = VGroup(badge_low_bg, badge_low_txt)

        t_content_low = (
            VGroup(t_val_low, badge_low)
            .arrange(DOWN, buff=0.20)
            .move_to(t_panel_bg.get_center())
        )

        # 2. Sağ Taraf: Olasılık Dağılımı Çubuk Grafiği (4 Dikey Çubuk)
        chart_title = self._build_header_with_symbol(
            "Olasılık Dağılımı", r"(P_i)", "P", "i", font_size=16, color=WHITE
        ).move_to(RIGHT * 3.85 + UP * 2.25)

        base_y = -1.42
        max_bar_h = 2.85  # %100 karşılığı
        x_coords = [1.95, 3.22, 4.49, 5.76]
        words = ["retrieval", "üretim", "hafıza", "halüsinasyon"]
        bar_cols = [GREEN_B, BLUE_B, PURPLE_B, RED_B]

        # Yatay kılavuz çizgileri ve eksen
        grid_lines = VGroup()
        for pct_tick in [0.25, 0.50, 0.75, 1.0]:
            gy = base_y + pct_tick * max_bar_h
            gl = DashedLine(
                np.array([1.25, gy, 0.0]),
                np.array([6.42, gy, 0.0]),
                dash_length=0.08,
                stroke_color=GREY_D,
                stroke_width=1.0,
                stroke_opacity=0.5,
            )
            grid_lines.add(gl)

        axis_line = Line(
            np.array([1.20, base_y, 0.0]),
            np.array([6.45, base_y, 0.0]),
            stroke_color=GREY_A,
            stroke_width=2.4,
        )

        word_labels = VGroup()
        for x, w_str, col in zip(x_coords, words, bar_cols):
            lbl = Text(w_str, font_size=13, color=WHITE, weight=BOLD)
            if lbl.width > 1.12:
                lbl.scale_to_fit_width(1.12)
            lbl.move_to(np.array([x, base_y - 0.30, 0.0]))
            word_labels.add(lbl)

        arr_soft_to_chart = Arrow(
            formula_card_bg.get_right(),
            np.array([1.22, 1.12, 0.0]),
            buff=0.10,
            color=BLUE_B,
            stroke_width=2.6,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(
            FadeIn(t_panel_bg),
            FadeIn(t_content_low, shift=UP * 0.15),
            GrowArrow(arr_soft_to_chart),
            FadeIn(chart_title),
            FadeIn(grid_lines),
            Create(axis_line),
            FadeIn(word_labels, shift=UP * 0.1),
            run_time=self.rt(1.0),
        )

        # DURUM 1: Düşük Temperature (T = 0.2 - RAG / Tutarlı Mod)
        probs_low = [0.92, 0.05, 0.02, 0.01]
        pct_txts_low = ["%92", "%5", "%2", "%1"]

        bars_curr = VGroup()
        pcts_curr = VGroup()

        for x, p, p_str, col in zip(x_coords, probs_low, pct_txts_low, bar_cols):
            bh = max(0.08, p * max_bar_h)
            bar = RoundedRectangle(
                corner_radius=0.06,
                width=0.76,
                height=bh,
                fill_color=col,
                fill_opacity=0.92,
                stroke_color=WHITE if p > 0.5 else col,
                stroke_width=2.0 if p > 0.5 else 1.4,
            ).move_to(np.array([x, base_y + bh / 2.0, 0.0]))
            p_lbl = Text(p_str, font_size=14, color=col, weight=BOLD).next_to(
                bar, UP, buff=0.10
            )
            bars_curr.add(bar)
            pcts_curr.add(p_lbl)

        # "retrieval" aniden %92'ye fırlar
        self.play(
            GrowFromEdge(bars_curr[0], DOWN),
            FadeIn(pcts_curr[0], shift=UP * 0.2),
            *[GrowFromEdge(bars_curr[i], DOWN) for i in range(1, 4)],
            *[FadeIn(pcts_curr[i]) for i in range(1, 4)],
            run_time=self.rt(0.95),
        )

        # Yeşille parlayan "retrieval" kelimesi kesin seçim olarak ayrılır
        sel_glow = SurroundingRectangle(
            word_labels[0],
            color=GREEN_A,
            fill_color=GREEN_E,
            fill_opacity=0.35,
            buff=0.09,
            corner_radius=0.07,
            stroke_width=2.4,
        )
        sel_tag = Text("Kesin Seçim", font_size=12, color=GREEN_A, weight=BOLD).next_to(
            pcts_curr[0], UP, buff=0.12
        )

        self.play(
            Create(sel_glow),
            FadeIn(sel_tag, shift=UP * 0.1),
            Indicate(bars_curr[0], color=GREEN_A, scale_factor=1.05),
            word_labels[0].animate.set_color(GREEN_A),
            run_time=self.rt(0.85),
        )
        self.pause(0.75)

        # DURUM 2: Yüksek Temperature Geçişi (T = 1.0 - Yaratıcı Mod)
        t_val_high = Text("T = 1.0", font_size=28, color=YELLOW, weight=BOLD)
        badge_high_bg = RoundedRectangle(
            corner_radius=0.08,
            width=3.55,
            height=0.64,
            fill_color="#2D1215",
            fill_opacity=0.95,
            stroke_color=RED_B,
            stroke_width=1.6,
        )
        badge_high_l1 = Text(
            "Yüksek T: Yaratıcı & Çeşitli",
            font_size=12,
            color=ORANGE,
            weight=BOLD,
        )
        badge_high_l2 = Text(
            "(Risk: Halüsinasyon)",
            font_size=11,
            color=RED_A,
            weight=BOLD,
        )
        badge_high_txt = (
            VGroup(badge_high_l1, badge_high_l2)
            .arrange(DOWN, buff=0.05)
            .move_to(badge_high_bg.get_center())
        )
        badge_high = VGroup(badge_high_bg, badge_high_txt)

        t_content_high = (
            VGroup(t_val_high, badge_high)
            .arrange(DOWN, buff=0.14)
            .move_to(t_panel_bg.get_center())
        )

        # Çubuklar dengelenir: retrieval %35, üretim %25, hafıza %20, halüsinasyon %20
        probs_high = [0.35, 0.25, 0.20, 0.20]
        pct_txts_high = ["%35", "%25", "%20", "%20"]

        bars_target = VGroup()
        pcts_target = VGroup()

        for x, p, p_str, col in zip(x_coords, probs_high, pct_txts_high, bar_cols):
            bh = max(0.08, p * max_bar_h)
            bar = RoundedRectangle(
                corner_radius=0.06,
                width=0.76,
                height=bh,
                fill_color=col,
                fill_opacity=0.90,
                stroke_color=col,
                stroke_width=1.8,
            ).move_to(np.array([x, base_y + bh / 2.0, 0.0]))
            p_lbl = Text(p_str, font_size=14, color=col, weight=BOLD).next_to(
                bar, UP, buff=0.10
            )
            bars_target.add(bar)
            pcts_target.add(p_lbl)

        self.play(
            FadeOut(sel_glow),
            FadeOut(sel_tag),
            word_labels[0].animate.set_color(WHITE),
            Transform(t_content_low, t_content_high),
            t_panel_bg.animate.set_fill("#241216", opacity=0.95).set_stroke(
                ORANGE, width=2.2
            ),
            Transform(bars_curr, bars_target),
            Transform(pcts_curr, pcts_target),
            run_time=self.rt(1.25),
        )

        # Rastgele seçim mekanizması (örnekleme/sampling) hafif parlamayla gösterilir
        sample_ring = SurroundingRectangle(
            bars_curr[1],
            color=YELLOW,
            buff=0.06,
            corner_radius=0.08,
            stroke_width=2.8,
        )
        self.play(FadeIn(sample_ring), run_time=self.rt(0.25))
        for idx_hop in [2, 3, 0, 1]:
            self.play(
                sample_ring.animate.move_to(
                    bars_curr[idx_hop].get_center()
                ).stretch_to_fit_height(bars_curr[idx_hop].height + 0.12),
                Indicate(bars_curr[idx_hop], color=YELLOW_A, scale_factor=1.06),
                run_time=self.rt(0.32),
            )
        self.play(
            Indicate(bars_curr[3], color=RED_A, scale_factor=1.08),
            word_labels[3].animate.set_color(RED_A),
            FadeOut(sample_ring),
            run_time=self.rt(0.55),
        )
        self.pause(0.4)

        # 4. Çıktı Cümlesi: Seçilen kelimeler birleşerek sağ altta cevap kutusunu tamamlar
        ans_bg = RoundedRectangle(
            corner_radius=0.14,
            width=6.6,
            height=0.76,
            fill_color="#0D2228",
            fill_opacity=0.96,
            stroke_color=TEAL_B,
            stroke_width=2.4,
        ).move_to(RIGHT * 2.55 + DOWN * 2.85)

        ans_prefix = Text("Cevap:", font_size=15, color=TEAL_A, weight=BOLD)
        ans_dots = Text('"...', font_size=15, color=GREY_A, weight=BOLD)
        ans_w1 = Text("retrieval", font_size=16, color=GREEN_A, weight=BOLD)
        ans_mid = Text("temelli", font_size=15, color=WHITE, weight=BOLD)
        ans_w2 = Text('üretim"', font_size=16, color=BLUE_A, weight=BOLD)

        self.play(
            FadeIn(ans_bg, shift=UP * 0.2),
            FadeIn(ans_prefix),
            FadeIn(ans_dots),
            run_time=self.rt(0.55),
        )
        self.play(
            TransformFromCopy(word_labels[0], ans_w1),
            FadeIn(ans_mid, shift=RIGHT * 0.1),
            TransformFromCopy(word_labels[1], ans_w2),
            Indicate(ans_bg, color=TEAL_A, scale_factor=1.03),
            run_time=self.rt(1.0),
        )
        self.pause(1.1)

        # 5. Sahne sade bir kararma (FadeOut) ile tamamlanır
        all_stage3 = VGroup(
            logit_column,
            arr_logit_to_soft,
            formula_card,
            t_panel_bg,
            t_content_low,
            arr_soft_to_chart,
            chart_title,
            grid_lines,
            axis_line,
            word_labels,
            bars_curr,
            pcts_curr,
            ans_bg,
            ans_prefix,
            ans_dots,
            ans_w1,
            ans_mid,
            ans_w2,
        )
        self.play(FadeOut(all_stage3, shift=DOWN * 0.15), run_time=self.rt(1.1))
        self.pause(0.4)
