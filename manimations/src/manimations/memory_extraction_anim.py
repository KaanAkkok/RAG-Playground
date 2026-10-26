from __future__ import annotations

from typing import Any

import numpy as np
from manim import *

from manimations.base import RAGScene

# Hız sabiti: Tüm animasyon ve bekleme süreleri bu çarpanla ayarlanır.
SPEED: float = 0.88


class MemoryExtractionScene(RAGScene):
    """
    RAG Pipeline 5. Aşama: Bellek Mimarisi & Semantik Bellek Çıkarımı (Memory Extraction).
    - Sayfa 1: Konuşma Geçmişi & Kısa Vadeli Bellek (Short-Term Memory / Sliding Window)
    - Sayfa 2: Semantik Bellek Çıkarımı (Extraction via LLM)
    - Sayfa 3: Uzun Vadeli Bellek (Long-Term Memory Depolama & İki Belleğin Entegrasyonu)
    """

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(data=data, **kwargs)

    def add_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        """2D Scene / RAGScene üzerinde sabit başlık çağrısıyla tam uyumluluk sağlar."""
        super_fn = getattr(super(), "add_fixed_in_frame_mobjects", None)
        if callable(super_fn):
            super_fn(*mobjects)

    def rt(self, seconds: float) -> float:
        return max(0.12, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.12, seconds / SPEED))

    @staticmethod
    def _make_chat_bubble(
        label_str: str,
        is_user: bool,
        width: float = 3.55,
        height: float = 0.54,
    ) -> VGroup:
        """
        Kullanıcı (Mavi) veya Model (Gri/Teal) için sembolik satır çizgili
        kompakt sohbet balonu oluşturur.
        """
        stroke_col = BLUE_C if is_user else TEAL_C
        fill_hex = "#1E3A8A" if is_user else "#134E4A"
        tag_col = BLUE_A if is_user else TEAL_A

        bubble_bg = RoundedRectangle(
            corner_radius=0.16,
            width=width,
            height=height,
            fill_color=fill_hex,
            fill_opacity=0.94,
            stroke_color=stroke_col,
            stroke_width=2.0,
        )

        lbl = Text(label_str, font_size=13, color=WHITE, weight=BOLD)
        lbl.move_to(bubble_bg.get_left() + RIGHT * 0.72)

        # Sembolik mesaj satır çizgileri (uzun metin yerine geometrik temsil)
        line1 = Line(
            ORIGIN,
            RIGHT * 1.45,
            stroke_color=tag_col,
            stroke_width=3.0,
            stroke_opacity=0.85,
        )
        line2 = Line(
            ORIGIN,
            RIGHT * 0.95,
            stroke_color=GREY_B,
            stroke_width=2.4,
            stroke_opacity=0.65,
        )
        sym_lines = (
            VGroup(line1, line2)
            .arrange(DOWN, buff=0.10, aligned_edge=LEFT)
            .next_to(lbl, RIGHT, buff=0.28)
        )

        role_dot = Dot(
            bubble_bg.get_left() + RIGHT * 0.22,
            radius=0.055,
            color=tag_col,
        )
        return VGroup(bubble_bg, role_dot, lbl, sym_lines)

    def construct(self) -> None:
        self.camera.background_color = BLACK

        # =====================================================================
        # SABİT BAŞLIK KODU (Prompt kuralı - Aynen korundu)
        # =====================================================================
        title = Text("5. Memory", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text("Memory Extraction", font_size=18, color=RED_B)
        title_group = (
            VGroup(title, subtitle)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .to_corner(UL, buff=0.5)
        )
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(FadeIn(title_group, shift=DOWN * 0.2), run_time=1.2)
        self.wait(0.3)

        # 3 Sayfayı sırayla çalıştır
        chat_history_block, bubbles_list = self.stage_1_short_term_memory()
        fact_pills, chat_compact, llm_extractor = self.stage_2_semantic_extraction(
            chat_history_block, bubbles_list
        )
        self.stage_3_long_term_memory_and_integration(
            fact_pills, chat_compact, llm_extractor
        )

    # =========================================================================
    # SAYFA 1: Konuşma Geçmişi & Kısa Vadeli Bellek (Short-Term Memory)
    # =========================================================================
    def stage_1_short_term_memory(self) -> tuple[VGroup, list[VGroup]]:
        # Sol Taraf: Beyaz RoundedRectangle arka plan üzerinde konuşma geçmişi
        panel_w, panel_h = 4.65, 5.15
        chat_panel_bg = RoundedRectangle(
            corner_radius=0.18,
            width=panel_w,
            height=panel_h,
            fill_color="#F1F5F9",
            fill_opacity=0.96,
            stroke_color=GREY_B,
            stroke_width=2.5,
        ).move_to(LEFT * 2.65 + DOWN * 0.58)

        panel_hdr = Text(
            "Konuşma Geçmişi", font_size=15, color="#0F172A", weight=BOLD
        ).next_to(chat_panel_bg.get_top(), DOWN, buff=0.20)

        panel_divider = Line(
            chat_panel_bg.get_left() + RIGHT * 0.25 + UP * 2.02,
            chat_panel_bg.get_right() + LEFT * 0.25 + UP * 2.02,
            stroke_color="#CBD5E1",
            stroke_width=1.8,
        )

        # 6 adet dönüşümlü sohbet balonu (Kullanıcı = Mavi, Model = Gri/Teal)
        msg_specs = [
            ("Soru 1", True),
            ("Cevap 1", False),
            ("Soru 2", True),
            ("Cevap 2", False),
            ("Soru 3", True),
            ("Cevap 3", False),
        ]

        bubbles_list: list[VGroup] = []
        bubbles_vgroup = VGroup()

        for idx, (m_lbl, is_user) in enumerate(msg_specs):
            b_item = self._make_chat_bubble(m_lbl, is_user)
            bubbles_list.append(b_item)
            bubbles_vgroup.add(b_item)

        bubbles_vgroup.arrange(DOWN, buff=0.16).move_to(
            chat_panel_bg.get_center() + DOWN * 0.22
        )

        # Kullanıcı mesajlarını hafif sola, Model mesajlarını hafif sağa yasla (sohbet akışı hissi)
        for idx, b_item in enumerate(bubbles_list):
            shift_dir = LEFT * 0.28 if idx % 2 == 0 else RIGHT * 0.28
            b_item.shift(shift_dir)

        # 1. Beyaz arka plan ve sohbet baloncukları yukarıdan aşağıya sırayla dökülür
        self.play(
            FadeIn(chat_panel_bg, scale=0.96),
            FadeIn(panel_hdr),
            Create(panel_divider),
            run_time=self.rt(0.85),
        )
        self.play(
            LaggedStart(
                *[FadeIn(b, shift=DOWN * 0.28) for b in bubbles_list],
                lag_ratio=0.16,
            ),
            run_time=self.rt(1.5),
        )
        self.pause(0.3)

        # 2. Kayan Çerçeve (Sliding Window - Son 3-4 mesajı saran parlak kesikli kapsül)
        recent_bubbles = VGroup(*bubbles_list[2:])  # Son 4 mesaj (Soru 2..Cevap 3)
        raw_window = SurroundingRectangle(
            recent_bubbles,
            color=YELLOW_B,
            buff=0.13,
            corner_radius=0.16,
            stroke_width=3.4,
        )
        sliding_window = DashedVMobject(raw_window, num_dashes=38, dashed_ratio=0.62)

        # Sağ tarafta Short-Term Memory açıklama kartı ve rozeti
        stm_tag_bg = RoundedRectangle(
            corner_radius=0.12,
            width=4.55,
            height=0.82,
            fill_color="#1E1B0C",
            fill_opacity=0.95,
            stroke_color=YELLOW_B,
            stroke_width=2.2,
        ).move_to(RIGHT * 3.15 + DOWN * 0.55)

        stm_tag_title = Text(
            "Short-Term Memory (Son K Mesaj)",
            font_size=15,
            color=YELLOW_B,
            weight=BOLD,
        )
        stm_tag_sub = Text(
            "Kayan Pencere (Sliding Window)",
            font_size=12,
            color=GREY_A,
        )
        stm_tag_txt = (
            VGroup(stm_tag_title, stm_tag_sub)
            .arrange(DOWN, buff=0.08)
            .move_to(stm_tag_bg.get_center())
        )
        stm_card = VGroup(stm_tag_bg, stm_tag_txt)

        win_connector = Arrow(
            raw_window.get_right(),
            stm_tag_bg.get_left(),
            buff=0.12,
            color=YELLOW_B,
            stroke_width=2.6,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(
            Create(sliding_window),
            GrowArrow(win_connector),
            FadeIn(stm_card, shift=RIGHT * 0.25),
            run_time=self.rt(1.1),
        )

        # 3. Çerçevenin içindeki mesajlar parlar; altına küçük bir rozet gelir
        ctx_badge_bg = RoundedRectangle(
            corner_radius=0.10,
            width=4.55,
            height=0.62,
            fill_color="#0B2624",
            fill_opacity=0.95,
            stroke_color=TEAL_B,
            stroke_width=2.0,
        ).next_to(stm_card, DOWN, buff=0.25)

        ctx_badge_txt = Text(
            "Doğrudan Prompt Bağlamına Eklenir",
            font_size=13,
            color=TEAL_A,
            weight=BOLD,
        ).move_to(ctx_badge_bg.get_center())
        ctx_badge = VGroup(ctx_badge_bg, ctx_badge_txt)

        self.play(
            *[Indicate(b, color=YELLOW_A, scale_factor=1.04) for b in bubbles_list[2:]],
            FadeIn(ctx_badge, shift=UP * 0.15),
            run_time=self.rt(0.95),
        )
        self.pause(0.35)

        # 4. Çerçeve dışındaki eski mesajlar hafifçe solar (opaklık 0.3'e düşer)
        old_note = Text(
            "Eski Mesajlar: Bağlam Dışı Riski",
            font_size=13,
            color=RED_B,
            weight=BOLD,
        ).move_to(RIGHT * 3.15 + UP * 1.15)
        old_arrow = Arrow(
            old_note.get_left(),
            bubbles_list[0].get_right() + RIGHT * 0.15,
            buff=0.12,
            color=RED_B,
            stroke_width=2.2,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(
            bubbles_list[0].animate.set_opacity(0.3),
            bubbles_list[1].animate.set_opacity(0.3),
            FadeIn(old_note, shift=LEFT * 0.2),
            GrowArrow(old_arrow),
            run_time=self.rt(1.0),
        )
        self.pause(0.85)

        # Sayfa 2'ye geçiş için yardımcı etiketleri temizle
        self.play(
            FadeOut(sliding_window),
            FadeOut(win_connector),
            FadeOut(stm_card),
            FadeOut(ctx_badge),
            FadeOut(old_note),
            FadeOut(old_arrow),
            run_time=self.rt(0.75),
        )

        chat_history_block = VGroup(
            chat_panel_bg, panel_hdr, panel_divider, bubbles_vgroup
        )
        return chat_history_block, bubbles_list

    # =========================================================================
    # SAYFA 2: Semantik Bellek Çıkarımı (Extraction via LLM)
    # =========================================================================
    def stage_2_semantic_extraction(
        self,
        chat_history_block: VGroup,
        bubbles_list: list[VGroup],
    ) -> tuple[VGroup, VGroup, VGroup]:
        # Sol Taraf: Tüm konuşma geçmişi kompakt tek bir blok olarak sola yerleşir ("Tam Konuşma")
        full_hdr_badge = RoundedRectangle(
            corner_radius=0.08,
            width=2.95,
            height=0.44,
            fill_color="#172554",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=1.8,
        )
        full_hdr_txt = Text(
            "Tam Konuşma", font_size=14, color=BLUE_A, weight=BOLD
        ).move_to(full_hdr_badge.get_center())
        full_hdr = VGroup(full_hdr_badge, full_hdr_txt)

        # Eski mesajlar dahil tüm konuşmayı tekrar netleştirip sol bloğa küçült
        self.play(
            bubbles_list[0].animate.set_opacity(0.85),
            bubbles_list[1].animate.set_opacity(0.85),
            chat_history_block.animate.scale(0.74).move_to(LEFT * 4.85 + DOWN * 0.55),
            run_time=self.rt(0.95),
        )
        full_hdr.next_to(chat_history_block, UP, buff=0.14)
        self.play(FadeIn(full_hdr, shift=DOWN * 0.1), run_time=self.rt(0.45))

        # Merkez: "LLM (Extractor)" Kutusu (Mor/Neon kenarlık) + İç Sinyal Noktaları
        ext_w, ext_h = 2.95, 2.35
        ext_glow = RoundedRectangle(
            corner_radius=0.22,
            width=ext_w + 0.18,
            height=ext_h + 0.18,
            fill_opacity=0.0,
            stroke_color=PURPLE_A,
            stroke_width=2.4,
            stroke_opacity=0.42,
        )
        ext_box = RoundedRectangle(
            corner_radius=0.18,
            width=ext_w,
            height=ext_h,
            fill_color="#1A102B",
            fill_opacity=0.96,
            stroke_color=PURPLE_B,
            stroke_width=2.8,
        )
        ext_title = Text("LLM (Extractor)", font_size=17, color=PURPLE_A, weight=BOLD)
        ext_title.next_to(ext_box.get_top(), DOWN, buff=0.22)

        # LLM içi reaksiyon düğümleri (Dot dalgası)
        inner_dots = VGroup()
        inner_lines = VGroup()
        dot_coords = [
            np.array([-0.75, 0.05, 0.0]),
            np.array([-0.75, -0.45, 0.0]),
            np.array([0.0, 0.20, 0.0]),
            np.array([0.0, -0.20, 0.0]),
            np.array([0.0, -0.60, 0.0]),
            np.array([0.75, -0.05, 0.0]),
            np.array([0.75, -0.45, 0.0]),
        ]
        for pt in dot_coords:
            inner_dots.add(Dot(pt, radius=0.065, color=PURPLE_B))

        edges = [
            (0, 2),
            (0, 3),
            (1, 3),
            (1, 4),
            (2, 5),
            (3, 5),
            (3, 6),
            (4, 6),
        ]
        for i_src, i_dst in edges:
            inner_lines.add(
                Line(
                    dot_coords[i_src],
                    dot_coords[i_dst],
                    stroke_color=PURPLE_C,
                    stroke_width=1.6,
                    stroke_opacity=0.45,
                )
            )

        inner_net = VGroup(inner_lines, inner_dots).move_to(
            ext_box.get_center() + DOWN * 0.22
        )

        # LLM kutusunun altında küçük etiket
        ext_sub_badge = RoundedRectangle(
            corner_radius=0.08,
            width=3.65,
            height=0.46,
            fill_color="#231438",
            fill_opacity=0.95,
            stroke_color=PURPLE_B,
            stroke_width=1.5,
        ).next_to(ext_box, DOWN, buff=0.22)
        ext_sub_txt = Text(
            "Semantik Özetleme & Varlık Çıkarımı",
            font_size=12,
            color=PURPLE_A,
            weight=BOLD,
        )
        if ext_sub_txt.width > 3.4:
            ext_sub_txt.scale_to_fit_width(3.4)
        ext_sub_txt.move_to(ext_sub_badge.get_center())
        ext_sub = VGroup(ext_sub_badge, ext_sub_txt)

        llm_extractor = VGroup(
            ext_glow, ext_box, ext_title, inner_net, ext_sub
        ).move_to(LEFT * 0.55 + DOWN * 0.45)

        extract_arrow = Arrow(
            chat_history_block.get_right(),
            ext_box.get_left(),
            buff=0.14,
            color=PURPLE_B,
            stroke_width=3.2,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(
            GrowArrow(extract_arrow),
            FadeIn(llm_extractor, scale=0.94),
            run_time=self.rt(1.0),
        )

        # 1 & 2. Konuşma geçmişinin tamamından mor renkli veri parçacıkları (Dot stream) çıkar ve LLM'e akar
        stream_paths = VGroup()
        stream_dots = VGroup()
        target_in = ext_box.get_left() + RIGHT * 0.15

        for b_item in bubbles_list:
            start_p = b_item.get_right()
            c1 = start_p + RIGHT * 0.75
            c2 = target_in + LEFT * 0.75
            path_curve = CubicBezier(
                start_p,
                c1,
                c2,
                target_in,
                stroke_color=PURPLE_A,
                stroke_width=1.8,
                stroke_opacity=0.45,
            )
            stream_paths.add(path_curve)
            stream_dots.add(Dot(start_p, radius=0.075, color=PURPLE_A))

        self.play(
            LaggedStart(*[Create(p) for p in stream_paths], lag_ratio=0.06),
            FadeIn(stream_dots),
            run_time=self.rt(0.65),
        )
        self.play(
            *[
                MoveAlongPath(d, p, rate_func=smooth)
                for d, p in zip(stream_dots, stream_paths)
            ],
            run_time=self.rt(0.95),
        )

        # 3. LLM kutusu sinyali alır, içindeki sinyal noktaları parlar ve kutu nabız gibi atar
        self.play(
            FadeOut(stream_dots),
            FadeOut(stream_paths),
            Indicate(ext_box, color=PURPLE_A, scale_factor=1.06),
            *[d.animate.set_color(YELLOW_B).scale(1.35) for d in inner_dots],
            *[
                l.animate.set_stroke(color=YELLOW_A, width=2.8, opacity=0.95)
                for l in inner_lines
            ],
            Flash(
                ext_box.get_center(), color=PURPLE_A, flash_radius=1.25, num_lines=12
            ),
            run_time=self.rt(0.95),
        )

        # 4. Ham konuşma metinleri damıtılarak 3 adet konsantre bilgi hapına dönüşür
        pill_specs = [
            ("Kullanıcı Tercihi: Python", TEAL_B, "#0B2928"),
            ("Proje: RAG Mimarisi", BLUE_B, "#0D233A"),
            ("Hedef: Düşük Gecikme", YELLOW_B, "#2B210A"),
        ]

        fact_pills = VGroup()
        for p_text, p_col, p_fill in pill_specs:
            p_bg = RoundedRectangle(
                corner_radius=0.22,
                width=3.85,
                height=0.72,
                fill_color=p_fill,
                fill_opacity=0.96,
                stroke_color=p_col,
                stroke_width=2.4,
            )
            p_dot = Dot(p_bg.get_left() + RIGHT * 0.32, radius=0.07, color=p_col)
            p_lbl = Text(p_text, font_size=14, color=WHITE, weight=BOLD)
            if p_lbl.width > 3.1:
                p_lbl.scale_to_fit_width(3.1)
            p_lbl.next_to(p_dot, RIGHT, buff=0.18)
            fact_pills.add(VGroup(p_bg, p_dot, p_lbl))

        fact_pills.arrange(DOWN, buff=0.26).move_to(RIGHT * 4.15 + DOWN * 0.35)
        pills_hdr = Text(
            "Semantik Çıkarım (Bilgi Hapları)",
            font_size=14,
            color=YELLOW_B,
            weight=BOLD,
        ).next_to(fact_pills, UP, buff=0.24)

        out_arrows = VGroup()
        for pill in fact_pills:
            out_arrows.add(
                Arrow(
                    ext_box.get_right(),
                    pill[0].get_left(),
                    buff=0.12,
                    color=pill[0].get_stroke_color(),
                    stroke_width=2.4,
                    max_tip_length_to_length_ratio=0.16,
                )
            )

        self.play(
            FadeIn(pills_hdr, shift=DOWN * 0.1),
            LaggedStart(*[GrowArrow(a) for a in out_arrows], lag_ratio=0.12),
            LaggedStart(
                *[TransformFromCopy(inner_net, pill) for pill in fact_pills],
                lag_ratio=0.16,
            ),
            run_time=self.rt(1.3),
        )
        self.pause(0.85)

        chat_compact = VGroup(full_hdr, chat_history_block, extract_arrow)
        llm_group_all = VGroup(llm_extractor, out_arrows, pills_hdr)
        return fact_pills, chat_compact, llm_group_all

    # =========================================================================
    # SAYFA 3: Uzun Vadeli Bellek (Long-Term Memory Depolama & Entegrasyon)
    # =========================================================================
    def stage_3_long_term_memory_and_integration(
        self,
        fact_pills: VGroup,
        chat_compact: VGroup,
        llm_group_all: VGroup,
    ) -> None:
        # 1. Sol bloğu ve LLM Extractor'ı kaldır, bilgi haplarını merkezin soluna taşı
        self.play(
            FadeOut(chat_compact, shift=LEFT * 0.35),
            FadeOut(llm_group_all),
            fact_pills.animate.move_to(LEFT * 2.95 + DOWN * 0.35),
            run_time=self.rt(0.95),
        )

        # Sağ Taraf: Uzun Vadeli Bellek / Vektör Veritabanı (Izgara / Katmanlı Blok)
        db_w, db_h = 4.65, 3.65
        db_glow = RoundedRectangle(
            corner_radius=0.22,
            width=db_w + 0.18,
            height=db_h + 0.18,
            fill_opacity=0.0,
            stroke_color=GREEN_A,
            stroke_width=2.2,
            stroke_opacity=0.35,
        )
        db_frame = RoundedRectangle(
            corner_radius=0.18,
            width=db_w,
            height=db_h,
            fill_color="#091E1A",
            fill_opacity=0.96,
            stroke_color=GREEN_B,
            stroke_width=2.8,
        )
        db_title = Text("Long-Term Memory", font_size=18, color=GREEN_A, weight=BOLD)
        db_sub = Text("Kalıcı / Vektörel Depolama", font_size=12, color=GREY_A)
        db_header = (
            VGroup(db_title, db_sub)
            .arrange(DOWN, buff=0.06)
            .next_to(db_frame.get_top(), DOWN, buff=0.20)
        )

        # Veritabanı içi 3 satırlı vektör ızgarası (Grid Slots)
        db_slots = VGroup()
        slot_colors = [TEAL_B, BLUE_B, YELLOW_B]
        vec_previews = [
            "[0.81, -0.24, ...]",
            "[0.64,  0.52, ...]",
            "[-0.19, 0.88, ...]",
        ]

        for s_col, v_str in zip(slot_colors, vec_previews):
            slot_bg = RoundedRectangle(
                corner_radius=0.10,
                width=4.15,
                height=0.68,
                fill_color="#0F2924",
                fill_opacity=0.9,
                stroke_color=GREEN_C,
                stroke_width=1.6,
            )
            vec_tag = Text(v_str, font_size=11, color=GREEN_B, weight=BOLD).move_to(
                slot_bg.get_right() + LEFT * 0.82
            )
            db_slots.add(VGroup(slot_bg, vec_tag))

        db_slots.arrange(DOWN, buff=0.16).move_to(db_frame.get_center() + DOWN * 0.34)
        ltm_db_group = VGroup(db_glow, db_frame, db_header, db_slots).move_to(
            RIGHT * 3.15 + DOWN * 0.35
        )

        store_arrows = VGroup()
        for i in range(3):
            store_arrows.add(
                Arrow(
                    fact_pills[i].get_right(),
                    db_slots[i][0].get_left(),
                    buff=0.12,
                    color=GREEN_B,
                    stroke_width=2.4,
                    max_tip_length_to_length_ratio=0.16,
                )
            )

        self.play(
            FadeIn(ltm_db_group, shift=LEFT * 0.3),
            LaggedStart(*[GrowArrow(a) for a in store_arrows], lag_ratio=0.12),
            run_time=self.rt(1.0),
        )

        # 2. Bilgi hapları Long-Term Memory ızgarasının içine yerleşir ve ızgara parlar
        stored_items = VGroup()
        for i, (pill, s_col) in enumerate(zip(fact_pills, slot_colors)):
            target_slot = db_slots[i][0]
            lbl_copy = pill[2].copy().scale(0.88)
            lbl_copy.move_to(target_slot.get_left() + RIGHT * 1.38)
            dot_copy = Dot(
                target_slot.get_left() + RIGHT * 0.22, radius=0.06, color=s_col
            )
            stored_items.add(VGroup(dot_copy, lbl_copy))

        self.play(
            *[ReplacementTransform(fact_pills[i], stored_items[i]) for i in range(3)],
            FadeOut(store_arrows),
            *[
                db_slots[i][0].animate.set_stroke(color=GREEN_A, width=2.4)
                for i in range(3)
            ],
            Indicate(db_frame, color=GREEN_A, scale_factor=1.04),
            Flash(db_frame.get_center(), color=GREEN_A, flash_radius=1.5, num_lines=14),
            run_time=self.rt(1.25),
        )
        self.pause(0.6)

        # 3 & 4. İki Belleğin Entegrasyonu:
        # Üstte Short-Term kutusu ("Geçici & Anlık"), Altta Long-Term kutusu ("Kalıcı & Semantik")
        full_ltm = VGroup(ltm_db_group, stored_items)
        self.play(FadeOut(full_ltm, scale=0.92), run_time=self.rt(0.65))

        # Üst Hat: Short-Term Memory
        stm_box_bg = RoundedRectangle(
            corner_radius=0.16,
            width=4.15,
            height=1.55,
            fill_color="#1F1A0B",
            fill_opacity=0.96,
            stroke_color=YELLOW_B,
            stroke_width=2.6,
        )
        stm_hdr = Text("Short-Term Memory", font_size=17, color=YELLOW_B, weight=BOLD)
        stm_badge_bg = RoundedRectangle(
            corner_radius=0.08,
            width=2.45,
            height=0.42,
            fill_color="#33290E",
            fill_opacity=0.95,
            stroke_color=YELLOW_A,
            stroke_width=1.4,
        )
        stm_badge_txt = Text(
            "Geçici & Anlık", font_size=13, color=YELLOW_A, weight=BOLD
        ).move_to(stm_badge_bg.get_center())
        stm_badge = VGroup(stm_badge_bg, stm_badge_txt)
        stm_sub = Text("Son K Mesaj (Kayan Pencere)", font_size=11, color=GREY_A)
        stm_inner = (
            VGroup(stm_hdr, stm_badge, stm_sub)
            .arrange(DOWN, buff=0.12)
            .move_to(stm_box_bg.get_center())
        )
        stm_summary_box = VGroup(stm_box_bg, stm_inner).move_to(LEFT * 3.55 + UP * 0.72)

        # Alt Hat: Long-Term Memory
        ltm_box_bg = RoundedRectangle(
            corner_radius=0.16,
            width=4.15,
            height=1.55,
            fill_color="#09221C",
            fill_opacity=0.96,
            stroke_color=GREEN_B,
            stroke_width=2.6,
        )
        ltm_hdr = Text("Long-Term Memory", font_size=17, color=GREEN_A, weight=BOLD)
        ltm_badge_bg = RoundedRectangle(
            corner_radius=0.08,
            width=2.65,
            height=0.42,
            fill_color="#0F382E",
            fill_opacity=0.95,
            stroke_color=GREEN_A,
            stroke_width=1.4,
        )
        ltm_badge_txt = Text(
            "Kalıcı & Semantik", font_size=13, color=GREEN_A, weight=BOLD
        ).move_to(ltm_badge_bg.get_center())
        ltm_badge = VGroup(ltm_badge_bg, ltm_badge_txt)
        ltm_sub = Text(
            "Vektör DB (Damıtılmış Bilgi Hapları)", font_size=11, color=GREY_A
        )
        ltm_inner = (
            VGroup(ltm_hdr, ltm_badge, ltm_sub)
            .arrange(DOWN, buff=0.12)
            .move_to(ltm_box_bg.get_center())
        )
        ltm_summary_box = VGroup(ltm_box_bg, ltm_inner).move_to(
            LEFT * 3.55 + DOWN * 1.48
        )

        # Sağ Taraf: Ortak Yanıt Üretim Hedefi (Prompt Bağlamı + LLM)
        target_bg = RoundedRectangle(
            corner_radius=0.20,
            width=3.65,
            height=2.85,
            fill_color="#0D192B",
            fill_opacity=0.96,
            stroke_color=BLUE_B,
            stroke_width=3.0,
        )
        target_glow = RoundedRectangle(
            corner_radius=0.22,
            width=3.83,
            height=3.03,
            fill_opacity=0.0,
            stroke_color=BLUE_A,
            stroke_width=2.0,
            stroke_opacity=0.35,
        )
        t_title = Text("Zengin Bağlam", font_size=18, color=BLUE_A, weight=BOLD)
        t_mid = Text("+", font_size=20, color=WHITE, weight=BOLD)
        t_llm = Text("LLM Yanıt Üretimi", font_size=16, color=TEAL_A, weight=BOLD)
        t_desc = Text("Kişiselleştirilmiş & Tutarlı", font_size=12, color=GREY_A)
        t_content = (
            VGroup(t_title, t_mid, t_llm, t_desc)
            .arrange(DOWN, buff=0.14)
            .move_to(target_bg.get_center())
        )
        target_block = VGroup(target_glow, target_bg, t_content).move_to(
            RIGHT * 3.85 + DOWN * 0.38
        )

        # İki kutudan ortak hedefe çıkan oklar
        arr_upper = Arrow(
            stm_box_bg.get_right(),
            target_bg.get_left() + UP * 0.55,
            buff=0.14,
            color=YELLOW_B,
            stroke_width=3.4,
            max_tip_length_to_length_ratio=0.16,
        )
        lbl_upper = Text(
            "Doğrudan Prompt'a", font_size=12, color=YELLOW_A, weight=BOLD
        ).next_to(arr_upper, UP, buff=0.10)

        arr_lower = Arrow(
            ltm_box_bg.get_right(),
            target_bg.get_left() + DOWN * 0.55,
            buff=0.14,
            color=GREEN_B,
            stroke_width=3.4,
            max_tip_length_to_length_ratio=0.16,
        )
        lbl_lower = Text(
            "Retrieval (Arama) ile DB'den", font_size=12, color=GREEN_A, weight=BOLD
        ).next_to(arr_lower, DOWN, buff=0.10)

        self.play(
            FadeIn(stm_summary_box, shift=RIGHT * 0.3),
            FadeIn(ltm_summary_box, shift=RIGHT * 0.3),
            FadeIn(target_block, shift=LEFT * 0.3),
            run_time=self.rt(1.0),
        )
        self.play(
            GrowArrow(arr_upper),
            FadeIn(lbl_upper, shift=DOWN * 0.1),
            GrowArrow(arr_lower),
            FadeIn(lbl_lower, shift=UP * 0.1),
            run_time=self.rt(1.0),
        )

        # Ortak besleme sinyali parlaması
        p_up = arr_upper.copy().set_stroke(color=WHITE, width=5.0)
        p_low = arr_lower.copy().set_stroke(color=WHITE, width=5.0)
        self.play(
            ShowPassingFlash(p_up, time_width=0.6),
            ShowPassingFlash(p_low, time_width=0.6),
            Indicate(target_block, color=TEAL_A, scale_factor=1.05),
            Flash(
                target_bg.get_center(), color=BLUE_A, flash_radius=1.35, num_lines=12
            ),
            run_time=self.rt(1.1),
        )
        self.pause(1.1)

        # 5. Yumuşak bir kararma (FadeOut) ile animasyon sonlanır
        all_stage3 = VGroup(
            stm_summary_box,
            ltm_summary_box,
            target_block,
            arr_upper,
            lbl_upper,
            arr_lower,
            lbl_lower,
        )
        self.play(FadeOut(all_stage3, shift=DOWN * 0.15), run_time=self.rt(1.1))
        self.pause(0.4)
