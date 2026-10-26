from __future__ import annotations

from typing import Any

from manim import *

# Hız sabiti: Tüm animasyon ve bekleme süreleri bu çarpanla ayarlanır.
SPEED: float = 1.0

DEFAULT_QUERY = "RAG sisteminde chunk boyutu nasıl seçilmeli?"
DEFAULT_INTENT = "knowledge_question"
DEFAULT_CONFIDENCE = 0.98
DEFAULT_RESOLVED_QUERY = "RAG mimarisinde chunk boyutu seçimi"


class IntentClassificationScene(ThreeDScene):
    """
    Query Processing aşamasında Intent Classification (Niyet Sınıflandırma)
    sürecini anlatan Manim sahnesi.
    """

    def __init__(
        self,
        query: str | None = None,
        intent: str | None = None,
        confidence: float | None = None,
        resolved_query: str | None = None,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_query = query
        self.input_intent = intent
        self.input_confidence = confidence
        self.input_resolved_query = resolved_query
        self.data: dict[str, Any] = data or {}

    def rt(self, seconds: float) -> float:
        return max(0.1, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.1, seconds / SPEED))

    def _resolve_inputs(self) -> tuple[str, str, float, str]:
        d = self.data if isinstance(self.data, dict) else {}
        q = self.input_query or d.get("query") or DEFAULT_QUERY
        intent = self.input_intent or d.get("intent") or DEFAULT_INTENT
        conf = (
            self.input_confidence
            if self.input_confidence is not None
            else d.get("confidence", DEFAULT_CONFIDENCE)
        )
        res_q = (
            self.input_resolved_query
            or d.get("resolved_query")
            or DEFAULT_RESOLVED_QUERY
        )
        return str(q), str(intent), float(conf), str(res_q)

    def construct(self) -> None:
        self.camera.background_color = BLACK

        # =====================================================================
        # SABİT BAŞLIK (Aynen kullanılacak)
        # =====================================================================
        title = Text("2. Query Processing", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text("Intent Classification", font_size=18, color=RED_B)
        title_group = (
            VGroup(title, subtitle)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .to_corner(UL, buff=0.5)
        )
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(FadeIn(title_group, shift=DOWN * 0.2), run_time=1.2)
        self.wait(0.3)

        query_str, intent_str, conf_val, resolved_str = self._resolve_inputs()

        # =====================================================================
        # 1. KULLANICI SORGUSU (SOL TARAF)
        # =====================================================================
        q_card_w = 3.85
        q_card_h = 2.35
        q_bg = RoundedRectangle(
            corner_radius=0.16,
            width=q_card_w,
            height=q_card_h,
            fill_color="#121824",
            fill_opacity=0.95,
            stroke_color=BLUE_B,
            stroke_width=2.2,
        )
        q_header_badge = RoundedRectangle(
            corner_radius=0.08,
            width=2.2,
            height=0.38,
            fill_color=BLUE_E,
            fill_opacity=0.6,
            stroke_color=BLUE_B,
            stroke_width=1.2,
        )
        q_header_txt = Text(
            "Kullanıcı Sorgusu", font_size=15, color=BLUE_A, weight=BOLD
        )
        q_header_txt.move_to(q_header_badge.get_center())
        q_header = VGroup(q_header_badge, q_header_txt).next_to(
            q_bg.get_top(), DOWN, buff=0.22
        )

        # Sorgu metnini iki satıra sararak kart içine sığdır
        q_words = query_str.split()
        mid_idx = max(1, len(q_words) // 2)
        line1_str = " ".join(q_words[:mid_idx])
        line2_str = " ".join(q_words[mid_idx:])

        q_lines = VGroup(
            Text(f'"{line1_str}', font_size=16, color=WHITE),
            Text(f'{line2_str}"', font_size=16, color=WHITE),
        ).arrange(DOWN, buff=0.12)
        if q_lines.width > q_card_w - 0.45:
            q_lines.scale_to_fit_width(q_card_w - 0.45)
        q_lines.next_to(q_header, DOWN, buff=0.32)

        query_card = VGroup(q_bg, q_header, q_lines).move_to(LEFT * 4.55 + DOWN * 0.45)

        self.play(FadeIn(query_card, shift=RIGHT * 0.25), run_time=self.rt(1.3))
        self.pause(0.8)

        # =====================================================================
        # 2. LLM İŞLEMCİ BLOĞU (MERKEZ)
        # =====================================================================
        llm_w = 3.1
        llm_h = 2.15
        llm_bg = RoundedRectangle(
            corner_radius=0.2,
            width=llm_w,
            height=llm_h,
            fill_color="#1A1426",
            fill_opacity=0.95,
            stroke_color=PURPLE_B,
            stroke_width=2.6,
        )
        llm_glow = RoundedRectangle(
            corner_radius=0.22,
            width=llm_w + 0.16,
            height=llm_h + 0.16,
            fill_opacity=0.0,
            stroke_color=PURPLE_A,
            stroke_width=1.2,
            stroke_opacity=0.45,
        )
        llm_title = Text("LLM", font_size=26, color=PURPLE_A, weight=BOLD)
        llm_sub = Text("(Intent Classifier)", font_size=15, color=GOLD_B, weight=BOLD)
        llm_labels = VGroup(llm_title, llm_sub).arrange(DOWN, buff=0.14)
        llm_labels.move_to(llm_bg.get_center())

        llm_block = VGroup(llm_glow, llm_bg, llm_labels).move_to(
            LEFT * 0.2 + DOWN * 0.45
        )

        # Sol karttan LLM'e akış oku
        arrow_in = Arrow(
            query_card.get_right(),
            llm_block.get_left(),
            buff=0.16,
            color=BLUE_B,
            stroke_width=3.2,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(
            FadeIn(llm_block, scale=0.92),
            GrowArrow(arrow_in),
            run_time=self.rt(1.2),
        )
        self.pause(0.4)

        # Kullanıcı sorgusu akış oku eşliğinde LLM içine girer
        query_packet = q_lines.copy()
        packet_box = SurroundingRectangle(
            query_packet,
            color=BLUE_B,
            fill_color="#121824",
            fill_opacity=0.9,
            buff=0.12,
            corner_radius=0.08,
        )
        moving_query = VGroup(packet_box, query_packet)

        self.play(
            Indicate(query_card, color=BLUE_A, scale_factor=1.03),
            FadeIn(moving_query, scale=0.95),
            run_time=self.rt(0.6),
        )
        self.play(
            moving_query.animate.move_to(llm_block.get_center()).scale(0.35),
            run_time=self.rt(1.1),
        )
        self.play(
            FadeOut(moving_query, scale=0.2),
            run_time=self.rt(0.3),
        )

        # LLM üzerinde kısa analiz / işleme parlaması
        proc_tag = Text("Analiz ediliyor...", font_size=14, color=YELLOW_B).next_to(
            llm_block, UP, buff=0.2
        )
        pulse_ring = RoundedRectangle(
            corner_radius=0.22,
            width=llm_w,
            height=llm_h,
            stroke_color=YELLOW_B,
            stroke_width=3.5,
            fill_opacity=0.0,
        ).move_to(llm_block.get_center())

        self.play(
            FadeIn(proc_tag, shift=UP * 0.1),
            Create(pulse_ring),
            llm_bg.animate.set_stroke(color=YELLOW_B, width=3.5),
            run_time=self.rt(0.7),
        )
        self.play(
            Flash(
                llm_block.get_center(), color=YELLOW_B, flash_radius=1.35, num_lines=12
            ),
            Indicate(llm_labels, color=YELLOW_B, scale_factor=1.08),
            pulse_ring.animate.scale(1.12).set_stroke(opacity=0.0),
            run_time=self.rt(0.9),
        )
        self.play(
            FadeOut(proc_tag),
            FadeOut(pulse_ring),
            llm_bg.animate.set_stroke(color=PURPLE_B, width=2.6),
            run_time=self.rt(0.5),
        )

        # =====================================================================
        # 3. YAPILANDIRILMIŞ ŞEMA ÇIKTISI (SAĞ TARAF - intent_schema.py)
        # =====================================================================
        out_w = 4.45
        out_h = 2.95
        out_bg = RoundedRectangle(
            corner_radius=0.16,
            width=out_w,
            height=out_h,
            fill_color="#101C1A",
            fill_opacity=0.96,
            stroke_color=TEAL_B,
            stroke_width=2.3,
        )

        schema_badge_bg = RoundedRectangle(
            corner_radius=0.08,
            width=2.95,
            height=0.38,
            fill_color="#17332F",
            fill_opacity=0.85,
            stroke_color=TEAL_B,
            stroke_width=1.2,
        )
        schema_badge_txt = Text(
            "intent_schema.py (JSON)", font_size=14, color=TEAL_A, weight=BOLD
        )
        schema_badge_txt.move_to(schema_badge_bg.get_center())
        schema_header = VGroup(schema_badge_bg, schema_badge_txt).next_to(
            out_bg.get_top(), DOWN, buff=0.2
        )

        # JSON / Pydantic alanları:
        # - intent: "knowledge_question"
        # - confidence: 0.98
        # - resolved_query: "RAG mimarisinde chunk boyutu seçimi"
        brace_open = Text("{", font_size=15, color=GREY_B)

        row_intent = VGroup(
            Text('  "intent": ', font_size=14, color=BLUE_B, weight=BOLD),
            Text(f'"{intent_str}",', font_size=14, color=GREEN_B),
        ).arrange(RIGHT, buff=0.06)

        row_conf = VGroup(
            Text('  "confidence": ', font_size=14, color=BLUE_B, weight=BOLD),
            Text(f"{conf_val:.2f},", font_size=14, color=YELLOW_B, weight=BOLD),
        ).arrange(RIGHT, buff=0.06)

        row_res_key = Text(
            '  "resolved_query":', font_size=14, color=BLUE_B, weight=BOLD
        )
        row_res_val = Text(f'    "{resolved_str}"', font_size=13, color=ORANGE)
        if row_res_val.width > out_w - 0.45:
            row_res_val.scale_to_fit_width(out_w - 0.45)

        brace_close = Text("}", font_size=15, color=GREY_B)

        json_body = VGroup(
            brace_open,
            row_intent,
            row_conf,
            row_res_key,
            row_res_val,
            brace_close,
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        if json_body.width > out_w - 0.4:
            json_body.scale_to_fit_width(out_w - 0.4)
        json_body.next_to(schema_header, DOWN, buff=0.22).align_to(
            out_bg.get_left() + RIGHT * 0.25, LEFT
        )

        output_card = VGroup(out_bg, schema_header, json_body).move_to(
            RIGHT * 4.25 + DOWN * 0.45
        )

        arrow_out = Arrow(
            llm_block.get_right(),
            output_card.get_left(),
            buff=0.16,
            color=TEAL_B,
            stroke_width=3.2,
            max_tip_length_to_length_ratio=0.18,
        )

        # LLM'den sağa akış oku ve çıktı kartının üretilmesi
        self.play(
            GrowArrow(arrow_out),
            FadeIn(out_bg, shift=RIGHT * 0.25),
            FadeIn(schema_header, shift=RIGHT * 0.25),
            run_time=self.rt(1.1),
        )
        self.play(
            LaggedStart(
                FadeIn(brace_open),
                FadeIn(row_intent, shift=RIGHT * 0.15),
                FadeIn(row_conf, shift=RIGHT * 0.15),
                FadeIn(row_res_key, shift=RIGHT * 0.15),
                FadeIn(row_res_val, shift=RIGHT * 0.15),
                FadeIn(brace_close),
                lag_ratio=0.18,
            ),
            run_time=self.rt(1.8),
        )

        # Onay rozeti ve belirginleşme efekti
        check_circle = Circle(
            radius=0.22,
            fill_color=GREEN_E,
            fill_opacity=0.95,
            stroke_color=GREEN_B,
            stroke_width=2.2,
        )
        check_mark = Text("✓", font_size=22, color=GREEN_A, weight=BOLD).move_to(
            check_circle.get_center()
        )
        check_badge = VGroup(check_circle, check_mark).move_to(
            out_bg.get_corner(UR) + LEFT * 0.2 + DOWN * 0.2
        )

        self.play(
            Circumscribe(
                output_card, color=GREEN_B, time_width=1.2, run_time=self.rt(1.2)
            ),
            out_bg.animate.set_stroke(color=GREEN_B, width=3.0),
            FadeIn(check_badge, scale=1.4),
            run_time=self.rt(1.2),
        )
        self.pause(2.0)
