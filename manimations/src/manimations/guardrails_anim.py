from __future__ import annotations

from typing import Any

import numpy as np
from manim import *

from manimations.base import RAGScene

# Hız sabiti: Animasyon ve bekleme sürelerini çarpmak/bölmek için
SPEED: float = 1.0


def create_shield(
    width: float = 2.4,
    height: float = 2.8,
    color: Any = BLUE_C,
    fill_color: str = "#0d1b2a",
    fill_opacity: float = 0.85,
    stroke_width: float = 3.5,
) -> VMobject:
    """Modern, pürüzsüz kavisli bir güvenlik kalkanı (Shield) oluşturur."""
    shield = VMobject()
    half_w = width / 2
    half_h = height / 2

    # Üst düz çizgi ve üst köşeler
    shield.set_points_as_corners(
        [
            np.array([-half_w, half_h, 0]),
            np.array([half_w, half_h, 0]),
            np.array([half_w, half_h * 0.1, 0]),
        ]
    )
    # Sağ kenardan alt uca kavis
    shield.add_cubic_bezier_curve_to(
        np.array([half_w * 0.95, -half_h * 0.45, 0]),
        np.array([half_w * 0.45, -half_h * 0.85, 0]),
        np.array([0, -half_h, 0]),
    )
    # Alt uçtan sol kenara kavis
    shield.add_cubic_bezier_curve_to(
        np.array([-half_w * 0.45, -half_h * 0.85, 0]),
        np.array([-half_w * 0.95, -half_h * 0.45, 0]),
        np.array([-half_w, half_h * 0.1, 0]),
    )
    # Sol üst köşeye kapanış
    shield.add_line_to(np.array([-half_w, half_h, 0]))
    shield.set_fill(fill_color, opacity=fill_opacity)
    shield.set_stroke(color, width=stroke_width)
    return shield


class GuardrailsScene(RAGScene):
    """
    Kullanıcı girdisindeki hassas verilerin (TCKN ve E-posta) Guardrails
    kalkanından geçerken tespit edilip maskelenmesini ([MASKED]) ve ardından
    güvenli hale gelen prompt'un LLM'e iletilmesini anlatan açıklayıcı animasyon sahnesi.
    """

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        bg_color: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data=data, bg_color=bg_color or "#000000", **kwargs)

    def rt(self, seconds: float) -> float:
        return max(0.1, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.1, seconds / SPEED))

    def construct(self) -> None:
        # Dinamik parametreler (Notebook'tan veya default)
        user_tckn = str(self.get_param("tckn", "27384910294"))
        user_email = str(self.get_param("email", "user1@example1.com"))
        mask_text = str(self.get_param("mask_placeholder", "[MASKED]"))

        # =====================================================================
        # 1. BAŞLIK VE ALT BAŞLIK
        # =====================================================================
        title = Text("2. Query Processing", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text(
            "PII Tespiti, Doğrulama & Veri Maskeleme (Guardrails)",
            font_size=18,
            color=RED_B,
        )
        header_group = (
            VGroup(title, subtitle)
            .arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            .to_corner(UL, buff=0.45)
        )
        self.play(FadeIn(header_group, shift=DOWN * 0.2), run_time=self.rt(0.8))
        self.pause(0.2)

        # =====================================================================
        # 2. ÜÇ ANA İSTASYON / MERKEZ (Layout Kurulumu)
        # =====================================================================
        # Sol İstasyon: Kullanıcı Girdisi Yuvası
        left_anchor_x = -4.3
        center_anchor_x = 0.0
        right_anchor_x = 4.3

        user_station_box = RoundedRectangle(
            corner_radius=0.15,
            width=3.4,
            height=3.8,
            fill_color="#121620",
            fill_opacity=0.6,
            stroke_color=GREY_C,
            stroke_width=1.5,
        ).move_to([left_anchor_x, -0.3, 0])

        user_station_label = Text(
            "Kullanıcı İstemi", font_size=15, weight=BOLD, color=GREY_A
        ).next_to(user_station_box.get_top(), DOWN, buff=0.2)
        user_station_badge = Text("Raw Input", font_size=11, color=GREY_B).next_to(
            user_station_label, DOWN, buff=0.08
        )
        user_station_group = VGroup(
            user_station_box, user_station_label, user_station_badge
        )

        # Orta İstasyon: Guardrails Kalkanı
        shield_outer = create_shield(
            width=2.8, height=3.2, color=BLUE_B, fill_color="#0a192f", fill_opacity=0.8
        )
        shield_inner = create_shield(
            width=2.4,
            height=2.8,
            color=TEAL_B,
            fill_color="#0e243d",
            fill_opacity=0.9,
            stroke_width=2,
        )
        shield_mobj = VGroup(shield_outer, shield_inner).move_to(
            [center_anchor_x, -0.3, 0]
        )

        shield_title = Text(
            "GUARDRAILS", font_size=15, weight=BOLD, color=WHITE
        ).move_to(shield_mobj.get_center() + UP * 0.5)
        shield_sub = Text("Security Shield", font_size=11, color=TEAL_C).next_to(
            shield_title, DOWN, buff=0.1
        )
        shield_regex_badge = RoundedRectangle(
            corner_radius=0.1,
            width=2.1,
            height=0.45,
            fill_color="#093145",
            fill_opacity=0.9,
            stroke_color=TEAL_A,
            stroke_width=1,
        ).move_to(shield_mobj.get_center() + DOWN * 0.45)
        shield_rule_text = Text(
            "PII Filter & Validator", font_size=10, weight=BOLD, color=WHITE
        ).move_to(shield_regex_badge.get_center())
        shield_group = VGroup(
            shield_mobj, shield_title, shield_sub, shield_regex_badge, shield_rule_text
        )

        # Sağ İstasyon: LLM Modeli
        llm_box = RoundedRectangle(
            corner_radius=0.2,
            width=3.2,
            height=3.8,
            fill_color="#18132b",
            fill_opacity=0.7,
            stroke_color=PURPLE_B,
            stroke_width=2,
        ).move_to([right_anchor_x, -0.3, 0])

        llm_label = Text(
            "LLM Engine", font_size=16, weight=BOLD, color=PURPLE_A
        ).next_to(llm_box.get_top(), DOWN, buff=0.22)
        llm_sub = Text("Yapay Zeka Modeli", font_size=11, color=GREY_A).next_to(
            llm_label, DOWN, buff=0.08
        )
        llm_core_rect = RoundedRectangle(
            corner_radius=0.12,
            width=2.4,
            height=1.4,
            fill_color="#2b1a4a",
            fill_opacity=0.85,
            stroke_color=PINK,
            stroke_width=1.5,
        ).move_to(llm_box.get_center() + DOWN * 0.25)
        llm_core_text = Text(
            "Safe Context\nReady",
            font_size=13,
            weight=BOLD,
            color=WHITE,
            line_spacing=0.8,
        ).move_to(llm_core_rect.get_center())
        llm_group = VGroup(llm_box, llm_label, llm_sub, llm_core_rect, llm_core_text)

        # Bağlantı Okları
        arrow_1 = Arrow(
            start=user_station_box.get_right() + RIGHT * 0.1,
            end=shield_outer.get_left() + LEFT * 0.1,
            color=GREY_B,
            buff=0,
            stroke_width=2.5,
            max_tip_length_to_length_ratio=0.2,
        )
        arrow_2 = Arrow(
            start=shield_outer.get_right() + RIGHT * 0.1,
            end=llm_box.get_left() + LEFT * 0.1,
            color=GREY_B,
            buff=0,
            stroke_width=2.5,
            max_tip_length_to_length_ratio=0.2,
        )

        self.play(
            FadeIn(user_station_group, shift=RIGHT * 0.2),
            FadeIn(shield_group, shift=UP * 0.2),
            FadeIn(llm_group, shift=LEFT * 0.2),
            Create(arrow_1),
            Create(arrow_2),
            run_time=self.rt(1.2),
        )
        self.pause(0.3)

        # =====================================================================
        # 3. KULLANICI GİRDİSİ (RAW PAYLOAD - TCKN ve E-POSTA İÇEREN KART)
        # =====================================================================
        payload_bg = RoundedRectangle(
            corner_radius=0.12,
            width=2.9,
            height=2.3,
            fill_color="#1e1822",
            fill_opacity=0.95,
            stroke_color=RED_C,
            stroke_width=2,
        ).move_to(user_station_box.get_center() + DOWN * 0.3)

        # Metin parçaları
        txt_prefix = Text("Fatura sorgusu:", font_size=11, color=GREY_B)

        # TCKN satırı
        tckn_lbl = Text("TCKN: ", font_size=11, color=WHITE, weight=BOLD)
        tckn_val = Text(user_tckn, font_size=11, color=RED_A, weight=BOLD)
        tckn_line = VGroup(tckn_lbl, tckn_val).arrange(RIGHT, buff=0.08)

        # Email satırı
        email_lbl = Text("E-mail: ", font_size=11, color=WHITE, weight=BOLD)
        email_val = Text(user_email, font_size=10, color=RED_A, weight=BOLD)
        email_line = VGroup(email_lbl, email_val).arrange(RIGHT, buff=0.08)

        # Uyarı etiketi
        warning_tag = RoundedRectangle(
            corner_radius=0.08,
            width=2.4,
            height=0.32,
            fill_color="#451212",
            fill_opacity=0.9,
            stroke_color=RED_D,
            stroke_width=1,
        )
        warning_tag_text = Text(
            "[!] PII: Hassas Veri", font_size=10, color=RED_A, weight=BOLD
        ).move_to(warning_tag.get_center())
        warning_group = VGroup(warning_tag, warning_tag_text)

        payload_content = (
            VGroup(txt_prefix, tckn_line, email_line, warning_group)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .move_to(payload_bg.get_center())
        )

        raw_payload_card = VGroup(payload_bg, payload_content)

        self.play(
            FadeIn(raw_payload_card, shift=UP * 0.3),
            run_time=self.rt(1.0),
        )
        # TCKN ve Email üzerinde dikkat çekme
        self.play(
            Indicate(tckn_val, color=RED, scale_factor=1.15),
            Indicate(email_val, color=RED, scale_factor=1.15),
            run_time=self.rt(0.9),
        )
        self.pause(0.3)

        # =====================================================================
        # 4. GİRDİNİN KALKANA (GUARDRAILS) DOĞRU İLERLEMESİ
        # =====================================================================
        flow_target_center = shield_mobj.get_center()
        arrow_1_active = arrow_1.copy().set_color(ORANGE).set_stroke(width=4)

        self.play(
            Transform(arrow_1, arrow_1_active),
            raw_payload_card.animate.move_to(flow_target_center).scale(0.9),
            run_time=self.rt(1.3),
        )
        self.pause(0.2)

        # Kalkan aktifleşir ve tarama çizgisi (Scan Beam) geçer
        shield_glow = shield_outer.copy().set_stroke(color=YELLOW_B, width=6)
        scan_line = Line(
            start=raw_payload_card.get_left() + UP * 1.1,
            end=raw_payload_card.get_right() + UP * 1.1,
            color=YELLOW_A,
            stroke_width=3,
        )

        self.play(
            Transform(shield_outer, shield_glow),
            FadeIn(scan_line),
            run_time=self.rt(0.4),
        )
        self.play(
            scan_line.animate.move_to(raw_payload_card.get_bottom() + DOWN * 0.1),
            run_time=self.rt(0.8),
        )
        self.play(FadeOut(scan_line), run_time=self.rt(0.2))

        # =====================================================================
        # 5. MASKELEME İŞLEMİ (PII -> [MASKED])
        # =====================================================================
        # Yeni maskelenmiş metin nesneleri
        masked_tckn_val = Text(
            mask_text, font_size=10, color=GREEN_A, weight=BOLD
        ).move_to(tckn_val.get_center())
        masked_email_val = Text(
            mask_text, font_size=10, color=GREEN_A, weight=BOLD
        ).move_to(email_val.get_center())

        # Uyarı etiketi güvenli yeşil etikete dönüşür
        safe_tag = RoundedRectangle(
            corner_radius=0.08,
            width=2.4,
            height=0.32,
            fill_color="#0e3d23",
            fill_opacity=0.9,
            stroke_color=GREEN_D,
            stroke_width=1,
        ).move_to(warning_tag.get_center())
        safe_tag_text = Text(
            "[OK] Maskelendi (Guvenli)", font_size=10, color=GREEN_A, weight=BOLD
        ).move_to(safe_tag.get_center())

        # Kart çerçevesi yeşile döner
        sanitized_bg = (
            payload_bg.copy()
            .set_stroke(color=GREEN_B, width=2)
            .set_fill(color="#0c2016", opacity=0.95)
        )
        shield_safe_glow = shield_outer.copy().set_stroke(color=GREEN_A, width=4)

        self.play(
            Transform(tckn_val, masked_tckn_val),
            Transform(email_val, masked_email_val),
            Transform(warning_tag, safe_tag),
            Transform(warning_tag_text, safe_tag_text),
            Transform(payload_bg, sanitized_bg),
            Transform(shield_outer, shield_safe_glow),
            run_time=self.rt(1.2),
        )
        self.play(
            Circumscribe(raw_payload_card, color=GREEN_B, run_time=self.rt(0.8)),
        )
        self.pause(0.3)

        # =====================================================================
        # 6. GÜVENLİ İSTEĞİN LLM'E İLETİLMESİ
        # =====================================================================
        arrow_2_active = arrow_2.copy().set_color(GREEN_B).set_stroke(width=4)
        llm_target_pos = llm_box.get_center() + DOWN * 0.25

        self.play(
            Transform(arrow_2, arrow_2_active),
            raw_payload_card.animate.move_to(llm_target_pos)
            .scale(0.85)
            .set_opacity(0.85),
            run_time=self.rt(1.3),
        )
        self.pause(0.2)

        # LLM girdi aldığında aydınlanır ve işleme başlar
        llm_active_glow = llm_box.copy().set_stroke(color=TEAL_A, width=4)
        llm_processed_text = Text(
            "PII-Free Query\nProcessing...",
            font_size=12,
            weight=BOLD,
            color=GREEN_A,
            line_spacing=0.8,
        ).move_to(llm_core_rect.get_center())

        self.play(
            Transform(llm_box, llm_active_glow),
            FadeOut(raw_payload_card, shift=DOWN * 0.1),
            Transform(llm_core_text, llm_processed_text),
            run_time=self.rt(0.9),
        )
        self.pause(0.2)

        # =====================================================================
        # 7. LLM GÜVENLİ YANIT ÜRETİMİ (GÜVENLİ DÖNGÜ TAMAMLANIYOR)
        # =====================================================================
        response_card = RoundedRectangle(
            corner_radius=0.12,
            width=3.6,
            height=1.2,
            fill_color="#112233",
            fill_opacity=0.95,
            stroke_color=TEAL_B,
            stroke_width=2,
        ).next_to(llm_box, DOWN, buff=0.35)

        resp_title = Text(
            "Guvenli LLM Yaniti:", font_size=11, color=TEAL_A, weight=BOLD
        )
        resp_msg = Text(
            "Fatura kaydınız güvenle işlendi.\nHerhangi bir kişisel veri sızdırılmadı.",
            font_size=10,
            color=WHITE,
            line_spacing=0.8,
        )
        resp_group = (
            VGroup(resp_title, resp_msg)
            .arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            .move_to(response_card.get_center())
        )
        full_response_bubble = VGroup(response_card, resp_group)

        self.play(
            FadeIn(full_response_bubble, shift=UP * 0.25),
            run_time=self.rt(1.0),
        )
        self.play(
            Flash(full_response_bubble.get_corner(UR), color=GREEN_B, flash_radius=0.4),
            run_time=self.rt(0.6),
        )
        self.pause(1.2)
