from __future__ import annotations

from manim import *

from manimations.base import RAGScene


class TextExtractionScene(RAGScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def construct(self) -> None:
        # 1. Başlık ve alt başlık (Sol üste tam sola yaslı, notebook ile birebir aynı)
        title = Text("1. Ingestion Pipeline", font_size=32, weight=BOLD, color=BLUE_B)
        subtitle = Text("Text Extraction", font_size=18, color=RED_B)
        title_group = (
            VGroup(title, subtitle)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .to_corner(UL, buff=0.5)
        )
        self.play(FadeIn(title_group, shift=DOWN * 0.2), run_time=1.2)
        self.wait(0.3)

        # 2. Orta sol tarafta kaynak doküman (senin oluşturduğun yapı)
        document = RoundedRectangle(
            corner_radius=0.2,
            width=3.0,
            height=4.0,
            color=WHITE,
            fill_opacity=1,
        ).shift(LEFT * 4.5 + DOWN * 0.3)

        lines = VGroup()
        for i in range(6):
            y_offset = 1.4 - i * 0.55
            line = Line(
                start=document.get_left() + RIGHT * 0.35 + UP * y_offset,
                end=document.get_right() + LEFT * 0.35 + UP * y_offset,
                color=BLACK,
                stroke_width=2.5,
            )
            lines.add(line)

        document_text = Text(
            "Document", font_size=22, color=BLACK, weight=BOLD
        ).move_to(document.get_center())
        document_group = VGroup(document, lines, document_text)
        self.play(FadeIn(document_group, shift=DOWN * 0.2), run_time=1.2)
        self.wait(0.3)

        # 3. Ortada OCR kutusu ve sağ tarafta hedef "Extracted Text" dokümanı
        ocr_box = RoundedRectangle(
            corner_radius=0.2,
            width=2.2,
            height=1.4,
            color=GREEN,
            fill_opacity=1,
        ).shift(DOWN * 0.3)
        ocr_text = Text("OCR", font_size=26, color=WHITE, weight=BOLD).move_to(
            ocr_box.get_center()
        )
        ocr_group = VGroup(ocr_box, ocr_text)

        extracted_doc = RoundedRectangle(
            corner_radius=0.2,
            width=3.2,
            height=4.0,
            color=BLUE_B,
            fill_color=DARK_GREY,
            fill_opacity=0.25,
            stroke_width=2.5,
        ).shift(RIGHT * 4.5 + DOWN * 0.3)
        extracted_doc_header = Text(
            "Extracted Text",
            font_size=20,
            color=BLUE_B,
            weight=BOLD,
        ).move_to(extracted_doc.get_top() + DOWN * 0.4)
        extracted_doc_divider = Line(
            extracted_doc.get_left() + RIGHT * 0.25 + UP * 1.3,
            extracted_doc.get_right() + LEFT * 0.25 + UP * 1.3,
            color=BLUE_B,
            stroke_width=1.5,
        )
        extracted_doc_group = VGroup(
            extracted_doc, extracted_doc_header, extracted_doc_divider
        )

        self.play(
            FadeIn(ocr_group, shift=DOWN * 0.2),
            FadeIn(extracted_doc_group, shift=DOWN * 0.2),
            run_time=1.2,
        )
        self.wait(0.3)

        # 4. OCR'ın köşelerinden çıkan ışınlar ve doküman üzerinde gezen küçük dikdörtgen
        # Dokümanı tam kapatmayan küçük bir tarama penceresi:
        scan_rect = Rectangle(
            width=1.1,
            height=0.65,
            color=YELLOW,
            fill_color=YELLOW,
            fill_opacity=0.3,
            stroke_width=2.5,
        ).move_to(document.get_center() + LEFT * 0.65 + UP * 1.1)

        def build_beams() -> VGroup:
            # OCR sol köşelerinden ve merkezlerinden tarama dikdörtgeninin köşelerine ışınlar
            ocr_ul = ocr_box.get_corner(UL)
            ocr_dl = ocr_box.get_corner(DL)
            ocr_ur = ocr_box.get_corner(UR)
            ocr_dr = ocr_box.get_corner(DR)

            sr_ul = scan_rect.get_corner(UL)
            sr_ur = scan_rect.get_corner(UR)
            sr_dl = scan_rect.get_corner(DL)
            sr_dr = scan_rect.get_corner(DR)

            # Işın konisi (yarı saydam yüzey + 4 köşe ışın çizgisi)
            cone = Polygon(
                ocr_ul,
                sr_ul,
                sr_dl,
                ocr_dl,
                color=GREEN_B,
                fill_color=YELLOW,
                fill_opacity=0.14,
                stroke_width=0,
            )
            b1 = Line(ocr_ul, sr_ul, color=YELLOW, stroke_width=2, stroke_opacity=0.75)
            b2 = Line(ocr_ul, sr_ur, color=YELLOW, stroke_width=2, stroke_opacity=0.75)
            b3 = Line(ocr_dl, sr_dl, color=YELLOW, stroke_width=2, stroke_opacity=0.75)
            b4 = Line(ocr_dl, sr_dr, color=YELLOW, stroke_width=2, stroke_opacity=0.75)
            b5 = Line(ocr_ur, sr_ur, color=YELLOW, stroke_width=2, stroke_opacity=0.75)
            b6 = Line(ocr_ur, sr_ul, color=YELLOW, stroke_width=2, stroke_opacity=0.75)
            b7 = Line(ocr_dr, sr_dr, color=YELLOW, stroke_width=2, stroke_opacity=0.75)
            b8 = Line(ocr_dr, sr_dl, color=YELLOW, stroke_width=2, stroke_opacity=0.75)
            return VGroup(cone, b1, b2, b3, b4, b5, b6, b7, b8)

        beams = always_redraw(build_beams)

        self.add(beams)
        self.add(ocr_group)

        self.play(FadeIn(scan_rect), run_time=0.6)

        # Sağdaki dokümanın içine yazılacak çıkarılmış metin satırları
        default_lines = [
            "# Başlık 1",
            "Metin satırı 1...",
            "Metin satırı 2...",
            "Metin satırı 3...",
            "Metin satırı 4...",
        ]
        custom_lines = self.get_param("extracted_lines", default_lines)

        def create_extracted_row(idx: int, label_str: str) -> VGroup:
            y_pos = 0.85 - idx * 0.52
            row_text = Text(
                label_str,
                font_size=15,
                color=WHITE,
            )
            row_text.move_to(extracted_doc.get_center() + UP * y_pos)
            row_text.align_to(extracted_doc.get_left() + RIGHT * 0.35, LEFT)
            return VGroup(row_text)

        # Tarama dikdörtgeni hareketleri: Sağ -> Aşağı -> Sol -> Yukarı
        doc_c = document.get_center()
        waypoints = [
            (doc_c + RIGHT * 0.65 + UP * 1.1, custom_lines[0]),  # Sağa
            (doc_c + RIGHT * 0.65 + DOWN * 0.2, custom_lines[1]),  # Aşağı
            (doc_c + LEFT * 0.65 + DOWN * 0.2, custom_lines[2]),  # Sola
            (doc_c + LEFT * 0.65 + UP * 0.45, custom_lines[3]),  # Yukarı
            (
                doc_c + RIGHT * 0.65 + UP * 0.45,
                custom_lines[4],
            ),  # Tekrar sağa (son satır)
        ]

        written_rows = VGroup()
        for idx, (target_pos, text_val) in enumerate(waypoints):
            row_item = create_extracted_row(idx, text_val)
            written_rows.add(row_item)

            # OCR'dan sağdaki dokümana kısa veri akışı çizgisi
            out_beam = DashedLine(
                ocr_box.get_right(),
                row_item.get_left() + LEFT * 0.1,
                color=BLUE_B,
                stroke_width=2,
            )
            self.play(
                scan_rect.animate.move_to(target_pos),
                Create(out_beam),
                FadeIn(row_item, shift=RIGHT * 0.15),
                run_time=0.85,
            )
            self.play(FadeOut(out_beam), run_time=0.2)

        # Tarama tamamlandığında ışınları ve tarama kutusunu kapat
        self.play(FadeOut(scan_rect), FadeOut(beams), run_time=0.6)
        self.wait(1.5)
