from __future__ import annotations

import math
from typing import Any

import numpy as np
from manim import *

from manimations.base import RAGScene

# Her sahne yaklasik bu sure (sn) sonunda biter; kalan sure sonda beklenerek tamamlanir.
TOTAL_SECONDS: float = 12.0
SPEED: float = 0.90  # orijinal EvaluationScene ile ayni hiz sabiti

# ---------------------------------------------------------------------------
# Ortak ornek veri (notebook'taki sample_retrieved / sample_expected ile ayni)
# R = [chunk_2, chunk_1, chunk_5], E = [chunk_1, chunk_3]
# ---------------------------------------------------------------------------
R_LIST = ["chunk_2", "chunk_1", "chunk_5"]
E_LIST = ["chunk_1", "chunk_3"]
N_R, N_E = len(R_LIST), len(E_LIST)
N_HIT = len(set(R_LIST) & set(E_LIST))  # 1

PREC = N_HIT / N_R  # 0.3333
REC = N_HIT / N_E  # 0.5
F1 = 2 * PREC * REC / (PREC + REC)  # 0.4
FIRST_RANK = next(i for i, c in enumerate(R_LIST, 1) if c in E_LIST)  # 2
RR = 1 / FIRST_RANK  # 0.5
REL = [1 if c in E_LIST else 0 for c in R_LIST]  # [0, 1, 0]
DISC = [1 / math.log2(i + 1) for i in range(1, N_R + 1)]  # 1, 0.6309, 0.5
DCG = sum(r * d for r, d in zip(REL, DISC))  # 0.6309
IDEAL = [1] * min(N_E, N_R) + [0] * (N_R - min(N_E, N_R))  # [1, 1, 0]
IDCG = sum(r * d for r, d in zip(IDEAL, DISC))  # 1.6309
NDCG = DCG / IDCG  # 0.3869

# Generation ornekleri (aciklayici, gercek notebook ciktisi degildir)
FAITH_SUP, FAITH_TOT = 3, 4
FAITH = FAITH_SUP / FAITH_TOT  # 0.75
REL_SIMS = [(0.92, 1), (0.88, 1), (0.60, 0)]  # (sim(q, q_i), rel_i)
RELEVANCY = sum(r * max(0.0, s) for s, r in REL_SIMS) / len(REL_SIMS)  # 0.6
TP, FP, FN = 3, 1, 2
Q_PREC = TP / (TP + FP)  # 0.75
Q_REC = TP / (TP + FN)  # 0.6  (Completeness)
Q_F1 = 2 * Q_PREC * Q_REC / (Q_PREC + Q_REC)  # 0.6667 (Accuracy)
SIM_A = 0.88
W_F, W_S = 0.75, 0.25
CORRECT = W_F * Q_F1 + W_S * SIM_A  # 0.72
Q_CLEAN = (CORRECT + Q_REC + Q_F1) / 3
NOISY = (0.55, 0.45, 0.50)  # correctness, completeness, accuracy (gurultulu baglam)
Q_NOISY = sum(NOISY) / 3
NOISE_SENS = max(0.0, min(1.0, (Q_CLEAN - Q_NOISY) / Q_CLEAN))


class CalcScene(RAGScene):
    """Tek bir metrigin hesaplanisini ~12 sn'de adim adim gosteren taban sahne."""

    ACCENT = BLUE_B
    SECTION = "Retrieval Evaluation"

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(data=data, **kwargs)

    def rt(self, seconds: float) -> float:
        return max(0.12, seconds / SPEED)

    def pause(self, seconds: float) -> None:
        self.wait(max(0.12, seconds / SPEED))

    def add_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        super_fn = getattr(super(), "add_fixed_in_frame_mobjects", None)
        if callable(super_fn):
            super_fn(*mobjects)

    # ---------------------------- yardimcilar ----------------------------
    @staticmethod
    def chip(
        text: str,
        color: ManimColor,
        w: float = 3.4,
        h: float = 0.46,
        fs: int = 14,
        fill: str = "#101620",
    ) -> VGroup:
        bg = RoundedRectangle(
            corner_radius=0.09,
            width=w,
            height=h,
            fill_color=fill,
            fill_opacity=0.95,
            stroke_color=color,
            stroke_width=1.8,
        )
        txt = Text(text, font_size=fs, color=WHITE, weight=BOLD).move_to(bg)
        if txt.width > w - 0.2:
            txt.scale_to_fit_width(w - 0.2)
        return VGroup(bg, txt)

    @staticmethod
    def lbl(text: str, color: ManimColor = WHITE, fs: int = 15) -> Text:
        return Text(text, font_size=fs, color=color, weight=BOLD)

    def header(self, title: str, subtitle: str = "") -> None:
        """Orijinal EvaluationScene basligi: sabit '6. Evaluation' + alt baslik (metrik adi)."""
        main = Text("6. Evaluation", font_size=32, weight=BOLD, color=BLUE_B)
        sub = Text(f"{self.SECTION} - {title}", font_size=18, color=RED_B)
        grp = (
            VGroup(main, sub)
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            .to_corner(UL, buff=0.5)
        )
        self.add_fixed_in_frame_mobjects(grp)
        self.play(FadeIn(grp, shift=DOWN * 0.2), run_time=self.rt(1.1))
        self.pause(0.3)

    def math(self, latex: str, fallback: str = "", fs: int = 30, color: ManimColor = WHITE) -> Mobject:
        """Formulleri LaTeX (MathTex) ile cizer. `fallback` eski imza uyumu icin tutuldu, kullanilmaz."""
        return MathTex(latex, font_size=fs, color=color)

    def mlbl(self, latex: str, color: ManimColor = WHITE, fs: int = 26) -> Mobject:
        return MathTex(latex, font_size=fs, color=color)

    @staticmethod
    def mchip(
        latex: str,
        color: ManimColor,
        w: float = 3.4,
        h: float = 0.46,
        fs: int = 24,
        fill: str = "#101620",
    ) -> VGroup:
        bg = RoundedRectangle(
            corner_radius=0.09,
            width=w,
            height=h,
            fill_color=fill,
            fill_opacity=0.95,
            stroke_color=color,
            stroke_width=1.8,
        )
        m = MathTex(latex, font_size=fs, color=WHITE).move_to(bg)
        if m.width > w - 0.2:
            m.scale_to_fit_width(w - 0.2)
        return VGroup(bg, m)

    def reveal_lines(
        self,
        specs: list[tuple[str, str, ManimColor]],
        y_top: float,
        gap: float = 0.8,
        run: float = 0.8,
        hold: float = 0.35,
        fs: int = 30,
    ) -> list[Mobject]:
        """Formul adimlarini yukaridan asagiya sirayla gosterir (genel -> yerine koyma -> sonuc)."""
        out: list[Mobject] = []
        for i, (latex, fb, col) in enumerate(specs):
            m = self.math(latex, fb, fs=fs, color=col)
            if m.width > 11.5:
                m.scale_to_fit_width(11.5)
            m.move_to(np.array([0.0, y_top - i * gap, 0.0]))
            self.play(FadeIn(m, shift=UP * 0.15), run_time=run)
            self.wait(hold)
            out.append(m)
        return out

    def box_result(self, mobj: Mobject, color: ManimColor = YELLOW_B) -> None:
        rect = SurroundingRectangle(mobj, color=color, buff=0.14, corner_radius=0.1)
        self.play(Create(rect), run_time=0.6)

    def finish(self) -> None:
        t = getattr(getattr(self, "renderer", None), "time", None)
        remaining = TOTAL_SECONDS - t if t is not None else 1.0
        self.wait(max(0.5, remaining))

    # ---- Retrieval icin ortak R / E kolonlari ----
    def sets_view(self, y: float = 1.35) -> dict[str, Any]:
        r_head = self.chip("Getirilen (R) - Top 3", BLUE_B, w=3.6, fs=14, fill="#162030")
        r_chips = VGroup(
            *[
                self.chip(f"{i}. {c}", GREEN_B if c in E_LIST else RED_C, w=3.6)
                for i, c in enumerate(R_LIST, 1)
            ]
        ).arrange(DOWN, buff=0.1)
        r_col = VGroup(r_head, r_chips).arrange(DOWN, buff=0.14).move_to([-3.4, y, 0])

        e_head = self.chip("Ground Truth (E)", GOLD_B, w=3.6, fs=14, fill="#162030")
        e_chips = VGroup(
            *[
                self.chip(c + ("" if c in R_LIST else "  (eksik)"), GREEN_B if c in R_LIST else ORANGE, w=3.6)
                for c in E_LIST
            ]
        ).arrange(DOWN, buff=0.1)
        e_col = VGroup(e_head, e_chips).arrange(DOWN, buff=0.14).move_to([3.4, y, 0])
        e_col.align_to(r_col, UP)
        return {"r_col": r_col, "r_chips": r_chips, "e_col": e_col, "e_chips": e_chips}

    def show_sets(self, v: dict[str, Any]) -> None:
        self.play(FadeIn(v["r_col"], shift=RIGHT * 0.25), FadeIn(v["e_col"], shift=LEFT * 0.25), run_time=0.9)

    def link_match(self, v: dict[str, Any]) -> Line:
        """R'deki chunk_1 ile E'deki chunk_1 arasina kesisim cizgisi cizer."""
        link = Line(v["r_chips"][1].get_right(), v["e_chips"][0].get_left(), color=GREEN_B, stroke_width=3)
        self.play(
            Create(link),
            Indicate(v["r_chips"][1], color=GREEN_B),
            Indicate(v["e_chips"][0], color=GREEN_B),
            run_time=0.9,
        )
        return link


# =============================================================================
# 1) CONTEXT PRECISION
# =============================================================================
class ContextPrecisionScene(CalcScene):
    ACCENT = BLUE_B

    def construct(self) -> None:
        self.camera.background_color = BLACK
        self.header("Context Precision", "Getirilen parcalarin kaci gercekten ilgili?")
        v = self.sets_view()
        self.show_sets(v)
        link = self.link_match(v)

        c_inter = self.mlbl(r"|R\cap E|=1", GREEN_A).move_to([0, v["r_col"].get_bottom()[1] - 0.35, 0])
        c_r = self.mlbl(r"|R|=3\;\text{(getirilen toplam)}", BLUE_B).next_to(v["r_col"], DOWN, buff=0.65)
        self.play(FadeIn(c_inter, shift=UP * 0.1), run_time=0.5)
        self.play(FadeIn(c_r, shift=UP * 0.1), run_time=0.5)

        lines = self.reveal_lines(
            [
                (r"\text{Precision}=\frac{|R\cap E|}{|R|}", "Precision = |R n E| / |R|", WHITE),
                (r"=\frac{1}{3}", "= 1 / 3", WHITE),
                (rf"=\mathbf{{{PREC:.4f}}}", f"= {PREC:.4f}", YELLOW_B),
            ],
            y_top=-1.65,
            gap=0.75,
        )
        self.box_result(lines[-1])
        self.finish()


# =============================================================================
# 2) CONTEXT RECALL
# =============================================================================
class ContextRecallScene(CalcScene):
    ACCENT = TEAL_B

    def construct(self) -> None:
        self.camera.background_color = BLACK
        self.header("Context Recall", "Beklenen parcalarin kaci getirilebildi?")
        v = self.sets_view()
        self.show_sets(v)
        self.link_match(v)

        missing = self.lbl("chunk_3 getirilemedi -> kayip", ORANGE, fs=14).next_to(v["e_col"], DOWN, buff=0.35)
        c_e = self.mlbl(r"|E|=2\;\text{(beklenen toplam)}", GOLD_B).next_to(missing, DOWN, buff=0.15)
        c_inter = self.mlbl(r"|R\cap E|=1", GREEN_A).move_to([-3.4, v["r_col"].get_bottom()[1] - 0.45, 0])
        self.play(FadeIn(c_inter), FadeIn(missing, shift=UP * 0.1), run_time=0.6)
        self.play(FadeIn(c_e, shift=UP * 0.1), run_time=0.5)

        lines = self.reveal_lines(
            [
                (r"\text{Recall}=\frac{|R\cap E|}{|E|}", "Recall = |R n E| / |E|", WHITE),
                (r"=\frac{1}{2}", "= 1 / 2", WHITE),
                (rf"=\mathbf{{{REC:.4f}}}", f"= {REC:.4f}", YELLOW_B),
            ],
            y_top=-2.0,
            gap=0.72,
        )
        self.box_result(lines[-1])
        self.finish()


# =============================================================================
# 3) F1 SCORE
# =============================================================================
class F1ScoreScene(CalcScene):
    ACCENT = GREEN_B

    def construct(self) -> None:
        self.camera.background_color = BLACK
        self.header("F1 Score", "Precision ve Recall'un harmonik ortalamasi")

        p_card = self.mchip(rf"\text{{Precision}}=\frac{{1}}{{3}}={PREC:.4f}", BLUE_B, w=4.4, h=0.9, fs=28, fill="#162030")
        r_card = self.mchip(rf"\text{{Recall}}=\frac{{1}}{{2}}={REC:.4f}", TEAL_B, w=4.4, h=0.9, fs=28, fill="#162030")
        cards = VGroup(p_card, r_card).arrange(RIGHT, buff=0.7).move_to([0, 1.5, 0])
        self.play(FadeIn(cards, shift=DOWN * 0.15), run_time=0.9)
        self.wait(0.3)

        num = 2 * PREC * REC
        den = PREC + REC
        lines = self.reveal_lines(
            [
                (r"F_1=\frac{2\cdot P\cdot R}{P+R}", "F1 = 2*P*R / (P + R)", WHITE),
                (
                    r"=\frac{2\cdot 0.3333\cdot 0.5000}{0.3333+0.5000}",
                    "= 2*0.3333*0.5000 / (0.3333+0.5000)",
                    WHITE,
                ),
                (
                    rf"=\frac{{{num:.4f}}}{{{den:.4f}}}",
                    f"= {num:.4f} / {den:.4f}",
                    WHITE,
                ),
                (rf"=\mathbf{{{F1:.4f}}}", f"= {F1:.4f}", YELLOW_B),
            ],
            y_top=0.1,
            gap=0.95,
            run=0.8,
            hold=0.3,
        )
        self.box_result(lines[-1])
        self.finish()


# =============================================================================
# 4) HIT RATE
# =============================================================================
class HitRateScene(CalcScene):
    ACCENT = GOLD_B

    def construct(self) -> None:
        self.camera.background_color = BLACK
        self.header("Hit Rate", "Getirilenler arasinda en az bir dogru parca var mi?")
        v = self.sets_view()
        self.show_sets(v)

        # Sirayla R'deki parcalari kontrol et
        scan = SurroundingRectangle(v["r_chips"][0], color=YELLOW_B, buff=0.05)
        self.play(Create(scan), run_time=0.4)
        for i in range(1, N_R):
            self.play(scan.animate.move_to(v["r_chips"][i]), run_time=0.45)
            if R_LIST[i] in E_LIST:
                break
        self.link_match(v)

        res = self.mlbl(r"|R\cap E|=1>0\;\Rightarrow\;\text{Hit!}", GREEN_A, fs=30).move_to([0, -1.1, 0])
        self.play(FadeIn(res, scale=0.9), run_time=0.5)

        lines = self.reveal_lines(
            [
                (r"\text{HitRate}=\mathbb{I}(|R\cap E|>0)", "HitRate = I(|R n E| > 0)", WHITE),
                (r"=\mathbb{I}(1>0)", "= I(1 > 0)", WHITE),
                (r"=\mathbf{1}", "= 1", YELLOW_B),
            ],
            y_top=-1.9,
            gap=0.7,
        )
        self.box_result(lines[-1])
        self.finish()


# =============================================================================
# 5) MRR
# =============================================================================
class MRRScene(CalcScene):
    ACCENT = PURPLE_B

    def construct(self) -> None:
        self.camera.background_color = BLACK
        self.header("MRR - Mean Reciprocal Rank", "Ilk dogru parca kacinci sirada?")

        row = VGroup(
            *[
                VGroup(
                    self.chip(c, GREEN_B if c in E_LIST else RED_C, w=2.6, h=0.7, fs=16),
                    self.lbl(f"rank {i}", GREY_B, fs=14),
                ).arrange(DOWN, buff=0.12)
                for i, c in enumerate(R_LIST, 1)
            ]
        ).arrange(RIGHT, buff=0.5).move_to([0, 1.3, 0])
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in row], lag_ratio=0.2), run_time=1.1)

        # Soldan saga tarama: alakasiz -> gec, ilgili -> dur
        scan = SurroundingRectangle(row[0][0], color=YELLOW_B, buff=0.06)
        self.play(Create(scan), run_time=0.4)
        for i in range(1, FIRST_RANK):
            self.play(scan.animate.move_to(row[i][0]), run_time=0.5)
        self.play(Indicate(row[FIRST_RANK - 1][0], color=GREEN_B), run_time=0.7)
        found = self.mlbl(rf"\text{{Ilk dogru parca: }}\text{{rank}}_q={FIRST_RANK}", GREEN_A, fs=30).move_to([0, -0.25, 0])
        self.play(FadeIn(found, shift=UP * 0.1), run_time=0.5)

        lines = self.reveal_lines(
            [
                (r"\text{RR}_q=\frac{1}{\text{rank}_q}", "RR = 1 / rank", WHITE),
                (rf"=\frac{{1}}{{{FIRST_RANK}}}=\mathbf{{{RR:.4f}}}", f"= 1 / {FIRST_RANK} = {RR:.4f}", YELLOW_B),
                (
                    r"\text{MRR}=\frac{1}{|Q|}\sum_{q\in Q}\text{RR}_q=\frac{0.5}{1}=0.5",
                    "MRR = (1/|Q|) * sum(RR_q) = 0.5 / 1 = 0.5",
                    PURPLE_A,
                ),
            ],
            y_top=-1.15,
            gap=0.8,
            fs=28,
        )
        self.box_result(lines[-1], color=PURPLE_B)
        self.finish()


# =============================================================================
# 6) BINARY NDCG
# =============================================================================
class NDCGScene(CalcScene):
    ACCENT = ORANGE

    def construct(self) -> None:
        self.camera.background_color = BLACK
        self.header("Binary NDCG", "Dogru parcalar ne kadar ustte? (logaritmik indirim)")

        head = [r"i", r"\text{rel}_i", r"\log_2(i+1)", r"\frac{\text{rel}_i}{\log_2(i+1)}"]
        cells = [self.mlbl(h, ORANGE, fs=26) for h in head]
        for i in range(N_R):
            cells += [
                self.lbl(str(i + 1), WHITE, 15),
                self.lbl(str(REL[i]), GREEN_B if REL[i] else RED_C, 15),
                self.lbl(f"{math.log2(i + 2):.3f}", WHITE, 15),
                self.lbl(f"{REL[i] * DISC[i]:.4f}", YELLOW_B if REL[i] else GREY_B, 15),
            ]
        table = VGroup(*cells).arrange_in_grid(rows=N_R + 1, cols=4, buff=(0.9, 0.2)).move_to([0.5, 1.45, 0])
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.04), run_time=1.6)
        self.play(Indicate(VGroup(*cells[8:12]), color=GREEN_B), run_time=0.7)

        d_terms = " + ".join(f"{r * d:.4f}" for r, d in zip(REL, DISC))
        i_terms = " + ".join(f"{r * d:.4f}" for r, d in zip(IDEAL, DISC))
        lines = self.reveal_lines(
            [
                (rf"\text{{DCG}}={d_terms}=\mathbf{{{DCG:.4f}}}", f"DCG = {d_terms} = {DCG:.4f}", WHITE),
                (
                    rf"\text{{IDCG}}\;(\text{{ideal: }}[1,1,0])={i_terms}=\mathbf{{{IDCG:.4f}}}",
                    f"IDCG (ideal [1,1,0]) = {i_terms} = {IDCG:.4f}",
                    GOLD_A,
                ),
                (
                    rf"\text{{NDCG}}=\frac{{{DCG:.4f}}}{{{IDCG:.4f}}}=\mathbf{{{NDCG:.4f}}}",
                    f"NDCG = {DCG:.4f} / {IDCG:.4f} = {NDCG:.4f}",
                    YELLOW_B,
                ),
            ],
            y_top=-0.55,
            gap=0.85,
            run=0.8,
            hold=0.3,
            fs=26,
        )
        self.box_result(lines[-1], color=ORANGE)
        self.finish()


# =============================================================================
# 7) FAITHFULNESS
# =============================================================================
class FaithfulnessScene(CalcScene):
    SECTION = "Generation Evaluation"
    ACCENT = GREEN_B

    def construct(self) -> None:
        self.camera.background_color = BLACK
        self.header("Faithfulness (Sadakat)", "Cevaptaki iddialarin kaynakca desteklenme orani")

        ctx = self.chip("Kaynak: RAG uc asamadan olusur: Retrieval, Augmentation, Generation.", TEAL_B, w=6.0, h=0.95, fs=13, fill="#162030")
        ctx.move_to([3.6, 1.2, 0])
        claims_spec = [
            ("RAG uc asamadan olusur", True),
            ("Ilk asama Retrieval'dir", True),
            ("Son asama Generation'dir", True),
            ("Sistem 2015'te gelistirildi", False),
        ]
        claim_rows = VGroup()
        for txt, ok in claims_spec:
            mark = self.lbl("OK" if ok else "X", GREEN_B if ok else RED_C, fs=15)
            chip = self.chip(txt, GREEN_B if ok else RED_C, w=4.2, h=0.46)
            claim_rows.add(VGroup(chip, mark).arrange(RIGHT, buff=0.2))
        claim_rows.arrange(DOWN, buff=0.12).move_to([-3.2, 1.0, 0])
        c_head = self.lbl("Cevaptan cikarilan iddialar (claims)", BLUE_B, fs=14).next_to(claim_rows, UP, buff=0.2)

        self.play(FadeIn(c_head), FadeIn(ctx, shift=LEFT * 0.2), run_time=0.8)
        for row, (_, ok) in zip(claim_rows, claims_spec):
            self.play(FadeIn(row[0], shift=RIGHT * 0.15), run_time=0.35)
            self.play(FadeIn(row[1], scale=1.3), run_time=0.25)

        cnt = self.mlbl(rf"|C_{{\text{{supported}}}}|={FAITH_SUP}\qquad|C_{{\text{{total}}}}|={FAITH_TOT}", YELLOW_B, fs=30).move_to([0, -0.9, 0])
        self.play(FadeIn(cnt, shift=UP * 0.1), run_time=0.5)

        lines = self.reveal_lines(
            [
                (
                    r"\text{Faithfulness}=\frac{|C_{\text{supported}}|}{|C_{\text{total}}|}",
                    "Faithfulness = C_supported / C_total",
                    WHITE,
                ),
                (rf"=\frac{{{FAITH_SUP}}}{{{FAITH_TOT}}}=\mathbf{{{FAITH:.4f}}}", f"= {FAITH_SUP} / {FAITH_TOT} = {FAITH:.4f}", YELLOW_B),
            ],
            y_top=-1.7,
            gap=0.85,
        )
        self.box_result(lines[-1])
        self.finish()


# =============================================================================
# 8) ANSWER RELEVANCY
# =============================================================================
class AnswerRelevancyScene(CalcScene):
    SECTION = "Generation Evaluation"
    ACCENT = BLUE_B

    def construct(self) -> None:
        self.camera.background_color = BLACK
        self.header("Answer Relevancy (Ilgililik)", "Cevaptan uretilen sorular asil soruya ne kadar benziyor?")

        q = self.chip("Asil soru q: RAG'in temel asamalari nelerdir?", BLUE_B, w=7.0, h=0.5, fs=14, fill="#162030")
        q.move_to([0, 2.05, 0])
        self.play(FadeIn(q, shift=DOWN * 0.15), run_time=0.7)

        gen_rows = VGroup()
        for i, (s, r) in enumerate(REL_SIMS, 1):
            col = GREEN_B if r else RED_C
            gq = self.chip(f"q{i} (cevaptan uretilen soru)", col, w=3.6, h=0.44, fs=13)
            sim = self.mlbl(rf"\text{{sim}}(q,q_{i})={s:.2f}", WHITE, fs=24)
            flag = self.mlbl(rf"\text{{rel}}_{i}={r}", col, fs=24)
            gen_rows.add(VGroup(gq, sim, flag).arrange(RIGHT, buff=0.5))
        gen_rows.arrange(DOWN, buff=0.14).move_to([0, 0.75, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.12) for r in gen_rows], lag_ratio=0.3), run_time=1.5)
        note = self.lbl("rel = 0 -> kacamak/alakasiz soru, katkisi 0", ORANGE, fs=13).next_to(gen_rows, DOWN, buff=0.15)
        self.play(FadeIn(note), run_time=0.4)

        terms = " + ".join(f"{r}\\cdot{s:.2f}" for s, r in REL_SIMS)
        terms_fb = " + ".join(f"{r}*{s:.2f}" for s, r in REL_SIMS)
        tot = sum(r * s for s, r in REL_SIMS)
        lines = self.reveal_lines(
            [
                (
                    r"\text{Relevancy}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(\text{rel}_i)\cdot\max(0,\text{sim}(q,q_i))",
                    "Relevancy = (1/N) * sum(I(rel_i) * max(0, sim))",
                    WHITE,
                ),
                (rf"=\frac{{1}}{{3}}\left({terms}\right)", f"= (1/3) * ({terms_fb})", WHITE),
                (rf"=\frac{{{tot:.2f}}}{{3}}=\mathbf{{{RELEVANCY:.4f}}}", f"= {tot:.2f} / 3 = {RELEVANCY:.4f}", YELLOW_B),
            ],
            y_top=-1.35,
            gap=0.8,
            run=0.7,
            hold=0.25,
            fs=26,
        )
        self.box_result(lines[-1])
        self.finish()


# =============================================================================
# 9) ANSWER QUALITY - TP / FP / FN -> Completeness, Accuracy
# =============================================================================
class AnswerQualityFactualScene(CalcScene):
    SECTION = "Generation Evaluation"
    ACCENT = GOLD_B

    def construct(self) -> None:
        self.camera.background_color = BLACK
        self.header("Answer Quality: Olgusal F1", "Uretilen cevap vs beklenen cevap -> TP / FP / FN")

        tp = self.mchip(rf"TP={TP}\;\text{{(dogru ifade)}}", GREEN_B, w=3.6, h=0.7, fs=26, fill="#12241B")
        fp = self.mchip(rf"FP={FP}\;\text{{(fazla / hatali)}}", RED_C, w=3.6, h=0.7, fs=26, fill="#2A1618")
        fn = self.mchip(rf"FN={FN}\;\text{{(eksik ifade)}}", ORANGE, w=3.6, h=0.7, fs=26, fill="#2A2012")
        trio = VGroup(tp, fp, fn).arrange(RIGHT, buff=0.35).move_to([0, 1.7, 0])
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.15) for c in trio], lag_ratio=0.25), run_time=1.5)
        self.wait(0.2)

        l1 = self.reveal_lines(
            [
                (
                    rf"\text{{Precision}}=\frac{{TP}}{{TP+FP}}=\frac{{{TP}}}{{{TP}+{FP}}}={Q_PREC:.4f}",
                    f"Precision = TP/(TP+FP) = {TP}/{TP + FP} = {Q_PREC:.4f}",
                    BLUE_B,
                ),
                (
                    rf"\text{{Recall (Completeness)}}=\frac{{TP}}{{TP+FN}}=\frac{{{TP}}}{{{TP}+{FN}}}={Q_REC:.4f}",
                    f"Recall = TP/(TP+FN) = {TP}/{TP + FN} = {Q_REC:.4f}",
                    TEAL_B,
                ),
                (
                    rf"F_{{1,\text{{factual}}}}=\frac{{2\cdot{Q_PREC:.2f}\cdot{Q_REC:.2f}}}{{{Q_PREC:.2f}+{Q_REC:.2f}}}=\mathbf{{{Q_F1:.4f}}}",
                    f"F1_factual = 2*{Q_PREC:.2f}*{Q_REC:.2f} / ({Q_PREC:.2f}+{Q_REC:.2f}) = {Q_F1:.4f}",
                    YELLOW_B,
                ),
            ],
            y_top=0.35,
            gap=1.0,
            run=0.9,
            hold=0.35,
            fs=27,
        )
        self.box_result(l1[-1])
        note = self.mlbl(r"\text{Accuracy}=F_1\qquad\text{Completeness}=\text{Recall}", GREY_B, fs=26).move_to([0, -3.1, 0])
        self.play(FadeIn(note), run_time=0.4)
        self.finish()


# =============================================================================
# 10) ANSWER CORRECTNESS (agirlikli toplam)
# =============================================================================
class AnswerCorrectnessScene(CalcScene):
    SECTION = "Generation Evaluation"
    ACCENT = GOLD_B

    def construct(self) -> None:
        self.camera.background_color = BLACK
        self.header("Answer Correctness", "Olgusal F1 (0.75) + anlamsal benzerlik (0.25)")

        inputs = VGroup(
            self.mchip(rf"F_{{1,\text{{factual}}}}={Q_F1:.4f}", GOLD_B, w=4.0, h=0.7, fs=28, fill="#162030"),
            self.mchip(rf"\text{{sim}}(a,a_{{\text{{exp}}}})={SIM_A:.2f}", TEAL_B, w=4.4, h=0.7, fs=28, fill="#162030"),
        ).arrange(RIGHT, buff=0.5).move_to([0, 1.9, 0])
        self.play(FadeIn(inputs, shift=DOWN * 0.15), run_time=0.8)

        l = self.reveal_lines(
            [
                (
                    r"\text{Correctness}=w_f\cdot F_{1,\text{factual}}+w_s\cdot\text{sim}(a,a_{\text{exp}})",
                    "Correctness = w_f*F1 + w_s*sim",
                    WHITE,
                ),
                (
                    rf"=0.75\cdot{Q_F1:.4f}+0.25\cdot{SIM_A:.2f}",
                    f"= 0.75*{Q_F1:.4f} + 0.25*{SIM_A:.2f}",
                    WHITE,
                ),
                (
                    rf"={W_F * Q_F1:.4f}+{W_S * SIM_A:.4f}=\mathbf{{{CORRECT:.4f}}}",
                    f"= {W_F * Q_F1:.4f} + {W_S * SIM_A:.4f} = {CORRECT:.4f}",
                    YELLOW_B,
                ),
            ],
            y_top=0.65,
            gap=0.85,
            run=0.8,
            hold=0.3,
            fs=27,
        )
        self.box_result(l[-1])

        # Yigilmis cubuk: 0..1 ekseni uzerinde iki katki
        W = 9.0
        x0 = -W / 2
        axis = Line([x0, -2.55, 0], [x0 + W, -2.55, 0], color=GREY_B)
        f_len, s_len = W * W_F * Q_F1, W * W_S * SIM_A
        bar_f = Rectangle(width=f_len, height=0.35, fill_color=BLUE_B, fill_opacity=0.9, stroke_width=0)
        bar_f.move_to([x0 + f_len / 2, -2.55 + 0.3, 0])
        bar_s = Rectangle(width=s_len, height=0.35, fill_color=TEAL_B, fill_opacity=0.9, stroke_width=0)
        bar_s.move_to([x0 + f_len + s_len / 2, -2.55 + 0.3, 0])
        t0 = self.lbl("0", GREY_B, 12).next_to(axis.get_start(), DOWN, buff=0.1)
        t1 = self.lbl("1", GREY_B, 12).next_to(axis.get_end(), DOWN, buff=0.1)
        lf = self.lbl(f"{W_F * Q_F1:.2f}", WHITE, 12).move_to(bar_f)
        ls = self.lbl(f"{W_S * SIM_A:.2f}", WHITE, 12).move_to(bar_s)
        self.play(Create(axis), FadeIn(t0), FadeIn(t1), run_time=0.4)
        self.play(GrowFromEdge(bar_f, LEFT), FadeIn(lf), run_time=0.6)
        self.play(GrowFromEdge(bar_s, LEFT), FadeIn(ls), run_time=0.6)
        self.finish()


# =============================================================================
# 11) NOISE SENSITIVITY
# =============================================================================
class NoiseSensitivityScene(CalcScene):
    SECTION = "Generation Evaluation"
    ACCENT = ORANGE

    def construct(self) -> None:
        self.camera.background_color = BLACK
        self.header("Noise Sensitivity", "Alakasiz chunk eklenince kalite ne kadar dustu?")

        clean = VGroup(
            self.lbl("Temiz bağlam", GREEN_B, 15),
            self.chip("chunk_1 (ilgili)", GREEN_B, w=3.2, h=0.4, fs=13),
            self.mchip(rf"Q_{{\text{{clean}}}}={Q_CLEAN:.4f}", GREEN_B, w=3.2, h=0.55, fs=26, fill="#12241B"),
        ).arrange(DOWN, buff=0.12)
        noisy = VGroup(
            self.lbl("Gürültülü bağlam", RED_C, 15),
            VGroup(
                self.chip("chunk_1 (ilgili)", GREEN_B, w=3.2, h=0.4, fs=13),
                self.chip("noise_a (alakasiz)", RED_C, w=3.2, h=0.4, fs=13),
                self.chip("noise_b (alakasiz)", RED_C, w=3.2, h=0.4, fs=13),
            ).arrange(DOWN, buff=0.07),
            self.mchip(rf"Q_{{\text{{noisy}}}}={Q_NOISY:.4f}", RED_C, w=3.2, h=0.55, fs=26, fill="#2A1618"),
        ).arrange(DOWN, buff=0.12)
        both = VGroup(clean, noisy).arrange(RIGHT, buff=1.2, aligned_edge=UP).move_to([0, 1.35, 0])
        self.play(FadeIn(clean, shift=RIGHT * 0.2), run_time=0.8)
        self.play(FadeIn(noisy, shift=LEFT * 0.2), run_time=0.8)

        qc_t = f"\\frac{{{CORRECT:.2f}+{Q_REC:.2f}+{Q_F1:.2f}}}{{3}}"
        qn_t = f"\\frac{{{NOISY[0]:.2f}+{NOISY[1]:.2f}+{NOISY[2]:.2f}}}{{3}}"
        lines = self.reveal_lines(
            [
                (
                    rf"Q_{{\text{{clean}}}}={qc_t}={Q_CLEAN:.4f}\qquad Q_{{\text{{noisy}}}}={qn_t}={Q_NOISY:.4f}",
                    f"Q_clean = ({CORRECT:.2f}+{Q_REC:.2f}+{Q_F1:.2f})/3 = {Q_CLEAN:.4f}   "
                    f"Q_noisy = ({NOISY[0]:.2f}+{NOISY[1]:.2f}+{NOISY[2]:.2f})/3 = {Q_NOISY:.4f}",
                    WHITE,
                ),
                (
                    r"\text{NS}=\max\!\left(0,\min\!\left(1,\frac{Q_{\text{clean}}-Q_{\text{noisy}}}{Q_{\text{clean}}}\right)\right)",
                    "NS = clip((Q_clean - Q_noisy) / Q_clean, 0, 1)",
                    WHITE,
                ),
                (
                    rf"=\frac{{{Q_CLEAN:.4f}-{Q_NOISY:.4f}}}{{{Q_CLEAN:.4f}}}=\mathbf{{{NOISE_SENS:.4f}}}",
                    f"= ({Q_CLEAN:.4f} - {Q_NOISY:.4f}) / {Q_CLEAN:.4f} = {NOISE_SENS:.4f}",
                    YELLOW_B,
                ),
            ],
            y_top=-0.75,
            gap=0.9,
            run=0.8,
            hold=0.3,
            fs=25,
        )
        self.box_result(lines[-1], color=ORANGE)
        note = self.lbl("Dusuk = iyi (sistem gurultuye dayanikli)", GREY_B, fs=14).move_to([0, -3.35, 0])
        self.play(FadeIn(note), run_time=0.4)
        self.finish()