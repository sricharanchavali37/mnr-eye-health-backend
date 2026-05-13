import os
import json
from datetime import datetime, timezone
from typing import Optional

REPORTS_DIR = "generated_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_report_pdf(
    report_id: str,
    user_name: str,
    user_email: str,
    screening_data: dict,
    risk_score: float,
    risk_level: str,
    flags: list,
    ai_summary: str,
    recommendations: list,
    urgent_referral: bool,
    referral_reason: Optional[str] = None,
) -> str:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

        BRAND_BLUE = HexColor("#1A6B9A")
        BRAND_TEAL = HexColor("#0EA5E9")
        BRAND_GREEN = HexColor("#16A34A")
        BRAND_ORANGE = HexColor("#EA580C")
        BRAND_RED = HexColor("#DC2626")
        BRAND_LIGHT = HexColor("#F0F9FF")
        GRAY = HexColor("#6B7280")
        DARK = HexColor("#1F2937")

        RISK_COLORS = {
            "low": BRAND_GREEN,
            "moderate": HexColor("#D97706"),
            "high": BRAND_ORANGE,
            "critical": BRAND_RED,
        }
        RISK_LABELS = {
            "low": "LOW RISK",
            "moderate": "MODERATE RISK",
            "high": "HIGH RISK",
            "critical": "CRITICAL RISK",
        }

        filename = f"{REPORTS_DIR}/mnr_report_{report_id}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=A4,
                                topMargin=1.5*cm, bottomMargin=2*cm,
                                leftMargin=2*cm, rightMargin=2*cm)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle("BrandTitle", fontSize=22, textColor=BRAND_BLUE,
                                  fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4))
        styles.add(ParagraphStyle("SubTitle", fontSize=11, textColor=GRAY,
                                  fontName="Helvetica", alignment=TA_CENTER, spaceAfter=2))
        styles.add(ParagraphStyle("SectionHead", fontSize=13, textColor=BRAND_BLUE,
                                  fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6))
        styles.add(ParagraphStyle("BodyText2", fontSize=10, textColor=DARK,
                                  fontName="Helvetica", leading=16, spaceAfter=4, alignment=TA_JUSTIFY))
        styles.add(ParagraphStyle("BulletItem", fontSize=10, textColor=DARK,
                                  fontName="Helvetica", leading=16, leftIndent=14,
                                  bulletIndent=4, spaceAfter=3))
        styles.add(ParagraphStyle("Footer", fontSize=8, textColor=GRAY,
                                  fontName="Helvetica", alignment=TA_CENTER))

        story = []
        story.append(Paragraph("MNR Eye Health Platform", styles["BrandTitle"]))
        story.append(Paragraph("AI-Powered Digital Vision Screening Report", styles["SubTitle"]))
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(width="100%", thickness=2, color=BRAND_TEAL))
        story.append(Spacer(1, 0.4*cm))

        generated_at = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
        info_data = [
            ["Patient Name", user_name, "Report ID", report_id[:8].upper()],
            ["Email", user_email, "Generated", generated_at],
        ]
        info_table = Table(info_data, colWidths=[3.5*cm, 7*cm, 3*cm, 5*cm])
        info_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, -1), DARK),
            ("TEXTCOLOR", (0, 0), (0, -1), BRAND_BLUE),
            ("TEXTCOLOR", (2, 0), (2, -1), BRAND_BLUE),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BRAND_LIGHT, white]),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#E5E7EB")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.5*cm))

        risk_color = RISK_COLORS.get(risk_level, GRAY)
        risk_label = RISK_LABELS.get(risk_level, risk_level.upper())
        risk_table = Table([[f"Risk Score: {risk_score}/100", risk_label]], colWidths=[9*cm, 8.5*cm])
        risk_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), risk_color),
            ("TEXTCOLOR", (0, 0), (-1, -1), white),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (0, 0), 16),
            ("FONTSIZE", (1, 0), (1, 0), 14),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 14),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 0.5*cm))

        story.append(Paragraph("Health Summary", styles["SectionHead"]))
        story.append(Paragraph(ai_summary or "No summary available.", styles["BodyText2"]))
        story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph("Screening Results", styles["SectionHead"]))
        def fmt(val, unit=""):
            return f"{val}{unit}" if val is not None else "Not tested"

        metrics = [
            ["Metric", "Left Eye", "Right Eye"],
            ["Visual Acuity", fmt(screening_data.get("visual_acuity_left")), fmt(screening_data.get("visual_acuity_right"))],
            ["Color Vision", fmt(screening_data.get("color_vision_score"), "/100"), "—"],
            ["Contrast Sensitivity", fmt(screening_data.get("contrast_sensitivity")), "—"],
            ["Field of View", fmt(screening_data.get("field_of_view_score"), "/100"), "—"],
        ]
        metrics_table = Table(metrics, colWidths=[7*cm, 5*cm, 5.5*cm])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, BRAND_LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#E5E7EB")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.4*cm))

        if flags:
            story.append(Paragraph("Clinical Flags", styles["SectionHead"]))
            for flag in flags:
                story.append(Paragraph(f"• {flag}", styles["BulletItem"]))
            story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph("Personalized Recommendations", styles["SectionHead"]))
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles["BulletItem"]))
        story.append(Spacer(1, 0.4*cm))

        if urgent_referral:
            referral_box = Table([[f"SPECIALIST REFERRAL RECOMMENDED\n{referral_reason or ''}"]], colWidths=[17.5*cm])
            referral_box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FEF2F2")),
                ("TEXTCOLOR", (0, 0), (-1, -1), BRAND_RED),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("PADDING", (0, 0), (-1, -1), 12),
                ("BOX", (0, 0), (-1, -1), 1.5, BRAND_RED),
            ]))
            story.append(referral_box)
            story.append(Spacer(1, 0.4*cm))

        story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#E5E7EB")))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            "This report is generated by MNR Eye Health Platform for informational purposes only. "
            "It is not a substitute for professional medical advice, diagnosis, or treatment.",
            styles["Footer"]
        ))
        story.append(Paragraph("© MNR Eye Health Platform | odos.health", styles["Footer"]))

        doc.build(story)
        return filename

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"PDF generation failed: {e}")
        raise
