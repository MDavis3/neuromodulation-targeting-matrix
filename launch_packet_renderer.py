from __future__ import annotations

import argparse
import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


LOGGER = logging.getLogger("neuromodulation_targeting_matrix.launch_packet_renderer")

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_PACKET_DIR = PROCESSED_DIR / "launch_packet"
DEFAULT_SUMMARY = DEFAULT_PACKET_DIR / "executive_summary.md"
DEFAULT_BRIEFS_DIR = DEFAULT_PACKET_DIR / "site_briefs"
DEFAULT_LEDGER = DEFAULT_PACKET_DIR / "launch_priority_ledger.csv"
DEFAULT_LAUNCH_CHART = DEFAULT_PACKET_DIR / "top_launch_sites.png"
DEFAULT_DRG_CHART = PROCESSED_DIR / "drg_economics_visual.png"
DEFAULT_OUTPUT_PDF = DEFAULT_PACKET_DIR / "executive_launch_packet.pdf"
DEFAULT_EMAIL_TXT = DEFAULT_PACKET_DIR / "launch_packet_email.txt"
DEFAULT_MANIFEST_TXT = DEFAULT_PACKET_DIR / "attachment_manifest.txt"

PAGE_BG = colors.HexColor("#1e1e1e")
TEXT = colors.HexColor("#f5f5f5")
MUTED = colors.HexColor("#cfcfcf")
CYAN = colors.HexColor("#00f0ff")
GRID = colors.HexColor("#4b4b4b")
CELL_BG = colors.HexColor("#252526")


@dataclass(frozen=True)
class PacketPaths:
    summary: Path
    briefs_dir: Path
    ledger: Path
    launch_chart: Path
    drg_chart: Path
    output_pdf: Path
    output_email: Path
    output_manifest: Path


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the launch selection outputs into a polished executive packet."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--briefs-dir", type=Path, default=DEFAULT_BRIEFS_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--launch-chart", type=Path, default=DEFAULT_LAUNCH_CHART)
    parser.add_argument("--drg-chart", type=Path, default=DEFAULT_DRG_CHART)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-email", type=Path, default=DEFAULT_EMAIL_TXT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_MANIFEST_TXT)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_styles() -> StyleSheet1:
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="PacketTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=TEXT,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PacketSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=MUTED,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PacketHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=CYAN,
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PacketSubheading",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=TEXT,
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PacketBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=TEXT,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PacketBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=TEXT,
            leftIndent=14,
            firstLineIndent=0,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PacketNumbered",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=TEXT,
            leftIndent=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PacketSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            spaceAfter=4,
        )
    )
    return styles


def add_page_background(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFillColor(PAGE_BG)
    canvas.rect(0, 0, LETTER[0], LETTER[1], stroke=0, fill=1)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(doc.leftMargin, 18, "Clinical Launch Packet")
    canvas.drawRightString(LETTER[0] - doc.rightMargin, 18, f"Page {doc.page}")
    canvas.restoreState()


def escape_text(text: str) -> str:
    return html.escape(text, quote=False)


def markdown_lines_to_flowables(text: str, styles: StyleSheet1) -> list:
    story: list = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            story.append(Spacer(1, 0.08 * inch))
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(escape_text(stripped[2:]), styles["PacketHeading"]))
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(escape_text(stripped[3:]), styles["PacketHeading"]))
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(escape_text(stripped[4:]), styles["PacketSubheading"]))
            continue

        if re.match(r"^\d+\.\s", stripped):
            story.append(Paragraph(escape_text(stripped), styles["PacketNumbered"]))
            continue

        if stripped.startswith("- "):
            story.append(Paragraph(f"&bull; {escape_text(stripped[2:])}", styles["PacketBullet"]))
            continue

        story.append(Paragraph(escape_text(stripped), styles["PacketBody"]))

    return story


def image_flowable(path: Path, max_width: float, max_height: float) -> Image:
    reader = ImageReader(str(path))
    width, height = reader.getSize()
    scale = min(max_width / width, max_height / height)
    image = Image(str(path), width=width * scale, height=height * scale)
    image.hAlign = "CENTER"
    return image


def build_top_sites_table(ledger: pl.DataFrame, top_n: int) -> Table:
    top_sites = ledger.head(top_n).select(
        [
            "Surgeon_Name",
            "Surgeon_City",
            "Launch_Priority_Score",
            "Launch_Wave",
            "Dyad_Partner_Name",
            "Net_Sourcing_Alpha",
        ]
    )

    rows = [
        [
            "Rank",
            "Lead Surgeon",
            "City",
            "Score",
            "Wave",
            "Dyad Partner",
            "Net Alpha",
        ]
    ]

    for index, row in enumerate(top_sites.to_dicts(), start=1):
        rows.append(
            [
                str(index),
                str(row["Surgeon_Name"]),
                str(row["Surgeon_City"]),
                f"{float(row['Launch_Priority_Score']):.2f}",
                str(row["Launch_Wave"]),
                str(row["Dyad_Partner_Name"]),
                f"{float(row['Net_Sourcing_Alpha']):,.0f}",
            ]
        )

    table = Table(
        rows,
        colWidths=[0.45 * inch, 1.75 * inch, 1.1 * inch, 0.75 * inch, 0.7 * inch, 1.65 * inch, 0.9 * inch],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), CYAN),
                ("TEXTCOLOR", (0, 0), (-1, 0), PAGE_BG),
                ("BACKGROUND", (0, 1), (-1, -1), CELL_BG),
                ("TEXTCOLOR", (0, 1), (-1, -1), TEXT),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_metric_table(ledger: pl.DataFrame, styles: StyleSheet1) -> Table:
    top_site = ledger.row(0, named=True)
    metrics = [
        ("Immediate outreach site", str(top_site["Surgeon_Name"])),
        ("Top launch score", f"{float(top_site['Launch_Priority_Score']):.2f}"),
        ("Best net alpha", f"{float(ledger['Net_Sourcing_Alpha'].max()):,.0f}"),
        ("Projected per-case profit uplift", f"{float(top_site['Projected_Profit_Uplift_Pct']):.2f}%"),
    ]

    rows = [[Paragraph(escape_text(label), styles["PacketSmall"]), Paragraph(escape_text(value), styles["PacketBody"])] for label, value in metrics]
    table = Table(rows, colWidths=[2.15 * inch, 4.65 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CELL_BG),
                ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def write_email_text(packet_paths: PacketPaths, ledger: pl.DataFrame, top_n: int) -> None:
    packet_paths.output_email.parent.mkdir(parents=True, exist_ok=True)
    top_sites = ledger.head(top_n).to_dicts()
    subject = "Clinical launch packet for next-gen wireless BCI site selection"

    lines = [
        f"Subject: {subject}",
        "",
        "Hi [Name],",
        "",
        "I built a launch-selection packet for a next-gen wireless BCI program that ranks initial clinical sites using three integrated lenses:",
        "- implant/referral dyad viability",
        "- competition-adjusted patient catchment",
        "- DRG-level hospital economics",
        "",
        "Attached are:",
        f"- {packet_paths.output_pdf.name}",
        f"- {packet_paths.ledger.name}",
        f"- {packet_paths.launch_chart.name}",
        "",
        "Top recommended sites:",
    ]

    for index, site in enumerate(top_sites, start=1):
        lines.append(
            f"- {index}. {site['Surgeon_Name']} | {site['Surgeon_City']}, {site['Surgeon_State']} "
            f"(score {site['Launch_Priority_Score']}, alpha {site['Net_Sourcing_Alpha']:.0f})"
        )

    lines.extend(
        [
            "",
            "The packet is designed to answer one practical question: which sites should launch first, and why will they recruit, withstand competitor overlap, and make financial sense for the hospital.",
            "",
            "Best,",
            "[Your Name]",
        ]
    )
    packet_paths.output_email.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote email draft to %s", packet_paths.output_email)


def write_attachment_manifest(packet_paths: PacketPaths) -> None:
    lines = [
        "Launch Packet Attachment Manifest",
        "",
        f"Primary packet: {packet_paths.output_pdf.name}",
        f"Ranked ledger: {packet_paths.ledger.name}",
        f"Launch chart: {packet_paths.launch_chart.name}",
        f"DRG economics chart: {packet_paths.drg_chart.name}",
        f"Email draft: {packet_paths.output_email.name}",
    ]
    packet_paths.output_manifest.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote attachment manifest to %s", packet_paths.output_manifest)


def build_packet_story(packet_paths: PacketPaths, ledger: pl.DataFrame, top_n: int) -> list:
    styles = build_styles()
    story: list = []
    timestamp = datetime.now().strftime("%B %d, %Y")

    story.extend(
        [
            Spacer(1, 0.35 * inch),
            Paragraph("Clinical Launch Packet", styles["PacketTitle"]),
            Paragraph(
                "Integrated site selection for a next-gen wireless BCI program",
                styles["PacketSubtitle"],
            ),
            Paragraph(
                f"Generated {timestamp}. This packet condenses site selection into one operating view: who can implant, who can refer, where the patient funnel exists after competitor overlap, and why the hospital economics work.",
                styles["PacketBody"],
            ),
            Spacer(1, 0.12 * inch),
            build_metric_table(ledger, styles),
            Spacer(1, 0.18 * inch),
            HRFlowable(width="100%", color=GRID, thickness=0.8),
            Spacer(1, 0.12 * inch),
            Paragraph("Top Launch Sites", styles["PacketHeading"]),
            build_top_sites_table(ledger, top_n=top_n),
            Spacer(1, 0.2 * inch),
        ]
    )

    if packet_paths.launch_chart.exists():
        story.extend(
            [
                Paragraph("Launch Ranking Visual", styles["PacketSubheading"]),
                image_flowable(packet_paths.launch_chart, max_width=6.9 * inch, max_height=4.8 * inch),
                Spacer(1, 0.16 * inch),
            ]
        )

    if packet_paths.drg_chart.exists():
        story.extend(
            [
                Paragraph("Hospital Economics Visual", styles["PacketSubheading"]),
                image_flowable(packet_paths.drg_chart, max_width=6.9 * inch, max_height=4.6 * inch),
                Spacer(1, 0.16 * inch),
            ]
        )

    story.append(PageBreak())
    summary_text = packet_paths.summary.read_text(encoding="utf-8")
    story.append(Paragraph("Executive Summary", styles["PacketTitle"]))
    story.extend(markdown_lines_to_flowables(summary_text, styles))

    for brief_path in sorted(packet_paths.briefs_dir.glob("*.md"))[:top_n]:
        story.append(PageBreak())
        brief_text = brief_path.read_text(encoding="utf-8")
        story.extend(markdown_lines_to_flowables(brief_text, styles))

    return story


def render_launch_packet(packet_paths: PacketPaths, top_n: int) -> tuple[Path, Path, Path]:
    packet_paths.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    ledger = pl.read_csv(packet_paths.ledger)

    doc = SimpleDocTemplate(
        str(packet_paths.output_pdf),
        pagesize=LETTER,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.45 * inch,
        title="Clinical Launch Packet",
        author="Neuromodulation Targeting Matrix",
    )
    story = build_packet_story(packet_paths, ledger, top_n=top_n)
    doc.build(story, onFirstPage=add_page_background, onLaterPages=add_page_background)
    LOGGER.info("Wrote executive PDF packet to %s", packet_paths.output_pdf)

    write_email_text(packet_paths, ledger, top_n=top_n)
    write_attachment_manifest(packet_paths)
    return packet_paths.output_pdf, packet_paths.output_email, packet_paths.output_manifest


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    packet_paths = PacketPaths(
        summary=args.summary,
        briefs_dir=args.briefs_dir,
        ledger=args.ledger,
        launch_chart=args.launch_chart,
        drg_chart=args.drg_chart,
        output_pdf=args.output_pdf,
        output_email=args.output_email,
        output_manifest=args.output_manifest,
    )
    outputs = render_launch_packet(packet_paths, top_n=args.top_n)
    LOGGER.info("Packet render complete: %s", outputs)


if __name__ == "__main__":
    main()
