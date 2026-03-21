from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import colors
import polars as pl


LOGGER = logging.getLogger("neuromodulation_targeting_matrix.launch_priority_engine")

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_TOP_TARGETS = PROCESSED_DIR / "clinical_top_targets.csv"
DEFAULT_DYAD_LEDGER = PROCESSED_DIR / "clinical_dyad_ledger.csv"
DEFAULT_CATCHMENT_LEDGER = PROCESSED_DIR / "viable_catchment_areas.csv"
DEFAULT_DRG_LEDGER = PROCESSED_DIR / "hospital_drg_economics.csv"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "launch_packet"


@dataclass(frozen=True)
class SourcePaths:
    top_targets: Path
    dyad_ledger: Path
    catchment_ledger: Path
    drg_ledger: Path
    output_dir: Path


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an executive-ready site launch package from the core clinical, catchment, and economics ledgers."
    )
    parser.add_argument("--top-targets", type=Path, default=DEFAULT_TOP_TARGETS)
    parser.add_argument("--dyad-ledger", type=Path, default=DEFAULT_DYAD_LEDGER)
    parser.add_argument("--catchment-ledger", type=Path, default=DEFAULT_CATCHMENT_LEDGER)
    parser.add_argument("--drg-ledger", type=Path, default=DEFAULT_DRG_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        infer_schema_length=5000,
        try_parse_dates=True,
        ignore_errors=True,
        encoding="utf8-lossy",
        null_values=["", "NULL", "null", "N/A", "n/a"],
    )


def ensure_columns(frame: pl.LazyFrame, required: Iterable[str], source_name: str) -> None:
    available = set(frame.collect_schema().names())
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}. "
            f"Available columns: {sorted(available)}"
        )


def normalize_text_expr(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
    )


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "site"


def format_optional_text(value: object) -> str:
    if value is None:
        return "Not surfaced"
    text = str(value).strip()
    return text if text else "Not surfaced"


def build_priority_rationale(site: dict[str, object]) -> str:
    rationales: list[str] = []

    if bool(site.get("Dual_Threat_Flag")):
        rationales.append("dual-threat clinical leadership")

    if float(site.get("Dyad_Partner_Intervention_Volume", 0.0)) >= 250:
        rationales.append("deep interventional psychiatry funnel")

    if float(site.get("Net_Sourcing_Alpha", 0.0)) >= 10_000:
        rationales.append("large post-competition patient alpha")

    if float(site.get("Active_Competitor_Trials", 0.0)) == 0:
        rationales.append("no active overlapping competitor trials")
    elif float(site.get("Active_Competitor_Trials", 0.0)) <= 2:
        rationales.append("manageable competitor overlap")

    if not rationales:
        rationales.append("balanced cross-functional launch profile")

    return ", ".join(rationales)


def build_holdback_rationale(site: dict[str, object]) -> str:
    holdbacks: list[str] = []

    if site.get("Trial_Site_Friction_Flag") == "High Friction Trial Site":
        holdbacks.append("no matched interventional referrer")

    if float(site.get("Active_Competitor_Trials", 0.0)) >= 3:
        holdbacks.append("heavy competitor trial overlap")

    if float(site.get("Net_Sourcing_Alpha", 0.0)) < 5_000:
        holdbacks.append("smaller competition-adjusted patient funnel")

    if float(site.get("Clinical_Suitability_Score", 0.0)) < 50:
        holdbacks.append("weaker clinical suitability signal")

    if not holdbacks:
        holdbacks.append("lower integrated launch score than the first-wave group")

    return ", ".join(holdbacks)


def load_top_targets(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "provider_name",
            "provider_city",
            "provider_state",
            "hospital_affiliation",
            "Clinical_Suitability_Score",
            "competitor_consulting_dollars",
            "financially_independent",
            "dual_threat_flag",
            "active_nih_grants",
            "total_surgical_volume",
        },
        "top target ledger",
    )
    return frame.select(
        pl.col("provider_name").alias("Surgeon_Name"),
        pl.col("provider_city").alias("Surgeon_City"),
        pl.col("provider_state").alias("Surgeon_State"),
        pl.col("hospital_affiliation").fill_null("").alias("Hospital_Affiliation"),
        pl.col("Clinical_Suitability_Score")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Clinical_Suitability_Score"),
        pl.col("competitor_consulting_dollars")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Competitor_Consulting_Dollars"),
        pl.col("financially_independent")
        .cast(pl.Boolean, strict=False)
        .fill_null(False)
        .alias("Financially_Independent"),
        pl.col("dual_threat_flag")
        .cast(pl.Boolean, strict=False)
        .fill_null(False)
        .alias("Dual_Threat_Flag"),
        pl.col("active_nih_grants")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Active_NIH_Grants"),
        pl.col("total_surgical_volume")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Surgeon_Volume"),
    ).with_columns(
        normalize_text_expr(pl.col("Surgeon_Name")).alias("surgeon_name_key"),
        normalize_text_expr(pl.col("Surgeon_City")).alias("surgeon_city_key"),
        normalize_text_expr(pl.col("Surgeon_State")).alias("surgeon_state_key"),
    )


def load_dyad_ledger(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Dyad_Partner_Name",
            "Dyad_Partner_Intervention_Volume",
            "Trial_Site_Friction_Flag",
        },
        "dyad ledger",
    )
    return frame.with_columns(
        normalize_text_expr(pl.col("Surgeon_Name")).alias("surgeon_name_key"),
        normalize_text_expr(pl.col("Surgeon_City")).alias("surgeon_city_key"),
        normalize_text_expr(pl.col("Surgeon_State")).alias("surgeon_state_key"),
    ).select(
        "surgeon_name_key",
        "surgeon_city_key",
        "surgeon_state_key",
        "Dyad_Partner_Name",
        "Dyad_Partner_Specialty",
        pl.col("Dyad_Partner_Intervention_Volume")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Dyad_Partner_Intervention_Volume"),
        "Dyad_Match_Type",
        "Trial_Site_Friction_Flag",
    )


def load_catchment_ledger(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Circuit_Level_TAM",
            "Net_Sourcing_Alpha",
            "Active_Competitor_Trials",
            "Trial_Cannibalization_Penalty",
        },
        "catchment ledger",
    )
    return frame.with_columns(
        normalize_text_expr(pl.col("Surgeon_Name")).alias("surgeon_name_key"),
        normalize_text_expr(pl.col("Surgeon_City")).alias("surgeon_city_key"),
        normalize_text_expr(pl.col("Surgeon_State")).alias("surgeon_state_key"),
    ).select(
        "surgeon_name_key",
        "surgeon_city_key",
        "surgeon_state_key",
        pl.col("Total_Beneficiaries")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Total_Beneficiaries"),
        pl.col("Circuit_Level_TAM")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Circuit_Level_TAM"),
        pl.col("Active_Competitor_Trials")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Active_Competitor_Trials"),
        pl.col("Trial_Cannibalization_Penalty")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Trial_Cannibalization_Penalty"),
        pl.col("Net_Sourcing_Alpha")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Net_Sourcing_Alpha"),
        "Catchment_Match_Status",
    )


def load_drg_metrics(path: Path) -> dict[str, float]:
    frame = scan_csv(path)
    ensure_columns(frame, {"Scenario", "Net_Hospital_Profit", "OR_Time_Hours"}, "DRG ledger")
    ledger = frame.collect()

    legacy_profit = (
        ledger.filter(pl.col("Scenario") == "Legacy System")
        .select("Net_Hospital_Profit")
        .item()
    )
    target_profit = (
        ledger.filter(pl.col("Scenario") == "Starfish Chip")
        .select("Net_Hospital_Profit")
        .item()
    )
    legacy_or_time = (
        ledger.filter(pl.col("Scenario") == "Legacy System")
        .select("OR_Time_Hours")
        .item()
    )
    target_or_time = (
        ledger.filter(pl.col("Scenario") == "Starfish Chip")
        .select("OR_Time_Hours")
        .item()
    )

    uplift_pct = round(((target_profit - legacy_profit) / abs(legacy_profit)) * 100, 2)
    return {
        "legacy_profit": float(legacy_profit),
        "target_profit": float(target_profit),
        "profit_delta": float(target_profit - legacy_profit),
        "profit_uplift_pct": uplift_pct,
        "or_time_saved_hours": float(legacy_or_time - target_or_time),
    }


def build_launch_priority_ledger(source_paths: SourcePaths) -> pl.DataFrame:
    top_targets = load_top_targets(source_paths.top_targets)
    dyad_ledger = load_dyad_ledger(source_paths.dyad_ledger)
    catchment_ledger = load_catchment_ledger(source_paths.catchment_ledger)
    drg_metrics = load_drg_metrics(source_paths.drg_ledger)

    return (
        top_targets.join(
            dyad_ledger,
            on=["surgeon_name_key", "surgeon_city_key", "surgeon_state_key"],
            how="left",
        )
        .join(
            catchment_ledger,
            on=["surgeon_name_key", "surgeon_city_key", "surgeon_state_key"],
            how="left",
        )
        .with_columns(
            pl.col("Trial_Site_Friction_Flag").fill_null("High Friction Trial Site"),
            pl.col("Dyad_Partner_Name").fill_null("No matched interventional referrer"),
            pl.col("Dyad_Partner_Specialty").fill_null("Unmatched"),
            pl.col("Dyad_Partner_Intervention_Volume").fill_null(0.0),
            pl.col("Dyad_Match_Type").fill_null("unmatched"),
            pl.col("Total_Beneficiaries").fill_null(0.0),
            pl.col("Circuit_Level_TAM").fill_null(0.0),
            pl.col("Active_Competitor_Trials").fill_null(0.0),
            pl.col("Trial_Cannibalization_Penalty").fill_null(0.0),
            pl.col("Net_Sourcing_Alpha").fill_null(0.0),
            pl.col("Catchment_Match_Status").fill_null("No Geographic Match"),
        )
        .with_columns(
            (pl.col("Clinical_Suitability_Score") / 100.0).clip(0.0, 1.0).alias(
                "clinical_signal"
            ),
            (
                pl.col("Dyad_Partner_Intervention_Volume")
                / pl.max_horizontal(pl.col("Dyad_Partner_Intervention_Volume").max(), pl.lit(1.0))
            )
            .clip(0.0, 1.0)
            .alias("dyad_signal"),
            (
                pl.col("Net_Sourcing_Alpha")
                / pl.max_horizontal(pl.col("Net_Sourcing_Alpha").max(), pl.lit(1.0))
            )
            .clip(0.0, 1.0)
            .alias("market_signal"),
            (
                1
                - (
                    pl.col("Active_Competitor_Trials")
                    / pl.max_horizontal(pl.col("Active_Competitor_Trials").max(), pl.lit(1.0))
                )
            )
            .clip(0.0, 1.0)
            .alias("competition_signal"),
            pl.lit(drg_metrics["target_profit"]).alias("Projected_Net_Hospital_Profit"),
            pl.lit(drg_metrics["legacy_profit"]).alias("Projected_Legacy_Hospital_Profit"),
            pl.lit(drg_metrics["profit_delta"]).alias("Projected_Profit_Delta"),
            pl.lit(drg_metrics["profit_uplift_pct"]).alias("Projected_Profit_Uplift_Pct"),
            pl.lit(drg_metrics["or_time_saved_hours"]).alias("Projected_OR_Time_Saved_Hours"),
        )
        .with_columns(
            (
                (
                    pl.col("clinical_signal") * 0.45
                    + pl.col("dyad_signal") * 0.20
                    + pl.col("market_signal") * 0.25
                    + pl.col("competition_signal") * 0.10
                )
                * 100
                + pl.when(pl.col("Dual_Threat_Flag")).then(4.0).otherwise(0.0)
                + pl.when(pl.col("Financially_Independent")).then(3.0).otherwise(0.0)
                - pl.when(pl.col("Trial_Site_Friction_Flag") == "High Friction Trial Site")
                .then(50.0)
                .otherwise(0.0)
            )
            .clip(lower_bound=0.0)
            .round(2)
            .alias("Launch_Priority_Score")
        )
        .with_columns(
            pl.when(pl.col("Launch_Priority_Score") >= 75)
            .then(pl.lit("Wave 1"))
            .when(pl.col("Launch_Priority_Score") >= 55)
            .then(pl.lit("Wave 2"))
            .otherwise(pl.lit("Wave 3"))
            .alias("Launch_Wave"),
            pl.when(pl.col("Launch_Priority_Score") >= 75)
            .then(pl.lit("Initiate outreach immediately"))
            .when(pl.col("Launch_Priority_Score") >= 55)
            .then(pl.lit("Validate after Wave 1"))
            .otherwise(pl.lit("Hold for later validation"))
            .alias("Executive_Recommendation"),
        )
        .sort(
            by=[
                "Launch_Priority_Score",
                "Net_Sourcing_Alpha",
                "Clinical_Suitability_Score",
            ],
            descending=[True, True, True],
        )
        .select(
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Hospital_Affiliation",
            "Surgeon_Volume",
            "Active_NIH_Grants",
            "Clinical_Suitability_Score",
            "Dyad_Partner_Name",
            "Dyad_Partner_Specialty",
            "Dyad_Partner_Intervention_Volume",
            "Dyad_Match_Type",
            "Trial_Site_Friction_Flag",
            "Total_Beneficiaries",
            "Circuit_Level_TAM",
            "Active_Competitor_Trials",
            "Trial_Cannibalization_Penalty",
            "Net_Sourcing_Alpha",
            "Financially_Independent",
            "Competitor_Consulting_Dollars",
            "Dual_Threat_Flag",
            "Projected_Legacy_Hospital_Profit",
            "Projected_Net_Hospital_Profit",
            "Projected_Profit_Delta",
            "Projected_Profit_Uplift_Pct",
            "Projected_OR_Time_Saved_Hours",
            "Launch_Priority_Score",
            "Launch_Wave",
            "Executive_Recommendation",
        )
        .collect()
    )


def create_launch_priority_chart(ledger: pl.DataFrame, output_path: Path, top_n: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = ledger.head(top_n).sort("Launch_Priority_Score")

    labels = [
        f"{surgeon} | {city}, {state}"
        for surgeon, city, state in zip(
            ranked["Surgeon_Name"].to_list(),
            ranked["Surgeon_City"].to_list(),
            ranked["Surgeon_State"].to_list(),
            strict=False,
        )
    ]
    scores = ranked["Launch_Priority_Score"].to_list()
    market_alpha = ranked["Net_Sourcing_Alpha"].to_list()

    fig, ax = plt.subplots(figsize=(15, 8), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    vmax = max(market_alpha) if market_alpha else 1.0
    vmin = min(market_alpha) if market_alpha else 0.0
    if vmax == vmin:
        vmax = vmin + 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["plasma"]
    bar_colors = [cmap(norm(value)) for value in market_alpha]

    bars = ax.barh(
        labels,
        scores,
        color=bar_colors,
        edgecolor=bar_colors,
        linewidth=1.0,
        alpha=0.98,
    )

    for bar, score, alpha_value in zip(bars, scores, market_alpha, strict=False):
        ax.text(
            score + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f} | alpha {alpha_value:,.0f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#f5f5f5",
        )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
    colorbar.set_label("Net Sourcing Alpha", color="#d4d4d4")
    colorbar.ax.yaxis.set_tick_params(color="#d4d4d4")
    plt.setp(colorbar.ax.get_yticklabels(), color="#d4d4d4")
    colorbar.outline.set_edgecolor("#666666")

    ax.set_title(
        f"Top {top_n} Recommended Launch Sites",
        loc="left",
        fontsize=18,
        color="#f5f5f5",
        pad=18,
        fontweight="bold",
    )
    ax.set_xlabel("Launch Priority Score", color="#d4d4d4", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="x", colors="#d4d4d4", labelsize=10)
    ax.tick_params(axis="y", colors="#f5f5f5", labelsize=10)
    ax.grid(axis="x", color="#3a3a3a", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    fig.text(
        0.01,
        0.015,
        "Launch Priority combines clinical suitability, dyad strength, market alpha, and competitive openness into one ranked operating view.",
        color="#cfcfcf",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    LOGGER.info("Wrote launch chart to %s", output_path)


def write_executive_summary(
    ledger: pl.DataFrame,
    output_path: Path,
    top_n: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_sites = ledger.head(top_n).to_dicts()
    reserve_sites = ledger.slice(top_n, 5).to_dicts()
    immediate_sites = [site for site in top_sites if site["Launch_Wave"] == "Wave 1"]
    diligence_sites = [site for site in top_sites if site["Launch_Wave"] == "Wave 2"]

    lines = [
        "# Executive Summary",
        "",
        "## Core Recommendation",
        "",
        "Use this package as the launch-selection layer for first-site planning. It collapses operator quality, matched referral depth, competition-adjusted patient catchment, and case-level hospital economics into one ranked operating view.",
        "",
        f"- Immediate outreach: {len(immediate_sites)} Wave 1 site(s).",
        f"- Parallel diligence: {len(diligence_sites)} Wave 2 site(s).",
        f"- Reserve queue shown below: next 5 ranked site(s) held behind the launch leaders.",
        "",
        "## Priority Order",
        "",
    ]

    for index, site in enumerate(top_sites, start=1):
        rationale = build_priority_rationale(site)
        lines.extend(
            [
                f"{index}. {site['Surgeon_Name']} | {site['Surgeon_City']}, {site['Surgeon_State']}",
                f"   - Launch priority score: {site['Launch_Priority_Score']} ({site['Launch_Wave']})",
                f"   - Referrer: {site['Dyad_Partner_Name']} ({site['Dyad_Partner_Specialty']})",
                f"   - Net sourcing alpha: {site['Net_Sourcing_Alpha']}",
                f"   - Active competitor trials: {int(site['Active_Competitor_Trials'])}",
                f"   - Projected hospital profit uplift: {site['Projected_Profit_Uplift_Pct']}%",
                f"   - Why this site rises: {rationale}",
            ]
        )

    if reserve_sites:
        lines.extend(
            [
                "",
                "## Why Sites 6-10 Are Not First",
                "",
            ]
        )
        for index, site in enumerate(reserve_sites, start=top_n + 1):
            lines.extend(
                [
                    f"{index}. {site['Surgeon_Name']} | {site['Surgeon_City']}, {site['Surgeon_State']}",
                    f"   - Current wave: {site['Launch_Wave']}",
                    f"   - Main holdback: {build_holdback_rationale(site)}",
                ]
            )

    lines.extend(
        [
            "",
            "## Why This Package Matters",
            "",
            "- The clinical ledger identifies who can implant and who can recruit.",
            "- The catchment ledger discounts patient demand for live competitor overlap.",
            "- The DRG model proves the financial case to hospital operators.",
            "- The final launch score compresses those signals into one ordering the executive team can act on immediately.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote executive summary to %s", output_path)


def write_site_briefs(ledger: pl.DataFrame, output_dir: Path, top_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, site in enumerate(ledger.head(top_n).to_dicts(), start=1):
        filename = f"{index:02d}_{slugify(site['Surgeon_City'])}_{slugify(site['Surgeon_Name'])}.md"
        path = output_dir / filename
        lines = [
            f"# Site Brief {index}: {site['Surgeon_Name']}",
            "",
            "## Overview",
            "",
            f"- Recommended wave: {site['Launch_Wave']}",
            f"- Executive recommendation: {site['Executive_Recommendation']}",
            f"- Location: {site['Surgeon_City']}, {site['Surgeon_State']}",
            "",
            "## Clinical Operator",
            "",
            f"- Surgeon volume: {site['Surgeon_Volume']}",
            f"- NIH grants: {site['Active_NIH_Grants']}",
            f"- Clinical suitability score: {site['Clinical_Suitability_Score']}",
            f"- Dual-threat flag: {site['Dual_Threat_Flag']}",
            "",
            "## Referral Funnel",
            "",
            f"- Dyad partner: {site['Dyad_Partner_Name']}",
            f"- Specialty: {site['Dyad_Partner_Specialty']}",
            f"- Interventional volume: {site['Dyad_Partner_Intervention_Volume']}",
            f"- Match type: {site['Dyad_Match_Type']}",
            f"- Trial friction flag: {site['Trial_Site_Friction_Flag']}",
            "",
            "## Market and Competition",
            "",
            f"- Total beneficiaries proxy: {site['Total_Beneficiaries']}",
            f"- Circuit-level TAM: {site['Circuit_Level_TAM']}",
            f"- Active competitor trials: {int(site['Active_Competitor_Trials'])}",
            f"- Cannibalization penalty: {site['Trial_Cannibalization_Penalty']}",
            f"- Net sourcing alpha: {site['Net_Sourcing_Alpha']}",
            "",
            "## Economics",
            "",
            f"- Projected next-gen wireless device profit per case: ${site['Projected_Net_Hospital_Profit']:,.0f}",
            f"- Legacy profit per case: ${site['Projected_Legacy_Hospital_Profit']:,.0f}",
            f"- Profit delta per case: ${site['Projected_Profit_Delta']:,.0f}",
            f"- Profit uplift: {site['Projected_Profit_Uplift_Pct']}%",
            f"- OR time saved: {site['Projected_OR_Time_Saved_Hours']} hours",
            "",
            "## Outreach Thesis",
            "",
            f"- Lead with a launch thesis centered on {site['Surgeon_City']} as a viable recruitment hub with a matched interventional referrer and a quantified patient funnel.",
            f"- Position the hospital case around ${site['Projected_Profit_Delta']:,.0f} incremental profit per case and {site['Projected_OR_Time_Saved_Hours']} hours of OR time recovered.",
            f"- Address competition directly: this site currently faces {int(site['Active_Competitor_Trials'])} active overlapping competitor trials.",
            f"- Suggested internal framing: {build_priority_rationale(site)}.",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote %s site briefs to %s", top_n, output_dir)


def write_outputs(
    ledger: pl.DataFrame,
    output_dir: Path,
    top_n: int,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    briefs_dir = output_dir / "site_briefs"
    ledger_path = output_dir / "launch_priority_ledger.csv"
    chart_path = output_dir / "top_launch_sites.png"
    summary_path = output_dir / "executive_summary.md"

    ledger.write_csv(ledger_path)
    create_launch_priority_chart(ledger, chart_path, top_n=top_n)
    write_executive_summary(ledger, summary_path, top_n=top_n)
    write_site_briefs(ledger, briefs_dir, top_n=top_n)
    LOGGER.info("Wrote launch priority ledger to %s", ledger_path)
    return ledger_path, chart_path, summary_path, briefs_dir


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    source_paths = SourcePaths(
        top_targets=args.top_targets,
        dyad_ledger=args.dyad_ledger,
        catchment_ledger=args.catchment_ledger,
        drg_ledger=args.drg_ledger,
        output_dir=args.output_dir,
    )
    ledger = build_launch_priority_ledger(source_paths)
    outputs = write_outputs(ledger, source_paths.output_dir, top_n=args.top_n)
    try:
        from launch_packet_renderer import PacketPaths, render_launch_packet

        packet_outputs = render_launch_packet(
            PacketPaths(
                summary=outputs[2],
                briefs_dir=outputs[3],
                ledger=outputs[0],
                launch_chart=outputs[1],
                drg_chart=PROCESSED_DIR / "drg_economics_visual.png",
                output_pdf=source_paths.output_dir / "executive_launch_packet.pdf",
                output_email=source_paths.output_dir / "launch_packet_email.txt",
                output_manifest=source_paths.output_dir / "attachment_manifest.txt",
            ),
            top_n=args.top_n,
        )
        LOGGER.info("Presentation outputs: %s", packet_outputs)
    except Exception as exc:
        LOGGER.warning("Packet render skipped: %s", exc)
    LOGGER.info("Launch package complete. Generated %s ranked sites.", len(ledger))
    LOGGER.info("Outputs: %s", outputs)


if __name__ == "__main__":
    main()
