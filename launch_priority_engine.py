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
DEFAULT_DYAD_LEDGER = PROCESSED_DIR / "clinical_dyad_ledger_v2.csv"
DEFAULT_CATCHMENT_LEDGER = PROCESSED_DIR / "viable_catchment_areas.csv"
DEFAULT_DRG_LEDGER = PROCESSED_DIR / "hospital_drg_site_adjusted.csv"
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
        description="Build an executive launch-selection package from the strengthened clinical, catchment, and site-economics ledgers."
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


def confidence_score_expr(column_name: str) -> pl.Expr:
    return (
        pl.when(pl.col(column_name) == "High")
        .then(1.0)
        .when(pl.col(column_name) == "Moderate")
        .then(0.7)
        .when(pl.col(column_name) == "Medium")
        .then(0.65)
        .when(pl.col(column_name) == "Low")
        .then(0.35)
        .otherwise(0.0)
    )


def build_priority_rationale(site: dict[str, object]) -> str:
    rationales: list[str] = []

    if site.get("Launch_Wave") == "Wave 1":
        rationales.append("meets full launch-ready evidence gates")
    if bool(site.get("Dual_Threat_Flag")):
        rationales.append("dual-threat operator signal")
    if site.get("Referral_Confidence_Tier") == "High":
        rationales.append("same-network referral confidence")
    if site.get("Referrer_Indication_Fit_Tier") == "High":
        rationales.append("strong interventional psychiatry fit")
    if float(site.get("Net_Sourcing_Alpha", 0.0)) >= 20_000:
        rationales.append("deep competition-adjusted patient funnel")
    if site.get("Competition_Environment") == "Blue Ocean":
        rationales.append("minimal live competitor crowding")
    if site.get("Economics_Evidence_Tier") == "High":
        rationales.append("hospital-adjusted economics validated")
    if not rationales:
        rationales.append("balanced cross-functional launch profile")
    return ", ".join(rationales)


def build_holdback_rationale(site: dict[str, object]) -> str:
    holdbacks: list[str] = []

    if site.get("Catchment_Evidence_Tier") != "High":
        holdbacks.append("epidemiology evidence still needs validation")
    if site.get("Referral_Confidence_Tier") not in {"High", "Medium"}:
        holdbacks.append("referral confidence is weaker than first-wave sites")
    if site.get("Referrer_Indication_Fit_Tier") == "Low":
        holdbacks.append("referrer fit to the target indication is shallow")
    if float(site.get("Competition_Pressure_Fraction", 0.0)) >= 0.18:
        holdbacks.append("competitor intensity is materially higher")
    if float(site.get("Net_Sourcing_Alpha", 0.0)) < 10_000:
        holdbacks.append("competition-adjusted funnel is smaller")
    if site.get("Economics_Evidence_Tier") != "High":
        holdbacks.append("site economics are still fallback-mode")
    if not holdbacks:
        holdbacks.append("lower integrated launch score than the first-wave group")
    return ", ".join(holdbacks)


def load_top_targets(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "npi",
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
        pl.col("npi").cast(pl.Utf8, strict=False).fill_null("").alias("Surgeon_NPI"),
        pl.col("provider_name").alias("Surgeon_Name"),
        pl.col("provider_city").alias("Surgeon_City"),
        pl.col("provider_state").alias("Surgeon_State"),
        pl.col("hospital_affiliation").fill_null("").alias("Hospital_Affiliation"),
        pl.col("Clinical_Suitability_Score").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("competitor_consulting_dollars")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Competitor_Consulting_Dollars"),
        pl.col("financially_independent")
        .cast(pl.Boolean, strict=False)
        .fill_null(False)
        .alias("Financially_Independent"),
        pl.col("dual_threat_flag").cast(pl.Boolean, strict=False).fill_null(False).alias("Dual_Threat_Flag"),
        pl.col("active_nih_grants").cast(pl.Float64, strict=False).fill_null(0.0).alias("Active_NIH_Grants"),
        pl.col("total_surgical_volume").cast(pl.Float64, strict=False).fill_null(0.0).alias("Surgeon_Volume"),
    )


def load_dyad_ledger(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "Surgeon_NPI",
            "Surgeon_City",
            "Surgeon_State",
            "Dyad_Partner_Name",
            "Dyad_Partner_Specialty",
            "Dyad_Partner_Intervention_Volume",
            "Dyad_Partner_Target_CPT_Count",
            "Dyad_Relationship_Score",
            "Referral_Confidence_Tier",
            "Dyad_Match_Type",
            "Trial_Site_Friction_Flag",
        },
        "dyad ledger",
    )
    return frame.select(
        pl.col("Surgeon_NPI").cast(pl.Utf8, strict=False).fill_null(""),
        pl.col("Surgeon_City").cast(pl.Utf8, strict=False).fill_null("").alias("Resolved_Surgeon_City"),
        pl.col("Surgeon_State").cast(pl.Utf8, strict=False).fill_null("").alias("Resolved_Surgeon_State"),
        "Dyad_Partner_Name",
        "Dyad_Partner_Specialty",
        pl.col("Dyad_Partner_Intervention_Volume").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Dyad_Partner_Target_CPT_Count").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Dyad_Relationship_Score").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Referral_Confidence_Tier").fill_null("None"),
        pl.col("Dyad_Match_Type").fill_null("unmatched"),
        pl.col("Trial_Site_Friction_Flag").fill_null("High Friction Trial Site"),
    )


def load_catchment_ledger(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "Surgeon_NPI",
            "Circuit_Level_TAM",
            "Net_Sourcing_Alpha",
            "Active_Competitor_Trials",
            "Trial_Cannibalization_Penalty",
            "Catchment_Evidence_Tier",
            "Competition_Pressure_Fraction",
            "Competition_Environment",
            "Catchment_Match_Status",
        },
        "catchment ledger",
    )
    return frame.select(
        pl.col("Surgeon_NPI").cast(pl.Utf8, strict=False).fill_null(""),
        pl.col("Circuit_Level_TAM").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Protocol_Eligible_Funnel_Estimate").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Active_Competitor_Trials").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Indication_Matched_Trials").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Facility_Overlap_Trials").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Competition_Intensity_Score").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Competition_Pressure_Fraction").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Trial_Cannibalization_Penalty").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Net_Sourcing_Alpha").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Catchment_Evidence_Tier").fill_null("Low"),
        pl.col("Competition_Environment").fill_null("Crowded"),
        pl.col("Catchment_Match_Status").fill_null("No Catchment Match"),
        pl.col("Local_Depression_Source").fill_null("Unknown"),
    )


def load_site_economics(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "Surgeon_NPI",
            "Hospital_Name",
            "Site_Adjusted_Medicare_Payout",
            "Site_Adjusted_OR_Hourly_Cost",
            "Legacy_Net_Hospital_Profit",
            "Target_Net_Hospital_Profit",
            "Projected_Profit_Delta",
            "Projected_Profit_Uplift_Pct",
            "Projected_OR_Time_Saved_Hours",
            "Economics_Evidence_Tier",
        },
        "site economics ledger",
    )
    return frame.select(
        pl.col("Surgeon_NPI").cast(pl.Utf8, strict=False).fill_null(""),
        pl.col("Hospital_Name").fill_null("Unresolved Hospital"),
        pl.col("Site_Adjusted_Medicare_Payout").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Site_Adjusted_OR_Hourly_Cost").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Legacy_Net_Hospital_Profit").cast(pl.Float64, strict=False).fill_null(0.0).alias("Projected_Legacy_Hospital_Profit"),
        pl.col("Target_Net_Hospital_Profit").cast(pl.Float64, strict=False).fill_null(0.0).alias("Projected_Net_Hospital_Profit"),
        pl.col("Projected_Profit_Delta").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Projected_Profit_Uplift_Pct").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Projected_OR_Time_Saved_Hours").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("Economics_Evidence_Tier").fill_null("Low"),
    )


def build_launch_priority_ledger(source_paths: SourcePaths) -> pl.DataFrame:
    top_targets = load_top_targets(source_paths.top_targets)
    dyad_ledger = load_dyad_ledger(source_paths.dyad_ledger)
    catchment_ledger = load_catchment_ledger(source_paths.catchment_ledger)
    site_economics = load_site_economics(source_paths.drg_ledger)

    return (
        top_targets.join(dyad_ledger, on="Surgeon_NPI", how="left")
        .join(catchment_ledger, on="Surgeon_NPI", how="left")
        .join(site_economics, on="Surgeon_NPI", how="left")
        .with_columns(
            pl.coalesce(["Resolved_Surgeon_City", "Surgeon_City"]).alias("Surgeon_City"),
            pl.coalesce(["Resolved_Surgeon_State", "Surgeon_State"]).alias("Surgeon_State"),
            pl.col("Dyad_Partner_Name").fill_null("No matched interventional referrer"),
            pl.col("Dyad_Partner_Specialty").fill_null("Unmatched"),
            pl.col("Dyad_Partner_Intervention_Volume").fill_null(0.0),
            pl.col("Dyad_Partner_Target_CPT_Count").fill_null(0.0),
            pl.col("Dyad_Relationship_Score").fill_null(0.0),
            pl.col("Referral_Confidence_Tier").fill_null("None"),
            pl.col("Dyad_Match_Type").fill_null("unmatched"),
            pl.col("Trial_Site_Friction_Flag").fill_null("High Friction Trial Site"),
            pl.col("Circuit_Level_TAM").fill_null(0.0),
            pl.col("Protocol_Eligible_Funnel_Estimate").fill_null(0.0),
            pl.col("Active_Competitor_Trials").fill_null(0.0),
            pl.col("Indication_Matched_Trials").fill_null(0.0),
            pl.col("Facility_Overlap_Trials").fill_null(0.0),
            pl.col("Competition_Intensity_Score").fill_null(0.0),
            pl.col("Competition_Pressure_Fraction").fill_null(0.0),
            pl.col("Trial_Cannibalization_Penalty").fill_null(0.0),
            pl.col("Net_Sourcing_Alpha").fill_null(0.0),
            pl.col("Catchment_Evidence_Tier").fill_null("Low"),
            pl.col("Competition_Environment").fill_null("Crowded"),
            pl.col("Catchment_Match_Status").fill_null("No Catchment Match"),
            pl.col("Local_Depression_Source").fill_null("Unknown"),
            pl.col("Projected_Legacy_Hospital_Profit").fill_null(0.0),
            pl.col("Projected_Net_Hospital_Profit").fill_null(0.0),
            pl.col("Projected_Profit_Delta").fill_null(0.0),
            pl.col("Projected_Profit_Uplift_Pct").fill_null(0.0),
            pl.col("Projected_OR_Time_Saved_Hours").fill_null(0.0),
            pl.col("Economics_Evidence_Tier").fill_null("Low"),
            pl.col("Hospital_Name").fill_null("Unresolved Hospital"),
            pl.when(pl.col("Hospital_Name") != "Unresolved Hospital")
            .then(pl.col("Hospital_Name"))
            .otherwise(pl.col("Hospital_Affiliation"))
            .alias("Hospital_Affiliation"),
        )
        .with_columns(
            pl.when(
                (pl.col("Dyad_Partner_Target_CPT_Count") >= 2)
                | (pl.col("Dyad_Partner_Intervention_Volume") >= 150)
            )
            .then(pl.lit("High"))
            .when(pl.col("Dyad_Partner_Intervention_Volume") >= 50)
            .then(pl.lit("Moderate"))
            .otherwise(pl.lit("Low"))
            .alias("Referrer_Indication_Fit_Tier"),
            confidence_score_expr("Referral_Confidence_Tier").alias("dyad_confidence_score"),
            confidence_score_expr("Catchment_Evidence_Tier").alias("catchment_confidence_score"),
            confidence_score_expr("Economics_Evidence_Tier").alias("economics_confidence_score"),
        )
        .with_columns(
            (
                (
                    pl.col("dyad_confidence_score")
                    + pl.col("catchment_confidence_score")
                    + pl.col("economics_confidence_score")
                )
                / 3
            )
            .round(3)
            .alias("Evidence_Confidence_Score"),
            (pl.col("Clinical_Suitability_Score") / 100.0).clip(0.0, 1.0).alias("clinical_signal"),
            (
                (
                    (
                        pl.col("Dyad_Relationship_Score")
                        / pl.max_horizontal(pl.col("Dyad_Relationship_Score").max(), pl.lit(1.0))
                    )
                    * 0.55
                    + (
                        pl.col("Dyad_Partner_Intervention_Volume")
                        / pl.max_horizontal(pl.col("Dyad_Partner_Intervention_Volume").max(), pl.lit(1.0))
                    )
                    * 0.45
                )
            )
            .clip(0.0, 1.0)
            .alias("dyad_signal"),
            (
                (
                    (
                        pl.col("Net_Sourcing_Alpha")
                        / pl.max_horizontal(pl.col("Net_Sourcing_Alpha").max(), pl.lit(1.0))
                    )
                    * 0.7
                    + (1 - pl.col("Competition_Pressure_Fraction").clip(0.0, 1.0)) * 0.3
                )
            )
            .clip(0.0, 1.0)
            .alias("market_signal"),
            (
                (
                    (
                        pl.col("Projected_Profit_Delta")
                        / pl.max_horizontal(pl.col("Projected_Profit_Delta").max(), pl.lit(1.0))
                    )
                    * 0.7
                    + (
                        pl.col("Projected_Net_Hospital_Profit")
                        / pl.max_horizontal(pl.col("Projected_Net_Hospital_Profit").max(), pl.lit(1.0))
                    )
                    * 0.3
                )
            )
            .clip(0.0, 1.0)
            .alias("economics_signal"),
        )
        .with_columns(
            (
                (
                    pl.col("clinical_signal") * 0.25
                    + pl.col("dyad_signal") * 0.25
                    + pl.col("market_signal") * 0.25
                    + pl.col("economics_signal") * 0.15
                    + pl.col("Evidence_Confidence_Score") * 0.10
                )
                * 100
                + pl.when(pl.col("Dual_Threat_Flag")).then(3.0).otherwise(0.0)
                + pl.when(pl.col("Financially_Independent")).then(2.5).otherwise(0.0)
                - pl.when(pl.col("Trial_Site_Friction_Flag") == "High Friction Trial Site")
                .then(40.0)
                .otherwise(0.0)
            )
            .clip(lower_bound=0.0)
            .round(2)
            .alias("Launch_Priority_Score")
        )
        .with_columns(
            pl.when(
                (pl.col("Trial_Site_Friction_Flag") != "High Friction Trial Site")
                & (pl.col("Referral_Confidence_Tier") == "High")
                & (pl.col("Catchment_Evidence_Tier") == "High")
                & (pl.col("Economics_Evidence_Tier") == "High")
                & (pl.col("Referrer_Indication_Fit_Tier") != "Low")
                & (pl.col("Net_Sourcing_Alpha") >= 15_000)
            )
            .then(pl.lit("Wave 1"))
            .when(
                (pl.col("Evidence_Confidence_Score") >= 0.70)
                & (pl.col("Net_Sourcing_Alpha") >= 8_000)
            )
            .then(pl.lit("Wave 2"))
            .otherwise(pl.lit("Wave 3"))
            .alias("Launch_Wave")
        )
        .with_columns(
            pl.when(pl.col("Launch_Wave") == "Wave 1")
            .then(pl.lit("Initiate outreach immediately"))
            .when(pl.col("Launch_Wave") == "Wave 2")
            .then(pl.lit("Run parallel diligence and site validation"))
            .otherwise(pl.lit("Keep in reserve until evidence strengthens"))
            .alias("Executive_Recommendation")
        )
        .with_columns(
            pl.when(pl.col("Launch_Wave") == "Wave 1")
            .then(1)
            .when(pl.col("Launch_Wave") == "Wave 2")
            .then(2)
            .otherwise(3)
            .alias("Launch_Wave_Order"),
        )
        .sort(
            by=[
                "Launch_Wave_Order",
                "Launch_Priority_Score",
                "Evidence_Confidence_Score",
                "Net_Sourcing_Alpha",
                "Projected_Profit_Delta",
            ],
            descending=[False, True, True, True, True],
        )
        .select(
            "Surgeon_NPI",
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Hospital_Affiliation",
            "Hospital_Name",
            "Surgeon_Volume",
            "Active_NIH_Grants",
            "Clinical_Suitability_Score",
            "Dyad_Partner_Name",
            "Dyad_Partner_Specialty",
            "Dyad_Partner_Intervention_Volume",
            "Dyad_Partner_Target_CPT_Count",
            "Referrer_Indication_Fit_Tier",
            "Dyad_Match_Type",
            "Referral_Confidence_Tier",
            "Dyad_Relationship_Score",
            "Trial_Site_Friction_Flag",
            "Catchment_Match_Status",
            "Catchment_Evidence_Tier",
            "Local_Depression_Source",
            "Circuit_Level_TAM",
            "Protocol_Eligible_Funnel_Estimate",
            "Active_Competitor_Trials",
            "Indication_Matched_Trials",
            "Facility_Overlap_Trials",
            "Competition_Intensity_Score",
            "Competition_Pressure_Fraction",
            "Competition_Environment",
            "Trial_Cannibalization_Penalty",
            "Net_Sourcing_Alpha",
            "Financially_Independent",
            "Competitor_Consulting_Dollars",
            "Dual_Threat_Flag",
            "Site_Adjusted_Medicare_Payout",
            "Site_Adjusted_OR_Hourly_Cost",
            "Projected_Legacy_Hospital_Profit",
            "Projected_Net_Hospital_Profit",
            "Projected_Profit_Delta",
            "Projected_Profit_Uplift_Pct",
            "Projected_OR_Time_Saved_Hours",
            "Economics_Evidence_Tier",
            "Evidence_Confidence_Score",
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
    evidence = ranked["Evidence_Confidence_Score"].to_list()

    fig, ax = plt.subplots(figsize=(15, 8), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    vmax = max(evidence) if evidence else 1.0
    vmin = min(evidence) if evidence else 0.0
    if vmax == vmin:
        vmax = vmin + 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["plasma"]
    bar_colors = [cmap(norm(value)) for value in evidence]

    bars = ax.barh(labels, scores, color=bar_colors, edgecolor=bar_colors, linewidth=1.0, alpha=0.98)
    for bar, score, evidence_value in zip(bars, scores, evidence, strict=False):
        ax.text(
            score + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f} | ev {evidence_value:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#f5f5f5",
        )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors="#d4d4d4", labelsize=9)
    colorbar.set_label("Evidence Confidence", color="#d4d4d4", fontsize=10)

    ax.set_title(
        "Launch Readiness | Ranked First-Site Operating View",
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
        "Launch Priority now gates for referral confidence, catchment evidence, and hospital-adjusted economics instead of treating the highest rank as automatically launch-ready.",
        color="#cfcfcf",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    LOGGER.info("Wrote launch chart to %s", output_path)


def write_executive_summary(ledger: pl.DataFrame, output_path: Path, top_n: int) -> None:
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
        "Use this package as the launch-selection layer for first-site planning. It now separates ranked operators from truly launch-ready hubs by requiring evidence across referral confidence, catchment quality, competition pressure, and hospital-adjusted economics.",
        "",
        f"- Immediate outreach: {len(immediate_sites)} Wave 1 site(s).",
        f"- Parallel diligence: {len(diligence_sites)} Wave 2 site(s).",
        f"- Reserve queue shown below: next 5 ranked site(s) held behind the launch leaders.",
        "",
        "## Priority Order",
        "",
    ]

    for index, site in enumerate(top_sites, start=1):
        lines.extend(
            [
                f"{index}. {site['Surgeon_Name']} | {site['Surgeon_City']}, {site['Surgeon_State']}",
                f"   - Launch priority score: {site['Launch_Priority_Score']} ({site['Launch_Wave']})",
                f"   - Evidence confidence: {site['Evidence_Confidence_Score']}",
                f"   - Referrer: {site['Dyad_Partner_Name']} ({site['Dyad_Partner_Specialty']})",
                f"   - Referral confidence: {site['Referral_Confidence_Tier']} | indication fit: {site['Referrer_Indication_Fit_Tier']}",
                f"   - Net sourcing alpha: {site['Net_Sourcing_Alpha']}",
                f"   - Competition environment: {site['Competition_Environment']}",
                f"   - Profit delta per case: ${site['Projected_Profit_Delta']:,.0f}",
                f"   - Why this site rises: {build_priority_rationale(site)}",
            ]
        )

    if reserve_sites:
        lines.extend(["", "## Why Sites 6-10 Are Not First", ""])
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
            "- The clinical ledger identifies who can implant and who can recruit inside the same operating network.",
            "- The catchment ledger uses measured local prevalence surfaces and discounts those funnels for live competitor overlap.",
            "- The economics ledger adjusts hospital profit by real IPPS provider characteristics instead of one national assumption.",
            "- The final launch score turns those signals into a launch-ready ordering rather than a generic top-5 list.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote executive summary to %s", output_path)


def write_site_briefs(ledger: pl.DataFrame, output_dir: Path, top_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing_brief in output_dir.glob("*.md"):
        existing_brief.unlink()

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
            f"- Evidence confidence: {site['Evidence_Confidence_Score']}",
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
            f"- Target CPT count: {site['Dyad_Partner_Target_CPT_Count']}",
            f"- Referral confidence: {site['Referral_Confidence_Tier']}",
            f"- Referrer indication fit: {site['Referrer_Indication_Fit_Tier']}",
            f"- Match type: {site['Dyad_Match_Type']}",
            f"- Trial friction flag: {site['Trial_Site_Friction_Flag']}",
            "",
            "## Market and Competition",
            "",
            f"- Protocol-eligible funnel estimate: {site['Protocol_Eligible_Funnel_Estimate']}",
            f"- Competition-adjusted alpha: {site['Net_Sourcing_Alpha']}",
            f"- Catchment evidence tier: {site['Catchment_Evidence_Tier']}",
            f"- Local prevalence source: {site['Local_Depression_Source']}",
            f"- Active competitor trials: {int(site['Active_Competitor_Trials'])}",
            f"- Competition pressure fraction: {site['Competition_Pressure_Fraction']}",
            f"- Cannibalization penalty: {site['Trial_Cannibalization_Penalty']}",
            "",
            "## Economics",
            "",
            f"- Hospital economics evidence: {site['Economics_Evidence_Tier']}",
            f"- Site-adjusted payout: ${site['Site_Adjusted_Medicare_Payout']:,.0f}",
            f"- Site-adjusted OR hourly cost: ${site['Site_Adjusted_OR_Hourly_Cost']:,.0f}",
            f"- Next-gen device profit per case: ${site['Projected_Net_Hospital_Profit']:,.0f}",
            f"- Legacy profit per case: ${site['Projected_Legacy_Hospital_Profit']:,.0f}",
            f"- Profit delta per case: ${site['Projected_Profit_Delta']:,.0f}",
            f"- Profit uplift: {site['Projected_Profit_Uplift_Pct']}%",
            f"- OR time saved: {site['Projected_OR_Time_Saved_Hours']} hours",
            "",
            "## Outreach Thesis",
            "",
            f"- Lead with a launch thesis centered on {site['Surgeon_City']} as a recruitable hub with same-network referral confidence and a quantified competition-adjusted funnel.",
            f"- Position the hospital case around ${site['Projected_Profit_Delta']:,.0f} incremental profit per case and {site['Projected_OR_Time_Saved_Hours']} hours of OR time recovered.",
            f"- Address competition directly: this site currently faces {int(site['Active_Competitor_Trials'])} active overlapping competitor trials and a pressure fraction of {site['Competition_Pressure_Fraction']}.",
            f"- Suggested internal framing: {build_priority_rationale(site)}.",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote %s site briefs to %s", top_n, output_dir)


def write_outputs(ledger: pl.DataFrame, output_dir: Path, top_n: int) -> tuple[Path, Path, Path, Path]:
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
