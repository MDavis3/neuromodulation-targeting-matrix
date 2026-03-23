from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import colors
import polars as pl


LOGGER = logging.getLogger("neuromodulation_targeting_matrix.catchment_engine")

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_DYAD_LEDGER = PROCESSED_DIR / "clinical_dyad_ledger_v2.csv"
DEFAULT_PATIENT_DENSITY = PROCESSED_DIR / "patient_density_v2.csv"
DEFAULT_COMPETITOR_TRIALS = PROCESSED_DIR / "competitor_neuromodulation_trials.csv"
DEFAULT_CARE_COMPARE = RAW_DIR / "care_compare" / "DAC_NationalDownloadableFile.csv"
DEFAULT_FACILITY_AFFILIATION = RAW_DIR / "care_compare" / "Facility_Affiliation.csv"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR

ACTIVE_COMPETITOR_STATUSES = {
    "RECRUITING",
    "ACTIVE, NOT RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "NOT YET RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING BY INVITATION",
    "ENROLLING_BY_INVITATION",
}
STATUS_WEIGHT_MAP = {
    "RECRUITING": 1.0,
    "ACTIVE, NOT RECRUITING": 0.7,
    "ACTIVE_NOT_RECRUITING": 0.7,
    "NOT YET RECRUITING": 0.85,
    "NOT_YET_RECRUITING": 0.85,
    "ENROLLING BY INVITATION": 0.55,
    "ENROLLING_BY_INVITATION": 0.55,
}


@dataclass(frozen=True)
class SourcePaths:
    dyad_ledger: Path
    patient_density: Path
    competitor_trials: Path
    care_compare: Path
    facility_affiliation: Path
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
        description="Build competition-adjusted patient catchment rankings for viable clinical dyad hubs."
    )
    parser.add_argument("--dyad-ledger", type=Path, default=DEFAULT_DYAD_LEDGER)
    parser.add_argument("--patient-density", type=Path, default=DEFAULT_PATIENT_DENSITY)
    parser.add_argument("--competitor-trials", type=Path, default=DEFAULT_COMPETITOR_TRIALS)
    parser.add_argument("--care-compare", type=Path, default=DEFAULT_CARE_COMPARE)
    parser.add_argument("--facility-affiliation", type=Path, default=DEFAULT_FACILITY_AFFILIATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def zip5_expr(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.replace_all(r"[^0-9]", "")
        .str.slice(0, 5)
    )


def facility_key_expr(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9 ]", " ")
        .str.replace_all(
            r"\b(THE|OF|AT|AND|HOSPITAL|MEDICAL|CENTER|CTR|HEALTH|SYSTEM|CLINIC|UNIVERSITY|DEPARTMENT|DEPT)\b",
            " ",
        )
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def load_viable_dyads(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "Surgeon_NPI",
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Dyad_Partner_Name",
            "Dyad_Partner_Specialty",
            "Trial_Site_Friction_Flag",
            "Referral_Confidence_Tier",
        },
        "clinical dyad ledger",
    )
    return (
        frame.filter(pl.col("Trial_Site_Friction_Flag") != "High Friction Trial Site")
        .select(
            pl.col("Surgeon_NPI").cast(pl.Utf8, strict=False).fill_null(""),
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Dyad_Partner_Name",
            "Dyad_Partner_Specialty",
            "Trial_Site_Friction_Flag",
            "Referral_Confidence_Tier",
            pl.col("Dyad_Relationship_Score")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("Dyad_Relationship_Score"),
        )
        .with_columns(
            normalize_text_expr(pl.col("Surgeon_City")).alias("site_city_key"),
            normalize_text_expr(pl.col("Surgeon_State")).alias("site_state_key"),
        )
    )


def load_patient_density(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "Surgeon_NPI",
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Protocol_Eligible_Funnel_Estimate",
            "Base_Catchment_Source",
            "Local_Depression_Source",
            "Depression_Prevalence_Fallback_Flag",
            "MHLTH_Prevalence_Fallback_Flag",
        },
        "patient density v2",
    )
    return frame.select(
        pl.col("Surgeon_NPI").cast(pl.Utf8, strict=False).fill_null(""),
        pl.col("Protocol_Eligible_Funnel_Estimate")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Protocol_Eligible_Funnel_Estimate"),
        pl.col("Protocol_Eligible_Funnel_Estimate")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Circuit_Level_TAM"),
        "Base_Catchment_Source",
        "Local_Depression_Source",
        pl.col("County").fill_null("").alias("Catchment_County"),
        pl.col("County_TotalPop18plus")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("County_TotalPop18plus"),
        pl.col("Base_Catchment_Adult_Pop18plus")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Base_Catchment_Adult_Pop18plus"),
        pl.col("Base_Depression_Prevalence_Pct")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Base_Depression_Prevalence_Pct"),
        pl.col("Base_MHLTH_Prevalence_Pct")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Base_MHLTH_Prevalence_Pct"),
        pl.col("Local_Depression_Prevalence_Pct")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Local_Depression_Prevalence_Pct"),
        pl.col("Local_MHLTH_Prevalence_Pct")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("Local_MHLTH_Prevalence_Pct"),
        pl.col("Local_Density_Adjustment")
        .cast(pl.Float64, strict=False)
        .fill_null(1.0)
        .alias("Local_Density_Adjustment"),
        pl.col("Mental_Distress_Adjustment")
        .cast(pl.Float64, strict=False)
        .fill_null(1.0)
        .alias("Mental_Distress_Adjustment"),
        pl.col("Depression_Prevalence_Fallback_Flag")
        .cast(pl.Boolean, strict=False)
        .fill_null(False)
        .alias("Depression_Prevalence_Fallback_Flag"),
        pl.col("MHLTH_Prevalence_Fallback_Flag")
        .cast(pl.Boolean, strict=False)
        .fill_null(False)
        .alias("MHLTH_Prevalence_Fallback_Flag"),
    )


def load_site_reference(care_compare_path: Path, facility_affiliation_path: Path) -> pl.LazyFrame:
    dac = scan_csv(care_compare_path)
    facility = scan_csv(facility_affiliation_path)
    ensure_columns(
        dac,
        {"NPI", "Facility Name", "ZIP Code", "City/Town", "State"},
        "Care Compare DAC file",
    )
    ensure_columns(
        facility,
        {"NPI", "facility_type", "Facility Affiliations Certification Number"},
        "facility affiliation file",
    )

    site_facility = (
        dac.select(
            pl.col("NPI").cast(pl.Utf8, strict=False).fill_null("").alias("Surgeon_NPI"),
            zip5_expr(pl.col("ZIP Code")).alias("Surgeon_Zip5"),
            pl.col("City/Town").cast(pl.Utf8, strict=False).fill_null("").alias("Care_City"),
            pl.col("State").cast(pl.Utf8, strict=False).fill_null("").alias("Care_State"),
            pl.col("Facility Name").cast(pl.Utf8, strict=False).fill_null("").alias("Site_Facility_Name"),
        )
        .filter(pl.col("Surgeon_NPI") != "")
        .with_columns(facility_key_expr(pl.col("Site_Facility_Name")).alias("site_facility_key"))
        .group_by("Surgeon_NPI")
        .agg(
            pl.col("Surgeon_Zip5").filter(pl.col("Surgeon_Zip5") != "").first(),
            pl.col("Care_City").filter(pl.col("Care_City") != "").first(),
            pl.col("Care_State").filter(pl.col("Care_State") != "").first(),
            pl.col("Site_Facility_Name").filter(pl.col("Site_Facility_Name") != "").first(),
            pl.col("site_facility_key").filter(pl.col("site_facility_key") != "").first(),
        )
    )

    site_hospital = (
        facility.select(
            pl.col("NPI").cast(pl.Utf8, strict=False).fill_null("").alias("Surgeon_NPI"),
            pl.col("facility_type").cast(pl.Utf8, strict=False).fill_null("").alias("facility_type"),
            pl.col("Facility Affiliations Certification Number")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .alias("Site_Hospital_CCN"),
        )
        .filter(pl.col("Surgeon_NPI") != "")
        .filter(pl.col("facility_type").str.to_lowercase().str.contains("hospital"))
        .group_by("Surgeon_NPI")
        .agg(
            pl.col("Site_Hospital_CCN")
            .filter(pl.col("Site_Hospital_CCN") != "")
            .first()
        )
    )

    return site_facility.join(site_hospital, on="Surgeon_NPI", how="left")


def indication_weight_expr(column_name: str) -> pl.Expr:
    therapy = pl.col(column_name).cast(pl.Utf8, strict=False).fill_null("").str.to_lowercase()
    return (
        pl.when(
            therapy.str.contains("treatment resistant depression|major depressive|depression|trd")
        )
        .then(1.0)
        .when(therapy.str.contains("ocd|obsessive"))
        .then(0.85)
        .when(therapy.str.contains("mood|psychiatric|mental"))
        .then(0.7)
        .when(therapy.str.contains("epilepsy|seizure"))
        .then(0.45)
        .when(therapy.str.contains("parkinson|movement|tremor|dystonia"))
        .then(0.25)
        .when(therapy.str.contains("pain|spinal|peripheral"))
        .then(0.1)
        .otherwise(0.3)
    )


def load_competitor_trials(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {"Trial_ID", "City", "State", "Overall_Status", "Sponsor", "Therapy_Area", "Facility"},
        "competitor trials",
    )
    return (
        frame.select(
            pl.col("Trial_ID").cast(pl.Utf8, strict=False).fill_null(""),
            pl.col("City").cast(pl.Utf8, strict=False).fill_null("").alias("trial_city"),
            pl.col("State").cast(pl.Utf8, strict=False).fill_null("").alias("trial_state"),
            pl.col("Overall_Status").cast(pl.Utf8, strict=False).fill_null("").alias("Overall_Status"),
            pl.col("Sponsor").cast(pl.Utf8, strict=False).fill_null("").alias("Sponsor"),
            pl.col("Therapy_Area").cast(pl.Utf8, strict=False).fill_null("").alias("Therapy_Area"),
            pl.col("Facility").cast(pl.Utf8, strict=False).fill_null("").alias("Trial_Facility_Name"),
        )
        .with_columns(
            normalize_text_expr(pl.col("trial_city")).alias("trial_city_key"),
            normalize_text_expr(pl.col("trial_state")).alias("trial_state_key"),
            normalize_text_expr(pl.col("Overall_Status")).alias("trial_status_key"),
            facility_key_expr(pl.col("Trial_Facility_Name")).alias("trial_facility_key"),
            indication_weight_expr("Therapy_Area").alias("indication_overlap_weight"),
        )
        .filter(pl.col("trial_status_key").is_in(ACTIVE_COMPETITOR_STATUSES))
        .with_columns(
            pl.col("trial_status_key")
            .replace_strict(STATUS_WEIGHT_MAP, default=0.5)
            .cast(pl.Float64)
            .alias("status_weight")
        )
    )


def build_competition_adjusted_catchment(source_paths: SourcePaths) -> pl.DataFrame:
    viable_sites = load_viable_dyads(source_paths.dyad_ledger)
    patient_density = load_patient_density(source_paths.patient_density)
    site_reference = load_site_reference(
        source_paths.care_compare,
        source_paths.facility_affiliation,
    )
    competitor_trials = load_competitor_trials(source_paths.competitor_trials)

    site_base = (
        viable_sites.join(patient_density, on="Surgeon_NPI", how="left")
        .join(site_reference, on="Surgeon_NPI", how="left")
        .with_columns(
            pl.coalesce(["Care_City", "Surgeon_City"]).alias("Surgeon_City"),
            pl.coalesce(["Care_State", "Surgeon_State"]).alias("Surgeon_State"),
            normalize_text_expr(pl.coalesce(["Care_City", "Surgeon_City"])).alias("site_city_key"),
            normalize_text_expr(pl.coalesce(["Care_State", "Surgeon_State"])).alias("site_state_key"),
            pl.col("Site_Facility_Name").fill_null(""),
            pl.col("site_facility_key").fill_null(""),
        )
        .with_columns(
            pl.when(
                pl.col("Depression_Prevalence_Fallback_Flag")
                | pl.col("MHLTH_Prevalence_Fallback_Flag")
            )
            .then(pl.lit("Moderate"))
            .when(pl.col("Local_Depression_Source") == "NationalFallback")
            .then(pl.lit("Low"))
            .otherwise(pl.lit("High"))
            .alias("Catchment_Evidence_Tier")
        )
    )

    competition = (
        site_base.join(
            competitor_trials,
            left_on=["site_city_key", "site_state_key"],
            right_on=["trial_city_key", "trial_state_key"],
            how="left",
        )
        .with_columns(
            pl.when(
                (pl.col("site_facility_key") != "")
                & (pl.col("site_facility_key") == pl.col("trial_facility_key"))
            )
            .then(1)
            .otherwise(0)
            .alias("Facility_Overlap_Flag"),
            pl.when(pl.col("indication_overlap_weight") >= 0.65)
            .then(1)
            .otherwise(0)
            .alias("Indication_Match_Flag"),
        )
        .with_columns(
            (
                pl.col("status_weight")
                * pl.col("indication_overlap_weight")
                * pl.when(pl.col("Facility_Overlap_Flag") == 1)
                .then(1.75)
                .otherwise(1.0)
            )
            .fill_null(0.0)
            .round(3)
            .alias("competition_intensity_component")
        )
        .group_by("Surgeon_NPI")
        .agg(
            pl.col("Trial_ID").drop_nulls().n_unique().alias("Active_Competitor_Trials"),
            pl.col("Indication_Match_Flag").sum().alias("Indication_Matched_Trials"),
            pl.col("Facility_Overlap_Flag").sum().alias("Facility_Overlap_Trials"),
            pl.col("competition_intensity_component")
            .sum()
            .round(3)
            .alias("Competition_Intensity_Score"),
        )
    )

    return (
        site_base.join(competition, on="Surgeon_NPI", how="left")
        .with_columns(
            pl.col("Active_Competitor_Trials").fill_null(0),
            pl.col("Indication_Matched_Trials").fill_null(0),
            pl.col("Facility_Overlap_Trials").fill_null(0),
            pl.col("Competition_Intensity_Score").fill_null(0.0),
        )
        .with_columns(
            (
                (
                    pl.col("Competition_Intensity_Score") * 0.07
                    + pl.col("Facility_Overlap_Trials") * 0.04
                )
                .clip(lower_bound=0.0, upper_bound=0.6)
            )
            .round(4)
            .alias("Competition_Pressure_Fraction"),
        )
        .with_columns(
            (
                pl.col("Circuit_Level_TAM") * pl.col("Competition_Pressure_Fraction")
            )
            .round(2)
            .alias("Trial_Cannibalization_Penalty"),
            (
                pl.col("Circuit_Level_TAM")
                - pl.col("Circuit_Level_TAM") * pl.col("Competition_Pressure_Fraction")
            )
            .clip(lower_bound=0.0)
            .round(2)
            .alias("Net_Sourcing_Alpha"),
        )
        .with_columns(
            pl.when(pl.col("Competition_Pressure_Fraction") <= 0.05)
            .then(pl.lit("Blue Ocean"))
            .when(pl.col("Competition_Pressure_Fraction") <= 0.18)
            .then(pl.lit("Manageable"))
            .otherwise(pl.lit("Crowded"))
            .alias("Competition_Environment"),
            pl.when(pl.col("Circuit_Level_TAM") > 0)
            .then(pl.lit("Matched"))
            .otherwise(pl.lit("No Catchment Match"))
            .alias("Catchment_Match_Status"),
        )
        .sort(
            by=["Net_Sourcing_Alpha", "Circuit_Level_TAM", "Dyad_Relationship_Score"],
            descending=[True, True, True],
        )
        .select(
            "Surgeon_NPI",
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Dyad_Partner_Name",
            "Dyad_Partner_Specialty",
            "Referral_Confidence_Tier",
            "Dyad_Relationship_Score",
            "Site_Facility_Name",
            "Site_Hospital_CCN",
            "Catchment_County",
            "Base_Catchment_Source",
            "Local_Depression_Source",
            "Catchment_Evidence_Tier",
            "Depression_Prevalence_Fallback_Flag",
            "MHLTH_Prevalence_Fallback_Flag",
            "County_TotalPop18plus",
            "Base_Catchment_Adult_Pop18plus",
            "Base_Depression_Prevalence_Pct",
            "Base_MHLTH_Prevalence_Pct",
            "Local_Depression_Prevalence_Pct",
            "Local_MHLTH_Prevalence_Pct",
            "Local_Density_Adjustment",
            "Mental_Distress_Adjustment",
            "Protocol_Eligible_Funnel_Estimate",
            "Circuit_Level_TAM",
            "Active_Competitor_Trials",
            "Indication_Matched_Trials",
            "Facility_Overlap_Trials",
            "Competition_Intensity_Score",
            "Competition_Pressure_Fraction",
            "Trial_Cannibalization_Penalty",
            "Net_Sourcing_Alpha",
            "Competition_Environment",
            "Catchment_Match_Status",
        )
        .collect()
    )


def create_catchment_chart(catchment_df: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = catchment_df.sort("Net_Sourcing_Alpha").tail(14)

    labels = [
        f"{name} | {city}, {state}"
        for name, city, state in zip(
            ranked["Surgeon_Name"].to_list(),
            ranked["Surgeon_City"].to_list(),
            ranked["Surgeon_State"].to_list(),
            strict=False,
        )
    ]
    alpha_values = ranked["Net_Sourcing_Alpha"].to_list()
    tam_values = ranked["Circuit_Level_TAM"].to_list()

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    vmax = max(tam_values) if tam_values else 1.0
    vmin = min(tam_values) if tam_values else 0.0
    if vmax == vmin:
        vmax = vmin + 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["magma"]
    bar_colors = [cmap(norm(value)) for value in tam_values]

    bars = ax.barh(
        labels,
        alpha_values,
        color=bar_colors,
        edgecolor=bar_colors,
        linewidth=1.0,
        alpha=0.98,
    )

    for bar, alpha_value, tam_value in zip(bars, alpha_values, tam_values, strict=False):
        ax.text(
            alpha_value + max(alpha_values) * 0.01 if alpha_values else 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{alpha_value:,.0f} | TAM {tam_value:,.0f}",
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
    colorbar.set_label("Protocol-Eligible Funnel Estimate", color="#d4d4d4", fontsize=10)

    ax.set_title(
        "Competition-Adjusted Clinical Catchment | Viable Trial Hubs",
        loc="left",
        fontsize=18,
        color="#f5f5f5",
        pad=18,
        fontweight="bold",
    )
    ax.set_xlabel("Net Sourcing Alpha", color="#d4d4d4", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="x", colors="#d4d4d4", labelsize=10)
    ax.tick_params(axis="y", colors="#f5f5f5", labelsize=10)
    ax.grid(axis="x", color="#3a3a3a", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    footer = (
        "Alpha reflects the protocol-eligible psychiatric funnel after weighting active competitor trials "
        "by status, indication overlap, and same-facility overlap."
    )
    fig.text(0.01, 0.015, footer, color="#cfcfcf", fontsize=9)

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    LOGGER.info("Wrote catchment chart to %s", output_path)


def write_outputs(catchment_df: pl.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "viable_catchment_areas.csv"
    chart_path = output_dir / "viable_catchment_areas.png"
    catchment_df.write_csv(ledger_path)
    create_catchment_chart(catchment_df, chart_path)
    LOGGER.info("Wrote catchment ledger to %s", ledger_path)
    return ledger_path, chart_path


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    source_paths = SourcePaths(
        dyad_ledger=args.dyad_ledger,
        patient_density=args.patient_density,
        competitor_trials=args.competitor_trials,
        care_compare=args.care_compare,
        facility_affiliation=args.facility_affiliation,
        output_dir=args.output_dir,
    )

    catchment_df = build_competition_adjusted_catchment(source_paths)
    write_outputs(catchment_df, source_paths.output_dir)
    LOGGER.info("Prepared %s viable competition-adjusted site rows.", len(catchment_df))


if __name__ == "__main__":
    main()
