from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import polars as pl


LOGGER = logging.getLogger("neuromodulation_targeting_matrix.drg_profitability_engine")

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_LEDGER_PATH = PROCESSED_DIR / "hospital_drg_economics.csv"
DEFAULT_SITE_LEDGER = PROCESSED_DIR / "clinical_dyad_ledger_v2.csv"
DEFAULT_SITE_OUTPUT_PATH = PROCESSED_DIR / "hospital_drg_site_adjusted.csv"
DEFAULT_CHART_PATH = PROCESSED_DIR / "drg_economics_visual.png"
DEFAULT_CARE_COMPARE = RAW_DIR / "care_compare" / "DAC_NationalDownloadableFile.csv"
DEFAULT_FACILITY_AFFILIATION = RAW_DIR / "care_compare" / "Facility_Affiliation.csv"
DEFAULT_IPPS_IMPACT = RAW_DIR / "ipps_2026" / "FY 2026 IPPS Final Rule Impact File.txt"


@dataclass(frozen=True)
class ScenarioAssumptions:
    scenario_name: str
    medicare_payout: float
    hardware_cost: float
    or_time_hours: float
    or_hourly_cost: float


SCENARIO_ASSUMPTIONS = (
    ScenarioAssumptions(
        scenario_name="Legacy System",
        medicare_payout=45_000.0,
        hardware_cost=28_000.0,
        or_time_hours=4.5,
        or_hourly_cost=2_500.0,
    ),
    ScenarioAssumptions(
        scenario_name="Target Device",
        medicare_payout=45_000.0,
        hardware_cost=18_000.0,
        or_time_hours=2.5,
        or_hourly_cost=2_500.0,
    ),
)


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
        description="Build deterministic national and site-adjusted hospital economics for a next-generation neuromodulation device."
    )
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--chart-path", type=Path, default=DEFAULT_CHART_PATH)
    parser.add_argument("--site-ledger", type=Path, default=DEFAULT_SITE_LEDGER)
    parser.add_argument("--site-output-path", type=Path, default=DEFAULT_SITE_OUTPUT_PATH)
    parser.add_argument("--facility-affiliation", type=Path, default=DEFAULT_FACILITY_AFFILIATION)
    parser.add_argument("--ipps-impact", type=Path, default=DEFAULT_IPPS_IMPACT)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def scan_csv(path: Path, **kwargs) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        infer_schema_length=5000,
        try_parse_dates=True,
        ignore_errors=True,
        encoding="utf8-lossy",
        null_values=["", "NULL", "null", "N/A", "n/a"],
        **kwargs,
    )


def ensure_columns(frame: pl.LazyFrame, required: Iterable[str], source_name: str) -> None:
    available = set(frame.collect_schema().names())
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}. "
            f"Available columns: {sorted(available)}"
        )


def build_financial_ledger() -> pl.DataFrame:
    scenario_rows = [
        {
            "Scenario": scenario.scenario_name,
            "Medicare_Payout": scenario.medicare_payout,
            "Hardware_Cost": scenario.hardware_cost,
            "OR_Time_Hours": scenario.or_time_hours,
            "OR_Hourly_Cost": scenario.or_hourly_cost,
        }
        for scenario in SCENARIO_ASSUMPTIONS
    ]

    return (
        pl.DataFrame(scenario_rows)
        .lazy()
        .with_columns(
            (pl.col("OR_Time_Hours") * pl.col("OR_Hourly_Cost"))
            .round(2)
            .alias("OR_Overhead_Cost"),
        )
        .with_columns(
            (pl.col("Hardware_Cost") + pl.col("OR_Overhead_Cost"))
            .round(2)
            .alias("Total_Procedure_Cost"),
        )
        .with_columns(
            (pl.col("Medicare_Payout") - pl.col("Total_Procedure_Cost"))
            .round(2)
            .alias("Net_Hospital_Profit"),
        )
        .collect()
    )


def calculate_profit_alpha(ledger: pl.DataFrame) -> float:
    legacy_profit = (
        ledger.filter(pl.col("Scenario") == "Legacy System")
        .select("Net_Hospital_Profit")
        .item()
    )
    target_profit = (
        ledger.filter(pl.col("Scenario") == "Target Device")
        .select("Net_Hospital_Profit")
        .item()
    )

    if legacy_profit == 0:
        raise ZeroDivisionError("Legacy net hospital profit is zero; alpha percentage is undefined.")

    return round(((target_profit - legacy_profit) / abs(legacy_profit)) * 100, 2)


def load_site_ledger(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(frame, {"Surgeon_NPI", "Surgeon_Name", "Surgeon_City", "Surgeon_State"}, "site ledger")
    return frame.select(
        pl.col("Surgeon_NPI").cast(pl.Utf8, strict=False).fill_null(""),
        "Surgeon_Name",
        "Surgeon_City",
        "Surgeon_State",
    )


def load_facility_affiliation(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {"NPI", "facility_type", "Facility Affiliations Certification Number"},
        "facility affiliation",
    )
    return (
        frame.select(
            pl.col("NPI").cast(pl.Utf8, strict=False).fill_null("").alias("Surgeon_NPI"),
            pl.col("facility_type").cast(pl.Utf8, strict=False).fill_null("").alias("facility_type"),
            pl.col("Facility Affiliations Certification Number")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.replace_all(r"[^0-9]", "")
            .str.zfill(6)
            .alias("Hospital_CCN"),
        )
        .filter(pl.col("Surgeon_NPI") != "")
        .filter(pl.col("Hospital_CCN") != "")
        .filter(pl.col("facility_type").str.to_lowercase().str.contains("hospital"))
        .group_by("Surgeon_NPI")
        .agg(pl.col("Hospital_CCN").first())
    )


def load_ipps_impact(path: Path) -> tuple[pl.LazyFrame, dict[str, float]]:
    frame = scan_csv(path, separator="\t", skip_rows=1)
    ensure_columns(
        frame,
        {
            "Provider Number",
            "Name",
            "FY 2026 Wage Index",
            "Operating CCR",
            "Proxy Value Based Purchasing Adjustment Factor",
            "Proxy Readmission Adjustment Factor",
            "Ownership Control Type",
            "URGEO",
            "Region",
        },
        "FY 2026 IPPS impact file",
    )

    prepared = (
        frame.select(
            pl.col("Provider Number")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.replace_all(r"[^0-9]", "")
            .str.zfill(6)
            .alias("Hospital_CCN"),
            pl.col("Name").cast(pl.Utf8, strict=False).fill_null("").alias("Hospital_Name"),
            pl.col("FY 2026 Wage Index")
            .cast(pl.Float64, strict=False)
            .fill_null(1.0)
            .alias("Site_Wage_Index"),
            pl.col("Operating CCR")
            .cast(pl.Float64, strict=False)
            .fill_null(0.3)
            .alias("Operating_CCR"),
            pl.col("Proxy Value Based Purchasing Adjustment Factor")
            .cast(pl.Float64, strict=False)
            .fill_null(1.0)
            .alias("Proxy_VBP_Factor"),
            pl.col("Proxy Readmission Adjustment Factor")
            .cast(pl.Float64, strict=False)
            .fill_null(1.0)
            .alias("Proxy_Readmission_Factor"),
            pl.col("Ownership Control Type")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .alias("Ownership_Control_Type"),
            pl.col("URGEO").cast(pl.Utf8, strict=False).fill_null("").alias("Urban_Rural_Flag"),
            pl.col("Region").cast(pl.Utf8, strict=False).fill_null("").alias("CMS_Region"),
        )
        .filter(pl.col("Hospital_CCN") != "")
    )

    medians = (
        prepared.select(
            pl.col("Site_Wage_Index").median().alias("median_wage_index"),
            pl.col("Operating_CCR").median().alias("median_operating_ccr"),
        )
        .collect()
        .to_dicts()[0]
    )
    return prepared, {
        "median_wage_index": float(medians["median_wage_index"] or 1.0),
        "median_operating_ccr": float(medians["median_operating_ccr"] or 0.3),
    }


def build_site_adjusted_economics(
    site_ledger_path: Path,
    facility_affiliation_path: Path,
    ipps_impact_path: Path,
) -> pl.DataFrame:
    legacy = next(s for s in SCENARIO_ASSUMPTIONS if s.scenario_name == "Legacy System")
    target = next(s for s in SCENARIO_ASSUMPTIONS if s.scenario_name == "Target Device")

    sites = load_site_ledger(site_ledger_path)
    site_ccn = load_facility_affiliation(facility_affiliation_path)
    ipps_impact, medians = load_ipps_impact(ipps_impact_path)

    return (
        sites.join(site_ccn, on="Surgeon_NPI", how="left")
        .join(ipps_impact, on="Hospital_CCN", how="left")
        .with_columns(
            pl.col("Site_Wage_Index").fill_null(medians["median_wage_index"]),
            pl.col("Operating_CCR").fill_null(medians["median_operating_ccr"]),
            pl.col("Proxy_VBP_Factor").fill_null(1.0),
            pl.col("Proxy_Readmission_Factor").fill_null(1.0),
            pl.col("Hospital_Name").fill_null("Unresolved Hospital"),
            pl.col("Ownership_Control_Type").fill_null("Unknown"),
            pl.col("Urban_Rural_Flag").fill_null("Unknown"),
            pl.col("CMS_Region").fill_null("Unknown"),
        )
        .with_columns(
            (
                pl.col("Site_Wage_Index") / pl.lit(medians["median_wage_index"])
            ).alias("Wage_Index_Factor"),
            (
                (pl.col("Operating_CCR") / pl.lit(medians["median_operating_ccr"]))
                .clip(lower_bound=0.9, upper_bound=1.1)
            ).alias("Cost_Complexity_Factor"),
        )
        .with_columns(
            (
                legacy.medicare_payout
                * pl.col("Proxy_VBP_Factor")
                * pl.col("Proxy_Readmission_Factor")
            )
            .round(2)
            .alias("Site_Adjusted_Medicare_Payout"),
            (
                legacy.or_hourly_cost
                * pl.col("Wage_Index_Factor")
                * pl.col("Cost_Complexity_Factor")
            )
            .round(2)
            .alias("Site_Adjusted_OR_Hourly_Cost"),
        )
        .with_columns(
            (pl.col("Site_Adjusted_OR_Hourly_Cost") * legacy.or_time_hours)
            .round(2)
            .alias("Legacy_OR_Overhead_Cost"),
            (pl.col("Site_Adjusted_OR_Hourly_Cost") * target.or_time_hours)
            .round(2)
            .alias("Target_OR_Overhead_Cost"),
        )
        .with_columns(
            (pl.lit(legacy.hardware_cost) + pl.col("Legacy_OR_Overhead_Cost"))
            .round(2)
            .alias("Legacy_Total_Procedure_Cost"),
            (pl.lit(target.hardware_cost) + pl.col("Target_OR_Overhead_Cost"))
            .round(2)
            .alias("Target_Total_Procedure_Cost"),
        )
        .with_columns(
            (pl.col("Site_Adjusted_Medicare_Payout") - pl.col("Legacy_Total_Procedure_Cost"))
            .round(2)
            .alias("Legacy_Net_Hospital_Profit"),
            (pl.col("Site_Adjusted_Medicare_Payout") - pl.col("Target_Total_Procedure_Cost"))
            .round(2)
            .alias("Target_Net_Hospital_Profit"),
        )
        .with_columns(
            (
                pl.col("Target_Net_Hospital_Profit") - pl.col("Legacy_Net_Hospital_Profit")
            )
            .round(2)
            .alias("Projected_Profit_Delta"),
            pl.lit(legacy.or_time_hours - target.or_time_hours).alias("Projected_OR_Time_Saved_Hours"),
        )
        .with_columns(
            pl.when(pl.col("Legacy_Net_Hospital_Profit") != 0)
            .then(
                (
                    pl.col("Projected_Profit_Delta")
                    / pl.col("Legacy_Net_Hospital_Profit").abs()
                    * 100
                ).round(2)
            )
            .otherwise(None)
            .alias("Projected_Profit_Uplift_Pct"),
            pl.when(
                pl.col("Hospital_CCN").is_null()
                | (pl.col("Hospital_CCN") == "")
                | (pl.col("Hospital_Name") == "Unresolved Hospital")
            )
            .then(pl.lit("Moderate"))
            .otherwise(pl.lit("High"))
            .alias("Economics_Evidence_Tier"),
        )
        .sort("Projected_Profit_Delta", descending=True)
        .collect()
    )


def write_ledger(ledger: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_csv(output_path)
    LOGGER.info("Wrote DRG economics ledger to %s", output_path)


def create_profitability_chart(
    ledger: pl.DataFrame,
    alpha_pct: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = ledger["Scenario"].to_list()
    hardware_costs = ledger["Hardware_Cost"].to_list()
    or_costs = ledger["OR_Overhead_Cost"].to_list()
    profits = ledger["Net_Hospital_Profit"].to_list()

    positions = list(range(len(scenarios)))
    bar_width = 0.24

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    hardware_bars = ax.bar(
        [position - bar_width for position in positions],
        hardware_costs,
        width=bar_width,
        color="#777777",
        edgecolor="#777777",
        label="Hardware Cost",
    )
    or_bars = ax.bar(
        positions,
        or_costs,
        width=bar_width,
        color="#bbbbbb",
        edgecolor="#bbbbbb",
        label="OR Overhead Cost",
    )
    profit_bars = ax.bar(
        [position + bar_width for position in positions],
        profits,
        width=bar_width,
        color="#00f0ff",
        edgecolor="#00f0ff",
        label="Contribution Margin",
    )

    for bars in (hardware_bars, or_bars, profit_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 600,
                f"${height:,.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#f5f5f5",
            )

    ax.set_title(
        "Hospital Economics | Modeled Per-Case Contribution Margin Scenario",
        loc="left",
        fontsize=18,
        color="#f5f5f5",
        pad=18,
        fontweight="bold",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(scenarios, color="#f5f5f5", fontsize=11)
    ax.set_ylabel("USD", color="#d4d4d4", fontsize=11)
    ax.tick_params(axis="y", colors="#d4d4d4", labelsize=10)
    ax.grid(axis="y", color="#3a3a3a", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    legend = ax.legend(frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color("#f5f5f5")

    fig.text(
        0.01,
        0.015,
        "Scenario comparison only. Values reflect modeled per-case contribution margins under the current hardware and OR-time assumptions.",
        color="#cfcfcf",
        fontsize=9,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    LOGGER.info("Wrote DRG economics visualization to %s", output_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    ledger = build_financial_ledger()
    alpha_pct = calculate_profit_alpha(ledger)
    site_adjusted = build_site_adjusted_economics(
        site_ledger_path=args.site_ledger,
        facility_affiliation_path=args.facility_affiliation,
        ipps_impact_path=args.ipps_impact,
    )

    print(f"Net Profit Increase %: {alpha_pct:.2f}%")

    write_ledger(ledger, args.ledger_path)
    write_ledger(site_adjusted, args.site_output_path)
    create_profitability_chart(ledger, alpha_pct, args.chart_path)


if __name__ == "__main__":
    main()
