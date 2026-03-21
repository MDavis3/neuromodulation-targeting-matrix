from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl


LOGGER = logging.getLogger("neuromodulation_targeting_matrix.patient_density_prep")

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DYAD_LEDGER = PROCESSED_DIR / "clinical_dyad_ledger.csv"
DEFAULT_GEOGRAPHY_SOURCE = (
    PROJECT_ROOT
    / "Medicare Geographic Variation - by National, State & County"
    / "2023"
    / "2014-2023 Medicare Fee-for-Service Geographic Variation Public Use File.csv"
)
DEFAULT_OUTPUT_PATH = PROCESSED_DIR / "medicare_patient_density.csv"

# The downloaded CMS geographic variation file is county-based rather than
# city-based. For the current top dyad hubs we therefore use a curated
# city-to-county crosswalk so each surgeon site can be anchored to the
# corresponding local county beneficiary base.
CITY_TO_COUNTY_CROSSWALK = [
    {"Surgeon_City": "San Francisco", "Surgeon_State": "CA", "County": "San Francisco"},
    {"Surgeon_City": "Stanford", "Surgeon_State": "CA", "County": "Santa Clara"},
    {"Surgeon_City": "Rochester", "Surgeon_State": "MN", "County": "Olmsted"},
    {"Surgeon_City": "Iowa City", "Surgeon_State": "IA", "County": "Johnson"},
    {"Surgeon_City": "Pittsburgh", "Surgeon_State": "PA", "County": "Allegheny"},
    {"Surgeon_City": "Los Angeles", "Surgeon_State": "CA", "County": "Los Angeles"},
    {"Surgeon_City": "New York", "Surgeon_State": "NY", "County": "New York"},
    {"Surgeon_City": "Durham", "Surgeon_State": "NC", "County": "Durham"},
    {"Surgeon_City": "Boston", "Surgeon_State": "MA", "County": "Suffolk"},
    {"Surgeon_City": "Cleveland", "Surgeon_State": "OH", "County": "Cuyahoga"},
    {"Surgeon_City": "Saint Louis", "Surgeon_State": "MO", "County": "St. Louis City"},
    {"Surgeon_City": "Philadelphia", "Surgeon_State": "PA", "County": "Philadelphia"},
    {"Surgeon_City": "Chapel Hill", "Surgeon_State": "NC", "County": "Orange"},
    {"Surgeon_City": "Kansas City", "Surgeon_State": "MO", "County": "Jackson"},
]


@dataclass(frozen=True)
class SourcePaths:
    dyad_ledger: Path
    geography_source: Path
    output_path: Path


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a site-level Medicare patient density proxy file."
    )
    parser.add_argument("--dyad-ledger", type=Path, default=DEFAULT_DYAD_LEDGER)
    parser.add_argument("--geography-source", type=Path, default=DEFAULT_GEOGRAPHY_SOURCE)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        infer_schema_length=5000,
        try_parse_dates=True,
        ignore_errors=True,
        encoding="utf8-lossy",
        null_values=["", "NULL", "null", "N/A", "n/a", "*"],
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


def load_viable_sites(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {"Surgeon_Name", "Surgeon_City", "Surgeon_State", "Trial_Site_Friction_Flag"},
        "clinical dyad ledger",
    )
    return (
        frame.select(
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Trial_Site_Friction_Flag",
        )
        .filter(pl.col("Trial_Site_Friction_Flag") != "High Friction Trial Site")
        .unique(subset=["Surgeon_City", "Surgeon_State"])
        .with_columns(
            normalize_text_expr(pl.col("Surgeon_City")).alias("city_key"),
            normalize_text_expr(pl.col("Surgeon_State")).alias("state_key"),
        )
    )


def load_site_crosswalk(viable_sites: pl.LazyFrame) -> pl.LazyFrame:
    crosswalk = pl.DataFrame(CITY_TO_COUNTY_CROSSWALK).lazy().with_columns(
        normalize_text_expr(pl.col("Surgeon_City")).alias("city_key"),
        normalize_text_expr(pl.col("Surgeon_State")).alias("state_key"),
        normalize_text_expr(pl.col("County")).alias("county_key"),
    )
    return viable_sites.join(crosswalk, on=["city_key", "state_key"], how="left")


def load_geography(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "YEAR",
            "BENE_GEO_LVL",
            "BENE_GEO_DESC",
            "BENE_AGE_LVL",
            "BENES_TOTAL_CNT",
            "BENE_AVG_AGE",
            "BENE_AVG_RISK_SCRE",
        },
        "Medicare geographic variation file",
    )
    return (
        frame.select(
            pl.col("YEAR").cast(pl.Int64, strict=False).alias("YEAR"),
            pl.col("BENE_GEO_LVL").cast(pl.Utf8, strict=False).alias("BENE_GEO_LVL"),
            pl.col("BENE_GEO_DESC").cast(pl.Utf8, strict=False).alias("BENE_GEO_DESC"),
            pl.col("BENE_AGE_LVL").cast(pl.Utf8, strict=False).alias("BENE_AGE_LVL"),
            pl.col("BENES_TOTAL_CNT")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("Total_Beneficiaries"),
            pl.col("BENE_AVG_AGE")
            .cast(pl.Float64, strict=False)
            .fill_null(72.0)
            .alias("BENE_AVG_AGE"),
            pl.col("BENE_AVG_RISK_SCRE")
            .cast(pl.Float64, strict=False)
            .fill_null(1.0)
            .alias("BENE_AVG_RISK_SCRE"),
        )
        .filter(pl.col("YEAR") == 2023)
        .filter(pl.col("BENE_GEO_LVL") == "County")
        .filter(pl.col("BENE_AGE_LVL") == "All")
        .with_columns(
            pl.col("BENE_GEO_DESC")
            .str.extract(r"^([A-Z]{2})-", 1)
            .alias("State"),
            pl.col("BENE_GEO_DESC")
            .str.extract(r"^[A-Z]{2}-(.*)$", 1)
            .alias("County"),
        )
        .with_columns(
            normalize_text_expr(pl.col("State")).alias("state_key"),
            normalize_text_expr(pl.col("County")).alias("county_key"),
            pl.col("BENE_AVG_RISK_SCRE").clip(lower_bound=0.75, upper_bound=1.35).alias(
                "risk_multiplier"
            ),
            (pl.col("BENE_AVG_AGE") / 72.0)
            .clip(lower_bound=0.90, upper_bound=1.15)
            .alias("age_multiplier"),
        )
        .with_columns(
            # The source file does not include disease-specific prevalence counts.
            # These counts are therefore modeled as beneficiary-base proxies using
            # conservative baseline prevalence assumptions adjusted by local CMS
            # risk and age intensity so the resulting TAM is geographically ranked
            # rather than treated as a literal epidemiology table.
            (pl.col("Total_Beneficiaries") * 0.14 * pl.col("risk_multiplier"))
            .round(2)
            .alias("Depression_Count"),
            (
                pl.col("Total_Beneficiaries")
                * 0.015
                * pl.col("risk_multiplier")
                * pl.col("age_multiplier")
            )
            .round(2)
            .alias("Parkinsons_Count"),
            (pl.col("Total_Beneficiaries") * 0.012 * pl.col("risk_multiplier"))
            .round(2)
            .alias("Epilepsy_Count"),
        )
    )


def build_patient_density_proxy(
    dyad_ledger_path: Path,
    geography_source_path: Path,
) -> pl.DataFrame:
    viable_sites = load_viable_sites(dyad_ledger_path)
    site_crosswalk = load_site_crosswalk(viable_sites)
    county_geo = load_geography(geography_source_path)

    result = (
        site_crosswalk.join(
            county_geo,
            on=["state_key", "county_key"],
            how="left",
        )
        .select(
            pl.col("State"),
            pl.col("Surgeon_City").alias("City"),
            pl.col("County"),
            pl.col("Total_Beneficiaries"),
            pl.col("Depression_Count"),
            pl.col("Parkinsons_Count"),
            pl.col("Epilepsy_Count"),
            pl.col("risk_multiplier").alias("Risk_Multiplier"),
            pl.col("age_multiplier").alias("Age_Multiplier"),
        )
        .collect()
    )

    missing = result.filter(pl.col("County").is_null() | pl.col("State").is_null())
    if missing.height:
        missing_sites = missing.select("City").to_series().to_list()
        raise ValueError(f"Missing county crosswalk or geography rows for sites: {missing_sites}")
    return result


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    source_paths = SourcePaths(
        dyad_ledger=args.dyad_ledger,
        geography_source=args.geography_source,
        output_path=args.output_path,
    )
    patient_density = build_patient_density_proxy(
        dyad_ledger_path=source_paths.dyad_ledger,
        geography_source_path=source_paths.geography_source,
    )
    source_paths.output_path.parent.mkdir(parents=True, exist_ok=True)
    patient_density.write_csv(source_paths.output_path)
    LOGGER.info("Wrote patient density proxy file to %s", source_paths.output_path)
    LOGGER.info("Prepared %s site-level density rows.", len(patient_density))


if __name__ == "__main__":
    main()
