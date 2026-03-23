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
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_DYAD_LEDGER = PROCESSED_DIR / "clinical_dyad_ledger.csv"
DEFAULT_CARE_COMPARE = RAW_DIR / "care_compare" / "DAC_NationalDownloadableFile.csv"
DEFAULT_COUNTY_PLACES = (
    RAW_DIR / "cdc_places" / "PLACES__Local_Data_for_Better_Health,_County_Data,_2025_release_20260323.csv"
)
DEFAULT_PLACE_PLACES = RAW_DIR / "cdc_places" / "places_mental_health_2025_wide.csv"
DEFAULT_ZCTA_PLACES = RAW_DIR / "cdc_places" / "zcta_mental_health_2025_wide.csv"
DEFAULT_OUTPUT_PATH = PROCESSED_DIR / "patient_density_v2.csv"

# County remains the most defensible public-data catchment surface for site
# sizing. The crosswalk only anchors each surgical hub to its principal county;
# local place/ZCTA prevalence then adjusts that county funnel for within-market
# density instead of pretending the entire county is equally reachable.
CITY_TO_COUNTY_CROSSWALK = [
    {"Surgeon_City": "San Francisco", "Surgeon_State": "CA", "County": "San Francisco"},
    {"Surgeon_City": "Stanford", "Surgeon_State": "CA", "County": "Santa Clara"},
    {"Surgeon_City": "Rochester", "Surgeon_State": "MN", "County": "Olmsted"},
    {"Surgeon_City": "Iowa City", "Surgeon_State": "IA", "County": "Johnson"},
    {"Surgeon_City": "Pittsburgh", "Surgeon_State": "PA", "County": "Allegheny"},
    {"Surgeon_City": "Los Angeles", "Surgeon_State": "CA", "County": "Los Angeles"},
    {"Surgeon_City": "Dallas", "Surgeon_State": "TX", "County": "Dallas"},
    {"Surgeon_City": "Fort Worth", "Surgeon_State": "TX", "County": "Tarrant"},
    {"Surgeon_City": "New York", "Surgeon_State": "NY", "County": "New York"},
    {"Surgeon_City": "Durham", "Surgeon_State": "NC", "County": "Durham"},
    {"Surgeon_City": "Boston", "Surgeon_State": "MA", "County": "Suffolk"},
    {"Surgeon_City": "Cleveland", "Surgeon_State": "OH", "County": "Cuyahoga"},
    {"Surgeon_City": "Saint Louis", "Surgeon_State": "MO", "County": "St. Louis"},
    {"Surgeon_City": "Philadelphia", "Surgeon_State": "PA", "County": "Philadelphia"},
    {"Surgeon_City": "Chapel Hill", "Surgeon_State": "NC", "County": "Orange"},
    {"Surgeon_City": "Kansas City", "Surgeon_State": "MO", "County": "Jackson"},
]


@dataclass(frozen=True)
class SourcePaths:
    dyad_ledger: Path
    care_compare: Path
    county_places: Path
    place_places: Path
    zcta_places: Path
    output_path: Path


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a psychiatric launch-funnel density file from CDC PLACES place/ZCTA/county data."
    )
    parser.add_argument("--dyad-ledger", type=Path, default=DEFAULT_DYAD_LEDGER)
    parser.add_argument("--care-compare", type=Path, default=DEFAULT_CARE_COMPARE)
    parser.add_argument("--county-places", type=Path, default=DEFAULT_COUNTY_PLACES)
    parser.add_argument("--place-places", type=Path, default=DEFAULT_PLACE_PLACES)
    parser.add_argument("--zcta-places", type=Path, default=DEFAULT_ZCTA_PLACES)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def scan_csv(path: Path, **kwargs) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        infer_schema_length=5000,
        try_parse_dates=True,
        ignore_errors=True,
        encoding="utf8-lossy",
        null_values=["", "NULL", "null", "N/A", "n/a", "*"],
        **kwargs,
    )


def read_csv(path: Path, **kwargs) -> pl.DataFrame:
    return pl.read_csv(
        path,
        infer_schema_length=5000,
        try_parse_dates=True,
        ignore_errors=True,
        encoding="utf8-lossy",
        null_values=["", "NULL", "null", "N/A", "n/a", "*"],
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


def normalize_text_expr(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
        .str.replace_all(r"\bSAINT\b", "ST")
    )


def zip5_expr(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.replace_all(r"[^0-9]", "")
        .str.slice(0, 5)
    )


def load_viable_sites(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {"Surgeon_NPI", "Surgeon_Name", "Surgeon_City", "Surgeon_State", "Trial_Site_Friction_Flag"},
        "clinical dyad ledger",
    )
    return (
        frame.select(
            pl.col("Surgeon_NPI").cast(pl.Utf8, strict=False).fill_null("").alias("Surgeon_NPI"),
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Trial_Site_Friction_Flag",
        )
        .filter(pl.col("Trial_Site_Friction_Flag") != "High Friction Trial Site")
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


def load_care_compare_zip(path: Path) -> pl.LazyFrame:
    dac = scan_csv(path)
    ensure_columns(dac, {"NPI", "ZIP Code", "City/Town", "State"}, "Care Compare DAC file")
    return (
        dac.select(
            pl.col("NPI").cast(pl.Utf8, strict=False).fill_null("").alias("Surgeon_NPI"),
            zip5_expr(pl.col("ZIP Code")).alias("Surgeon_Zip5"),
            pl.col("City/Town").cast(pl.Utf8, strict=False).fill_null("").alias("Care_City"),
            pl.col("State").cast(pl.Utf8, strict=False).fill_null("").alias("Care_State"),
        )
        .filter(pl.col("Surgeon_NPI") != "")
        .group_by("Surgeon_NPI")
        .agg(
            pl.col("Surgeon_Zip5").filter(pl.col("Surgeon_Zip5") != "").first(),
            pl.col("Care_City").filter(pl.col("Care_City") != "").first(),
            pl.col("Care_State").filter(pl.col("Care_State") != "").first(),
        )
    )


def load_county_places(path: Path) -> pl.LazyFrame:
    frame = read_csv(path).lazy()
    ensure_columns(
        frame,
        {"Year", "StateAbbr", "LocationName", "MeasureId", "Data_Value", "TotalPop18plus"},
        "CDC PLACES county file",
    )

    prepared = (
        frame.select(
            pl.col("Year").cast(pl.Int64, strict=False).alias("Year"),
            pl.col("StateAbbr").cast(pl.Utf8, strict=False).alias("StateAbbr"),
            pl.col("LocationName").cast(pl.Utf8, strict=False).alias("County"),
            pl.col("MeasureId").cast(pl.Utf8, strict=False).alias("MeasureId"),
            pl.col("Data_Value").cast(pl.Float64, strict=False).alias("Data_Value"),
            pl.col("TotalPop18plus")
            .cast(pl.Utf8, strict=False)
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False)
            .alias("TotalPop18plus"),
        )
        .with_columns(
            normalize_text_expr(pl.col("StateAbbr")).alias("state_key"),
            normalize_text_expr(pl.col("County")).alias("county_key"),
        )
    )

    county_population = prepared.group_by(
        ["StateAbbr", "County", "state_key", "county_key"]
    ).agg(
        pl.col("TotalPop18plus").sort_by("Year").last().alias("County_TotalPop18plus"),
        pl.col("Year").max().alias("County_Data_Year"),
    )

    county_prevalence = (
        prepared.filter(pl.col("MeasureId").is_in(["DEPRESSION", "MHLTH"]))
        .group_by(["StateAbbr", "County", "state_key", "county_key", "MeasureId"])
        .agg(
            pl.col("Data_Value").sort_by("Year").last().alias("Data_Value"),
            pl.col("Year").max().alias("Measure_Year"),
        )
        .group_by(["StateAbbr", "County", "state_key", "county_key"])
        .agg(
            pl.when(pl.col("MeasureId") == "DEPRESSION")
            .then(pl.col("Data_Value"))
            .otherwise(None)
            .max()
            .alias("County_Depression_Prevalence_Pct"),
            pl.when(pl.col("MeasureId") == "MHLTH")
            .then(pl.col("Data_Value"))
            .otherwise(None)
            .max()
            .alias("County_MHLTH_Prevalence_Pct"),
            pl.when(pl.col("MeasureId") == "DEPRESSION")
            .then(pl.col("Measure_Year"))
            .otherwise(None)
            .max()
            .alias("County_Depression_Data_Year"),
            pl.when(pl.col("MeasureId") == "MHLTH")
            .then(pl.col("Measure_Year"))
            .otherwise(None)
            .max()
            .alias("County_MHLTH_Data_Year"),
        )
    )

    return county_population.join(
        county_prevalence,
        on=["StateAbbr", "County", "state_key", "county_key"],
        how="left",
    ).select(
        "StateAbbr",
        "County",
        "state_key",
        "county_key",
        "County_TotalPop18plus",
        "County_Data_Year",
        "County_Depression_Prevalence_Pct",
        "County_MHLTH_Prevalence_Pct",
        "County_Depression_Data_Year",
        "County_MHLTH_Data_Year",
    )


def load_place_places(path: Path) -> pl.LazyFrame:
    frame = read_csv(path).lazy()
    ensure_columns(
        frame,
        {"StateAbbr", "LocationName", "TotalPop18plus", "PrevalencePct_DEPRESSION", "PrevalencePct_MHLTH"},
        "CDC PLACES place file",
    )
    return (
        frame.select(
            pl.col("StateAbbr").cast(pl.Utf8, strict=False).alias("StateAbbr"),
            pl.col("LocationName").cast(pl.Utf8, strict=False).alias("Place"),
            pl.col("TotalPop18plus").cast(pl.Float64, strict=False).alias("Place_TotalPop18plus"),
            pl.col("PrevalencePct_DEPRESSION")
            .cast(pl.Float64, strict=False)
            .alias("Place_Depression_Prevalence_Pct"),
            pl.col("PrevalencePct_MHLTH")
            .cast(pl.Float64, strict=False)
            .alias("Place_MHLTH_Prevalence_Pct"),
        )
        .with_columns(
            normalize_text_expr(pl.col("StateAbbr")).alias("state_key"),
            normalize_text_expr(pl.col("Place")).alias("city_key"),
        )
    )


def load_zcta_places(path: Path) -> pl.LazyFrame:
    frame = read_csv(path).lazy()
    ensure_columns(
        frame,
        {"LocationID", "TotalPop18plus", "PrevalencePct_DEPRESSION", "PrevalencePct_MHLTH"},
        "CDC PLACES ZCTA file",
    )
    return frame.select(
        pl.col("LocationID").cast(pl.Utf8, strict=False).alias("Surgeon_Zip5"),
        pl.col("TotalPop18plus").cast(pl.Float64, strict=False).alias("ZCTA_TotalPop18plus"),
        pl.col("PrevalencePct_DEPRESSION")
        .cast(pl.Float64, strict=False)
        .alias("ZCTA_Depression_Prevalence_Pct"),
        pl.col("PrevalencePct_MHLTH").cast(pl.Float64, strict=False).alias("ZCTA_MHLTH_Prevalence_Pct"),
    )


def build_patient_density_proxy(
    dyad_ledger_path: Path,
    care_compare_path: Path,
    county_places_path: Path,
    place_places_path: Path,
    zcta_places_path: Path,
) -> pl.DataFrame:
    viable_sites = load_viable_sites(dyad_ledger_path)
    site_crosswalk = load_site_crosswalk(viable_sites)
    care_compare_zip = load_care_compare_zip(care_compare_path)
    county_places = load_county_places(county_places_path)
    place_places = load_place_places(place_places_path)
    zcta_places = load_zcta_places(zcta_places_path)

    result = (
        site_crosswalk.join(care_compare_zip, on="Surgeon_NPI", how="left")
        .with_columns(
            pl.coalesce(["Care_City", "Surgeon_City"]).alias("Surgeon_City"),
            pl.coalesce(["Care_State", "Surgeon_State"]).alias("Surgeon_State"),
            normalize_text_expr(pl.coalesce(["Care_City", "Surgeon_City"])).alias("city_key"),
            normalize_text_expr(pl.coalesce(["Care_State", "Surgeon_State"])).alias("state_key"),
        )
        .join(county_places, on=["state_key", "county_key"], how="left")
        .join(place_places, on=["state_key", "city_key"], how="left")
        .join(zcta_places, on="Surgeon_Zip5", how="left")
        .with_columns(
            (
                pl.col("County_Depression_Prevalence_Pct").is_null()
                & pl.col("Place_Depression_Prevalence_Pct").is_null()
                & pl.col("ZCTA_Depression_Prevalence_Pct").is_null()
            ).alias("Depression_Prevalence_Fallback_Flag"),
            (
                pl.col("County_MHLTH_Prevalence_Pct").is_null()
                & pl.col("Place_MHLTH_Prevalence_Pct").is_null()
                & pl.col("ZCTA_MHLTH_Prevalence_Pct").is_null()
            ).alias("MHLTH_Prevalence_Fallback_Flag"),
            pl.coalesce(
                [
                    pl.col("County_TotalPop18plus"),
                    pl.col("Place_TotalPop18plus"),
                    pl.col("ZCTA_TotalPop18plus"),
                ]
            ).alias("Base_Catchment_Adult_Pop18plus"),
            pl.coalesce(
                [
                    pl.col("County_Depression_Prevalence_Pct"),
                    pl.col("Place_Depression_Prevalence_Pct"),
                    pl.col("ZCTA_Depression_Prevalence_Pct"),
                    pl.col("County_Depression_Prevalence_Pct").mean(),
                ]
            ).alias("Base_Depression_Prevalence_Pct"),
            pl.coalesce(
                [
                    pl.col("County_MHLTH_Prevalence_Pct"),
                    pl.col("Place_MHLTH_Prevalence_Pct"),
                    pl.col("ZCTA_MHLTH_Prevalence_Pct"),
                    pl.col("County_MHLTH_Prevalence_Pct").mean(),
                ]
            ).alias("Base_MHLTH_Prevalence_Pct"),
            pl.when(pl.col("County_TotalPop18plus").is_not_null())
            .then(pl.lit("County"))
            .when(pl.col("Place_TotalPop18plus").is_not_null())
            .then(pl.lit("Place"))
            .otherwise(pl.lit("ZCTA"))
            .alias("Base_Catchment_Source"),
            pl.coalesce(
                [
                    pl.col("ZCTA_Depression_Prevalence_Pct"),
                    pl.col("Place_Depression_Prevalence_Pct"),
                    pl.col("County_Depression_Prevalence_Pct"),
                    pl.col("County_Depression_Prevalence_Pct").mean(),
                ]
            ).alias("Local_Depression_Prevalence_Pct"),
            pl.coalesce(
                [
                    pl.col("ZCTA_MHLTH_Prevalence_Pct"),
                    pl.col("Place_MHLTH_Prevalence_Pct"),
                    pl.col("County_MHLTH_Prevalence_Pct"),
                    pl.col("County_MHLTH_Prevalence_Pct").mean(),
                ]
            ).alias("Local_MHLTH_Prevalence_Pct"),
            pl.when(pl.col("ZCTA_Depression_Prevalence_Pct").is_not_null())
            .then(pl.lit("ZCTA"))
            .when(pl.col("Place_Depression_Prevalence_Pct").is_not_null())
            .then(pl.lit("Place"))
            .when(pl.col("County_Depression_Prevalence_Pct").is_not_null())
            .then(pl.lit("County"))
            .otherwise(pl.lit("NationalFallback"))
            .alias("Local_Depression_Source"),
        )
        .with_columns(
            # County adult depression prevalence is the strongest public-data
            # surface for site-level psychiatric opportunity because it combines
            # a real local prevalence estimate with a real adult population
            # denominator. We still use a 15% TRD subset multiplier because
            # public PLACES data is depression prevalence, not treatment
            # resistance; this is now anchored to measured prevalence rather than
            # a generic beneficiary proxy.
            (
                pl.col("Base_Catchment_Adult_Pop18plus")
                * (pl.col("Base_Depression_Prevalence_Pct") / 100.0)
            )
            .round(2)
            .alias("County_Depressed_Adults_Estimate"),
            (
                pl.col("Base_Catchment_Adult_Pop18plus")
                * (pl.col("Base_Depression_Prevalence_Pct") / 100.0)
                * 0.15
            )
            .round(2)
            .alias("County_TRD_Adults_Estimate"),
        )
        .with_columns(
            (
                (
                    pl.col("Local_Depression_Prevalence_Pct")
                    / pl.col("Base_Depression_Prevalence_Pct")
                )
                .clip(lower_bound=0.85, upper_bound=1.15)
            )
            .fill_nan(1.0)
            .fill_null(1.0)
            .alias("Local_Density_Adjustment"),
            (
                (
                    pl.col("Local_MHLTH_Prevalence_Pct")
                    / pl.col("Base_MHLTH_Prevalence_Pct")
                )
                .clip(lower_bound=0.90, upper_bound=1.10)
            )
            .fill_nan(1.0)
            .fill_null(1.0)
            .alias("Mental_Distress_Adjustment"),
        )
        .with_columns(
            (
                pl.col("County_TRD_Adults_Estimate")
                * pl.col("Local_Density_Adjustment")
                * pl.col("Mental_Distress_Adjustment")
            )
            .round(2)
            .alias("Protocol_Eligible_Funnel_Estimate")
        )
        .select(
            "Surgeon_NPI",
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Surgeon_Zip5",
            "County",
            "County_TotalPop18plus",
            "County_Depression_Prevalence_Pct",
            "County_MHLTH_Prevalence_Pct",
            "Place_TotalPop18plus",
            "Place_Depression_Prevalence_Pct",
            "Place_MHLTH_Prevalence_Pct",
            "ZCTA_TotalPop18plus",
            "ZCTA_Depression_Prevalence_Pct",
            "ZCTA_MHLTH_Prevalence_Pct",
            "Local_Depression_Source",
            "Local_Depression_Prevalence_Pct",
            "Local_MHLTH_Prevalence_Pct",
            "Local_Density_Adjustment",
            "Mental_Distress_Adjustment",
            "Base_Catchment_Source",
            "Base_Catchment_Adult_Pop18plus",
            "Base_Depression_Prevalence_Pct",
            "Base_MHLTH_Prevalence_Pct",
            "Depression_Prevalence_Fallback_Flag",
            "MHLTH_Prevalence_Fallback_Flag",
            "County_Depressed_Adults_Estimate",
            "County_TRD_Adults_Estimate",
            "Protocol_Eligible_Funnel_Estimate",
        )
        .collect()
    )

    missing = result.filter(
        pl.col("Base_Catchment_Adult_Pop18plus").is_null()
        | pl.col("Base_Depression_Prevalence_Pct").is_null()
    )
    if missing.height:
        missing_sites = missing.select("Surgeon_City").to_series().to_list()
        raise ValueError(
            f"Missing county crosswalk or county PLACES depression rows for sites: {missing_sites}"
        )
    return result


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    source_paths = SourcePaths(
        dyad_ledger=args.dyad_ledger,
        care_compare=args.care_compare,
        county_places=args.county_places,
        place_places=args.place_places,
        zcta_places=args.zcta_places,
        output_path=args.output_path,
    )
    patient_density = build_patient_density_proxy(
        dyad_ledger_path=source_paths.dyad_ledger,
        care_compare_path=source_paths.care_compare,
        county_places_path=source_paths.county_places,
        place_places_path=source_paths.place_places,
        zcta_places_path=source_paths.zcta_places,
    )
    source_paths.output_path.parent.mkdir(parents=True, exist_ok=True)
    patient_density.write_csv(source_paths.output_path)
    LOGGER.info("Wrote patient density v2 file to %s", source_paths.output_path)
    LOGGER.info("Prepared %s site-level density rows.", len(patient_density))


if __name__ == "__main__":
    main()
