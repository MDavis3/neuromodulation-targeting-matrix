from __future__ import annotations

import argparse
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import polars as pl


LOGGER = logging.getLogger("neuromodulation_targeting_matrix.fetch_cdc_places")

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cdc_places"
SOCRATA_RESOURCE_ROOT = "https://data.cdc.gov/resource"

# These are the only PLACES measures needed for the current psychiatric
# launch-funnel upgrade. Pulling the full catalog would add a large amount of
# noise and file size without improving the sourcing model.
DEFAULT_MEASURES = ("DEPRESSION", "MHLTH")
DEFAULT_YEAR = "2023"
DEFAULT_DATA_VALUE_TYPE = "CrdPrv"

DATASET_SPECS = {
    "places": {
        "dataset_id": "eav7-hnsx",
        "label": "PLACES place-level data",
        "output_name": "places_mental_health_2025.csv",
        "select_columns": (
            "year",
            "stateabbr",
            "statedesc",
            "locationname",
            "locationid",
            "datasource",
            "category",
            "measure",
            "measureid",
            "data_value_unit",
            "data_value_type",
            "data_value",
            "low_confidence_limit",
            "high_confidence_limit",
            "totalpopulation",
            "totalpop18plus",
            "datavaluetypeid",
            "short_question_text",
            "geolocation",
        ),
    },
    "zcta": {
        "dataset_id": "qnzd-25i4",
        "label": "PLACES ZCTA-level data",
        "output_name": "zcta_mental_health_2025.csv",
        "select_columns": (
            "year",
            "locationname",
            "locationid",
            "datasource",
            "category",
            "measure",
            "measureid",
            "data_value_unit",
            "data_value_type",
            "data_value",
            "low_confidence_limit",
            "high_confidence_limit",
            "totalpopulation",
            "totalpop18plus",
            "datavaluetypeid",
            "short_question_text",
            "geolocation",
        ),
    },
}


@dataclass(frozen=True)
class FetchSpec:
    dataset_id: str
    label: str
    output_path: Path
    select_columns: tuple[str, ...]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch filtered CDC PLACES mental-health prevalence files for place and ZCTA geographies."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=["places", "zcta"],
    )
    parser.add_argument(
        "--measures",
        nargs="+",
        default=list(DEFAULT_MEASURES),
        help="PLACES MeasureId values to fetch. Defaults to the psychiatric launch funnel measures only.",
    )
    parser.add_argument("--year", default=DEFAULT_YEAR)
    parser.add_argument("--data-value-type-id", default=DEFAULT_DATA_VALUE_TYPE)
    parser.add_argument("--page-size", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_fetch_specs(dataset_names: list[str], output_dir: Path) -> list[FetchSpec]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs: list[FetchSpec] = []
    for dataset_name in dataset_names:
        spec = DATASET_SPECS[dataset_name]
        specs.append(
            FetchSpec(
                dataset_id=spec["dataset_id"],
                label=spec["label"],
                output_path=output_dir / spec["output_name"],
                select_columns=spec["select_columns"],
            )
        )
    return specs


def build_where_clause(year: str, data_value_type_id: str, measures: list[str]) -> str:
    quoted_measures = ", ".join(f"'{measure.upper()}'" for measure in measures)
    return (
        f"Year = '{year}' "
        f"AND DataValueTypeID = '{data_value_type_id}' "
        f"AND MeasureId IN ({quoted_measures})"
    )


def build_query_url(
    spec: FetchSpec,
    year: str,
    data_value_type_id: str,
    measures: list[str],
    page_size: int,
    offset: int,
) -> str:
    params = {
        "$select": ", ".join(spec.select_columns),
        "$where": build_where_clause(year, data_value_type_id, measures),
        "$order": "locationid, measureid",
        "$limit": str(page_size),
        "$offset": str(offset),
    }
    return (
        f"{SOCRATA_RESOURCE_ROOT}/{spec.dataset_id}.json?"
        f"{urllib.parse.urlencode(params)}"
    )


def fetch_json(url: str) -> list[dict]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "neuromodulation-targeting-matrix/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_dataset(
    spec: FetchSpec,
    year: str,
    data_value_type_id: str,
    measures: list[str],
    page_size: int,
) -> pl.DataFrame:
    offset = 0
    all_rows: list[dict] = []

    while True:
        url = build_query_url(
            spec=spec,
            year=year,
            data_value_type_id=data_value_type_id,
            measures=measures,
            page_size=page_size,
            offset=offset,
        )
        LOGGER.info(
            "Fetching %s rows %s-%s",
            spec.label,
            offset + 1,
            offset + page_size,
        )
        rows = fetch_json(url)
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    if not all_rows:
        raise ValueError(
            f"No rows returned for {spec.label}. Check the measure ids and year filter."
        )

    return (
        pl.DataFrame(all_rows)
        .select(
            pl.col("year").cast(pl.Int64, strict=False).alias("Year"),
            (
                pl.col("stateabbr").cast(pl.Utf8, strict=False)
                if "stateabbr" in all_rows[0]
                else pl.lit("")
            ).alias("StateAbbr"),
            (
                pl.col("statedesc").cast(pl.Utf8, strict=False)
                if "statedesc" in all_rows[0]
                else pl.lit("")
            ).alias("StateDesc"),
            pl.col("locationname").cast(pl.Utf8, strict=False).alias("LocationName"),
            pl.col("locationid").cast(pl.Utf8, strict=False).alias("LocationID"),
            pl.col("datasource").cast(pl.Utf8, strict=False).alias("DataSource"),
            pl.col("category").cast(pl.Utf8, strict=False).alias("Category"),
            pl.col("measure").cast(pl.Utf8, strict=False).alias("Measure"),
            pl.col("measureid").cast(pl.Utf8, strict=False).alias("MeasureId"),
            pl.col("data_value_unit").cast(pl.Utf8, strict=False).alias("Data_Value_Unit"),
            pl.col("data_value_type").cast(pl.Utf8, strict=False).alias("Data_Value_Type"),
            pl.col("data_value").cast(pl.Float64, strict=False).alias("Data_Value"),
            pl.col("low_confidence_limit")
            .cast(pl.Float64, strict=False)
            .alias("Low_Confidence_Limit"),
            pl.col("high_confidence_limit")
            .cast(pl.Float64, strict=False)
            .alias("High_Confidence_Limit"),
            pl.col("totalpopulation").cast(pl.Float64, strict=False).alias("TotalPopulation"),
            pl.col("totalpop18plus").cast(pl.Float64, strict=False).alias("TotalPop18plus"),
            pl.col("datavaluetypeid").cast(pl.Utf8, strict=False).alias("DataValueTypeID"),
            pl.col("short_question_text")
            .cast(pl.Utf8, strict=False)
            .alias("Short_Question_Text"),
            pl.col("geolocation").cast(pl.Utf8, strict=False).alias("Geolocation"),
        )
        .sort(["StateAbbr", "LocationName", "MeasureId"])
    )


def write_outputs(frame: pl.DataFrame, spec: FetchSpec) -> tuple[Path, Path]:
    long_path = spec.output_path
    wide_path = spec.output_path.with_name(spec.output_path.stem + "_wide.csv")

    frame.write_csv(long_path)

    wide = (
        frame.select(
            "Year",
            "StateAbbr",
            "StateDesc",
            "LocationName",
            "LocationID",
            "TotalPopulation",
            "TotalPop18plus",
            "MeasureId",
            pl.col("Data_Value").alias("PrevalencePct"),
            "Low_Confidence_Limit",
            "High_Confidence_Limit",
        )
        .pivot(
            index=[
                "Year",
                "StateAbbr",
                "StateDesc",
                "LocationName",
                "LocationID",
                "TotalPopulation",
                "TotalPop18plus",
            ],
            on="MeasureId",
            values=["PrevalencePct", "Low_Confidence_Limit", "High_Confidence_Limit"],
        )
        .sort(["StateAbbr", "LocationName"])
    )
    wide.write_csv(wide_path)

    LOGGER.info("Wrote %s rows to %s", frame.height, long_path)
    LOGGER.info("Wrote wide view to %s", wide_path)
    return long_path, wide_path


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    specs = build_fetch_specs(args.datasets, args.output_dir)

    measures = [measure.upper() for measure in args.measures]
    LOGGER.info(
        "Fetching CDC PLACES data for measures=%s year=%s type=%s",
        measures,
        args.year,
        args.data_value_type_id,
    )

    for spec in specs:
        frame = fetch_dataset(
            spec=spec,
            year=args.year,
            data_value_type_id=args.data_value_type_id,
            measures=measures,
            page_size=args.page_size,
        )
        write_outputs(frame, spec)


if __name__ == "__main__":
    main()
