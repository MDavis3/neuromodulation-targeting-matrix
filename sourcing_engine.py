from __future__ import annotations

"""Neuromodulation GTM and Clinical Sourcing Engine.

This build targets the official raw files under `data/raw/` and uses Polars
lazy execution for every flat-file source. AACT is supplied as a PostgreSQL
custom dump, so the script performs a targeted extraction pass for only the
tables required by the sourcing model.

Important modeling note:
The public MAUDE release does not expose physician NPIs. This engine therefore
implements a state-level competitor-friction proxy from MAUDE and joins that
signal back to physicians through their primary Medicare practice state.
"""

import argparse
import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import polars as pl


LOGGER = logging.getLogger("neuromodulation_targeting_matrix.sourcing_engine")

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AACT_EXTRACT_DIR = PROCESSED_DIR / "aact_extract"

TARGET_CPT_CODES = {"61867", "61868", "61885", "61886", "95822"}
TARGET_GRANT_ACTIVITY_CODES = {"R01", "U01"}
TARGET_PAYMENT_MANUFACTURERS = [
    "medtronic",
    "abbott",
    "st. jude",
    "st jude",
    "boston scientific",
]
TARGET_PAYMENT_NATURE_TERMS = [
    "consulting",
    "honoraria",
    "royalty",
    "speaker",
    "education",
    "travel",
    "food",
    "beverage",
]
TARGET_MAUDE_PROBLEM_CODES = {"1395", "2885", "1291", "1930"}
TARGET_MAUDE_DEVICE_TERMS = [
    "deep brain",
    "dbs",
    "spinal cord",
    "neurostimulator",
    "stimulator",
    "medtronic",
    "abbott",
    "boston scientific",
]
TARGET_NIH_KEYWORDS = [
    "neuromodulation",
    "brain-computer interface",
    "brain computer interface",
    "bci",
    "closed-loop",
    "closed loop",
    "neural interface",
    "functional neurosurgery",
    "deep brain stimulation",
]
LOW_RECRUITMENT_TERMS = [
    "low recruitment",
    "poor recruitment",
    "slow recruitment",
    "insufficient enrollment",
    "low enrollment",
    "accrual",
]
AACT_ACTIVE_STATUSES = {
    "recruiting",
    "not yet recruiting",
    "enrolling by invitation",
    "active, not recruiting",
}
DEFAULT_TOP_N = 15


@dataclass(frozen=True)
class SourcePaths:
    medicare: Path
    nih_projects: Path
    nih_abstracts: Path
    open_payments_general: tuple[Path, ...]
    maude_master: Path
    maude_devices: tuple[Path, ...]
    maude_problem_links: Path
    aact_dump: Path
    output_dir: Path


@dataclass(frozen=True)
class ScoringWeights:
    volume: float = 0.42
    nih: float = 0.28
    aact: float = 0.18
    maude: float = 0.12
    payments_penalty: float = 0.33
    low_recruitment_penalty: float = 0.22
    dual_threat_bonus: float = 10.0
    free_agent_bonus: float = 5.0
    friction_multiplier: float = 0.25


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
        description="Build the clinical sourcing ledger from raw official data."
    )
    parser.add_argument(
        "--medicare",
        type=Path,
        default=RAW_DIR / "medicare" / "MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv",
    )
    parser.add_argument(
        "--nih-projects",
        type=Path,
        default=RAW_DIR / "nih" / "RePORTER_PRJ_C_FY2025.csv",
    )
    parser.add_argument(
        "--nih-abstracts",
        type=Path,
        default=RAW_DIR / "nih" / "RePORTER_PRJABS_C_FY2025.csv",
    )
    parser.add_argument(
        "--open-payments-dir",
        type=Path,
        default=RAW_DIR / "open_payments",
    )
    parser.add_argument(
        "--maude-dir",
        type=Path,
        default=RAW_DIR / "maude",
    )
    parser.add_argument(
        "--aact-dump",
        type=Path,
        default=RAW_DIR / "aact" / "postgres.dmp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_source_paths(args: argparse.Namespace) -> SourcePaths:
    general_payment_files = tuple(
        sorted(args.open_payments_dir.glob("OP_DTL_GNRL_*.csv"))
    )
    maude_device_files = tuple(sorted(args.maude_dir.glob("DEVICE*.txt")))
    if not general_payment_files:
        raise FileNotFoundError("No Open Payments general payment CSV files found.")
    if not maude_device_files:
        raise FileNotFoundError("No MAUDE DEVICE*.txt files found.")
    return SourcePaths(
        medicare=args.medicare,
        nih_projects=args.nih_projects,
        nih_abstracts=args.nih_abstracts,
        open_payments_general=general_payment_files,
        maude_master=args.maude_dir / "mdrfoiThru2025.txt",
        maude_devices=maude_device_files,
        maude_problem_links=args.maude_dir / "foidevproblem.txt",
        aact_dump=args.aact_dump,
        output_dir=args.output_dir,
    )


def scan_csv(
    path: Path,
    *,
    separator: str = ",",
    has_header: bool = True,
    new_columns: list[str] | None = None,
    quote_char: str | None = '"',
) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        separator=separator,
        has_header=has_header,
        new_columns=new_columns,
        quote_char=quote_char,
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


def normalize_npi_expr(column_name: str) -> pl.Expr:
    return (
        pl.col(column_name)
        .cast(pl.Utf8, strict=False)
        .str.replace_all(r"\.0$", "")
        .str.replace_all(r"[^0-9]", "")
        .str.zfill(10)
    )


def normalize_text_expr(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.to_uppercase()
        .str.replace_all(r"\(CONTACT\)", "")
        .str.replace_all(
            r"\b(MD|M D|PHD|DO|D O|MSC|MS|MBA|MPH|JD|DDS|DMD)\b",
            "",
        )
        .str.replace_all(r"[^A-Z0-9 ]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def first_token_expr(expr: pl.Expr) -> pl.Expr:
    return normalize_text_expr(expr).str.extract(r"^([A-Z0-9]+)", 1).fill_null("")


def second_token_expr(expr: pl.Expr) -> pl.Expr:
    return (
        normalize_text_expr(expr)
        .str.extract(r"^[A-Z0-9]+\s+([A-Z0-9]+)", 1)
        .fill_null("")
    )


def last_token_expr(expr: pl.Expr) -> pl.Expr:
    return normalize_text_expr(expr).str.extract(r"([A-Z0-9]+)$", 1).fill_null("")


def middle_initial_expr(expr: pl.Expr) -> pl.Expr:
    return second_token_expr(expr).str.slice(0, 1).fill_null("")


def contains_any_text(column_name: str, terms: list[str]) -> pl.Expr:
    return pl.any_horizontal(
        [pl.col(column_name).str.contains(term, literal=True) for term in terms]
    )


def truthy_expr(column_name: str) -> pl.Expr:
    return pl.col(column_name).cast(pl.Utf8, strict=False).str.to_lowercase().is_in(
        ["t", "true", "1", "yes", "y"]
    )


def normalized_log_expr(column_name: str) -> pl.Expr:
    return (
        pl.when(pl.col(column_name) > 0)
        .then((pl.col(column_name) + 1).log(base=10))
        .otherwise(0.0)
        / pl.when(pl.col(column_name).max() > 0)
        .then((pl.col(column_name).max() + 1).log(base=10))
        .otherwise(1.0)
    )


def regex_escape_joined(terms: Iterable[str]) -> str:
    return "|".join(re.escape(term) for term in terms)


def parse_copy_columns(copy_stmt: str) -> list[str]:
    match = re.search(r"\((.*)\)\s+FROM\s+stdin;", copy_stmt, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse COPY statement: {copy_stmt}")
    return [column.strip() for column in match.group(1).split(",")]


def load_aact_metadata_only(dump_path: Path):
    from pgdumplib import constants
    from pgdumplib import dump as dumpmod

    dump = dumpmod.Dump()
    dump.entries = []
    dump._handle = open(dump_path, "rb")
    dump._read_header()
    if dump.version >= (1, 15, 0):
        dump.compression_algorithm = constants.COMPRESSION_ALGORITHMS[
            dump._compression_algorithm
        ]
    else:
        dump.compression_algorithm = (
            constants.COMPRESSION_GZIP
            if dump._read_int() != 0
            else constants.COMPRESSION_NONE
        )
    dump.timestamp = dump._read_timestamp()
    dump.dbname = dump._read_bytes().decode(dump.encoding)
    dump.server_version = dump._read_bytes().decode(dump.encoding)
    dump.dump_version = dump._read_bytes().decode(dump.encoding)
    dump._read_entries()
    dump._set_encoding()
    return dump


def cleanup_aact_dump(dump) -> None:
    if getattr(dump, "_handle", None) is not None and not dump._handle.closed:
        dump._handle.close()
    temp_dir = getattr(dump, "_temp_dir", None)
    if temp_dir is not None:
        try:
            temp_dir.cleanup()
        except Exception:
            LOGGER.debug("AACT temp cleanup hit a non-fatal error.", exc_info=True)


def extract_aact_table_from_dump(
    dump_path: Path,
    table_name: str,
    output_csv: Path,
) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and output_csv.stat().st_size > 0:
        LOGGER.info("Using cached AACT extract: %s", output_csv)
        return output_csv

    LOGGER.info("Extracting AACT table '%s' from %s", table_name, dump_path)
    from pgdumplib import constants

    dump = load_aact_metadata_only(dump_path)
    try:
        entry = next(
            e
            for e in dump.entries
            if e.namespace == "ctgov"
            and e.tag == table_name
            and e.desc == constants.TABLE_DATA
        )
        columns = parse_copy_columns(entry.copy_stmt)
        dump._handle.seek(entry.offset)
        _block_type, dump_id = dump._read_block_header()
        if dump_id != entry.dump_id:
            raise RuntimeError(
                f"Unexpected dump id for AACT table {table_name}: "
                f"{dump_id} != {entry.dump_id}"
            )
        dump._cache_table_data(dump_id)

        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in dump.table_data("ctgov", table_name):
                writer.writerow(row)
    finally:
        cleanup_aact_dump(dump)

    LOGGER.info("Wrote AACT extract: %s", output_csv)
    return output_csv


def prepare_aact_extracts(dump_path: Path, output_dir: Path) -> dict[str, Path]:
    return {
        "studies": extract_aact_table_from_dump(
            dump_path,
            "studies",
            output_dir / "studies.csv",
        ),
        "overall_officials": extract_aact_table_from_dump(
            dump_path,
            "overall_officials",
            output_dir / "overall_officials.csv",
        ),
        "facilities": extract_aact_table_from_dump(
            dump_path,
            "facilities",
            output_dir / "facilities.csv",
        ),
    }


def process_medicare_volume(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "Rndrng_NPI",
            "Rndrng_Prvdr_Last_Org_Name",
            "Rndrng_Prvdr_First_Name",
            "Rndrng_Prvdr_MI",
            "Rndrng_Prvdr_State_Abrvtn",
            "Rndrng_Prvdr_City",
            "Rndrng_Prvdr_Type",
            "HCPCS_Cd",
            "Tot_Srvcs",
        },
        "Medicare",
    )

    prepared = (
        frame.select(
            normalize_npi_expr("Rndrng_NPI").alias("npi"),
            pl.col("Rndrng_Prvdr_Last_Org_Name").alias("provider_last_name"),
            pl.col("Rndrng_Prvdr_First_Name").alias("provider_first_name"),
            pl.col("Rndrng_Prvdr_MI").alias("provider_middle_name"),
            pl.col("Rndrng_Prvdr_State_Abrvtn")
            .cast(pl.Utf8, strict=False)
            .alias("provider_state"),
            pl.col("Rndrng_Prvdr_City").cast(pl.Utf8, strict=False).alias("provider_city"),
            pl.col("Rndrng_Prvdr_Type").cast(pl.Utf8, strict=False).alias("provider_type"),
            pl.col("HCPCS_Cd").cast(pl.Utf8, strict=False).alias("hcpcs_code"),
            pl.col("Tot_Srvcs")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("total_services"),
        )
        .filter(pl.col("hcpcs_code").is_in(TARGET_CPT_CODES))
        .filter(pl.col("provider_type") == "Neurosurgery")
    )

    return (
        prepared.group_by("npi")
        .agg(
            pl.col("provider_last_name").first(),
            pl.col("provider_first_name").first(),
            pl.col("provider_middle_name").first(),
            pl.col("provider_state").first(),
            pl.col("provider_city").first(),
            pl.col("provider_type").first(),
            pl.col("total_services").sum().alias("total_surgical_volume"),
            pl.when(pl.col("hcpcs_code").is_in(["61867", "61868"]))
            .then(pl.col("total_services"))
            .otherwise(0.0)
            .sum()
            .alias("implant_lead_volume"),
            pl.when(pl.col("hcpcs_code").is_in(["61885", "61886"]))
            .then(pl.col("total_services"))
            .otherwise(0.0)
            .sum()
            .alias("ipg_volume"),
            pl.when(pl.col("hcpcs_code") == "95822")
            .then(pl.col("total_services"))
            .otherwise(0.0)
            .sum()
            .alias("intraop_monitoring_volume"),
        )
        .with_columns(
            normalize_text_expr(pl.col("provider_last_name")).alias("provider_last_norm"),
            first_token_expr(pl.col("provider_first_name")).alias("provider_first_norm"),
            middle_initial_expr(pl.col("provider_middle_name")).alias(
                "provider_middle_initial"
            ),
            pl.concat_str(
                [
                    normalize_text_expr(pl.col("provider_first_name")),
                    pl.lit(" "),
                    normalize_text_expr(pl.col("provider_last_name")),
                ]
            )
            .str.strip_chars()
            .alias("provider_name"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("provider_last_norm"),
                    pl.lit("|"),
                    pl.col("provider_first_norm"),
                    pl.lit("|"),
                    pl.col("provider_middle_initial"),
                ]
            ).alias("name_key_strict"),
            pl.concat_str(
                [
                    pl.col("provider_last_norm"),
                    pl.lit("|"),
                    pl.col("provider_first_norm"),
                ]
            ).alias("name_key_loose"),
        )
    )


def build_provider_aliases(provider_directory: pl.LazyFrame) -> pl.LazyFrame:
    strict_aliases = provider_directory.select(
        "npi",
        "provider_name",
        "provider_state",
        "provider_city",
        "total_surgical_volume",
        pl.col("name_key_strict").alias("name_key"),
        pl.lit(2).alias("match_quality"),
    )
    loose_aliases = provider_directory.select(
        "npi",
        "provider_name",
        "provider_state",
        "provider_city",
        "total_surgical_volume",
        pl.col("name_key_loose").alias("name_key"),
        pl.lit(1).alias("match_quality"),
    )
    return pl.concat([strict_aliases, loose_aliases]).unique(["npi", "name_key"])


def process_nih_reporter(
    projects_path: Path,
    abstracts_path: Path,
    provider_aliases: pl.LazyFrame,
) -> pl.LazyFrame:
    projects = scan_csv(projects_path)
    abstracts = scan_csv(abstracts_path)
    ensure_columns(
        projects,
        {
            "APPLICATION_ID",
            "ACTIVITY",
            "ORG_STATE",
            "PI_NAMEs",
            "PROJECT_TITLE",
            "PROJECT_TERMS",
            "TOTAL_COST",
        },
        "NIH projects",
    )
    ensure_columns(abstracts, {"APPLICATION_ID", "ABSTRACT_TEXT"}, "NIH abstracts")

    abstracts_prepared = abstracts.select(
        pl.col("APPLICATION_ID").cast(pl.Utf8, strict=False).alias("application_id"),
        pl.col("ABSTRACT_TEXT")
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .alias("abstract_text"),
    )

    combined_text = pl.concat_str(
        [
            pl.col("project_title"),
            pl.lit(" "),
            pl.col("project_terms"),
            pl.lit(" "),
            pl.col("abstract_text"),
        ]
    ).str.to_lowercase()

    filtered = (
        projects.select(
            pl.col("APPLICATION_ID").cast(pl.Utf8, strict=False).alias("application_id"),
            pl.col("ACTIVITY").cast(pl.Utf8, strict=False).str.to_uppercase().alias("activity_code"),
            pl.col("ORG_STATE")
            .cast(pl.Utf8, strict=False)
            .str.to_uppercase()
            .fill_null("")
            .alias("org_state"),
            pl.col("PI_NAMEs").cast(pl.Utf8, strict=False).fill_null("").alias("pi_names"),
            pl.col("PROJECT_TITLE").cast(pl.Utf8, strict=False).fill_null("").alias("project_title"),
            pl.col("PROJECT_TERMS").cast(pl.Utf8, strict=False).fill_null("").alias("project_terms"),
            pl.col("TOTAL_COST").cast(pl.Float64, strict=False).fill_null(0.0).alias("total_cost"),
        )
        .join(abstracts_prepared, on="application_id", how="left")
        .with_columns(combined_text.alias("research_text"))
        .filter(pl.col("activity_code").is_in(TARGET_GRANT_ACTIVITY_CODES))
        .filter(contains_any_text("research_text", TARGET_NIH_KEYWORDS))
        .with_columns(pl.col("pi_names").str.split(";").alias("pi_name_list"))
        .explode("pi_name_list")
        .with_columns(
            pl.col("pi_name_list").str.split_exact(",", 1).alias("pi_split"),
        )
        .with_columns(
            normalize_text_expr(pl.col("pi_split").struct.field("field_0")).alias("pi_last_norm"),
            normalize_text_expr(pl.col("pi_split").struct.field("field_1")).alias("pi_rest_norm"),
        )
        .with_columns(
            first_token_expr(pl.col("pi_rest_norm")).alias("pi_first_norm"),
            middle_initial_expr(pl.col("pi_rest_norm")).alias("pi_middle_initial"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("pi_last_norm"),
                    pl.lit("|"),
                    pl.col("pi_first_norm"),
                    pl.lit("|"),
                    pl.col("pi_middle_initial"),
                ]
            ).alias("name_key_strict"),
            pl.concat_str(
                [
                    pl.col("pi_last_norm"),
                    pl.lit("|"),
                    pl.col("pi_first_norm"),
                ]
            ).alias("name_key_loose"),
        )
    )

    strict_matches = filtered.join(
        provider_aliases.filter(pl.col("match_quality") == 2),
        left_on="name_key_strict",
        right_on="name_key",
        how="inner",
    ).with_columns(pl.lit(2).alias("key_rank"))

    loose_matches = filtered.join(
        provider_aliases.filter(pl.col("match_quality") == 1),
        left_on="name_key_loose",
        right_on="name_key",
        how="inner",
    ).with_columns(pl.lit(1).alias("key_rank"))

    return (
        pl.concat([strict_matches, loose_matches])
        .with_columns(
            (pl.col("org_state") == pl.col("provider_state")).cast(pl.Int8).alias(
                "state_match"
            )
        )
        .group_by("application_id", "pi_name_list")
        .agg(
            pl.col("npi")
            .sort_by(
                ["state_match", "key_rank", "total_surgical_volume"],
                descending=[True, True, True],
            )
            .first()
            .alias("npi"),
            pl.col("activity_code").first(),
            pl.col("total_cost").max().alias("nih_total_cost"),
        )
        .group_by("npi")
        .agg(
            pl.len().alias("active_nih_grants"),
            pl.col("nih_total_cost").sum().alias("nih_total_award_amount"),
            pl.col("activity_code").n_unique().alias("nih_unique_mechanisms"),
        )
    )


def process_open_payments(paths: tuple[Path, ...]) -> pl.LazyFrame:
    frames = []
    competitor_regex = regex_escape_joined(TARGET_PAYMENT_MANUFACTURERS)
    payment_regex = regex_escape_joined(TARGET_PAYMENT_NATURE_TERMS)

    for path in paths:
        frame = scan_csv(path)
        ensure_columns(
            frame,
            {
                "Covered_Recipient_NPI",
                "Covered_Recipient_Specialty_1",
                "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
                "Total_Amount_of_Payment_USDollars",
                "Nature_of_Payment_or_Transfer_of_Value",
                "Program_Year",
            },
            f"Open Payments ({path.name})",
        )
        frames.append(
            frame.select(
                normalize_npi_expr("Covered_Recipient_NPI").alias("npi"),
                pl.col("Covered_Recipient_Specialty_1")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .fill_null("")
                .alias("specialty"),
                pl.col("Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .fill_null("")
                .alias("manufacturer_name"),
                pl.col("Nature_of_Payment_or_Transfer_of_Value")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .fill_null("")
                .alias("payment_nature"),
                pl.col("Total_Amount_of_Payment_USDollars")
                .cast(pl.Float64, strict=False)
                .fill_null(0.0)
                .alias("payment_amount"),
                pl.col("Program_Year").cast(pl.Int32, strict=False).alias("program_year"),
            )
            .filter(pl.col("specialty").str.contains("neurological surgery"))
            .filter(pl.col("manufacturer_name").str.contains(competitor_regex))
            .filter(pl.col("payment_nature").str.contains(payment_regex))
        )

    return (
        pl.concat(frames)
        .group_by("npi")
        .agg(
            pl.col("payment_amount").sum().alias("competitor_consulting_dollars"),
            pl.len().alias("competitor_payment_events"),
            pl.col("program_year").n_unique().alias("competitor_payment_years"),
        )
    )


def process_maude(
    master_path: Path,
    device_paths: tuple[Path, ...],
    problem_links_path: Path,
) -> pl.LazyFrame:
    master = scan_csv(master_path, separator="|", quote_char=None)
    ensure_columns(
        master,
        {"MDR_REPORT_KEY", "EVENT_TYPE", "REPORTER_STATE_CODE", "MANUFACTURER_NAME"},
        "MAUDE master",
    )

    device_frames = []
    for path in device_paths:
        device = scan_csv(path, separator="|", quote_char=None)
        ensure_columns(
            device,
            {
                "MDR_REPORT_KEY",
                "BRAND_NAME",
                "GENERIC_NAME",
                "MANUFACTURER_D_NAME",
                "DEVICE_REPORT_PRODUCT_CODE",
            },
            f"MAUDE device ({path.name})",
        )
        device_frames.append(
            device.select(
                pl.col("MDR_REPORT_KEY").cast(pl.Utf8, strict=False).alias("mdr_report_key"),
                pl.col("BRAND_NAME")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .fill_null("")
                .alias("brand_name"),
                pl.col("GENERIC_NAME")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .fill_null("")
                .alias("generic_name"),
                pl.col("MANUFACTURER_D_NAME")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .fill_null("")
                .alias("device_manufacturer_name"),
                pl.col("DEVICE_REPORT_PRODUCT_CODE")
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .alias("product_code"),
            )
        )

    device_all = pl.concat(device_frames)
    problems = scan_csv(
        problem_links_path,
        separator="|",
        has_header=False,
        new_columns=["MDR_REPORT_KEY", "problem_code"],
        quote_char=None,
    ).select(
        pl.col("MDR_REPORT_KEY").cast(pl.Utf8, strict=False).alias("mdr_report_key"),
        pl.col("problem_code").cast(pl.Utf8, strict=False).alias("problem_code"),
    )

    return (
        master.select(
            pl.col("MDR_REPORT_KEY").cast(pl.Utf8, strict=False).alias("mdr_report_key"),
            pl.col("EVENT_TYPE")
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .fill_null("")
            .alias("event_type"),
            pl.col("REPORTER_STATE_CODE")
            .cast(pl.Utf8, strict=False)
            .str.to_uppercase()
            .fill_null("")
            .alias("provider_state"),
            pl.col("MANUFACTURER_NAME")
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .fill_null("")
            .alias("manufacturer_name"),
        )
        .join(problems, on="mdr_report_key", how="inner")
        .join(device_all, on="mdr_report_key", how="left")
        .with_columns(
            pl.concat_str(
                [
                    pl.col("manufacturer_name"),
                    pl.lit(" "),
                    pl.col("device_manufacturer_name"),
                    pl.lit(" "),
                    pl.col("brand_name"),
                    pl.lit(" "),
                    pl.col("generic_name"),
                ]
            ).alias("device_text")
        )
        .filter(pl.col("problem_code").is_in(TARGET_MAUDE_PROBLEM_CODES))
        .filter(contains_any_text("device_text", TARGET_MAUDE_DEVICE_TERMS))
        .filter(pl.col("provider_state") != "")
        .unique(["mdr_report_key", "problem_code", "provider_state"])
        .group_by("provider_state")
        .agg(
            pl.len().alias("state_legacy_hardware_failures"),
            pl.when(pl.col("event_type").str.contains("injury", literal=True))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("state_maude_injury_events"),
            pl.when(pl.col("problem_code").is_in(["1395", "1291"]))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("state_mechanical_failure_events"),
        )
    )


def process_aact(
    dump_path: Path,
    provider_aliases: pl.LazyFrame,
    output_dir: Path,
) -> pl.LazyFrame:
    extracted = prepare_aact_extracts(dump_path, output_dir)
    studies = scan_csv(extracted["studies"])
    officials = scan_csv(extracted["overall_officials"])
    facilities = scan_csv(extracted["facilities"])

    ensure_columns(
        studies,
        {
            "nct_id",
            "study_type",
            "overall_status",
            "phase",
            "why_stopped",
            "is_fda_regulated_device",
            "brief_title",
        },
        "AACT studies extract",
    )
    ensure_columns(
        officials,
        {"nct_id", "role", "name", "affiliation"},
        "AACT overall_officials extract",
    )
    ensure_columns(facilities, {"nct_id", "name"}, "AACT facilities extract")

    facility_counts = facilities.select(
        pl.col("nct_id").cast(pl.Utf8, strict=False).alias("nct_id")
    ).group_by("nct_id").agg(pl.len().alias("facility_count"))

    studies_prepared = (
        studies.select(
            pl.col("nct_id").cast(pl.Utf8, strict=False).alias("nct_id"),
            pl.col("study_type")
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .fill_null("")
            .alias("study_type"),
            pl.col("overall_status")
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .fill_null("")
            .alias("overall_status"),
            pl.col("phase")
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .fill_null("")
            .alias("phase"),
            pl.col("why_stopped")
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .fill_null("")
            .alias("why_stopped"),
            pl.col("brief_title").cast(pl.Utf8, strict=False).fill_null("").alias("brief_title"),
            pl.col("is_fda_regulated_device")
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .fill_null("")
            .alias("is_fda_regulated_device"),
        )
        .join(facility_counts, on="nct_id", how="left")
        .with_columns(pl.col("facility_count").fill_null(0))
        .filter(pl.col("study_type") == "interventional")
        .filter(truthy_expr("is_fda_regulated_device"))
        .with_columns(
            pl.concat_str(
                [
                    pl.col("why_stopped"),
                    pl.lit(" "),
                    pl.col("brief_title").str.to_lowercase(),
                ]
            ).alias("trial_text")
        )
    )

    officials_prepared = (
        officials.select(
            pl.col("nct_id").cast(pl.Utf8, strict=False).alias("nct_id"),
            pl.col("role")
            .cast(pl.Utf8, strict=False)
            .str.to_uppercase()
            .fill_null("")
            .alias("role"),
            pl.col("name").cast(pl.Utf8, strict=False).fill_null("").alias("official_name"),
            pl.col("affiliation")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .alias("affiliation"),
        )
        .filter(pl.col("role").str.contains("PRINCIPAL_INVESTIGATOR"))
        .with_columns(
            normalize_text_expr(pl.col("official_name").str.replace_all(r",.*$", ""))
            .alias("official_name_clean")
        )
        .with_columns(
            first_token_expr(pl.col("official_name_clean")).alias("official_first_norm"),
            middle_initial_expr(pl.col("official_name_clean")).alias(
                "official_middle_initial"
            ),
            last_token_expr(pl.col("official_name_clean")).alias("official_last_norm"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("official_last_norm"),
                    pl.lit("|"),
                    pl.col("official_first_norm"),
                    pl.lit("|"),
                    pl.col("official_middle_initial"),
                ]
            ).alias("name_key_strict"),
            pl.concat_str(
                [
                    pl.col("official_last_norm"),
                    pl.lit("|"),
                    pl.col("official_first_norm"),
                ]
            ).alias("name_key_loose"),
        )
    )

    strict_matches = officials_prepared.join(
        provider_aliases.filter(pl.col("match_quality") == 2),
        left_on="name_key_strict",
        right_on="name_key",
        how="inner",
    ).with_columns(pl.lit(2).alias("key_rank"))

    loose_matches = officials_prepared.join(
        provider_aliases.filter(pl.col("match_quality") == 1),
        left_on="name_key_loose",
        right_on="name_key",
        how="inner",
    ).with_columns(pl.lit(1).alias("key_rank"))

    official_matches = (
        pl.concat([strict_matches, loose_matches])
        .group_by("nct_id", "official_name")
        .agg(
            pl.col("npi")
            .sort_by(["key_rank", "total_surgical_volume"], descending=[True, True])
            .first()
            .alias("npi")
        )
    )

    return (
        official_matches.join(studies_prepared, on="nct_id", how="inner")
        .group_by("npi")
        .agg(
            pl.when(pl.col("overall_status") == "completed")
            .then(1)
            .otherwise(0)
            .sum()
            .alias("completed_device_trials"),
            pl.when(pl.col("overall_status").is_in(sorted(AACT_ACTIVE_STATUSES)))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("active_device_trials"),
            pl.when(
                (pl.col("overall_status") == "terminated")
                & contains_any_text("trial_text", LOW_RECRUITMENT_TERMS)
            )
            .then(1)
            .otherwise(0)
            .sum()
            .alias("terminated_low_recruitment_trials"),
            pl.col("facility_count").max().alias("max_facility_count"),
        )
        .with_columns(
            (
                (pl.col("completed_device_trials") >= 1)
                & (pl.col("terminated_low_recruitment_trials") == 0)
            ).alias("has_successful_device_trial_infrastructure")
        )
    )


def build_master_ledger(
    medicare: pl.LazyFrame,
    nih: pl.LazyFrame,
    open_payments: pl.LazyFrame,
    maude: pl.LazyFrame,
    aact: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        medicare.join(nih, on="npi", how="left")
        .join(aact, on="npi", how="left")
        .join(open_payments, on="npi", how="left")
        .join(maude, on="provider_state", how="left")
        .with_columns(
            pl.col("active_nih_grants").fill_null(0),
            pl.col("nih_total_award_amount").fill_null(0.0),
            pl.col("nih_unique_mechanisms").fill_null(0),
            pl.col("completed_device_trials").fill_null(0),
            pl.col("active_device_trials").fill_null(0),
            pl.col("terminated_low_recruitment_trials").fill_null(0),
            pl.col("max_facility_count").fill_null(0),
            pl.col("has_successful_device_trial_infrastructure").fill_null(False),
            pl.col("competitor_consulting_dollars").fill_null(0.0),
            pl.col("competitor_payment_events").fill_null(0),
            pl.col("competitor_payment_years").fill_null(0),
            pl.col("state_legacy_hardware_failures").fill_null(0),
            pl.col("state_maude_injury_events").fill_null(0),
            pl.col("state_mechanical_failure_events").fill_null(0),
        )
    )


def filter_implanting_neurosurgeons(ledger: pl.LazyFrame) -> pl.LazyFrame:
    """Exclude non-implanting neurologists before scoring and ranking.

    A provider must have at least one implant lead or IPG procedure in the
    Medicare-derived features to remain in the targeting universe.
    """

    return ledger.filter((pl.col("implant_lead_volume") + pl.col("ipg_volume")) > 0)


def score_ledger(ledger: pl.LazyFrame, weights: ScoringWeights) -> pl.LazyFrame:
    return (
        ledger.with_columns(
            (pl.col("total_surgical_volume").rank("average") / pl.len()).alias(
                "volume_percentile"
            ),
            (
                pl.when(pl.col("active_nih_grants") > 0)
                .then(
                    (pl.col("active_nih_grants").clip(upper_bound=3) / 3.0) * 0.65
                    + normalized_log_expr("nih_total_award_amount") * 0.35
                )
                .otherwise(0.0)
            ).alias("nih_signal"),
            (
                (
                    (pl.col("completed_device_trials").clip(upper_bound=3) / 3.0)
                    * 0.65
                    + (pl.col("active_device_trials").clip(upper_bound=2) / 2.0)
                    * 0.20
                    + (pl.col("max_facility_count").clip(upper_bound=10) / 10.0)
                    * 0.15
                ).clip(upper_bound=1.0)
            ).alias("aact_signal"),
            normalized_log_expr("state_legacy_hardware_failures")
            .clip(upper_bound=1.0)
            .alias("maude_signal"),
            normalized_log_expr("competitor_consulting_dollars")
            .clip(upper_bound=1.0)
            .alias("payment_penalty"),
            pl.when(pl.col("terminated_low_recruitment_trials") > 0)
            .then(1.0)
            .otherwise(0.0)
            .alias("low_recruitment_penalty"),
            (
                pl.col("total_surgical_volume")
                >= pl.col("total_surgical_volume").quantile(0.90)
            ).alias("top_decile_volume"),
            (
                (pl.col("competitor_consulting_dollars") < 10_000)
                & (pl.col("competitor_payment_events") <= 5)
            ).alias("financially_independent"),
        )
        .with_columns(
            pl.when(
                (pl.col("total_surgical_volume") > 20)
                & (pl.col("active_nih_grants") >= 1)
            )
            .then(True)
            .otherwise(False)
            .alias("dual_threat_flag")
        )
        .with_columns(
            (
                (
                    pl.col("volume_percentile") * weights.volume
                    + pl.col("nih_signal") * weights.nih
                    + pl.col("aact_signal") * weights.aact
                    + pl.col("maude_signal") * weights.maude
                    - pl.col("payment_penalty") * weights.payments_penalty
                    - pl.col("low_recruitment_penalty")
                    * weights.low_recruitment_penalty
                )
                * (1 + pl.col("maude_signal") * weights.friction_multiplier)
                * 100
                + pl.when(pl.col("dual_threat_flag"))
                .then(weights.dual_threat_bonus)
                .otherwise(0.0)
                + pl.when(pl.col("financially_independent"))
                .then(weights.free_agent_bonus)
                .otherwise(0.0)
            )
            .clip(lower_bound=0.0)
            .round(2)
            .alias("Clinical_Suitability_Score")
        )
    )


def select_top_targets(scored_ledger: pl.LazyFrame, top_n: int) -> pl.DataFrame:
    return (
        scored_ledger.sort(
            by=[
                "Clinical_Suitability_Score",
                "total_surgical_volume",
                "active_nih_grants",
                "completed_device_trials",
            ],
            descending=[True, True, True, True],
        )
        .select(
            "npi",
            "provider_name",
            "provider_state",
            "provider_city",
            "total_surgical_volume",
            "implant_lead_volume",
            "ipg_volume",
            "intraop_monitoring_volume",
            "active_nih_grants",
            "nih_total_award_amount",
            "completed_device_trials",
            "active_device_trials",
            "terminated_low_recruitment_trials",
            "competitor_consulting_dollars",
            "competitor_payment_events",
            "state_legacy_hardware_failures",
            "state_mechanical_failure_events",
            "dual_threat_flag",
            "financially_independent",
            "Clinical_Suitability_Score",
        )
        .limit(top_n)
        .collect()
    )


def create_bloomberg_chart(targets: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_targets = (
        targets.sort("Clinical_Suitability_Score", descending=True)
        .head(10)
        .sort("Clinical_Suitability_Score")
    )
    labels = [
        f"{provider_name} - {provider_city}, {provider_state} "
        f"(Vol: {int(total_surgical_volume)} | NIH: {int(active_nih_grants)})"
        for provider_name, provider_city, provider_state, total_surgical_volume, active_nih_grants in zip(
            top_targets["provider_name"].fill_null("UNKNOWN").to_list(),
            top_targets["provider_city"].fill_null("Unknown City").to_list(),
            top_targets["provider_state"].fill_null("NA").to_list(),
            top_targets["total_surgical_volume"].fill_null(0).to_list(),
            top_targets["active_nih_grants"].fill_null(0).to_list(),
            strict=False,
        )
    ]
    scores = top_targets["Clinical_Suitability_Score"].to_list()
    bar_colors = [
        "#00f0ff" if is_dual_threat else "#555555"
        for is_dual_threat in top_targets["dual_threat_flag"].fill_null(False).to_list()
    ]

    fig, ax = plt.subplots(figsize=(18, 10), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    bars = ax.barh(
        labels,
        scores,
        color=bar_colors,
        edgecolor=bar_colors,
        linewidth=1.2,
        alpha=0.98,
    )

    for bar, score in zip(bars, scores, strict=False):
        ax.text(
            score + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#f5f5f5",
        )

    ax.set_title(
        "Clinical Sourcing Ledger | Top 10 Implanting Targets",
        loc="left",
        fontsize=18,
        color="#f5f5f5",
        pad=18,
        fontweight="bold",
    )
    ax.set_xlabel("Clinical Suitability Score", color="#d4d4d4", fontsize=11)
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
        "Electric blue bars indicate dual-threat surgeons: >20 total surgical volume and at least 1 active NIH grant."
    )
    fig.text(0.01, 0.015, footer, color="#cfcfcf", fontsize=9)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    LOGGER.info("Wrote executive chart to %s", output_path)


def write_outputs(
    scored_ledger: pl.LazyFrame,
    top_targets: pl.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_ledger_path = output_dir / "clinical_sourcing_ledger.parquet"
    top_csv_path = output_dir / "clinical_top_targets.csv"
    chart_path = output_dir / "clinical_top_targets.png"
    scored_ledger.collect().write_parquet(full_ledger_path)
    top_targets.write_csv(top_csv_path)
    LOGGER.info("Wrote full ledger to %s", full_ledger_path)
    LOGGER.info("Wrote top-target CSV to %s", top_csv_path)
    create_bloomberg_chart(top_targets, chart_path)
    return full_ledger_path, top_csv_path, chart_path


def run_pipeline(
    source_paths: SourcePaths,
    top_n: int,
    weights: ScoringWeights,
) -> tuple[pl.DataFrame, tuple[Path, Path, Path]]:
    source_paths.output_dir.mkdir(parents=True, exist_ok=True)
    medicare = process_medicare_volume(source_paths.medicare)
    provider_aliases = build_provider_aliases(medicare)
    nih = process_nih_reporter(
        source_paths.nih_projects,
        source_paths.nih_abstracts,
        provider_aliases,
    )
    open_payments = process_open_payments(source_paths.open_payments_general)
    maude = process_maude(
        source_paths.maude_master,
        source_paths.maude_devices,
        source_paths.maude_problem_links,
    )
    aact = process_aact(
        source_paths.aact_dump,
        provider_aliases,
        AACT_EXTRACT_DIR,
    )
    ledger = build_master_ledger(medicare, nih, open_payments, maude, aact)
    implanting_ledger = filter_implanting_neurosurgeons(ledger)
    scored = score_ledger(implanting_ledger, weights=weights)
    top_targets = select_top_targets(scored, top_n)
    outputs = write_outputs(scored, top_targets, source_paths.output_dir)
    return top_targets, outputs


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    source_paths = build_source_paths(args)
    top_targets, outputs = run_pipeline(
        source_paths=source_paths,
        top_n=args.top_n,
        weights=ScoringWeights(),
    )
    LOGGER.info("Pipeline complete. Generated %s ranked targets.", len(top_targets))
    LOGGER.info("Outputs: %s", outputs)


if __name__ == "__main__":
    main()
