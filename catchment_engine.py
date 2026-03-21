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

DEFAULT_DYAD_LEDGER = PROCESSED_DIR / "clinical_dyad_ledger.csv"
DEFAULT_PATIENT_DENSITY = PROJECT_ROOT / "medicare_patient_density.csv"


@dataclass(frozen=True)
class SourcePaths:
    dyad_ledger: Path
    patient_density: Path
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
        description="Build viable patient catchment areas for clinical dyad hubs."
    )
    parser.add_argument("--dyad-ledger", type=Path, default=DEFAULT_DYAD_LEDGER)
    parser.add_argument(
        "--patient-density",
        type=Path,
        default=DEFAULT_PATIENT_DENSITY,
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
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


def resolve_density_columns(frame: pl.LazyFrame) -> dict[str, str]:
    available = frame.collect_schema().names()
    available_map = {column.lower(): column for column in available}

    def pick(*aliases: str) -> str:
        for alias in aliases:
            if alias.lower() in available_map:
                return available_map[alias.lower()]
        raise ValueError(
            f"Could not resolve any of {aliases} from patient density columns: {available}"
        )

    return {
        "state": pick("State"),
        "location": pick("City", "County"),
        "total_beneficiaries": pick("Total_Beneficiaries"),
        "depression_count": pick("Depression_Count"),
        "parkinsons_count": pick("Parkinsons_Count"),
        "epilepsy_count": pick("Epilepsy_Count"),
    }


def load_viable_dyad_sites(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    ensure_columns(
        frame,
        {
            "Surgeon_Name",
            "Surgeon_City",
            "Surgeon_State",
            "Dyad_Partner_Name",
            "Trial_Site_Friction_Flag",
        },
        "clinical dyad ledger",
    )

    return (
        frame.select(
            pl.col("Surgeon_Name").cast(pl.Utf8, strict=False).alias("Surgeon_Name"),
            pl.col("Surgeon_Volume")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("Surgeon_Volume"),
            pl.col("Surgeon_City").cast(pl.Utf8, strict=False).alias("Surgeon_City"),
            pl.col("Surgeon_State").cast(pl.Utf8, strict=False).alias("Surgeon_State"),
            pl.col("Dyad_Partner_Name")
            .cast(pl.Utf8, strict=False)
            .alias("Dyad_Partner_Name"),
            pl.col("Dyad_Partner_Specialty")
            .cast(pl.Utf8, strict=False)
            .alias("Dyad_Partner_Specialty"),
            pl.col("Trial_Site_Friction_Flag")
            .cast(pl.Utf8, strict=False)
            .alias("Trial_Site_Friction_Flag"),
        )
        # Friction sites are intentionally excluded here because those hubs do not
        # currently have a validated referrer funnel and therefore should not be
        # counted as viable recruiting sites in the patient catchment TAM.
        .filter(pl.col("Trial_Site_Friction_Flag") != "High Friction Trial Site")
        .with_columns(
            normalize_text_expr(pl.col("Surgeon_City")).alias("surgeon_city_key"),
            normalize_text_expr(pl.col("Surgeon_State")).alias("surgeon_state_key"),
        )
    )


def load_patient_density(path: Path) -> pl.LazyFrame:
    frame = scan_csv(path)
    column_map = resolve_density_columns(frame)

    return (
        frame.select(
            pl.col(column_map["state"]).cast(pl.Utf8, strict=False).alias("State"),
            pl.col(column_map["location"]).cast(pl.Utf8, strict=False).alias("Location"),
            pl.col(column_map["total_beneficiaries"])
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("Total_Beneficiaries"),
            pl.col(column_map["depression_count"])
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("Depression_Count"),
            pl.col(column_map["parkinsons_count"])
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("Parkinsons_Count"),
            pl.col(column_map["epilepsy_count"])
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("Epilepsy_Count"),
        )
        .with_columns(
            normalize_text_expr(pl.col("State")).alias("surgeon_state_key"),
            normalize_text_expr(pl.col("Location")).alias("surgeon_city_key"),
            # Only a fraction of depression prevalence is likely to meet a
            # treatment-resistant threshold. The 0.15 multiplier approximates
            # that severe subset rather than overstating the directly addressable
            # psychiatric population in the catchment estimate.
            (
                pl.col("Parkinsons_Count")
                + pl.col("Epilepsy_Count")
                + (pl.col("Depression_Count") * 0.15)
            )
            .round(2)
            .alias("Circuit_Level_TAM"),
        )
        .group_by("surgeon_state_key", "surgeon_city_key")
        .agg(
            pl.col("State").first(),
            pl.col("Location").first(),
            pl.col("Total_Beneficiaries").sum().round(2).alias("Total_Beneficiaries"),
            pl.col("Depression_Count").sum().round(2).alias("Depression_Count"),
            pl.col("Parkinsons_Count").sum().round(2).alias("Parkinsons_Count"),
            pl.col("Epilepsy_Count").sum().round(2).alias("Epilepsy_Count"),
            pl.col("Circuit_Level_TAM").sum().round(2).alias("Circuit_Level_TAM"),
        )
    )


def build_catchment_ledger(
    dyad_ledger_path: Path,
    patient_density_path: Path,
) -> pl.DataFrame:
    viable_dyads = load_viable_dyad_sites(dyad_ledger_path)
    patient_density = load_patient_density(patient_density_path)

    return (
        viable_dyads.join(
            patient_density,
            on=["surgeon_state_key", "surgeon_city_key"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("Circuit_Level_TAM").is_null())
            .then(pl.lit("No Geographic Match"))
            .otherwise(pl.lit("Matched"))
            .alias("Catchment_Match_Status"),
            pl.col("Total_Beneficiaries").fill_null(0.0),
            pl.col("Depression_Count").fill_null(0.0),
            pl.col("Parkinsons_Count").fill_null(0.0),
            pl.col("Epilepsy_Count").fill_null(0.0),
            pl.col("Circuit_Level_TAM").fill_null(0.0),
            pl.col("State").fill_null(pl.col("Surgeon_State")),
            pl.col("Location").fill_null(pl.col("Surgeon_City")),
        )
        .sort("Circuit_Level_TAM", descending=True)
        .select(
            "Surgeon_Name",
            "Surgeon_Volume",
            "Surgeon_City",
            "Surgeon_State",
            "Dyad_Partner_Name",
            "Dyad_Partner_Specialty",
            "Total_Beneficiaries",
            "Depression_Count",
            "Parkinsons_Count",
            "Epilepsy_Count",
            "Circuit_Level_TAM",
            "Catchment_Match_Status",
        )
        .collect()
    )


def create_catchment_chart(catchment_df: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked = catchment_df.sort("Circuit_Level_TAM").tail(14)
    labels = [
        f"{surgeon} + {partner} | {city}, {state}"
        for surgeon, partner, city, state in zip(
            ranked["Surgeon_Name"].to_list(),
            ranked["Dyad_Partner_Name"].to_list(),
            ranked["Surgeon_City"].to_list(),
            ranked["Surgeon_State"].to_list(),
            strict=False,
        )
    ]
    tam_values = ranked["Circuit_Level_TAM"].to_list()

    fig, ax = plt.subplots(figsize=(18, 10), facecolor="#1e1e1e")
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
        tam_values,
        color=bar_colors,
        edgecolor=bar_colors,
        linewidth=1.0,
        alpha=0.98,
    )

    for bar, value in zip(bars, tam_values, strict=False):
        ax.text(
            value + (vmax * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.0f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#f5f5f5",
        )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
    colorbar.set_label("Circuit-Level TAM", color="#d4d4d4")
    colorbar.ax.yaxis.set_tick_params(color="#d4d4d4")
    plt.setp(colorbar.ax.get_yticklabels(), color="#d4d4d4")
    colorbar.outline.set_edgecolor("#666666")

    ax.set_title(
        "Viable Clinical Dyad Hubs by Circuit-Level TAM",
        loc="left",
        fontsize=18,
        color="#f5f5f5",
        pad=18,
        fontweight="bold",
    )
    ax.set_xlabel("Circuit-Level TAM", color="#d4d4d4", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="x", colors="#d4d4d4", labelsize=10)
    ax.tick_params(axis="y", colors="#f5f5f5", labelsize=9)
    ax.grid(axis="x", color="#3a3a3a", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    fig.text(
        0.01,
        0.015,
        "Depression is weighted at 15% to proxy the treatment-resistant subset rather than the full prevalence pool.",
        color="#cfcfcf",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    LOGGER.info("Wrote catchment chart to %s", output_path)


def write_outputs(catchment_df: pl.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "viable_catchment_areas.csv"
    chart_path = output_dir / "viable_catchment_areas.png"
    catchment_df.write_csv(csv_path)
    create_catchment_chart(catchment_df, chart_path)
    LOGGER.info("Wrote viable catchment CSV to %s", csv_path)
    return csv_path, chart_path


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    source_paths = SourcePaths(
        dyad_ledger=args.dyad_ledger,
        patient_density=args.patient_density,
        output_dir=args.output_dir,
    )
    catchment_df = build_catchment_ledger(
        dyad_ledger_path=source_paths.dyad_ledger,
        patient_density_path=source_paths.patient_density,
    )
    outputs = write_outputs(catchment_df, source_paths.output_dir)
    LOGGER.info("Catchment engine complete. Generated %s viable hubs.", len(catchment_df))
    LOGGER.info("Outputs: %s", outputs)


if __name__ == "__main__":
    main()
