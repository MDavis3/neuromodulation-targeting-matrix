from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


LOGGER = logging.getLogger("neuromodulation_targeting_matrix.drg_profitability_engine")

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_LEDGER_PATH = PROCESSED_DIR / "hospital_drg_economics.csv"
DEFAULT_CHART_PATH = PROCESSED_DIR / "drg_economics_visual.png"


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
        scenario_name="Starfish Chip",
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
        description="Build a deterministic hospital DRG profitability comparison."
    )
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--chart-path", type=Path, default=DEFAULT_CHART_PATH)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


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
    starfish_profit = (
        ledger.filter(pl.col("Scenario") == "Starfish Chip")
        .select("Net_Hospital_Profit")
        .item()
    )

    if legacy_profit == 0:
        raise ZeroDivisionError("Legacy net hospital profit is zero; alpha percentage is undefined.")

    return round(((starfish_profit - legacy_profit) / abs(legacy_profit)) * 100, 2)


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
        label="Net Hospital Profit",
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
        f"Hospital DRG Economics | Starfish Alpha +{alpha_pct:.2f}%",
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

    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    LOGGER.info("Wrote DRG economics visualization to %s", output_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    ledger = build_financial_ledger()
    alpha_pct = calculate_profit_alpha(ledger)

    print(f"Starfish Alpha (Net Profit Increase %): {alpha_pct:.2f}%")

    write_ledger(ledger, args.ledger_path)
    create_profitability_chart(ledger, alpha_pct, args.chart_path)


if __name__ == "__main__":
    main()
