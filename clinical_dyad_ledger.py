from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


# High-acuity referring specialists for circuit-level therapeutics typically
# surface through TMS, ECT, or advanced EEG-related billing activity.
ADVANCED_INTERVENTION_CPT_CODES = {
    "90867",  # TMS initial mapping and treatment
    "90868",  # TMS subsequent delivery
    "90869",  # TMS motor threshold redetermination
    "90870",  # ECT
    "95951",  # prolonged EEG monitoring with video
    "95953",  # ambulatory EEG monitoring
    "95956",  # EEG monitoring by technologist
}

TARGET_SPECIALTY_PATTERN = r"psychiatry|neurology"


@dataclass(frozen=True)
class SurgeonSchema:
    name: str = "provider_name"
    volume: str = "total_surgical_volume"
    city: str = "provider_city"
    state: str = "provider_state"
    hospital: str | None = "hospital_affiliation"


@dataclass(frozen=True)
class MedicareSchema:
    npi: str = "Rndrng_NPI"
    first_name: str = "Rndrng_Prvdr_First_Name"
    last_name: str = "Rndrng_Prvdr_Last_Org_Name"
    specialty: str = "Rndrng_Prvdr_Type"
    city: str = "Rndrng_Prvdr_City"
    state: str = "Rndrng_Prvdr_State_Abrvtn"
    cpt_code: str = "HCPCS_Cd"
    billed_volume: str = "Tot_Srvcs"
    hospital: str | None = None


def _norm_text(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
    )


def _specialist_name_expr(schema: MedicareSchema) -> pl.Expr:
    return (
        pl.concat_str(
            [
                pl.col(schema.first_name).cast(pl.Utf8, strict=False).fill_null(""),
                pl.lit(" "),
                pl.col(schema.last_name).cast(pl.Utf8, strict=False).fill_null(""),
            ]
        )
        .str.strip_chars()
        .alias("dyad_partner_name")
    )


def build_referring_specialist_funnel(
    medicare_path: str | Path,
    schema: MedicareSchema = MedicareSchema(),
    min_intervention_volume: int = 10,
) -> pl.LazyFrame:
    """Build a physician-level referral funnel for Psychiatry/Neurology.

    The funnel isolates physicians billing for advanced interventions that are
    more likely to feed refractory psychiatric or complex neurology patients
    into an experimental neuromodulation trial.
    """

    medicare = pl.scan_csv(medicare_path, infer_schema_length=10_000)

    hospital_expr = (
        _norm_text(pl.col(schema.hospital)).alias("hospital_key")
        if schema.hospital
        else pl.lit("").alias("hospital_key")
    )

    return (
        medicare.with_columns(
            pl.col(schema.npi).cast(pl.Utf8, strict=False).alias("dyad_partner_npi"),
            _specialist_name_expr(schema),
            pl.col(schema.specialty)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .alias("dyad_partner_specialty"),
            _norm_text(pl.col(schema.city)).alias("partner_city_key"),
            _norm_text(pl.col(schema.state)).alias("partner_state_key"),
            hospital_expr,
            pl.col(schema.cpt_code).cast(pl.Utf8, strict=False).alias("target_cpt_code"),
            pl.col(schema.billed_volume)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("intervention_volume"),
        )
        .filter(
            pl.col("dyad_partner_specialty")
            .str.to_lowercase()
            .str.contains(TARGET_SPECIALTY_PATTERN)
        )
        .filter(pl.col("target_cpt_code").is_in(ADVANCED_INTERVENTION_CPT_CODES))
        .group_by(
            [
                "dyad_partner_npi",
                "dyad_partner_name",
                "dyad_partner_specialty",
                "partner_city_key",
                "partner_state_key",
                "hospital_key",
            ]
        )
        .agg(
            pl.col("intervention_volume")
            .sum()
            .round(2)
            .alias("dyad_partner_intervention_volume"),
            pl.col("target_cpt_code")
            .n_unique()
            .alias("dyad_partner_target_cpt_count"),
        )
        .filter(pl.col("dyad_partner_intervention_volume") >= min_intervention_volume)
    )


def build_clinical_dyad_ledger(
    surgeons_df: pl.DataFrame | pl.LazyFrame,
    medicare_path: str | Path,
    surgeon_schema: SurgeonSchema = SurgeonSchema(),
    medicare_schema: MedicareSchema = MedicareSchema(),
    min_intervention_volume: int = 10,
) -> pl.DataFrame:
    """Pair each top surgeon with the best local psychiatric/neurology referrer.

    Matching preference:
    1. Same hospital / health-system affiliation, if both inputs carry it.
    2. Exact same city and state.
    """

    surgeons_lf = surgeons_df.lazy() if isinstance(surgeons_df, pl.DataFrame) else surgeons_df
    surgeons_prepared = surgeons_lf.with_row_count("surgeon_row_id").with_columns(
        pl.col(surgeon_schema.name).cast(pl.Utf8, strict=False).alias("surgeon_name"),
        pl.col(surgeon_schema.volume)
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
        .alias("surgeon_volume"),
        pl.col(surgeon_schema.city).cast(pl.Utf8, strict=False).alias("surgeon_city"),
        pl.col(surgeon_schema.state).cast(pl.Utf8, strict=False).alias("surgeon_state"),
        _norm_text(pl.col(surgeon_schema.city)).alias("surgeon_city_key"),
        _norm_text(pl.col(surgeon_schema.state)).alias("surgeon_state_key"),
        (
            _norm_text(pl.col(surgeon_schema.hospital)).alias("surgeon_hospital_key")
            if surgeon_schema.hospital
            else pl.lit("").alias("surgeon_hospital_key")
        ),
    )

    specialists = build_referring_specialist_funnel(
        medicare_path=medicare_path,
        schema=medicare_schema,
        min_intervention_volume=min_intervention_volume,
    )

    candidate_matches: list[pl.LazyFrame] = []

    if surgeon_schema.hospital and medicare_schema.hospital:
        same_hospital = (
            surgeons_prepared.filter(pl.col("surgeon_hospital_key") != "")
            .join(
                specialists.filter(pl.col("hospital_key") != ""),
                left_on="surgeon_hospital_key",
                right_on="hospital_key",
                how="left",
            )
            .with_columns(
                pl.lit(0).alias("match_priority"),
                pl.lit("same_hospital").alias("match_type"),
            )
        )
        candidate_matches.append(same_hospital)

    same_city_state = (
        surgeons_prepared.join(
            specialists,
            left_on=["surgeon_city_key", "surgeon_state_key"],
            right_on=["partner_city_key", "partner_state_key"],
            how="left",
        )
        .with_columns(
            pl.lit(1).alias("match_priority"),
            pl.lit("same_city_state").alias("match_type"),
        )
    )
    candidate_matches.append(same_city_state)

    best_partner = (
        pl.concat(candidate_matches, how="diagonal_relaxed")
        .sort(
            by=[
                "surgeon_row_id",
                "match_priority",
                "dyad_partner_intervention_volume",
            ],
            descending=[False, False, True],
            nulls_last=True,
        )
        .unique(subset=["surgeon_row_id"], keep="first", maintain_order=True)
        .select(
            "surgeon_row_id",
            "dyad_partner_name",
            "dyad_partner_specialty",
            "dyad_partner_intervention_volume",
            "dyad_partner_npi",
            "match_type",
        )
    )

    return (
        surgeons_prepared.join(best_partner, on="surgeon_row_id", how="left")
        .with_columns(
            pl.when(pl.col("dyad_partner_name").is_null())
            .then(pl.lit("High Friction Trial Site"))
            .otherwise(pl.lit("Matched Local Referrer"))
            .alias("trial_site_friction_flag")
        )
        .select(
            pl.col("surgeon_name").alias("Surgeon_Name"),
            pl.col("surgeon_volume").alias("Surgeon_Volume"),
            pl.col("surgeon_city").alias("Surgeon_City"),
            pl.col("surgeon_state").alias("Surgeon_State"),
            pl.col("dyad_partner_name").alias("Dyad_Partner_Name"),
            pl.col("dyad_partner_specialty").alias("Dyad_Partner_Specialty"),
            pl.col("dyad_partner_intervention_volume").alias(
                "Dyad_Partner_Intervention_Volume"
            ),
            pl.col("dyad_partner_npi").alias("Dyad_Partner_NPI"),
            pl.col("match_type").alias("Dyad_Match_Type"),
            pl.col("trial_site_friction_flag").alias("Trial_Site_Friction_Flag"),
        )
        .collect()
    )


if __name__ == "__main__":
    surgeons_df = pl.read_csv("top15_surgeons.csv")
    dyad_ledger = build_clinical_dyad_ledger(
        surgeons_df=surgeons_df,
        medicare_path="medicare_physicians.csv",
        surgeon_schema=SurgeonSchema(hospital="hospital_affiliation"),
        medicare_schema=MedicareSchema(
            hospital=None,  # Set this if your physician file carries hospital NPI/affiliation.
        ),
        min_intervention_volume=10,
    )
    dyad_ledger.write_csv("clinical_dyad_ledger.csv")
