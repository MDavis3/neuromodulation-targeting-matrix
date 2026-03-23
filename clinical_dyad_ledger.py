from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_CARE_COMPARE_PATH = RAW_DIR / "care_compare" / "DAC_NationalDownloadableFile.csv"
DEFAULT_FACILITY_AFFILIATION_PATH = RAW_DIR / "care_compare" / "Facility_Affiliation.csv"
DEFAULT_SHARED_PATIENT_PATTERNS_PATH = (
    RAW_DIR / "referral_patterns" / "physician-shared-patient-patterns-2015-days90.txt"
)

# The psychiatric referral funnel should be restricted to physicians who
# demonstrably treat refractory depression rather than general mental-health or
# diagnostic neurology volume. TMS and ECT billing are used here as an
# operational proof point for interventional psychiatry capacity.
INTERVENTIONAL_PSYCHIATRY_CPT_CODES = {
    "90867",  # TMS initial mapping and treatment
    "90868",  # TMS subsequent delivery
    "90870",  # ECT
}

TARGET_SPECIALTY_PATTERN = r"psychiatry"


@dataclass(frozen=True)
class SurgeonSchema:
    npi: str | None = "npi"
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


def _scan_csv(path: str | Path, **kwargs) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        infer_schema_length=10_000,
        ignore_errors=True,
        try_parse_dates=True,
        encoding="utf8-lossy",
        null_values=["", "NULL", "null", "N/A", "n/a"],
        **kwargs,
    )


def _norm_text(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
    )


def _zip5_expr(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.replace_all(r"[^0-9]", "")
        .str.slice(0, 5)
    )


def _pair_low_expr(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return pl.when(left <= right).then(left).otherwise(right)


def _pair_high_expr(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return pl.when(left <= right).then(right).otherwise(left)


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


def load_care_compare_provider_reference(path: str | Path = DEFAULT_CARE_COMPARE_PATH) -> pl.LazyFrame:
    dac = _scan_csv(path)
    return (
        dac.select(
            pl.col("NPI").cast(pl.Utf8, strict=False).alias("provider_npi"),
            _norm_text(pl.col("Provider Last Name")).alias("provider_last_name"),
            _norm_text(pl.col("Provider First Name")).alias("provider_first_name"),
            _norm_text(pl.col("pri_spec")).alias("primary_specialty"),
            _norm_text(pl.col("Facility Name")).alias("care_facility_name"),
            pl.col("org_pac_id").cast(pl.Utf8, strict=False).fill_null("").alias("org_pac_id"),
            pl.col("City/Town").cast(pl.Utf8, strict=False).fill_null("").alias("care_city"),
            pl.col("State").cast(pl.Utf8, strict=False).fill_null("").alias("care_state"),
            _zip5_expr(pl.col("ZIP Code")).alias("care_zip5"),
        )
        .filter(pl.col("provider_npi") != "")
        .group_by("provider_npi")
        .agg(
            pl.col("provider_last_name").first(),
            pl.col("provider_first_name").first(),
            pl.col("primary_specialty").first(),
            pl.col("care_facility_name").filter(pl.col("care_facility_name") != "").first(),
            pl.col("org_pac_id").filter(pl.col("org_pac_id") != "").first(),
            pl.col("care_city").filter(pl.col("care_city") != "").first(),
            pl.col("care_state").filter(pl.col("care_state") != "").first(),
            pl.col("care_zip5").filter(pl.col("care_zip5") != "").first(),
        )
        .with_columns(
            _norm_text(pl.col("care_city")).alias("care_city_key"),
            _norm_text(pl.col("care_state")).alias("care_state_key"),
        )
    )


def load_provider_org_memberships(path: str | Path = DEFAULT_CARE_COMPARE_PATH) -> pl.LazyFrame:
    dac = _scan_csv(path)
    return (
        dac.select(
            pl.col("NPI").cast(pl.Utf8, strict=False).alias("provider_npi"),
            pl.col("org_pac_id").cast(pl.Utf8, strict=False).fill_null("").alias("org_pac_id"),
        )
        .filter(pl.col("provider_npi") != "")
        .filter(pl.col("org_pac_id") != "")
        .unique()
    )


def load_provider_hospital_affiliations(
    path: str | Path = DEFAULT_FACILITY_AFFILIATION_PATH,
) -> pl.LazyFrame:
    facility = _scan_csv(path)
    return (
        facility.select(
            pl.col("NPI").cast(pl.Utf8, strict=False).alias("provider_npi"),
            pl.col("facility_type").cast(pl.Utf8, strict=False).fill_null("").alias("facility_type"),
            pl.col("Facility Affiliations Certification Number")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .alias("hospital_ccn"),
        )
        .filter(pl.col("provider_npi") != "")
        .filter(pl.col("hospital_ccn") != "")
        .filter(pl.col("facility_type").str.to_lowercase().str.contains("hospital"))
        .unique()
    )


def build_referring_specialist_funnel(
    medicare_path: str | Path,
    schema: MedicareSchema = MedicareSchema(),
    care_compare_path: str | Path = DEFAULT_CARE_COMPARE_PATH,
    min_intervention_volume: int = 10,
) -> pl.LazyFrame:
    """Build an interventional-psychiatry funnel enriched with verified provider metadata.

    Medicare alone proves procedure volume, but not the provider's current
    practice location or organizational attachment. Care Compare improves that
    linkage so dyad matching is based on the psychiatrist's actual operating
    network rather than only a billing-city fallback.
    """

    medicare = _scan_csv(medicare_path)
    care_compare = load_care_compare_provider_reference(care_compare_path)

    base = (
        medicare.with_columns(
            pl.col(schema.npi).cast(pl.Utf8, strict=False).alias("dyad_partner_npi"),
            _specialist_name_expr(schema),
            pl.col(schema.specialty)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .alias("dyad_partner_specialty"),
            _norm_text(pl.col(schema.city)).alias("medicare_city_key"),
            _norm_text(pl.col(schema.state)).alias("medicare_state_key"),
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
        .filter(pl.col("target_cpt_code").is_in(INTERVENTIONAL_PSYCHIATRY_CPT_CODES))
        .group_by(
            [
                "dyad_partner_npi",
                "dyad_partner_name",
                "dyad_partner_specialty",
                "medicare_city_key",
                "medicare_state_key",
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

    return (
        base.join(care_compare, left_on="dyad_partner_npi", right_on="provider_npi", how="left")
        .with_columns(
            pl.coalesce(
                ["care_city_key", "medicare_city_key"],
            ).alias("partner_city_key"),
            pl.coalesce(
                ["care_state_key", "medicare_state_key"],
            ).alias("partner_state_key"),
            pl.col("care_zip5").fill_null("").alias("partner_zip5"),
            pl.col("org_pac_id").fill_null("").alias("partner_org_pac_id"),
            pl.col("care_facility_name").fill_null("").alias("partner_facility_name"),
        )
    )


def load_shared_patient_pairs(
    path: str | Path,
    surgeon_npis: list[str],
    partner_npis: list[str],
) -> pl.LazyFrame:
    if not surgeon_npis or not partner_npis:
        return pl.LazyFrame(
            {
                "pair_low_npi": [],
                "pair_high_npi": [],
                "shared_patient_count_proxy": [],
                "shared_services_proxy": [],
                "shared_episodes_proxy": [],
            }
        )

    relevant_npis = sorted(set(surgeon_npis) | set(partner_npis))
    shared = _scan_csv(
        path,
        has_header=False,
        new_columns=[
            "npi_1",
            "npi_2",
            "shared_patient_count_proxy",
            "shared_services_proxy",
            "shared_episodes_proxy",
        ],
    )

    return (
        shared.select(
            pl.col("npi_1").cast(pl.Utf8, strict=False).str.strip_chars().alias("npi_1"),
            pl.col("npi_2").cast(pl.Utf8, strict=False).str.strip_chars().alias("npi_2"),
            pl.col("shared_patient_count_proxy")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("shared_patient_count_proxy"),
            pl.col("shared_services_proxy")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("shared_services_proxy"),
            pl.col("shared_episodes_proxy")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("shared_episodes_proxy"),
        )
        .filter(pl.col("npi_1").is_in(relevant_npis) | pl.col("npi_2").is_in(relevant_npis))
        .filter(pl.col("npi_1") != pl.col("npi_2"))
        .with_columns(
            _pair_low_expr(pl.col("npi_1"), pl.col("npi_2")).alias("pair_low_npi"),
            _pair_high_expr(pl.col("npi_1"), pl.col("npi_2")).alias("pair_high_npi"),
        )
        .group_by(["pair_low_npi", "pair_high_npi"])
        .agg(
            pl.col("shared_patient_count_proxy").max(),
            pl.col("shared_services_proxy").max(),
            pl.col("shared_episodes_proxy").max(),
        )
    )


def build_clinical_dyad_ledger(
    surgeons_df: pl.DataFrame | pl.LazyFrame,
    medicare_path: str | Path,
    surgeon_schema: SurgeonSchema = SurgeonSchema(),
    medicare_schema: MedicareSchema = MedicareSchema(),
    min_intervention_volume: int = 10,
    care_compare_path: str | Path = DEFAULT_CARE_COMPARE_PATH,
    facility_affiliation_path: str | Path = DEFAULT_FACILITY_AFFILIATION_PATH,
    shared_patient_patterns_path: str | Path | None = DEFAULT_SHARED_PATIENT_PATTERNS_PATH,
) -> pl.DataFrame:
    """Pair each top surgeon with the strongest local interventional psychiatrist.

    Matching no longer relies on geography alone. Each candidate dyad is scored
    using public-network evidence in descending order of confidence:
    1. Shared affiliated hospital
    2. Shared group practice / org PAC
    3. Same ZIP
    4. Same city/state
    5. Historical shared-patient-pattern volume when available
    """

    care_compare = load_care_compare_provider_reference(care_compare_path)
    org_memberships = load_provider_org_memberships(care_compare_path)
    hospital_affiliations = load_provider_hospital_affiliations(facility_affiliation_path)

    surgeons_lf = surgeons_df.lazy() if isinstance(surgeons_df, pl.DataFrame) else surgeons_df
    surgeons_prepared = (
        surgeons_lf.with_row_count("surgeon_row_id")
        .with_columns(
            (
                pl.col(surgeon_schema.npi).cast(pl.Utf8, strict=False).fill_null("").alias("surgeon_npi")
                if surgeon_schema.npi
                else pl.lit("").alias("surgeon_npi")
            ),
            pl.col(surgeon_schema.name).cast(pl.Utf8, strict=False).alias("surgeon_name"),
            pl.col(surgeon_schema.volume)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("surgeon_volume"),
            pl.col(surgeon_schema.city).cast(pl.Utf8, strict=False).alias("surgeon_city"),
            pl.col(surgeon_schema.state).cast(pl.Utf8, strict=False).alias("surgeon_state"),
            (
                _norm_text(pl.col(surgeon_schema.hospital)).alias("surgeon_hospital_key")
                if surgeon_schema.hospital
                else pl.lit("").alias("surgeon_hospital_key")
            ),
        )
        .join(care_compare, left_on="surgeon_npi", right_on="provider_npi", how="left")
        .with_columns(
            pl.coalesce(
                ["care_city", "surgeon_city"],
            ).alias("surgeon_city"),
            pl.coalesce(
                ["care_state", "surgeon_state"],
            ).alias("surgeon_state"),
            pl.col("care_zip5").fill_null("").alias("surgeon_zip5"),
            pl.col("org_pac_id").fill_null("").alias("surgeon_org_pac_id"),
            _norm_text(pl.col("surgeon_city")).alias("surgeon_city_key"),
            _norm_text(pl.col("surgeon_state")).alias("surgeon_state_key"),
            pl.col("care_facility_name").fill_null("").alias("surgeon_care_facility_name"),
        )
    )

    specialists = build_referring_specialist_funnel(
        medicare_path=medicare_path,
        schema=medicare_schema,
        care_compare_path=care_compare_path,
        min_intervention_volume=min_intervention_volume,
    )

    surgeon_npis = (
        surgeons_prepared.select("surgeon_npi")
        .filter(pl.col("surgeon_npi") != "")
        .unique()
        .collect()
        .get_column("surgeon_npi")
        .to_list()
    )
    partner_npis = (
        specialists.select("dyad_partner_npi")
        .filter(pl.col("dyad_partner_npi") != "")
        .unique()
        .collect()
        .get_column("dyad_partner_npi")
        .to_list()
    )

    shared_pairs = (
        load_shared_patient_pairs(shared_patient_patterns_path, surgeon_npis, partner_npis)
        if shared_patient_patterns_path and Path(shared_patient_patterns_path).exists()
        else pl.LazyFrame(
            {
                "pair_low_npi": [],
                "pair_high_npi": [],
                "shared_patient_count_proxy": [],
                "shared_services_proxy": [],
                "shared_episodes_proxy": [],
            }
        )
    )

    surgeons_hospitals = surgeons_prepared.join(
        hospital_affiliations,
        left_on="surgeon_npi",
        right_on="provider_npi",
        how="left",
    ).select("surgeon_row_id", pl.col("hospital_ccn").alias("match_hospital_ccn"))

    specialists_hospitals = specialists.join(
        hospital_affiliations,
        left_on="dyad_partner_npi",
        right_on="provider_npi",
        how="left",
    ).select("dyad_partner_npi", pl.col("hospital_ccn").alias("match_hospital_ccn"))

    surgeons_orgs = surgeons_prepared.join(
        org_memberships,
        left_on="surgeon_npi",
        right_on="provider_npi",
        how="left",
    ).select("surgeon_row_id", pl.col("org_pac_id").alias("match_org_pac_id"))

    specialists_orgs = specialists.join(
        org_memberships,
        left_on="dyad_partner_npi",
        right_on="provider_npi",
        how="left",
    ).select("dyad_partner_npi", pl.col("org_pac_id").alias("match_org_pac_id"))

    candidate_matches: list[pl.LazyFrame] = []

    same_hospital = (
        surgeons_hospitals.filter(
            pl.col("match_hospital_ccn").is_not_null() & (pl.col("match_hospital_ccn") != "")
        )
        .join(
            specialists_hospitals.filter(
                pl.col("match_hospital_ccn").is_not_null() & (pl.col("match_hospital_ccn") != "")
            ),
            on="match_hospital_ccn",
            how="inner",
        )
        .join(surgeons_prepared, on="surgeon_row_id", how="inner")
        .join(specialists, on="dyad_partner_npi", how="inner")
        .with_columns(
            pl.lit(1).alias("shared_hospital_flag"),
            pl.lit(0).alias("shared_org_pac_flag"),
            pl.when(
                (pl.col("surgeon_zip5") != "") & (pl.col("surgeon_zip5") == pl.col("partner_zip5"))
            )
            .then(1)
            .otherwise(0)
            .alias("same_zip_flag"),
            pl.when(
                (pl.col("surgeon_city_key") == pl.col("partner_city_key"))
                & (pl.col("surgeon_state_key") == pl.col("partner_state_key"))
            )
            .then(1)
            .otherwise(0)
            .alias("same_city_state_flag"),
            pl.lit("shared_hospital_affiliation").alias("match_type"),
        )
    )
    candidate_matches.append(same_hospital)

    same_org = (
        surgeons_orgs.filter(
            pl.col("match_org_pac_id").is_not_null() & (pl.col("match_org_pac_id") != "")
        )
        .join(
            specialists_orgs.filter(
                pl.col("match_org_pac_id").is_not_null() & (pl.col("match_org_pac_id") != "")
            ),
            on="match_org_pac_id",
            how="inner",
        )
        .join(surgeons_prepared, on="surgeon_row_id", how="inner")
        .join(specialists, on="dyad_partner_npi", how="inner")
        .with_columns(
            pl.lit(0).alias("shared_hospital_flag"),
            pl.lit(1).alias("shared_org_pac_flag"),
            pl.when(
                (pl.col("surgeon_zip5") != "") & (pl.col("surgeon_zip5") == pl.col("partner_zip5"))
            )
            .then(1)
            .otherwise(0)
            .alias("same_zip_flag"),
            pl.when(
                (pl.col("surgeon_city_key") == pl.col("partner_city_key"))
                & (pl.col("surgeon_state_key") == pl.col("partner_state_key"))
            )
            .then(1)
            .otherwise(0)
            .alias("same_city_state_flag"),
            pl.lit("shared_group_practice").alias("match_type"),
        )
    )
    candidate_matches.append(same_org)

    same_zip = (
        surgeons_prepared.filter(pl.col("surgeon_zip5") != "")
        .join(
            specialists.filter(pl.col("partner_zip5") != ""),
            left_on="surgeon_zip5",
            right_on="partner_zip5",
            how="inner",
        )
        .with_columns(
            pl.lit(0).alias("shared_hospital_flag"),
            pl.lit(0).alias("shared_org_pac_flag"),
            pl.lit(1).alias("same_zip_flag"),
            pl.when(
                (pl.col("surgeon_city_key") == pl.col("partner_city_key"))
                & (pl.col("surgeon_state_key") == pl.col("partner_state_key"))
            )
            .then(1)
            .otherwise(0)
            .alias("same_city_state_flag"),
            pl.lit("same_zip").alias("match_type"),
        )
    )
    candidate_matches.append(same_zip)

    same_city_state = (
        surgeons_prepared.join(
            specialists,
            left_on=["surgeon_city_key", "surgeon_state_key"],
            right_on=["partner_city_key", "partner_state_key"],
            how="left",
        )
        .with_columns(
            pl.lit(0).alias("shared_hospital_flag"),
            pl.lit(0).alias("shared_org_pac_flag"),
            pl.when(
                (pl.col("surgeon_zip5") != "") & (pl.col("surgeon_zip5") == pl.col("partner_zip5"))
            )
            .then(1)
            .otherwise(0)
            .alias("same_zip_flag"),
            pl.lit(1).alias("same_city_state_flag"),
            pl.lit("same_city_state").alias("match_type"),
        )
    )
    candidate_matches.append(same_city_state)

    candidate_pool = (
        pl.concat(candidate_matches, how="diagonal_relaxed")
        .filter(pl.col("dyad_partner_npi").is_not_null())
        .filter(pl.col("dyad_partner_npi") != "")
        .with_columns(
            _pair_low_expr(pl.col("surgeon_npi"), pl.col("dyad_partner_npi")).alias("pair_low_npi"),
            _pair_high_expr(pl.col("surgeon_npi"), pl.col("dyad_partner_npi")).alias("pair_high_npi"),
        )
        .join(shared_pairs, on=["pair_low_npi", "pair_high_npi"], how="left")
        .with_columns(
            pl.col("shared_patient_count_proxy").fill_null(0.0),
            pl.col("shared_services_proxy").fill_null(0.0),
            pl.col("shared_episodes_proxy").fill_null(0.0),
        )
        .group_by(["surgeon_row_id", "dyad_partner_npi"])
        .agg(
            pl.col("surgeon_npi").first(),
            pl.col("surgeon_name").first(),
            pl.col("surgeon_volume").first(),
            pl.col("surgeon_city").first(),
            pl.col("surgeon_state").first(),
            pl.col("dyad_partner_name").first(),
            pl.col("dyad_partner_specialty").first(),
            pl.col("dyad_partner_intervention_volume").max(),
            pl.col("dyad_partner_target_cpt_count").max(),
            pl.col("shared_hospital_flag").max(),
            pl.col("shared_org_pac_flag").max(),
            pl.col("same_zip_flag").max(),
            pl.col("same_city_state_flag").max(),
            pl.col("shared_patient_count_proxy").max(),
            pl.col("shared_services_proxy").max(),
            pl.col("shared_episodes_proxy").max(),
            pl.col("match_type").first(),
        )
        .with_columns(
            (
                pl.col("shared_hospital_flag") * 1000
                + pl.col("shared_org_pac_flag") * 400
                + pl.col("same_zip_flag") * 120
                + pl.col("same_city_state_flag") * 60
                + pl.col("dyad_partner_intervention_volume").clip(upper_bound=1000) / 20
                + pl.col("shared_patient_count_proxy").log1p() * 25
            )
            .round(2)
            .alias("dyad_relationship_score")
        )
        .with_columns(
            pl.when(pl.col("shared_hospital_flag") == 1)
            .then(pl.lit("High"))
            .when(
                (pl.col("shared_org_pac_flag") == 1)
                | (pl.col("shared_patient_count_proxy") >= 50)
                | (pl.col("same_zip_flag") == 1)
            )
            .then(pl.lit("Medium"))
            .when(pl.col("same_city_state_flag") == 1)
            .then(pl.lit("Low"))
            .otherwise(pl.lit("None"))
            .alias("referral_confidence_tier")
        )
        .sort(
            by=[
                "surgeon_row_id",
                "dyad_relationship_score",
                "dyad_partner_intervention_volume",
            ],
            descending=[False, True, True],
            nulls_last=True,
        )
        .unique(subset=["surgeon_row_id"], keep="first", maintain_order=True)
    )

    return (
        surgeons_prepared.join(candidate_pool, on="surgeon_row_id", how="left")
        .with_columns(
            pl.when(pl.col("dyad_partner_name").is_null())
            .then(pl.lit("High Friction Trial Site"))
            .otherwise(pl.lit("Matched Local Referrer"))
            .alias("trial_site_friction_flag"),
            pl.col("dyad_relationship_score").fill_null(0.0),
            pl.col("shared_hospital_flag").fill_null(0),
            pl.col("shared_org_pac_flag").fill_null(0),
            pl.col("same_zip_flag").fill_null(0),
            pl.col("same_city_state_flag").fill_null(0),
            pl.col("shared_patient_count_proxy").fill_null(0.0),
            pl.col("shared_services_proxy").fill_null(0.0),
            pl.col("shared_episodes_proxy").fill_null(0.0),
            pl.col("referral_confidence_tier").fill_null("None"),
        )
        .select(
            pl.col("surgeon_npi").alias("Surgeon_NPI"),
            pl.col("surgeon_name").alias("Surgeon_Name"),
            pl.col("surgeon_volume").alias("Surgeon_Volume"),
            pl.col("surgeon_city").alias("Surgeon_City"),
            pl.col("surgeon_state").alias("Surgeon_State"),
            pl.col("dyad_partner_name").alias("Dyad_Partner_Name"),
            pl.col("dyad_partner_specialty").alias("Dyad_Partner_Specialty"),
            pl.col("dyad_partner_intervention_volume").alias(
                "Dyad_Partner_Intervention_Volume"
            ),
            pl.col("dyad_partner_target_cpt_count").alias("Dyad_Partner_Target_CPT_Count"),
            pl.col("dyad_partner_npi").alias("Dyad_Partner_NPI"),
            pl.col("match_type").alias("Dyad_Match_Type"),
            pl.col("dyad_relationship_score").alias("Dyad_Relationship_Score"),
            pl.col("referral_confidence_tier").alias("Referral_Confidence_Tier"),
            pl.col("shared_hospital_flag").alias("Shared_Hospital_Flag"),
            pl.col("shared_org_pac_flag").alias("Shared_Org_PAC_Flag"),
            pl.col("same_zip_flag").alias("Same_Zip_Flag"),
            pl.col("same_city_state_flag").alias("Same_City_State_Flag"),
            pl.col("shared_patient_count_proxy").alias("Shared_Patient_Count_Proxy"),
            pl.col("shared_services_proxy").alias("Shared_Services_Proxy"),
            pl.col("shared_episodes_proxy").alias("Shared_Episodes_Proxy"),
            pl.col("trial_site_friction_flag").alias("Trial_Site_Friction_Flag"),
        )
        .collect()
    )


if __name__ == "__main__":
    surgeons_df = pl.read_csv("top15_surgeons.csv")
    dyad_ledger = build_clinical_dyad_ledger(
        surgeons_df=surgeons_df,
        medicare_path="medicare_physicians.csv",
        surgeon_schema=SurgeonSchema(npi="npi", hospital="hospital_affiliation"),
        medicare_schema=MedicareSchema(),
        min_intervention_volume=10,
    )
    dyad_ledger.write_csv("clinical_dyad_ledger.csv")
