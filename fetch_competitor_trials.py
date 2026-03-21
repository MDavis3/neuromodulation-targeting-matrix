from __future__ import annotations

import argparse
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl


LOGGER = logging.getLogger("neuromodulation_targeting_matrix.fetch_competitor_trials")

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_PATH = PROCESSED_DIR / "competitor_neuromodulation_trials.csv"
CTG_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
TARGET_SPONSOR_QUERY = "(Medtronic OR Abbott)"
ACTIVE_STATUSES = {
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
}
STATUS_LABELS = {
    "RECRUITING": "Recruiting",
    "ACTIVE_NOT_RECRUITING": "Active, not recruiting",
    "NOT_YET_RECRUITING": "Not yet recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by invitation",
}
NEUROMODULATION_KEYWORDS = (
    "neuromod",
    "neurostimulation",
    "neurostimulator",
    "deep brain",
    "spinal cord",
    "brain stimulation",
    "vagus",
    "dorsal root ganglion",
    "sacral nerve",
    "responsive neurostimulation",
    "peripheral nerve",
    "tms",
)
TARGET_SPONSOR_NAMES = ("medtronic", "abbott")
US_STATE_NAME_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch current Medtronic/Abbott neuromodulation trials from ClinicalTrials.gov v2."
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "neuromodulation-targeting-matrix/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def build_query_url(page_size: int, page_token: str | None = None) -> str:
    params = {
        "query.spons": TARGET_SPONSOR_QUERY,
        "pageSize": str(page_size),
    }
    if page_token:
        params["pageToken"] = page_token
    return f"{CTG_BASE_URL}?{urllib.parse.urlencode(params)}"


def collect_candidate_studies(page_size: int) -> list[dict]:
    studies: list[dict] = []
    page_token: str | None = None

    while True:
        url = build_query_url(page_size=page_size, page_token=page_token)
        payload = fetch_json(url)
        batch = payload.get("studies", [])
        studies.extend(batch)
        LOGGER.info("Fetched %s studies from %s", len(batch), url)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return studies


def normalize_text(value: str | None) -> str:
    return (value or "").strip()


def normalize_state_value(value: str | None) -> str:
    cleaned = normalize_text(value)
    return US_STATE_NAME_TO_ABBR.get(cleaned, cleaned)


def extract_status(protocol_section: dict) -> str:
    status_module = protocol_section.get("statusModule", {})
    overall_status = normalize_text(status_module.get("overallStatus"))
    if overall_status in ACTIVE_STATUSES:
        return overall_status
    last_known_status = normalize_text(status_module.get("lastKnownStatus"))
    if last_known_status in ACTIVE_STATUSES:
        return last_known_status
    return overall_status or last_known_status


def extract_matching_sponsors(protocol_section: dict) -> list[str]:
    sponsor_module = protocol_section.get("sponsorCollaboratorsModule", {})
    names: list[str] = []

    lead_sponsor = sponsor_module.get("leadSponsor", {}) or {}
    if normalize_text(lead_sponsor.get("name")):
        names.append(normalize_text(lead_sponsor.get("name")))

    for collaborator in sponsor_module.get("collaborators", []) or []:
        name = normalize_text(collaborator.get("name"))
        if name:
            names.append(name)

    return [
        sponsor_name
        for sponsor_name in names
        if any(target in sponsor_name.lower() for target in TARGET_SPONSOR_NAMES)
    ]


def is_neuromodulation_study(protocol_section: dict) -> bool:
    identification_module = protocol_section.get("identificationModule", {})
    conditions_module = protocol_section.get("conditionsModule", {})
    interventions_module = protocol_section.get("armsInterventionsModule", {})

    searchable_values = [
        normalize_text(identification_module.get("briefTitle")),
        normalize_text(identification_module.get("officialTitle")),
        *[normalize_text(value) for value in conditions_module.get("conditions", []) or []],
        *[normalize_text(value) for value in conditions_module.get("keywords", []) or []],
        *[
            normalize_text(intervention.get("name"))
            for intervention in interventions_module.get("interventions", []) or []
        ],
        *[
            normalize_text(intervention.get("type"))
            for intervention in interventions_module.get("interventions", []) or []
        ],
    ]
    searchable_text = " ".join(searchable_values).lower()
    return any(keyword in searchable_text for keyword in NEUROMODULATION_KEYWORDS)


def extract_trial_rows(studies: list[dict]) -> list[dict]:
    rows: list[dict] = []

    for study in studies:
        protocol_section = study.get("protocolSection", {}) or {}
        status_value = extract_status(protocol_section)
        if status_value not in ACTIVE_STATUSES:
            continue

        matching_sponsors = extract_matching_sponsors(protocol_section)
        if not matching_sponsors:
            continue

        if not is_neuromodulation_study(protocol_section):
            continue

        identification_module = protocol_section.get("identificationModule", {}) or {}
        conditions_module = protocol_section.get("conditionsModule", {}) or {}
        locations_module = protocol_section.get("contactsLocationsModule", {}) or {}

        nct_id = normalize_text(identification_module.get("nctId"))
        if not nct_id:
            continue

        therapy_area = "; ".join(
            normalize_text(condition)
            for condition in (conditions_module.get("conditions", []) or [])
            if normalize_text(condition)
        )
        matched_sponsor = " / ".join(sorted(set(matching_sponsors)))

        for location in locations_module.get("locations", []) or []:
            country = normalize_text(location.get("country"))
            city = normalize_text(location.get("city"))
            state = normalize_state_value(location.get("state"))
            facility = normalize_text(location.get("facility"))

            if country != "United States" or not city or not state:
                continue

            rows.append(
                {
                    "Trial_ID": nct_id,
                    "City": city,
                    "State": state,
                    "Overall_Status": STATUS_LABELS.get(status_value, status_value.title()),
                    "Sponsor": matched_sponsor,
                    "Therapy_Area": therapy_area,
                    "Facility": facility,
                    "Source": "ClinicalTrials.gov v2 API",
                }
            )

    return rows


def write_trials_csv(rows: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        df = (
            pl.DataFrame(rows)
            .unique(subset=["Trial_ID", "City", "State"], keep="first")
            .sort(["State", "City", "Trial_ID"])
        )
    else:
        df = pl.DataFrame(
            schema={
                "Trial_ID": pl.Utf8,
                "City": pl.Utf8,
                "State": pl.Utf8,
                "Overall_Status": pl.Utf8,
                "Sponsor": pl.Utf8,
                "Therapy_Area": pl.Utf8,
                "Facility": pl.Utf8,
                "Source": pl.Utf8,
            }
        )

    df.write_csv(output_path)
    LOGGER.info("Wrote %s competitor trial rows to %s", df.height, output_path)
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    studies = collect_candidate_studies(page_size=args.page_size)
    rows = extract_trial_rows(studies)
    write_trials_csv(rows, args.output_path)


if __name__ == "__main__":
    main()
