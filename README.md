# Neuromodulation Targeting Matrix

An open-source, Polars-driven Go-To-Market (GTM) and clinical sourcing engine for Class III neuromodulation devices.

## Overview

This project demonstrates how to process large federal healthcare datasets with a memory-efficient Python and Polars pipeline to identify high-value physician targets for clinical partnerships, investigator outreach, and enterprise commercialization strategy.

The engine is designed for teams operating in regulated MedTech environments where commercial execution depends on finding physicians who combine:

- High procedural volume
- Relevant research activity
- Trial-site execution capability
- Low competitive financial entrenchment
- Observable market friction around incumbent hardware

The repository is structured as a portfolio-grade technical implementation rather than a productized application. It emphasizes scalable data engineering, reproducible feature construction, and executive-facing output generation.

## Architecture

The pipeline uses Polars lazy evaluation to minimize memory pressure and push filters, projections, and aggregations as far upstream as possible.

Core characteristics:

- Lazy ingestion with `pl.scan_csv()` for multi-gigabyte federal flat files
- Modular source-specific normalization functions
- Physician-level feature synthesis across five data pillars
- Deterministic scoring logic for clinical/commercial prioritization
- Executive-ready visualization output

## Data Pillars

The engine combines five public-data pillars into a unified physician targeting ledger:

1. Medicare utilization
   Measures procedural volume using CPT/HCPCS activity associated with neuromodulation-related surgical workflows.

2. NIH RePORTER / ExPORTER
   Identifies physicians associated with active research activity relevant to neuromodulation, neural interfaces, and closed-loop systems.

3. ClinicalTrials.gov / AACT
   Detects investigators and institutions with prior device-trial execution history, including completion and recruitment outcomes.

4. CMS Open Payments
   Quantifies financial ties to incumbent manufacturers to surface financially independent KOLs and reduce false-positive commercial targets.

5. FDA MAUDE
   Captures adverse-event and hardware-friction signals from legacy device ecosystems to inform competitive positioning.

## Scoring Model

The final ranking is based on a composite physician-level score built from:

- Surgical volume percentile
- Research signal from active NIH grants
- Trial infrastructure signal from AACT
- Competitive friction signal from MAUDE-derived features
- Penalties for high competitor consulting exposure

The purpose of the score is not to provide a clinical truth metric. It is a prioritization heuristic for GTM and clinical sourcing teams that need to narrow a national physician universe into a short list of high-probability targets.

## Why Polars

This implementation uses Polars instead of pandas because the workload is dominated by:

- Multi-gigabyte CSV inputs
- Repeated column pruning and predicate pushdown
- Large group-by aggregations
- Join-heavy feature engineering

Polars lazy execution makes the pipeline practical on commodity hardware while preserving clear, auditable transformation logic.

## Outputs

The pipeline produces:

- A physician-level ledger suitable for downstream analysis
- Ranked target lists for commercial or clinical sourcing workflows
- Executive-facing visual summaries of top-ranked targets

Raw and processed data are intentionally excluded from this public repository.

## Repository Scope

This public repo includes the pipeline code and project structure only.

It does not include:

- Raw source data
- Processed outputs
- Internal strategy notes
- Client-specific or company-specific targeting narratives

## Running the Pipeline

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the engine:

```bash
python sourcing_engine.py
```

The script expects local copies of the required public datasets. Those inputs are not distributed in this repository.

## Public-Repo Notes

If you are adapting this project for your own GTM or clinical-sourcing use case:

- Keep all raw federal files outside version control
- Treat any score as a prioritization layer, not a standalone decision system
- Validate all matching logic when joining public datasets with inconsistent identifiers

## License

Add the license appropriate for your intended public use before publishing.
