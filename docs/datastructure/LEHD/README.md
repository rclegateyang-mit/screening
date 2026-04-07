# LEHD Snapshot Documentation (S2024_R2025Q4)

Local copies of the U.S. Census Bureau LEHD Snapshot documentation pages, fetched from the official documentation site. Release version S2024_R2025Q4, last updated January 23, 2026.

Base URL: https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/

## Files

| File | Description | Source URL |
|------|-------------|------------|
| [index.md](index.md) | Main table of contents listing all sections and subsections of the LEHD Snapshot documentation | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/index.html |
| [introduction.md](introduction.md) | Overview of the LEHD Snapshot program: data organization, file types, coverage, exclusions, data formats (SAS/Parquet), quarter indexing (qtime), access requirements, and citation guidance | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/introduction.html |
| [jobs_level.md](jobs_level.md) | Section 2 landing page for jobs-level files: EHF (Employment History File), JHF (Job History File), EHF_US_INDICATORS, and EHF_ALL_AVAILABILITY | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/jobs_level.html |
| [employer_level.md](employer_level.md) | Section 3 landing page for employer-level files: ECF (Employer Characteristics File), SPF (Successor-Predecessor File), and QWI (Quarterly Workforce Indicators) | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/employer_level.html |
| [person_level.md](person_level.md) | Section 4 landing page for person-level files: ICF_US demographics, multiple imputation files (age/sex/POB, education, race/ethnicity), and residential geography files (CPR, RCF) | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/person_level.html |
| [acronyms.md](acronyms.md) | Complete glossary of acronyms used throughout the LEHD documentation (BR, CBSA, EHF, ICF, PIK, QCEW, SEIN, UI, etc.) | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/appendices/acronyms.html |
| [value_labels.md](value_labels.md) | Selected codebook value labels for categorical variables: state codes, CBSA codes, county codes, WIB codes, NAICS/SIC industry codes, and indexed quarter (qtime) values | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/appendices/value_labels.html |
| [version_history.md](version_history.md) | Release history and errata for snapshots S2021 through S2024, documenting structural changes, new features (Parquet format, NAICS updates, Connecticut county codes), and data coverage extensions | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/appendices/version_history.html |
| [bibliography.md](bibliography.md) | Full bibliography of academic papers and technical reports cited in the LEHD documentation, covering QWI methodology, confidentiality protection, LBD redesign, and residence file development | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/bibliography/bibliography.html |

## Detailed Codebooks

| File | Description | Source URL |
|------|-------------|------------|
| [jobs_level_ehf.md](jobs_level_ehf.md) | Full codebooks for EHF_ZZ (Employment History File), JHF_ZZ (Job History File), EHF_US_INDICATORS, and EHF_ALL_AVAILABILITY: variable definitions, types, lengths, imputation methodology, U2W establishment assignment, FID construction, and analytical guidance | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/jobs_level/ehf.html |
| [employer_level_ecf.md](employer_level_ecf.md) | Full codebooks for ECF_ZZ_SEINUNIT (establishment-level), ECF_ZZ_SEIN (tax-ID-level modal characteristics), and ECF_ZZ_SEIN_T26 (LBD-linked firm age and size): all variable definitions, geocoding quality codes, ownership codes, best_flag values, industry classification methodology, and FAS_EIN format | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/employer_level/ecf.html |
| [employer_level_spf.md](employer_level_spf.md) | Full codebook for SPF_ZZ (Successor-Predecessor File): all variable definitions, link_ui and succ_link_ui flag values, source flag values, size class codes, and methodology for identifying SEIN transitions from UI wage records and ES202 firm reports | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/employer_level/spf.html |
| [employer_level_qwi.md](employer_level_qwi.md) | Full codebook for QWI_ZZ_SEINUNIT (Quarterly Workforce Indicators): all variable definitions including complete set of worker/job flow measures (b, e, m, f, a, s, h, r, etc.), fuzzed delta_ counterparts, earnings measures, establishment-level calculated rates, and all codebook value details | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/employer_level/qwi.html |
| [person_level_icf.md](person_level_icf.md) | Full codebooks for ICF_US, ICF_US_IMPLICATES_AGE_SEX_POB, ICF_US_IMPLICATES_EDUCATION, and ICF_US_IMPLICATES_RACE_ETHNICITY: variable definitions, sex/race/ethnicity/education code values, imputation flag codes, missing data rates by characteristic, and imputation system descriptions (CART, KDE, Stage A/B/C) | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/person_level/icf.html |
| [person_level_icfres.md](person_level_icfres.md) | Full codebooks for ICF_US_RESIDENCE_CPR (1999-2011, CPR-sourced) and ICF_US_RESIDENCE_RCF (2012+, RCF-sourced): variable definitions, flag_latlong and flag_rcf code values, geographic precision levels, and notes on CPR discontinuation and Connecticut COG county changes | https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/person_level/icfres.html |

## Related Documentation

- [Census RDC computing environment](../../notes/census_rdc_computing_environment.md) -- OS, software, PBS Pro, Python packages
- [Census RDC disclosure rules](../../notes/census_rdc_disclosure_rules.md) -- output review, rounding, cell sizes, LEHD-specific rules
- [Census RDC workflow guide](../../notes/census_rdc_workflow.md) -- code input, batch jobs, data conversion, disclosure preparation
