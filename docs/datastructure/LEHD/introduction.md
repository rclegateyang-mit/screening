# LEHD Snapshot Documentation - Introduction

Source URL: https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/introduction.html

## Overview

The LEHD Snapshot represents the U.S. Census Bureau's quarterly database of linked employer-employee data. The program covers "over 95% of employment in the United States" by merging administrative data on jobs, businesses, and workers.

## Data Organization

The Snapshot organizes into three primary file types:

### Job-Level Files
- **Employment History File (EHF_ZZ)**: Quarterly earnings by job, reported to state UI systems
- **Job History File (JHF_ZZ)**: Wide-format earnings history with imputed establishment identifiers
- **Earnings Indicators (EHF_US_INDICATORS)**: National-scope PIK earnings presence flags
- **Earnings Availability (EHF_ALL_AVAILABILITY)**: Data availability ranges by state

### Employer-Level Files
- **Employer Characteristics File (ECF)**: Available at establishment and firm levels with industry/size/location data
- **Quarterly Workforce Indicators (QWI_ZZ_SEINUNIT)**: Establishment-level hires, separations, job growth, earnings
- **Successor-Predecessor File (SPF_ZZ)**: Employer flow relationships

### Person-Level Files
- **Individual Characteristics File (ICF_US)**: Worker demographics, birthplace, education
- **Multiple Imputation Files**: Separate tables for age/sex/place of birth, race/ethnicity, and education
- **Residence Geography Files**: CPR (1999-2011) and RCF (2012 forward) residence data

## Data Coverage & Exclusions

### Covered Employment
The data includes "most corporate officials, all executives, all supervisory personnel, all professionals" and wage earners under UI coverage. Notable exclusions are independent contractors, railroad workers, and federal government employees.

### Availability
Job and employer data spans from 1985 onward, with varying start dates by state. Most recent data extends through 2024. Residence data availability differs: CPR covers 1999-2011; RCF covers 2012-2023.

## Technical Specifications

### Data Formats
Files are provided in SAS format and—beginning with S2024—Parquet format.
- Parquet files are in a `parquet/` subdirectory alongside SAS tables
- Worker and job-level Parquet files are partitioned by first two PIK characters (00-99, plus AA for non-numeric)
- Example naming: `ehf_dc_00.parquet`
- Functionally identical to SAS with minor storage type differences
- Select metadata tables provided as comma-delimited text files

### Quarter Indexing
The system uses _qtime_ as a sequential quarter counter beginning with 1985Q1 = 1:
- `qtime = (year-1985) * 4 + quarter`
- `year = int((qtime-1)/4) + 1985`
- `quarter = mod(qtime-1, 4) + 1`

### Data Packages (approval-based delivery)

| Package | Contents | Approval |
|---------|----------|----------|
| Jobs Data | EHF_ZZ, JHF_ZZ, EHF_US_INDICATORS, EHF_ALL_AVAILABILITY | State, IRS, SSA |
| Employer Data (Standard) | ECF_ZZ_SEIN, ECF_ZZ_SEINUNIT, SPF_ZZ, QWI_ZZ_SEINUNIT | State |
| Employer Data (T26) | ECF_ZZ_SEIN_T26 | State + IRS |
| Person Demographics | ICF_US + 3 implicates files | State, IRS, SSA |
| Person Residence | ICF_US_RESIDENCE_CPR, ICF_US_RESIDENCE_RCF | State, IRS, SSA |

### State Coverage Examples

| State | ECF start | EHF start | QWI start | Notes |
|-------|-----------|-----------|-----------|-------|
| Maryland | 1985Q4 | 1985Q4 | 1990Q1 | |
| California | 1991Q1 | 1991Q3 | 1991Q3 | |
| Alaska | — | — | — | Inactive partner, data ends 2016Q2 |
| Massachusetts | — | — | — | Most recent data 2024Q3 |

## Access Requirements

Projects using LEHD Snapshot data require:
- Demonstration that public-use sources cannot support the research
- Standard FSRDC review and approval
- Additional approvals for: state-level tables (per state MOU), person demographics (SSA approval), and T26 employer data (IRS approval)

## Citation Guidance

Researchers should acknowledge NSF grants SAS-9978093, SES-0339191, ITR-0427889; NIH grant AG018854; and Sloan Foundation support. The suggested data citation format is: "U.S. Census Bureau. LEHD Snapshot Release S2024. \[Computer file\], U.S. Census Bureau, Center for Economic Studies, Research Data Centers, Washington, DC, 2025."
