# LEHD Snapshot Documentation: Employment History Files (Section 2.1)

Source URL: https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/jobs_level/ehf.html

Release: S2024_R2025Q4

---

## Overview

The Employment History Files comprise job-level data sourced from state unemployment insurance wage records. Four main tables exist:

**Employment History File (EHF_ZZ)**
- Annual records with quarterly earnings per covered job
- State scope; Key: PIK SEIN SEINUNIT YEAR
- Simple structure suitable for research analysis

**Job History File (JHF_ZZ)**
- Wide format with earnings in quarterly variables
- Includes imputed establishments for multi-unit employers
- State scope; Key: PIK SEIN SPELL_U2W
- Contains successor-predecessor linkage information

**Earnings Indicators (EHF_US_INDICATORS)**
- Flags indicating PIK earnings across LEHD states
- National scope; Key: PIK YEAR
- Identifies potential sample bias

**Earnings Availability (EHF_US_AVAILABILITY)**
- Quarter ranges for state data availability
- National scope; Key: STATE

## Key Technical Concepts

### Identifier Types
UI wage records use state tax identifiers (SEIN) as employers, except Minnesota which uses establishment-level reporting (SEINUNIT). These differ fundamentally from firm or establishment definitions.

### Longitudinal Job Construction
Job spells on JHF are arrayed quarterly. The SPELL_U2W variable forms a unique key with PIK-SEIN. The Firm Identifier (FID) links related records across SEIN changes, created by:

1. Sequential counter for all PIK records
2. Assignment of lowest counter value within related job sets
3. Character field prepended with state FIPS code for national uniqueness

### Imputation Methodology
The Worker-to-Establishment (U2W) imputation assigns workers to establishments based on relative size and residence-to-establishment distance. Ten implicates capture imputation uncertainty.

### Wage Record Gaps
Approximately 1.5 percent of jobs lack UI wage reports. Imputation identifies underreporting by comparing reported records to QCEW expectations, then conditions worker status on adjacent-quarter employment patterns.

### Successor-Predecessor Linking
Records across SEIN changes representing continuous employment receive identical FIDs. Linkage is confirmed when SPF-identified transitions coincide with PIK-level separation-accession patterns.

## EHF_ZZ Table Structure

**Access Requirements:** State, IRS, and SSA approval

**Variables:**

| Variable | Type | Length | Description |
|----------|------|--------|-------------|
| pik | char | 9 | Protected Identification Key |
| year | num | 4 | Calendar year |
| earn_ann | num | 8 | Annual earnings |
| earn1-earn4 | num | 5 | Quarterly earnings |
| sein | char | 12 | State Employer Identification Number |
| seinunit | char | 5 | State UI reporting unit |
| state | char | 2 | Geographic state (FIPS) |
| flag_impute1-4 | char | 1 | Imputation flags (0=reported, 1=imputed) |

## JHF_ZZ Table Structure

**Access Requirements:** State, IRS, and SSA approval

**Primary Variables:**

| Variable | Type | Length | Description |
|----------|------|--------|-------------|
| pik | char | 9 | Protected Identification Key |
| sein | char | 12 | State Employer Identification Number |
| spell_u2w | num | 3 | Spell identifier at SEIN |
| e21-e161 | num | 8 | Quarterly earnings (qtime-indexed) |
| fid | num | 5 | Linked job spell identifier |
| seinunit1-10 | char | 5 | Imputed establishment units |
| first_acc | num | 3 | First employment quarter |
| last_sep | num | 3 | Last employment quarter |
| flag_seinunit_imputed | num | 3 | Imputation flag (0=not imputed, 1=imputed) |
| random_pik_group | char | 2 | Random selector [00-99, AA] |

## EHF_US_INDICATORS Structure

**Scope:** National; **Key:** PIK YEAR

Variables track state count for earnings across quarters:
- num_states_earn1 through num_states_earn4

## EHF_ALL_AVAILABILITY Structure

**Scope:** National; **Key:** STATE

Provides availability windows:
- start_year, start_quarter, end_year, end_quarter (EHF)
- start_year_jhf through end_quarter_jhf (JHF)
- start_qtime_jhf, end_qtime_jhf (1985Q1=1 indexing)

## Analytical Guidance

**Subsample Selection:** Use RANDOM_PIK_GROUP from first two PIK digits (approximately uniform [00,99]; "AA" indicates no valid SSN).

**Establishment Characteristics:** Merge from ECF_SEIN (employment-weighted), ECF_SEIN_T26 (firm-level), ECF_SEINUNIT (establishment-level), or QWI_SEINUNIT tables.

**Multiple Imputation:** All 10 implicates recommended; if using single implicate, select first.

**Earnings Combination:** Sum earnings across multiple JHF records for same PIK-FID-quarter; weight job counts by earnings share and divide by 10 if using implicates.
