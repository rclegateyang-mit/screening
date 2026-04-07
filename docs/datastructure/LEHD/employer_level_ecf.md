# LEHD Snapshot Documentation: Employer Characteristics File (ECF)

Source URL: https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/employer_level/ecf.html

Release: S2024_R2025Q4

---

## Document Overview
This documentation covers the Employer Characteristics File (ECF) from the LEHD Snapshot, release S2024_R2025Q4. The ECF contains employer-level data sourced primarily from the Quarterly Census of Employment and Wages (QCEW).

## 3.1 Employer Characteristics File Structure

### File Types
The ECF includes three file variants:

1. **ECF_ST_SEINUNIT** - Establishment-level data
   - Key: SEIN, YEAR, QUARTER, SEINUNIT
   - Scope: State

2. **ECF_ST_SEIN** - Tax identifier level modal characteristics
   - Key: SEIN, YEAR, QUARTER
   - Scope: State

3. **ECF_ST_T26** - National firm age and size
   - Key: SEIN, YEAR, QUARTER
   - Scope: State
   - Requires IRS approval

## Key Conceptual Distinctions

The documentation clarifies terminology: "SEINs are fundamentally a tax entity and distinct from an establishment or a firm." Single-unit firms represent approximately half of all workers. Multi-unit firms may have multiple SEINs across or within states. Establishments within the same SEIN typically operate in the same industry, though exceptions exist.

## Geography and Geocoding

Establishments receive geographic coding based on physical address (preferred), mailing address, or reported county. Quality codes range from 1-4 for physical addresses and 5-8 for mailing addresses, reflecting precision levels. When geocoding fails, county-level assignment receives a quality flag of 9, and imputation produces a flag of 10.

The documentation notes "unusually high rates of flux in geographic assignments have been noted in certain state-quarters."

### Connecticut County Changes
Beginning with 2023 data, Connecticut's historic counties were replaced with nine Councils of Governments (COGs). This change reflects administrative reorganization.

## Industry Classification

Industries undergo translation across multiple coding systems including SIC (1987) and various NAICS versions (1997-2022). The process involves:
- Identifying reported coding systems
- Using Census concordance tables
- Calculating transition shares from observed data
- Applying longitudinal edits and imputation

Only final codes are retained on the ECF.

## Employment and Wage Variables

### SEINUNIT-Level Variables
- empl_month1, empl_month2, empl_month3: ES202 employment (edited)
- total_wages: ES202 wages (edited)
- best_emp1-3, best_wages: Combined UI/202 data

Each employment and wage variable includes a flag indicating whether data were reported or imputed, with codes including:
- R: Reported data
- M: Missing data
- H: Hand-imputed
- E: Imputed from parent record
- K: Catastrophe-related imputation

### SEIN-Level Variables
- sein_emp1-3, sein_wages: Aggregated 202 data
- seinsize_m, seinsize_b, seinsize_e: UI employment
- sein_best_emp1-3, sein_best_wages: Best combined data

## Ownership Classification

The documentation indicates that establishments within the same SEIN "almost never" report different ownership codes. Four categories exist:
- Code 1: Federal government
- Code 2: State government
- Code 3: Local government
- Code 5: Private ownership

Federal employers are included but rarely appear in wage records.

## Firm Age and Size Methodology

Age and size derive primarily from the Longitudinal Business Database (LBD), which requires IRS approval to access. The approach employs:
- Firm age: Years from first year with positive payroll
- National firm size: March 12 employment sum across all establishments

When LBD linkage fails, values are "developed from ECF data using a combination of edits and imputation."

## FAS_EIN Variable Structure

The FAS_EIN enables firm age and size linkage. Formats include:
- EINUS + 9 digits: Federal EIN (e.g., EINUS123456789)
- EIN[state] + encrypted EIN: Encrypted federal EIN (e.g., EINNY123456ABC)
- SEIN + increment: When federal EIN unavailable (e.g., 33ABCD1234EFGH01)

## ECF_ZZ_SEINUNIT Variable Definitions

| Variable | Type | Description |
|----------|------|-------------|
| es_state | char(2) | FIPS state code |
| sein | char(12) | State employer ID |
| seinunit | char(5) | Establishment identifier |
| year | num | Year (YYYY) |
| quarter | num | Quarter (1-4) |
| qtime | num | Quarter index (1985Q1=1) |
| in_202 | bool | SEIN in QCEW (0/1) |
| in_ui | bool | SEIN in UI wage data (0/1) |
| empl_month1-3 | int32 | Monthly employment (edited) |
| empl_month1_flg-empl_month3_flg | char(1) | Employment source flags (see below) |
| total_wages | float64 | Quarterly wages (edited) |
| total_wages_flg | char(1) | Wages source flag (see below) |
| best_emp1-3 | float64 | Best UI/202 monthly employment |
| best_wages | float64 | Best UI/202 wages |
| es_owner_code | char(1) | Ownership (1,2,3,5) |
| geoid_2020 | char(15) | Census block geocode: state(2)+county(3)+tract(6)+block(4) |
| leg_state | char(2) | Current FIPS state |
| leg_county | char(5) | Current county identifier |
| leg_tract | char(6) | Current Census tract |
| leg_blkgrp | char(1) | Current block group |
| leg_block | char(4) | Census block |
| leg_latitude, leg_longitude | float64 | Internal point (6 decimals) |
| leg_geo_qual | uint8 | Geocoding quality (1-10) |
| leg_cbsa | char(5) | Core-Based Statistical Area |
| leg_cbsa_memi | char(1) | CBSA type (1=Metro, 2=Micro, 9=Non-metro) |
| leg_wib | char(8) | Workforce Investment Board |
| leg_ssccc | char(5) | State-county identifier |
| sic1987fnl | char(6) | Final SIC code |
| naics1997fnl through naics2022fnl | char(6) | NAICS codes by year |
| ramp_sein | float64 | Ramp fuzz factor |
| uniform_sein | float64 | Draw from uniform distribution |

### Employment/Wage Source Flags

| Code | Meaning |
|------|---------|
| R | Reported data |
| E | Imputed worksite employment from parent record |
| H | Hand-imputed |
| W | Estimated from wage record |
| C | Changed/re-reported |
| M | Missing data |
| L | Late reported |
| P | Prorated from master |
| K | Catastrophe imputation |
| N | Zero-filled pending resolution |

## ECF_ZZ_SEIN Variable Definitions

| Variable | Type | Description |
|----------|------|-------------|
| sein_emp1-3 | uint32 | SEIN 202 monthly employment |
| sein_wages | float64 | SEIN 202 wages |
| seinsize_m, seinsize_b, seinsize_e | uint32 | UI employment measures |
| sein_best_emp1-3 | float64 | Combined best employment |
| sein_best_wages | float64 | Combined best wages |
| best_flag | uint8 | Data source combination (1-12) |
| multi_unit | bool | Multi-establishment (0/1) |
| num_estabs | uint32 | Establishment count |
| mode_es_owner_code_emp | char(1) | Employment-weighted modal ownership |
| mode_leg_county_emp, mode_leg_cbsa_emp | char | Modal geography |
| mode_naics*_emp | char(6) | Modal NAICS codes |

## ECF_ZZ_SEIN_T26 Variables (LBD-Linked)

| Variable | Type | Description |
|----------|------|-------------|
| firmid | char(10) | LBD firm alpha |
| fas_ein | char(14) | Firm age/size identifier |
| fas_ein_flag | uint8 | Source of EIN (1-5) |
| firmage | float64 | Calculated firm age |
| source_age | char(1) | Age source (1-4) |
| firmsize | float64 | Initial firm size |
| source_size | char(1) | Size source (1-4) |
| firm_birth_year, firm_birth_quarter | int32/int8 | Birth timing |
| firmsize_fuzz | float64 | Noise-infused size |
| lbd_match | uint8 | LBD match status (1-5) |
| multi_unit_lbd | bool | Multi-unit in LBD |

## Geographic Quality Codes

Quality codes indicate geocoding precision:
- 1-4: Physical address-based (1=complete address, 4=centroid)
- 5-8: Mailing address-based (5=complete address, 8=centroid)
- 9: County from QCEW
- 10: County imputation

## Best Flag Values (SEIN-Level)

The best_flag variable indicates data source combinations for employment and earnings:
- Codes 1-6: Single-unit firms
- Codes 7-12: Multi-unit firms
- Combinations specify UI vs. QCEW sources and data availability

## Data Access Requirements

- **ECF_ZZ_SEINUNIT**: State approval required
- **ECF_ZZ_SEIN**: State approval required
- **ECF_ZZ_SEIN_T26**: State and IRS approval required

## File Formats

All ECF files are available in:
- SAS Data Tables
- Parquet format

CSV codebooks are downloadable for each file and variable.
