# LEHD Snapshot Documentation: Quarterly Workforce Indicators (QWI) — Section 3.3

Source URL: https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/employer_level/qwi.html

Release: S2024_R2025Q4. Last updated: January 23, 2026.

---

## 3.3.1 Overview

The QWI table (QWI_ZZ_SEINUNIT) contains establishment-level workforce metrics calculated from unemployment insurance wage records. The internal reference is the Unit Flows File (UFF_B).

**Scope:** State
**Key:** SEIN YEAR QUARTER SEINUNIT

### Establishment-level measures include:
- Worker and Job Flows (accessions, separations, job creation, job destruction)
- Worker compensation for stocks and flows
- Dynamic worker compensation summary statistics

## 3.3.2 User Guidance

### QWI Calculation

QWI measures derive from job-level longitudinal work history, corrected for successor-predecessor relationships. Measures account for multiple imputation and reporting issues, then aggregate to establishment level.

### QWI Count Measures

| QWI Variable | Job Reference | Description | Leading Quarters | Lagging Quarters | Public Use QWI |
|---|---|---|---|---|---|
| B | b | Beginning-of-period employment | 1 | 0 | Emp |
| E | e | End-of-period employment | 0 | 1 | EmpEnd |
| M | m | Employment any time during period | 0 | 0 | EmpTotal |
| F | f | Full-quarter employment | 1 | 1 | EmpS |
| Fpv | - | Full-quarter employment - previous quarter | 2 | 1 | EmpSpv |
| A | a1 | Accessions | 1 | 0 | HirA |
| CA | a2 | Accession into Continuing Quarter Employment | 1 | 1 | HirAEnd |
| FA | a3 | Flow into full-quarter employment | 2 | 1 | HirAS |
| S | s1 | Separations | 0 | 1 | Sep |
| CS | s2 | Separations from Continuing Quarter Employment | 1 | 1 | SepBeg |
| FS | s3 | Flow out of full-quarter employment | 2 | 1 | SepS |
| FSnx | - | Flow out of full-quarter employment - next quarter | 1 | 2 | SepSnx |
| H | h1 | New hires | 4 | 0 | HirN |
| CH | h2 | New Hires into Continuous Quarter Employment | 4 | 1 | - |
| H3 | h3 | Full-quarter new hires | 5 | 1 | HirNS |
| R | r1 | Recalls | 4 | 0 | HirR |
| CR | r2 | Recalls into Continuous Quarter Employment | 4 | 1 | - |
| W1 | w1 | Total payroll of all employees | 0 | 0 | Payroll |

### QWI Earnings

| QWI Variable | Job Reference | Description | Leading Quarters | Lagging Quarters | Earnings Base | Public Use QWI |
|---|---|---|---|---|---|---|
| W2B | w2b | Average monthly earnings of beginning-of-period employees | 1 | 0 | B | EarnBeg |
| W3 | w3 | Average monthly earnings of full-quarter employees | 1 | 1 | F | EarnS |
| WFA | wa3 | Average monthly earnings of transits to full-quarter status | 2 | 1 | FA | EarnHirAS |
| WH3 | wh3 | Average monthly earnings of new hires to full-quarter status | 5 | 1 | H3 | EarnHirNS |
| WFS | ws3 | Average monthly earnings of separations from full-quarter status | 1 | 2 | FSnx | EarnSepS |

### Establishment-level Calculations

| QWI Variable | Description | Leading Quarters | Lagging Quarters | Calculation | Public Use QWI |
|---|---|---|---|---|---|
| Ebar | Average employment | 1 | 1 | (E + B) / 2 | - |
| JF | Net job flows | 1 | 1 | E-B | FrmJbC |
| JC | Job creation | 1 | 1 | max(0,JF) | FrmJbGn |
| JCR | Average job creation rate | 1 | 1 | JC /Ebar | - |
| JD | Job destruction | 1 | 1 | \|min(0,JF)\| | FrmJbLs |
| JDR | Average job destruction rate | 1 | 1 | JD /Ebar | - |
| Fbar | Average full-quarter employment | 2 | 1 | (F + Fpv)/2 | - |
| FJF | Net change in full-quarter employment | 2 | 1 | F-Fpv | FrmJbCS |
| FG | Average full-quarter employment growth rate | 2 | 1 | FJF /Fbar | - |
| FJC | Full-quarter job creation | 2 | 1 | max(0,FJF) | FrmJbGnS |
| FJCR | Average full-quarter job creation rate | 2 | 1 | FJC /Fbar | - |
| FJD | Full-quarter job destruction | 2 | 1 | \|min(0,FJF)\| | FrmJbLsS |
| FJDR | Average full-quarter job destruction rate | 2 | 1 | FJD / Fbar | - |
| AR | Average accession rate | 1 | 1 | A / Ebar | - |
| SR | Average separation rate | 1 | 1 | S / Ebar | - |
| FAR | Average rate of flow into full-quarter employment | 2 | 1 | FA / Fbar | - |
| FSR | Average rate of flow out of full-quarter employment | 2 | 1 | FS / Fbar | - |

### Establishment Characteristics on QWI Table

Establishment characteristics are obtained via lookup to the Employer Characteristics File using SEIN-SEINUNIT-YEAR-QUARTER key. When establishments don't exist in all quarters, longitudinal lookup retrieves characteristics from following quarters, noted in UNIT_QUARTERS_OFF variable.

## 3.3.3 Codebook

### Table Metadata: QWI_ZZ_SEINUNIT

**Access Requirements:** State Approval Required, IRS Approval Required, SSA Approval Required

**Description:** "The microdata version of the public use QWI data, containing establishment-level hires, separations, net job growth, and average earnings."

**Scope:** State
**Key:** sein year quarter seinunit
**Sort Order:** sein year quarter seinunit
**File Formats:** SAS Data Table, Parquet

### Variable Information for QWI_ZZ_SEINUNIT

| Variable Name | SAS Type | SAS Length | Parquet Type | Description |
|---|---|---|---|---|
| sein | char | 12 | string | State Employer ID Number |
| seinunit | char | 5 | string | State UI Reporting Unit Number |
| year | num | 4 | uint32 | Year YYYY |
| quarter | num | 4 | uint8 | Quarter QQ |
| es_owner_code | char | 1 | string | Cleaned OWNER_CODE O |
| geoid_2020 | char | 15 | string | Census geocode: Tabulation state FIPS (2) \|\| county FIPS (3) \|\| tract (6) \|\| block (4) |
| leg_geo_qual | num | 3 | uint8 | Quality of final geography |
| sic1987fnl | char | 6 | string | Cleaned SIC Code IIII |
| naics2022fnl | char | 6 | string | Final 2022 NAICS Code NNNNNN |
| delta | num | 8 | float64 | Ramp fuzz factor |
| deltaz | num | 8 | float64 | Ramp fuzz factor- Z measures |
| leg_cbsa | char | 5 | string | Core-Based Statistical Area |
| leg_state | char | 2 | string | ES202 FIPS State SS |
| leg_county | char | 5 | string | County designation |
| leg_wib | char | 8 | string | Workforce Investment Board |
| qwi_unit_weight | num | 3 | float32 | UI this quarter and ES in at least one quarter |
| es_state | char | 2 | string | State FIPS code |
| unit_quarters_off | num | 3 | float32 | Number of quarters away from data that establishment was found |
| qwi_final_weight | num | 3 | float32 | In-scope for QWI tabulation |
| b | num | 8 | float64 | Beginning-of-period employment |
| e | num | 8 | float64 | End-of-period employment |
| m | num | 8 | float64 | Employment any time during period |
| f | num | 8 | float64 | Full-quarter employment |
| h3 | num | 8 | float64 | Full-quarter new hires |
| w1 | num | 8 | float64 | Total payroll of all employees |
| w2b | num | 8 | float64 | Average monthly earnings of beginning-of-period employees |
| w3 | num | 8 | float64 | Average monthly earnings of full-quarter employees |
| wh3 | num | 8 | float64 | Average monthly earnings of new hires to full-quarter status |
| delta_b | num | 8 | float64 | Fuzzed: Beginning-of-period employment |
| delta_e | num | 8 | float64 | Fuzzed: End-of-period employment |
| delta_m | num | 8 | float64 | Fuzzed: Employment any time during period |
| delta_f | num | 8 | float64 | Fuzzed: Full-quarter employment |
| fpv | num | 8 | float64 | Full-quarter employment - previous quarter |
| delta_fpv | num | 8 | float64 | Fuzzed: Full-quarter employment - previous quarter |
| a | num | 8 | float64 | Accessions |
| delta_a | num | 8 | float64 | Fuzzed: Accessions |
| ca | num | 8 | float64 | Accession into Continuing Quarter Employment |
| delta_ca | num | 8 | float64 | Fuzzed: Accession into Continuing Quarter Employment |
| fa | num | 8 | float64 | Flow into full-quarter employment |
| delta_fa | num | 8 | float64 | Fuzzed: Flow into full-quarter employment |
| s | num | 8 | float64 | Separations |
| delta_s | num | 8 | float64 | Fuzzed: Separations |
| cs | num | 8 | float64 | Separations from Continuing Quarter Employment |
| delta_cs | num | 8 | float64 | Fuzzed: Separations from Continuing Quarter Employment |
| fs | num | 8 | float64 | Flow out of full-quarter employment |
| delta_fs | num | 8 | float64 | Fuzzed: Flow out of full-quarter employment |
| fsnx | num | 8 | float64 | Flow out of full-quarter employment - next quarter |
| delta_fsnx | num | 8 | float64 | Fuzzed: Flow out of full-quarter employment - next quarter |
| h | num | 8 | float64 | New hires |
| delta_h | num | 8 | float64 | Fuzzed: New hires |
| ch | num | 8 | float64 | New Hires into Continuous Quarter Employment |
| delta_h3 | num | 8 | float64 | Fuzzed: Full-quarter new hires |
| r | num | 8 | float64 | Recalls |
| delta_r | num | 8 | float64 | Fuzzed: Recalls |
| cr | num | 8 | float64 | Recalls into Continuous Quarter Employment |
| delta_w1 | num | 8 | float64 | Fuzzed: Total payroll of all employees |
| delta_w2b | num | 8 | float64 | Fuzzed: Average monthly earnings of beginning-of-period employees |
| delta_w3 | num | 8 | float64 | Fuzzed: Average monthly earnings of full-quarter employees |
| wfa | num | 8 | float64 | Average monthly earnings of transits to full-quarter status |
| delta_wfa | num | 8 | float64 | Fuzzed: Average monthly earnings of transits to full-quarter status |
| delta_wh3 | num | 8 | float64 | Fuzzed: Average monthly earnings of new hires to full-quarter status |
| wfs | num | 8 | float64 | Average monthly earnings of separations from full-quarter status |
| delta_wfs | num | 8 | float64 | Fuzzed: Average monthly earnings of separations from full-quarter status |
| ebar | num | 8 | float64 | Average employment |
| delta_ebar | num | 8 | float64 | Fuzzed: Average employment |
| jf | num | 8 | float64 | Net job flows |
| delta_jf | num | 8 | float64 | Fuzzed: Net job flows |
| jc | num | 8 | float64 | Job creation |
| delta_jc | num | 8 | float64 | Fuzzed: Job creation |
| jcr | num | 8 | float64 | Average job creation rate |
| jd | num | 8 | float64 | Job destruction |
| delta_jd | num | 8 | float64 | Fuzzed: Job destruction |
| jdr | num | 8 | float64 | Average job destruction rate |
| fbar | num | 8 | float64 | Average full-quarter employment |
| delta_fbar | num | 8 | float64 | Fuzzed: Average full-quarter employment |
| fjf | num | 8 | float64 | Net change in full-quarter employment |
| delta_fjf | num | 8 | float64 | Fuzzed: Net change in full-quarter employment |
| fg | num | 8 | float64 | Average full-quarter employment growth rate |
| fjc | num | 8 | float64 | Full-quarter job creation |
| delta_fjc | num | 8 | float64 | Fuzzed: Full-quarter job creation |
| fjcr | num | 8 | float64 | Average full-quarter job creation rate |
| fjd | num | 8 | float64 | Full-quarter job destruction |
| delta_fjd | num | 8 | float64 | Fuzzed: Full-quarter job destruction |
| fjdr | num | 8 | float64 | Average full-quarter job destruction rate |
| ar | num | 8 | float64 | Average accession rate |
| sr | num | 8 | float64 | Average separation rate |
| far | num | 8 | float64 | Average rate of flow into full-quarter employment |
| fsr | num | 8 | float64 | Average rate of flow out of full-quarter employment |

### Details for es_owner_code

**Description:** Cleaned OWNER_CODE O

| Code | Label |
|---|---|
| 1 | Federal government |
| 2 | State government |
| 3 | Local government |
| 5 | Private ownership |

### Details for leg_geo_qual

**Description:** Quality of final geography

| Code | Label |
|---|---|
| 1 | Physical address location based on complete street address |
| 2 | Physical address location based on ZIP+4, ZIP+2, or USPS finance area centroid |
| 3 | Physical address location based on street or ZIP Code centroid with block group accuracy |
| 4 | Physical address location based on street centroid within tract, ZIP Code, or city; 5-digit ZIP Code; or municipality centroid |
| 5 | Mailing address location based on complete street address |
| 6 | Mailing address location based on ZIP+4, ZIP+2, or USPS finance area centroid |
| 7 | Mailing address location based on street or ZIP Code centroid with block group accuracy |
| 8 | Mailing address location based on street centroid within tract, ZIP Code, or city; 5-digit ZIP Code; or municipality centroid |
| 9 | County assignment from QCEW |
| 10 | County imputation |
