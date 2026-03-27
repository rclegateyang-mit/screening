# LEHD Person-Level Files Overview

**Sources:**
- [LEHD Snapshot Documentation S2024 - Person Level](https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/person_level.html)
- [LEHD Snapshot Documentation S2024 - Introduction](https://lehd.ces.census.gov/data/lehd-snapshot-doc/latest/sections/introduction.html)

---

## Overview

The person-level section of the LEHD Snapshot documentation is an index page that links to two main categories of person-level files:

### 1. Individual Characteristics Files - Demographics (ICF)

This section covers the core demographic data and imputation system. Sub-pages include:

| Sub-page | Content |
|---|---|
| **General overview and user guidance** | Description of imputation methodology, data quality guidance, full vs. limited information imputation |
| **ICF_US File codebook** | Variable definitions for the core demographics file (pik, dob, sex, pob, race, ethnicity, educ_c, imputation flags) |
| **ICF_US_IMPLICATES_AGE_SEX_POB** | 10 implicates for date of birth, sex, and place of birth |
| **ICF_US_IMPLICATES_EDUCATION** | 10 implicates for education (educ_c1-educ_c10, workers age 25+ only) |
| **ICF_US_IMPLICATES_RACE_ETHNICITY** | 10 implicates for race and ethnicity |

The detailed imputation methodology is documented on the ICF sub-page (`person_level/icf.html`), which covers:
- Three-stage imputation process (Stage A: sex/DOB/POB, Stage B: race/ethnicity, Stage C: education)
- CART + KDE methodology
- Full information vs. limited information imputation systems
- Cell variables for each stage
- Data quality guidance

### 2. Individual Characteristics Files - Residential Geography (ICF Residence)

This section covers residential location data. Sub-pages include:

| Sub-page | Content |
|---|---|
| **ICF_US_RESIDENCE_CPR File codebook** | Residential geography from the Census Bureau's Composite Person Record |
| **ICF_US_RESIDENCE_RCF File codebook** | Residential geography from the Resident Candidate File |

---

## LEHD Snapshot Data Architecture (Context)

The LEHD Snapshot is a quarterly database of linked employer-employee data covering over 95% of U.S. employment. It integrates administrative data from multiple federal and state agencies without additional respondent burden.

### Three File Categories

1. **Job-Level Data**: Earnings from UI wage records, organized by worker-employer links. Covers UI-covered employment only (excludes federal workers, independent contractors, self-employed).

2. **Employer-Level Data**: Firm characteristics including industry (NAICS), size, location, and aggregated employment measures.

3. **Person-Level Data**: Demographic characteristics and residence geography sourced from SSA and Census records. This is where the ICF files reside.

### Current Release

**S2024_R2025Q4** contains earnings data through 2024, with residence data available through 2023.

---

## Person-Level Data: Key Points for Users

### What is observed vs. imputed

The person-level files combine:
- **Observed data** from administrative records (Numident for sex, DOB, POB) and survey matches (Census 2000, ACS for race, ethnicity, education)
- **Imputed data** using a multi-stage CART + KDE methodology for unmatched workers

### Imputation rates by variable

| Variable | Approximate Imputation Rate | Stage |
|---|---|---|
| Sex | ~5% | A |
| Date of Birth | ~5% | A |
| Place of Birth | ~5% | A |
| Race | ~20% | B |
| Ethnicity | ~20% | B |
| Education | ~80-85% | C |

### Education data quality warning

The documentation explicitly warns that education data quality varies significantly:

> "A good rule of thumb is: the older the earnings year and/or the older average age of the workers in an estimation cell, the higher quality the education data."

This is because:
1. The full-information imputation model was last run in 2010 and is no longer operational
2. Workers entering from 2009 onward receive only limited-information imputation (conditioning only on age, sex, POB, race, ethnicity -- not on earnings, industry, or co-worker/co-resident characteristics)
3. Over time, the share of workers with only limited-information imputation grows

### Using multiple implicates

For any analysis stratified by imputed demographics (especially education), researchers should:
1. Run the analysis separately on each of the 10 implicates
2. Combine results using Rubin (1987) combining rules
3. This produces standard errors that properly account for imputation uncertainty

The core ICF_US file contains the **first implicate** as the default value. The separate IMPLICATES tables contain all 10 implicates for each variable set.

### Imputation flags

Each demographic variable has an associated `_imputed` flag:
- **1** = Observed (from administrative or survey data)
- **2** = Imputed (missing data, model-generated value)
- **3** = Imputed after observed data failed consistency checks

### Access requirements

All ICF files require:
- State Approval
- IRS Approval (for Title 26 data elements)
- SSA Approval

Files are available in SAS Data Table and Parquet formats, partitioned by PIK group.

---

## Key References

- Abowd, J.M., Stephens, B.E., Vilhuber, L., Andersson, F., McKinney, K.L., Roemer, M., and Woodcock, S. (2009). "The LEHD Infrastructure Files and the Creation of the Quarterly Workforce Indicators." In *Producer Dynamics: New Evidence from Micro Data*, pp. 149-230. University of Chicago Press.
- Graham, M. (2022). "LEHD Snapshot Documentation, Release S2021_R2022Q4." CES Working Paper 22-51.
- McKinney, K.L. et al. (2021). "Total Error and Variability Measures for the Quarterly Workforce Indicators and LEHD Origin-Destination Employment Statistics in OnTheMap." *Journal of Survey Statistics and Methodology*.
- Rubin, D.B. (1987). *Multiple Imputation for Nonresponse in Surveys*. New York: Wiley.
