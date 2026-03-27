# LEHD Snapshot Documentation: Individual Characteristics Files — Demographics (Section 4.1)

Source URL: https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/person_level/icf.html

Release: S2024_R2025Q4

---

## Overview

The Individual Characteristics Files (ICF) contain person-specific demographic information including sex, age/date-of-birth, place-of-birth, race, ethnicity, and education for all workers appearing in wage data.

**Core Files:**
- **ICF_US**: National scope file with one record per person (key: PIK)
- **ICF_US_IMPLICATES_AGE_SEX_POB**: Date of birth, sex, place of birth implicates
- **ICF_US_IMPLICATES_EDUCATION**: Education implicates
- **ICF_US_IMPLICATES_RACE_ETHNICITY**: Race and ethnicity implicates

## Key Concepts

### Data Sources
Demographic characteristics derive from:
- Census Numident (edited SSA transaction data)
- Decennial Census (2000, 2010 short forms; 2000 long form)
- American Community Survey (2001-present)

### Missing Data Rates

| Characteristic | Percent Missing | Model Stage |
|---|---|---|
| Sex | ~5% | A |
| Date of Birth | ~5% | A |
| Place of Birth | ~5% | A |
| Race | ~20% | B |
| Ethnicity | ~20% | B |
| Education | ~80% | C |

### Imputation Flags
Each demographic characteristic includes an imputation flag with values:
- **1**: Observed data
- **2**: Imputed data
- **3**: Imputed (observed data failed consistency checks)

### PIK Assignment
Workers receive Protected Identification Keys (9-digit codes replacing SSNs) through the Census Bureau's Person Identification Validation System (PVS), with match rates exceeding 90%.

## Imputation Systems

**Full Information Imputation (2010):** Uses Classification and Regression Trees (CART) to cluster similar workers, then applies Kernel Density Estimation to generate implicates.

**Limited Information Imputation (2009+):** Lower-quality method for new workers; conditions only on observed characteristics without earnings or co-worker data.

### Stage A Variables (Sex, DOB, POB)
- Modal Non-Native status
- Co-Worker Fraction Male
- New-Worker indicator

### Stage B Variables (Race, Ethnicity)
- Sex, DOB, Bestrace
- Annual earnings (quartiles)
- POB_race categories
- Co-resident/co-worker demographic groups

### Stage C Variables (Education, age 25+)
- Sex, DOB, Race/Ethnicity
- POB_race_educ (immigrant education by country)
- Annual earnings (ventiles)
- Industry (collapsed NAICS)
- Co-resident/co-worker groups

## ICF_US File Specification

**Access Requirements:** State, IRS, and SSA approval required

**File Formats:** SAS Data Table, Parquet (PIK-partitioned)

**Key:** pik | **Sort Order:** pik

### Variables

| Variable | Type | Length | Description |
|---|---|---|---|
| pik | char | 9 | Protected Identification Key |
| dob | num | 4 | Date of birth |
| sex | char | 1 | Sex (F/M) |
| pob | char | 1 | Place of birth |
| race | char | 1 | Race (codes 1-7) |
| ethnicity | char | 1 | Ethnicity (H/N) |
| educ_c | char | 1 | Highest education, age 25+ (1-4) |
| sex_imputed | char | 1 | Imputation status (1-3) |
| race_imputed | char | 1 | Imputation status (1-3) |
| ethnicity_imputed | char | 1 | Imputation status (1-3) |
| educ_c_imputed | char | 1 | Imputation status (1-3) |
| pob_imputed | char | 1 | Imputation status (1-3) |
| dob_imputed | char | 1 | Imputation status (1-3) |

### Sex Codes
- F: Female
- M: Male

### Place of Birth Codes
Numeric (1-9) for regions; alphabetic (A-Z) for specific countries/territories.

### Race Codes
1. White
2. Black
3. American Indian/Alaska Native
4. Asian
5. Native Hawaiian/Pacific Islander
7. Two or more races

### Ethnicity Codes
- H: Hispanic
- N: Non-Hispanic

### Education Codes (Age 25+)
1. Less than high school
2. High school
3. Some college
4. Bachelor's degree or higher

## Implicates Files

### ICF_US_IMPLICATES_AGE_SEX_POB
**Variables:** pik; dob1-dob10; sex1-sex10; pob1-pob10

### ICF_US_IMPLICATES_EDUCATION
**Variables:** pik; educ_c1-educ_c10

### ICF_US_IMPLICATES_RACE_ETHNICITY
**Variables:** pik; race1-race10; ethnicity1-ethnicity10

All implicates files contain 10 draws from Posterior Predictive Distributions for workers with missing characteristics.

## Data Quality Notes

Research staff last updated full information imputation in 2010. The documentation indicates "the quality of the imputed data has deteriorated as younger birth cohorts who receive only the limited information impute enter the data." Education data quality is most affected given its 80% missing rate. Researchers plan to update and integrate the full information system into production by 2025.
