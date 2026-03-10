# LEHD Individual Characteristics Files (ICF) - Detailed Imputation Methodology

**Sources:**
- [LEHD Snapshot Documentation S2024 - ICF Demographics](https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/person_level/icf.html)
- [LEHD Snapshot Documentation S2023/Latest - ICF Demographics](https://lehd.ces.census.gov/data/lehd-snapshot-doc/latest/sections/person_level/icf.html)
- [LEHD Snapshot Documentation S2021 - ICF Demographics](https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2021/sections/person_level/icf.html)

---

## Overview

The Individual Characteristics File (ICF_US) provides demographic data for all workers appearing in UI wage records. It contains one record per person on the Employer History File (EHF) with characteristics completed through observed data and imputation. Three additional IMPLICATES tables provide multiple imputed values (10 draws from the Posterior Predictive Distribution, or PPD) for workers with incomplete observed data.

## Data Sources

Demographic information originates from three primary sources:

1. **Numident** (Census Numident, derived from Social Security Administration data) -- source for sex, date of birth, place of birth
2. **Decennial Census** (2000 short form, 2000 long form, 2010 short form) -- source for race, ethnicity, education (2000 long form only)
3. **American Community Survey (ACS)** (2001-present) -- source for race, ethnicity, education

## Protected Identification Key (PIK) Assignment

The system replaces Social Security Numbers with 9-digit PIKs maintaining one-to-one mapping. Assignment uses the Census Bureau's Person Identification Validation System (PVS). Primary matching attempts SSN, name, and date-of-birth verification against the Numident. For survey data lacking these elements, the **GeoSearch method** limits matching candidates to records at identical physical addresses, followed by general name and DOB searches. Match rates exceed 90% of survey respondents, though assignment may not be exclusive -- multiple survey records can receive the same PIK at a single time point.

LEHD performs unduplication using additional characteristics on both Numident and survey data, producing one record per PIK per data source-year.

## Missing Data Patterns

Missing data follows a **monotone pattern**, enabling staged imputation:

| Characteristic | Primary Source | Missing Rate | Imputation Stage |
|---|---|---|---|
| Sex | Numident | ~5% | A |
| Date of Birth | Numident | ~5% | A |
| Place of Birth | Numident, Decennial, ACS | ~5% | A |
| Race | Decennial, ACS | ~20% | B |
| Ethnicity | Decennial, ACS | ~20% | B |
| Education | Decennial (2000 long form), ACS | ~80-85% | C |

This hierarchical structure allows completion "in stages, starting from the least missing characteristics and then moving on to the next set of variables."

---

## Three-Stage Imputation Process

### General Methodology: CART + KDE

The system employs a **Classification and Regression Tree (CART)** based approach to cluster groups of similar workers together into cells/bins. The imputation models are defined by the cells workers are assigned to. The more homogeneity of the imputation variable within a cell, the greater the predictive power of the model. Ideally, once workers are assigned to cells, the variability in outcomes will be large across cells and small within cells.

Within each cell, a **joint Kernel Density Estimate (KDE)** is estimated for the variables that need to be completed at that stage. The estimated joint densities are used to generate implicates for workers with missing data by cell ID by sampling randomly from the density.

**Critical assumption:** Data is **Missing at Random (MAR)** within each cell ID.

### Multiple Implicate Generation

When a variable is missing, **10 implicates** (draws from the Posterior Predictive Distribution, or PPD) are provided. This conforms to the Rubin (1987) multiple imputation framework, where each implicate is the output of formal probability models that address coverage, edit, and imputation errors. The Dirichlet prior distribution is used for imputation probabilities, with imputations made by sampling from the posterior predictive distribution (which is also Dirichlet).

---

### Stage A: Sex, Date of Birth, Place of Birth (~5% Missing)

**Nature of missing data:** Stage A missing data differs fundamentally from later stages. Most workers lacking PIKs on the Numident are likely undocumented residents. The system assumes "recent legal immigrants have similar characteristics as recent illegal immigrants," particularly when employed at firms with high concentrations of recent legal immigrants.

**Cell Variables for Stage A:**

| Variable | Description |
|---|---|
| Modal Non-Native | Set to "U.S." if co-worker fraction native exceeds 0.5; otherwise set to the modal value across all co-workers at all jobs for that worker |
| Co-Worker Fraction Male | Coded "high" if >0.5 male; otherwise "low" |
| New-Worker | Binary indicator: 1 if entered data after first quarter state appears in records; 0 otherwise. Workers appearing after initial state reporting year are more likely immigrants. |

---

### Stage B: Race, Ethnicity (~20% Missing)

**Nature of missing data:** Missing data comes from survey non-response or non-sampling. These workers are "much more likely to be MAR than the workers in Stage A." Missing data represents combinations of persons not sampled and non-respondents.

**Cell Variables for Stage B:**

| Variable | Description |
|---|---|
| Sex | From Stage A (observed or imputed) |
| DOB | Date of birth from Stage A |
| Bestrace | Collapsed Numident race information |
| Earnings | Average annual earnings across all jobs (quartiles), used when bestrace is missing |
| POB_race | Collapsed place-of-birth categories based on modal race per place of birth |
| Co-Resident demographics | Black/White, Hispanic/Not-Hispanic group characteristics of co-residents |
| Co-Worker demographics | Black/White, Hispanic/Not-Hispanic group characteristics of co-workers |

---

### Stage C: Education for Workers Age 25+ (~80-85% Missing)

Education has "significantly less observed data available to build a model" due to the very high missing rate. Only workers age 25+ receive education imputation (since younger workers may not have completed their education).

**Dual Methodological Approach:**

1. **Log-linear model:** Imputes education without requiring fully interacted effects across all conditioning variables, providing greater modeling flexibility given the limited observed education data. This approach does not require the fully non-parametric structure used in Stages A and B.

2. **Parallel non-parametric CART/KDE approach:** Uses the same CART + KDE methodology as Stages A and B.

**Cell Variables for Stage C:**

| Variable | Description |
|---|---|
| Sex | From Stage A |
| DOB | Date of birth from Stage A |
| Race/Ethnicity | From Stage B |
| POB_race_educ | Collapsed place-of-birth categories based on country-specific immigrant education levels |
| Earnings | Average annual earnings across all jobs (**ventiles** -- 20 quantile groups) |
| Earnings_FB | Low and high earnings categories specifically for foreign-born workers |
| Industry | Collapsed NAICS industry groups based on observed education distribution |
| Co-Resident demographics | Black/White, Hispanic/Not-Hispanic group characteristics |
| Co-Worker demographics | Black/White, Hispanic/Not-Hispanic group characteristics |

**Education Categories (educ_c):**

| Code | Description |
|---|---|
| 1 | Less than high school |
| 2 | High school diploma or equivalent |
| 3 | Some college or associate's degree |
| 4 | Bachelor's degree or higher |

---

## Full Information vs. Limited Information Imputation

### Full Information Imputation (Operational through 2010)

The full information imputation system uses the complete set of conditioning variables described above for each stage, including:
- Earnings history (ventiles for Stage C)
- Co-worker characteristics
- Co-resident characteristics
- Industry information
- Place-of-birth collapsed by education levels

This system was developed and **run once by the research staff in 2010** and **is no longer operational**.

Workers who had positive UI earnings **prior to 2009** and no observed education received this full information imputation.

### Limited Information Imputation (2009 Forward)

Workers entering the Snapshot from **2009 onward** with missing education receive the lower-quality "limited" information imputation. This model:

- Conditions **only** on observed characteristics: **age, sex, place of birth, race, ethnicity**
- Draws missing characteristics using a set of posterior predictive distributions (PPDs) estimated from the 2010 model run
- Does **NOT** condition on:
  - Annual earnings
  - Co-worker characteristics
  - Co-resident characteristics
  - Industry information

This represents a substantially degraded imputation quality compared to the full information system.

---

## Data Quality Issues and Known Limitations

### Progressive Quality Degradation

The documentation explicitly states: "the full information imputation system was only run once by the research staff in 2010 and is no longer operational." Over the following 15+ years, data quality has deteriorated as:

- **Younger birth cohorts** who receive only the limited information imputation **enter** the data
- **Older birth cohorts** who received the full information imputation **exit** (retire, die)

### Education Quality is Most Severely Affected

For education specifically, "the quality impact is more severe" due to the ~80-85% missing rate. The document provides this guidance:

> "A good rule of thumb is: the older the earnings year and/or the older average age of the workers in an estimation cell, the higher quality the education data."

This means:
- **Education data for recent years and younger workers is lowest quality** -- these workers almost entirely have limited-information imputation
- **Education data for earlier years and older workers is higher quality** -- these workers are more likely to have full-information imputation or observed data from the 2000 Census long form

### Quality Particularly Severe for Recent Estimation Cells

The documentation warns that quality is "particularly severe for more recent earnings year estimation cells composed primarily of younger workers."

### Standard Error Adjustment Required

Estimated standard errors require adjustment when using ICF characteristics to account for imputation variability. McKinney et al. (2021) provides methods for this adjustment.

### Planned Update

A Census research team is actively working to update and reintegrate the full information imputation system into production. Completion was expected in 2025, but the current (S2024) documentation still describes the system as not operational.

---

## ICF_US Core File Variables

**Key:** pik (Protected Identification Key, 9-character string)

| Variable | Type | Description |
|---|---|---|
| pik | char(9) | Protected Identification Key |
| dob | numeric(4) | Date of birth (year) |
| sex | char(1) | Sex (F=Female, M=Male) |
| pob | char(1) | Place of birth (coded A-Z, 1-9; see POB coding below) |
| race | char(1) | Race (1=White, 2=Black, 3=American Indian/Alaska Native, 4=Asian, 5=Native Hawaiian/Pacific Islander, 7=Two or more races) |
| ethnicity | char(1) | Ethnicity (H=Hispanic, N=Non-Hispanic) |
| educ_c | char(1) | Highest educational attainment, age 25+ (1-4 scale) |
| sex_imputed | char(1) | Imputation flag for sex |
| dob_imputed | char(1) | Imputation flag for DOB |
| pob_imputed | char(1) | Imputation flag for POB |
| race_imputed | char(1) | Imputation flag for race |
| ethnicity_imputed | char(1) | Imputation flag for ethnicity |
| educ_c_imputed | char(1) | Imputation flag for education |

### Imputation Flag Values

| Flag | Meaning |
|---|---|
| 1 | Demographic characteristic observed |
| 2 | Demographic characteristic imputed |
| 3 | Demographic characteristic was observed but failed internal consistency checks, then imputed |

---

## IMPLICATES Tables

### ICF_US_IMPLICATES_AGE_SEX_POB
Provides 10 implicates each for dob, sex, and pob for workers with missing values.
- Variables: dob1-dob10, sex1-sex10, pob1-pob10
- Coding identical to ICF_US core file

### ICF_US_IMPLICATES_EDUCATION
Provides 10 implicates for education for workers age 25+ with missing education.
- Variables: educ_c1-educ_c10
- Coding: same 1-4 scale as core file
- Only populated for workers age 25+

### ICF_US_IMPLICATES_RACE_ETHNICITY
Provides 10 implicates each for race and ethnicity.
- Variables: race1-race10, ethnicity1-ethnicity10
- Coding identical to ICF_US core file

**Note:** The core ICF_US file uses the **first implicate** as the default value, with the imputation flags indicating whether the value is observed or imputed.

---

## Place of Birth Coding (Collapsed 27 Categories)

| Code | Region/Country |
|---|---|
| 1 | Central Asia |
| 2 | Southeast Asia |
| 3 | Middle East/North Africa |
| 4 | Caribbean |
| 5 | Central America |
| 6 | South America |
| 7 | Africa |
| 8 | Oceania |
| 9 | Not Specified |
| A | United States/territories |
| B | Mexico |
| C | Philippines |
| D | Vietnam |
| E | India |
| F | Germany |
| G | Puerto Rico |
| H | El Salvador |
| I | Cuba |
| J | United Kingdom |
| K | Canada |
| L | China |
| M | South Korea |
| N | Taiwan |
| O | Guatemala |
| P | Japan |
| Q | Haiti |
| R | USSR Core |
| S | Jamaica |
| T | Colombia |
| U | Poland |
| V | Iran |
| W | Dominican Republic |
| X | Italy |
| Y | Former Socialist Europe |
| Z | Western Europe |

---

## Access Requirements

All ICF files require:
- State Approval
- IRS Approval
- SSA Approval

Files available in SAS Data Table and Parquet formats (partitioned by PIK group). Sort order is by pik; scope is National.

---

## Key References

- Abowd, J.M., Stephens, B.E., Vilhuber, L., Andersson, F., McKinney, K.L., Roemer, M., and Woodcock, S. (2009). "The LEHD Infrastructure Files and the Creation of the Quarterly Workforce Indicators." In *Producer Dynamics: New Evidence from Micro Data*, pp. 149-230. University of Chicago Press. (NBER Chapter c0485)
- McKinney, K.L. et al. (2017). "Total Error and Variability Measures with Integrated Disclosure Limitation for Quarterly Workforce Indicators and LEHD Origin Destination Employment Statistics in OnTheMap." CES Working Paper 17-71.
- McKinney, K.L. et al. (2021). "Total Error and Variability Measures for the Quarterly Workforce Indicators and LEHD Origin-Destination Employment Statistics in OnTheMap." *Journal of Survey Statistics and Methodology*.
- Rubin, D.B. (1987). *Multiple Imputation for Nonresponse in Surveys*. New York: Wiley.
