# Abowd, Stephens, Vilhuber et al. (2009) - LEHD Infrastructure Files: ICF Imputation System

**Full Citation:** Abowd, J.M., Stephens, B.E., Vilhuber, L., Andersson, F., McKinney, K.L., Roemer, M., and Woodcock, S. (2009). "The LEHD Infrastructure Files and the Creation of the Quarterly Workforce Indicators." In *Producer Dynamics: New Evidence from Micro Data*, T. Dunne, J.B. Jensen, and M.J. Roberts, eds., pp. 149-230. University of Chicago Press. (Studies in Income and Wealth, Vol. 68.)

**Sources:**
- [NBER Book Chapter Page](https://www.nber.org/books-and-chapters/producer-dynamics-new-evidence-micro-data/lehd-infrastructure-files-and-creation-quarterly-workforce-indicators)
- [NBER Chapter PDF](https://www.nber.org/system/files/chapters/c0485/c0485.pdf) (binary PDF, not directly web-extractable)
- [LEHD Technical Paper TP-2006-01](https://lehd.ces.census.gov/doc/technical_paper/tp-2006-01.pdf) (earlier working paper version)
- [IDEAS/RePEc Entry](https://ideas.repec.org/h/nbr/nberch/0485.html)
- [De Gruyter](https://www.degruyterbrill.com/document/doi/10.7208/9780226172576-008/html?lang=en) (paywalled)

**Note:** The full text of this chapter was not directly extractable via web fetch (PDF binary encoding). The content below is assembled from the NBER page metadata, the LEHD Snapshot documentation that implements this methodology, web search results, and the paper's table of contents structure.

---

## Paper Overview

This is the foundational technical paper documenting the LEHD Infrastructure File system. Chapter 3 (Section 3.2) covers the Individual Characteristics File (ICF) and its imputation system in detail, with subsections:

- 3.2.1: Age and sex imputation
- 3.2.2: Place of residence imputation
- 3.2.3: Education imputation

The paper covers pp. 149-230 of the NBER volume, providing extensive technical detail on the full LEHD data infrastructure.

---

## ICF Imputation System (as described in this paper and implemented in LEHD production)

### Fundamental Architecture

The ICF imputation system addresses the problem that administrative records (UI wage records, SSA Numident) contain limited demographic information. The system links these records to survey data (Decennial Census, ACS) via the Protected Identification Key (PIK) to obtain demographics. For workers not matched to survey data, demographics are imputed.

### Missing Data Framework

The paper establishes the monotone missing data pattern:
- **Sex, DOB, POB**: ~5% missing (from Numident)
- **Race, Ethnicity**: ~20% missing (from surveys)
- **Education**: ~80% missing (from Census 2000 long form and ACS)

This monotone pattern enables staged imputation where each subsequent stage conditions on previously completed variables.

### CART Methodology

The Classification and Regression Tree (CART) approach:
1. **Partitions** workers into subgroups (cells/bins) based on observable characteristics
2. **Optimizes** cell boundaries to maximize between-cell variability and minimize within-cell variability for the target imputation variable
3. **Assigns** each worker to exactly one cell based on their observable characteristics

The CART approach is essentially a non-parametric classification method that creates data-driven groupings of "similar" workers without imposing parametric distributional assumptions.

### KDE Methodology

Within each CART-defined cell:
1. A **joint Kernel Density Estimate (KDE)** is computed for all variables requiring imputation at that stage
2. The KDE provides smooth probability distributions over the joint space of missing variables
3. **Implicates are generated** by sampling randomly from these estimated densities

This two-step process (CART for cell assignment, KDE for within-cell density estimation) combines:
- The flexibility of non-parametric tree-based methods for defining relevant conditioning groups
- The smoothness of kernel density estimation for generating plausible imputed values

### Multiple Imputation Framework

Following Rubin (1987):
- **10 implicates** (complete datasets) are generated through repeated sampling from the posterior predictive distribution (PPD)
- Each implicate represents a plausible complete dataset under the MAR assumption
- The **Dirichlet prior distribution** is used for imputation probabilities
- Imputations are made by sampling from the posterior predictive distribution, which is also Dirichlet
- Standard errors can be computed using Rubin's combining rules across the 10 implicates

### Stage A: Sex, Date of Birth, Place of Birth

**Conditioning/Cell Variables:**
- Modal Non-Native status (based on co-worker nativity composition)
- Co-Worker Fraction Male (binary: high >0.5, low <=0.5)
- New-Worker indicator (entered after state's first reporting quarter)

**Key insight:** Stage A missing data is fundamentally different -- these workers likely lack SSNs on the Numident, suggesting undocumented immigration status. The assumption is that recent legal immigrants share characteristics with recent undocumented immigrants, especially at workplaces with high immigrant concentrations.

### Stage B: Race, Ethnicity

**Conditioning/Cell Variables:**
- Sex, DOB (from Stage A)
- Bestrace (collapsed Numident race)
- Average annual earnings (quartiles, when bestrace missing)
- POB_race (collapsed place-of-birth by modal race)
- Co-resident and co-worker racial/ethnic group characteristics

**Key insight:** Stage B missing data (survey non-response/non-sampling) is more likely MAR than Stage A.

### Stage C: Education (Age 25+)

**Dual approach due to severe data scarcity (80-85% missing):**

1. **Log-linear model**: Does not require fully interacted effects across all conditioning variables, providing modeling flexibility given limited observed education data.

2. **Non-parametric CART/KDE**: Standard approach used in Stages A and B, applied in parallel.

**Conditioning/Cell Variables:**
- Sex, DOB (from Stage A)
- Race/Ethnicity (from Stage B)
- POB_race_educ (collapsed place-of-birth based on country-specific immigrant education levels)
- Average annual earnings across all jobs (**ventiles** -- 20 quantile groups)
- Earnings_FB (low/high earnings categories for foreign-born workers)
- Industry (collapsed NAICS groups based on observed education distribution)
- Co-resident racial/ethnic group characteristics
- Co-worker racial/ethnic group characteristics

**Key insight:** Education is the most challenging imputation because (a) the missing rate is extremely high (~80-85%), (b) the observed education data comes primarily from the Census 2000 long form (1-in-6 sample) and ACS (annual rolling sample), and (c) the relationship between education and other characteristics is complex.

---

## Related Working Papers

### McKinney et al. (2017, 2021) - Total Error Framework

**Citation:** McKinney, K.L., Green, A.S., Vilhuber, L., and Abowd, J.M. (2017/2021). "Total Error and Variability Measures for the Quarterly Workforce Indicators and LEHD Origin-Destination Employment Statistics in OnTheMap."

- [CES Working Paper 17-71](https://ideas.repec.org/p/cen/wpaper/17-71.html)
- [CES Working Paper 20-30](https://ideas.repec.org/p/cen/wpaper/20-30.html)
- [Journal of Survey Statistics and Methodology (2021)](https://academic.oup.com/jssam/advance-article-abstract/doi/10.1093/jssam/smaa029/5955529)

This paper conducts the first comprehensive total quality evaluation of QWI indicators. The evaluation is conducted by generating **multiple threads of the edit and imputation models** used in the LEHD Infrastructure File System. Each thread or implicate is the output of formal probability models that address coverage, edit, and imputation errors.

Key findings relevant to education imputation:
- The multiple implicate framework allows proper variance estimation accounting for imputation uncertainty
- Total variability measures decompose into components from sampling, imputation, and disclosure limitation
- Education-stratified QWI indicators have higher total error due to the high imputation rate for education

### Graham (2022) - LEHD Snapshot Documentation

**Citation:** Graham, M. (2022). "LEHD Snapshot Documentation, Release S2021_R2022Q4." CES Working Paper 22-51.

- [Census Working Paper](https://www2.census.gov/ces/wp/2022/CES-WP-22-51.pdf)

This documents the complete LEHD Snapshot data infrastructure, including the ICF imputation system as implemented in production.

---

## Key Takeaways for Researchers Using LEHD Education Data

1. **Education is ~80-85% imputed** -- the vast majority of education values in LEHD are model-generated, not observed.

2. **The full-information imputation model was last run in 2010** and has not been updated since. Workers entering after 2009 receive only limited-information imputation.

3. **Limited-information imputation does not use earnings or workplace characteristics** -- it conditions only on age, sex, place of birth, race, and ethnicity, substantially reducing quality.

4. **Quality degrades over time** as newer cohorts with limited-information imputation replace older cohorts with full-information imputation.

5. **Use multiple implicates** to properly account for imputation uncertainty in standard errors (Rubin combining rules).

6. **Older workers and earlier years have better education data** -- a direct consequence of the 2010 model freeze.

7. **The 4-category education variable is coarse** -- even with perfect imputation, the categories (less than HS, HS, some college, BA+) limit the granularity of education analysis.
