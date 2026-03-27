# Education Imputation in LEHD (2000-2018)

**Sources:** McKinney, Green, Vilhuber, and Abowd (2017, CES-WP-17-71); Abowd et al. (2009); LEHD Snapshot Documentation (Graham 2022).

---

## 1. Why Education Must Be Imputed

LEHD is built from state unemployment insurance (UI) wage records, which contain only worker identifiers, employer identifiers, and quarterly earnings -- no demographic information. Education data must be linked in from external sources via Protected Identification Keys (PIKs):

- **Census 2000 Long Form** (1-in-6 household sample): primary source, provides ~7% coverage of LEHD workers
- **American Community Survey (ACS)** (2001+): rolling annual sample, raises coverage to ~15%

The remaining **~80-85% of workers** have no observed education and must receive imputed values.

## 2. Education Categories

Education is collapsed into four ordered categories:

| Code | Description |
|------|-------------|
| 1 | Less than high school diploma |
| 2 | High school graduate, no college |
| 3 | Some college or associate's degree |
| 4 | Bachelor's degree or above |

Education is only imputed for workers **age 25 and older**, since younger workers may not have completed their education. QWI/LODES education-stratified tables restrict to this population.

## 3. Imputation Architecture: Three-Stage Monotone Process

Education imputation is the **third and final stage** of a sequential multiple imputation process that exploits the monotone missing data pattern:

| Stage | Variables | Missing Rate | Method |
|-------|-----------|-------------|--------|
| A | Sex, date of birth, place of birth | ~5% | Non-parametric KDE |
| B | Race, ethnicity | ~20% | Non-parametric KDE |
| **C** | **Education** | **~80-85%** | **Log-linear model + KDE** |

Each stage conditions on completed values from prior stages. This monotone structure arises because workers missing education are a superset of those missing race/ethnicity, who are a superset of those missing sex/DOB.

## 4. Education Imputation Method (Stage C) -- Full Information

### 4.1 Conditioning Variables (Stratifiers)

The education imputation conditions on:

- **Sex** (from Stage A)
- **Date of birth / age** (from Stage A)
- **Race and ethnicity** (from Stage B)
- **Place of birth by income quantile** (collapsed country-of-birth categories based on immigrant education levels)
- **Native / non-native status**
- **Modal NAICS industry** (6 collapsed categories) for dominant job
- **Collapsed race/ethnicity cells**
- **Full-quarter earnings deciles** (continuous, discretized)
- **Coworker fraction male**
- **Co-resident demographics** (fraction white, fraction Hispanic)
- **Co-worker demographics** (fraction white, fraction Hispanic)

### 4.2 Statistical Model

Two approaches are used in parallel:

1. **Log-linear model with flat priors (primary):** A fully interacted log-linear model with a reduced parameter set. This was chosen over the KDE approach because:
   - When stratifiers create many cells (especially with detailed geography), cells become too small for KDE
   - The log-linear model can include stratifiers as main effects or with limited interactions
   - Acts as a small-area estimator: main effects from stratifiers, local effects from the log-linear structure

2. **Non-parametric CART + KDE (secondary):** The same approach used in Stages A and B. Workers are partitioned into cells via Classification and Regression Trees (CART), then within each cell a joint Kernel Density Estimate generates imputed values.

#### Formal Setup

Let $c \in \{1,2,3,4\}$ index the four education categories (< HS, HS, Some College, BA+). Define a cell $g$ by the full interaction of all conditioning variables (sex, age group, race, ethnicity, POB, earnings decile, industry, coworker/co-resident demographics). Within cell $g$, let $n_g^{obs}$ denote the number of workers with observed education, and let $n_{g,c}^{obs}$ be the count observed in category $c$.

**Multinomial-Dirichlet model.** The observed education counts within cell $g$ are modeled as multinomial:

$$
(n_{g,1}^{obs}, \ldots, n_{g,4}^{obs}) \mid \boldsymbol{\pi}_g \sim \text{Multinomial}(n_g^{obs},\; \boldsymbol{\pi}_g)
$$

with a symmetric Dirichlet prior (flat, i.e., $\alpha_c = 1$ for all $c$):

$$
\boldsymbol{\pi}_g \sim \text{Dirichlet}(\alpha_1, \ldots, \alpha_4), \quad \alpha_c = 1
$$

The posterior is then:

$$
\boldsymbol{\pi}_g \mid \text{data} \sim \text{Dirichlet}(n_{g,1}^{obs} + 1, \ldots, n_{g,4}^{obs} + 1)
$$

with posterior mean for category $c$:

$$
\hat{\pi}_{g,c} = \frac{n_{g,c}^{obs} + 1}{n_g^{obs} + 4}
$$

**Log-linear variant.** When cells are too sparse for the fully interacted model, a log-linear model with reduced interactions is used instead. The log-linear model estimates:

$$
\ln \pi_{g,c} = \mu_c + \sum_k \beta_{k,c} \, x_{g,k} + \text{(selected interactions)}
$$

where $x_{g,k}$ are the stratifier values for cell $g$. This allows borrowing strength across cells via main effects while retaining local flexibility through limited interactions. The posterior predictive distribution is still Dirichlet, with parameters derived from the fitted model.

**Generating implicates.** For each worker $i$ with missing education in cell $g$:
1. Draw $\boldsymbol{\pi}_g^{(l)}$ from the posterior $\text{Dirichlet}(n_{g,1}^{obs}+1, \ldots, n_{g,4}^{obs}+1)$
2. Draw $e_i^{(l)} \sim \text{Categorical}(\boldsymbol{\pi}_g^{(l)})$

This is repeated $L=10$ times, producing implicates $e_i^{(1)}, \ldots, e_i^{(10)}$.

### 4.3 Multiple Implicates and Rubin Combining Rules

Following Rubin (1987), $L = 10$ implicates (complete datasets) are generated. Each implicate $l$ produces a complete set of education values -- observed values are retained, missing values are drawn from the PPD. The first implicate is stored as the default in the ICF core file; all 10 are available in `ICF_US_IMPLICATES_EDUCATION` (`educ_c1` through `educ_c10`).

#### Point Estimates

For any quantity of interest $Q$ (e.g., employment count in a cell stratified by education), the point estimate is the average across implicates:

$$
\bar{Q} = \frac{1}{L} \sum_{l=1}^{L} \hat{Q}^{(l)}
$$

For QWI employment counts, the fuzzed-weighted count in cell $k$ for implicate $l$ is:

$$
B_k^{(l)*} = \sum_{(i,j) \in \{\text{def } k\}} b_{i,j}^{(l)} \, w_j \, \delta_j
$$

where $b_{i,j}^{(l)}$ is the indicator that person $i$ at establishment $j$ belongs to cell $k$ in implicate $l$ (this indicator depends on the imputed education value), $w_j$ is the QWI weight, and $\delta_j$ is the noise-infusion fuzz factor. The published statistic is:

$$
B_k^* = \frac{1}{L} \sum_{l=1}^{L} B_k^{(l)*}
$$

#### Between-Implicate Variance

Captures uncertainty due to imputation (the dominant source for education):

$$
b_k^* = \frac{1}{L-1} \sum_{l=1}^{L} \left( B_k^{(l)*} - \bar{B}_k^* \right)^2
$$

#### Within-Implicate Variance

Captures sampling variability (small for LEHD since it is near-census):

$$
\bar{v}_k^* = \frac{1}{L} \sum_{l=1}^{L} v_k^{(l)*}
$$

where the within-implicate sampling variance for each implicate uses the finite-population-corrected multinomial formula:

$$
v_k^{(l)*} = N_{WB}^2 \left( \frac{p_k^{(l)*}(1 - p_k^{(l)*})}{N_{UB}} \right) \left( \frac{N_{WB} - N_{UB}}{N_{WB} - 1} \right)
$$

with $p_k^{(l)*} = B_k^{(l)*} / N_{WB}$ being the estimated proportion in cell $k$, $N_{WB}$ the fuzzed-weighted population total, and $N_{UB}$ the fuzzed-unweighted sample total.

#### Total Variance (Rubin's Formula)

$$
TV_k^* = \bar{v}_k^* + \frac{L+1}{L} \, b_k^*
$$

The factor $(L+1)/L = 11/10$ corrects for the finite number of implicates.

#### Missingness Ratio

Quantifies the fraction of total variability attributable to imputation:

$$
\text{MR}_k^* = \frac{\frac{L+1}{L} \, b_k^*}{TV_k^*}
$$

For education-stratified QWI cells, this is typically **> 95%**, meaning imputation uncertainty dominates.

#### Degrees of Freedom

Approximate degrees of freedom for confidence intervals (Rubin and Schenker, 1986):

$$
df_k^* = (L-1) \left(1 + \frac{L}{L+1} \cdot \frac{\bar{v}_k^*}{b_k^*}\right)^2
$$

When the missingness ratio is high (as it is for education), $df_k^*$ approaches $L - 1 = 9$, meaning confidence intervals should use a $t$-distribution with approximately 9 degrees of freedom rather than a normal distribution.

#### Confidence Intervals

An approximate 90% confidence interval for the count in cell $k$:

$$
B_k^* \pm t_{0.05, \, df_k^*} \cdot \sqrt{TV_k^*}
$$

## 5. Full Information vs. Limited Information Imputation

This distinction is **critical** for understanding education data quality over the 2000-2018 period.

### Full Information Imputation (workers with pre-2009 earnings)

- Uses the complete set of conditioning variables described in Section 4.1 above
- Incorporates earnings, industry, coworker/co-resident characteristics
- Was developed and **run once by the research staff in 2010**
- **Is no longer operational** -- the system has not been updated since

### Limited Information Imputation (workers entering from 2009 onward)

- Conditions **only** on: age, sex, place of birth, race, ethnicity
- **Does NOT use:** earnings, industry, coworker characteristics, co-resident characteristics
- Draws from posterior predictive distributions estimated during the 2010 model run
- Represents **substantially degraded imputation quality**

#### Formal Comparison

Define the full conditioning set $\mathbf{X}_i^{full}$ and limited conditioning set $\mathbf{X}_i^{lim}$ for worker $i$:

$$
\mathbf{X}_i^{full} = (\text{sex}, \text{age}, \text{race}, \text{ethnicity}, \text{POB}, \text{earnings decile}, \text{industry}, \text{coworker demo}, \text{co-resident demo})
$$

$$
\mathbf{X}_i^{lim} = (\text{sex}, \text{age}, \text{race}, \text{ethnicity}, \text{POB})
$$

The full-information model estimates cell-specific probabilities:

$$
\Pr(\text{educ} = c \mid \mathbf{X}_i^{full}) = \pi_{g(i),c}^{full}
$$

where $g(i)$ maps worker $i$ to a cell defined by the full interaction of all conditioning variables. The limited-information model instead estimates:

$$
\Pr(\text{educ} = c \mid \mathbf{X}_i^{lim}) = \pi_{g'(i),c}^{lim}
$$

where $g'(i)$ maps worker $i$ to a coarser cell defined only by the five demographic variables. Since $g'$ partitions workers into far fewer cells than $g$, the limited-information probabilities are averages over the omitted conditioning variables:

$$
\pi_{g',c}^{lim} = \mathbb{E}\!\left[\pi_{g,c}^{full} \mid g \in g'\right] = \sum_{g \in g'} \frac{n_g^{obs}}{n_{g'}^{obs}} \, \pi_{g,c}^{full}
$$

This pooling means that **within a limited-information cell, a high-earning worker in finance and a low-earning worker in food service receive the same education distribution** -- the model cannot distinguish them. Concretely, earnings deciles alone explain a large share of education variation (BA+ rates range from ~10% in the lowest decile to ~60% in the highest), so dropping earnings removes the single most informative predictor.

#### Implicates Under Limited Information

The procedure is identical in form: for worker $i$ in limited-information cell $g'$,

1. Draw $\boldsymbol{\pi}_{g'}^{(l)}$ from the stored PPD $\text{Dirichlet}(n_{g',1}^{obs}+1, \ldots, n_{g',4}^{obs}+1)$
2. Draw $e_i^{(l)} \sim \text{Categorical}(\boldsymbol{\pi}_{g'}^{(l)})$

The PPD parameters $(n_{g',c}^{obs})$ were estimated once in 2010 from the Census 2000 long form + ACS data and are **frozen** -- they are not updated as new ACS waves arrive. This means the limited-information model also cannot capture secular trends in educational attainment (e.g., rising BA+ rates among younger cohorts after 2010).

### Timeline of Quality Degradation

```
2000 --|-- Census 2000 long form collected (1-in-6 sample)
       |   ~7% of LEHD workers have observed education
2001 --|-- ACS begins; observed education coverage rises toward ~15%
       |
2009 --|-- Last year for which workers receive full-information imputation
2010 --|-- Full-information imputation model run for the last time
       |   Workers entering after this point get limited-information imputation
       |
2018 --|-- By this point, a large share of the workforce (especially younger
       |   workers) has only limited-information education imputation
```

## 6. Imputation Quality Assessment

### Out-of-Sample Validation Against ACS

The imputation model was validated by comparing imputed education to ACS responses (which were not used to fit the model). Results from Table A.1 of McKinney et al. (2017):

| True Education (ACS) | Imputed into correct category | Target (non-imputed) accuracy |
|-----------------------|------------------------------|------------------------------|
| < High School | 26.0% | 80.4% |
| High School | 35.4% | 81.2% |
| Some College | 29.7% | 84.8% |
| Bachelor's+ | 46.9% | 94.3% |

**Key findings:**
- Imputed accuracy is far below the target (non-imputed) accuracy for all categories
- Bachelor's degree and above has the highest imputed accuracy (~47%), likely because high earnings are a strong predictor
- "Less than High School" and "Some College" categories are hardest to impute correctly (~26-30%)
- The imputation model **is more informative than random assignment** but accuracy is modest

### QWI-Level Validation

At the statewide level, comparing QWI indicators computed with reported vs. imputed education:
- Full-quarter wages within education categories: differences range from **-8.1% to +9.4%**
- Beginning-of-quarter employment shares by education: differences range from **-5.3 to +6.6 percentage points**, with most cells within 2 percentage points

### Missingness Ratios

From the total variability analysis (Tables 1-5 of the paper):
- **Gender x Education** tabulations have a median Rubin missingness ratio of **96.8%** -- meaning imputation accounts for virtually all of the total variability
- This is by far the highest missingness ratio among all QWI stratification schemes
- Despite this, Gender x Education tabulations have acceptable total variation for cells with 10+ workers (coefficient of variation median ~0.3%)

## 7. Implications for Researchers Using LEHD Education Data (2000-2018)

### Quality Gradient by Time and Age

> "A good rule of thumb is: the older the earnings year and/or the older average age of the workers in an estimation cell, the higher quality the education data."

- **Pre-2009 data with older workers:** Highest quality -- full-information imputation or observed data from Census 2000
- **Post-2009 data with younger workers:** Lowest quality -- limited-information imputation without earnings or workplace characteristics
- Quality **degrades monotonically** from 2009 onward as full-information cohorts exit and limited-information cohorts enter

### Standard Errors

- Researchers **must** use all 10 implicates with Rubin's combining rules to properly account for imputation uncertainty
- Single-implicate standard errors will dramatically understate uncertainty for education-stratified analyses
- The between-implicate variance component (capturing imputation uncertainty) dominates total variability for education: missingness ratio > 95%

### Practical Considerations

1. **~85% of education values are imputed** -- treat education-stratified results as model-dependent
2. **The 4-category variable is coarse** -- even perfect imputation cannot distinguish associate's from some college, or master's from bachelor's
3. **Education is modeled as time-invariant** for each worker -- the system does not update education as workers age (this matters less for workers 25+, but creates issues for workers who attain degrees after their first observed job)
4. **Quality is particularly poor for "recent estimation cells composed primarily of younger workers"** (post-2009 cohorts with limited-information imputation)

## 8. Key References

- Abowd, J.M., B.E. Stephens, L. Vilhuber, F. Andersson, K.L. McKinney, M. Roemer, and S. Woodcock (2009). "The LEHD Infrastructure Files and the Creation of the Quarterly Workforce Indicators." In *Producer Dynamics*, pp. 149-230. University of Chicago Press.
- McKinney, K.L., A.S. Green, L. Vilhuber, and J.M. Abowd (2017). "Total Error and Variability Measures with Integrated Disclosure Limitation for QWI and LEHD ODES in OnTheMap." CES Working Paper 17-71.
- McKinney, K.L., A.S. Green, L. Vilhuber, and J.M. Abowd (2021). Published version in *Journal of Survey Statistics and Methodology*.
- Graham, M. (2022). "LEHD Snapshot Documentation, Release S2021_R2022Q4." CES Working Paper 22-51.
- Rubin, D.B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.
