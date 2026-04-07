# Macro Moments for $(\alpha, \sigma_e)$: Three Equations and the $\gamma_0$ Normalization

## Model primitives

Screening FOC: $w_j = \bar{q}_j \cdot A_j(1-\alpha)(Q_jL_j)^{-\alpha}$

Production: $R_j = A_j(Q_jL_j)^{1-\alpha}$

Under Spec M with standardized micro estimates $(\tilde{q}_j, \tilde{Q}_j^M)$:
- $\ln \bar{q}_j = \gamma_0 + \sigma_e\tilde{q}_j$
- $\ln Q_j = \gamma_0 + \ln \tilde{Q}_j^M(\sigma_e)$

## Three equations

**Combined (screening FOC + production, eliminates $A_j$):**

$$\sigma_e\tilde{q}_j - \ln \tilde{Q}_j^M(\sigma_e) = \ln w_j + \ln L_j - \ln(1-\alpha) - \ln R_j$$

- No residual. Holds exactly at every firm at the true parameters.
- $\gamma_0$ cancels algebraically (appears in both $\ln \bar{q}_j$ and $\ln Q_j$, subtracts out).
- Identifies: $\sigma_e$ from cross-firm variation, $\alpha$ from the level.

**Production:**

$$\ln R_j - (1-\alpha)\ln(\tilde{Q}_j^M(\sigma_e) \cdot L_j) = \ln A_j + (1-\alpha)\gamma_0$$

- Residual: $\ln A_j + (1-\alpha)\gamma_0$.
- Requires instruments orthogonal to $\ln A_j$.

**Screening FOC:**

$$\sigma_e\tilde{q}_j - \alpha\ln(\tilde{Q}_j^M(\sigma_e) \cdot L_j) - \ln w_j + \ln(1-\alpha) + (1-\alpha)\gamma_0 = -\ln A_j$$

- Residual: $-\ln A_j - (1-\alpha)\gamma_0$.
- Requires instruments orthogonal to $\ln A_j$.

Note: Combined = Screening + Production. Only 2 of the 3 equations are independent.

## The $\gamma_0$ normalization

Under Spec M, $\gamma_0$ and $\{A_j\}$ are not separately identified: shifting $\gamma_0 \to \gamma_0 + c$ is equivalent to $A_j \to A_j \cdot \exp(-(1-\alpha)c)$. We normalize $\gamma_0 = 0$. This normalization is a **substitute** for the $E[\ln A_j]=0$ normalization.

**Effect on each equation:**

- **Combined:** No effect. $\gamma_0$ was never present — it cancels algebraically regardless of its value.
- **Production:** Residual becomes $\ln A_j + (1-\alpha)\gamma_0^{\text{true}}$. The level shift $(1-\alpha)\gamma_0^{\text{true}}$ is a nonzero constant if $\gamma_0^{\text{true}} \neq 0$.
- **Screening FOC:** Residual becomes $-\ln A_j - (1-\alpha)\gamma_0^{\text{true}}$. Same level shift with opposite sign.

## Demeaning rule

The level shift $(1-\alpha)\gamma_0^{\text{true}}$ is constant across firms (within a market). Any instrument that includes a constant will pick up this shift and violate the moment condition. Fix: demean all instruments used with the primitive equations.

| Equation | Instruments must be demeaned? | Reason |
|----------|-------------------------------|--------|
| Combined | No | $\gamma_0$ cancels; no residual at all |
| Production | Yes | Residual has constant shift $(1-\alpha)\gamma_0^{\text{true}}$ |
| Screening FOC | Yes | Residual has constant shift $-(1-\alpha)\gamma_0^{\text{true}}$ |

In multi-market estimation, demean within market to additionally absorb market-level TFP differences.

## Practical upshot

Denote $\ddot{x}_j \equiv x_j - \bar{x}$ (demeaned). Define residuals at candidate $(\alpha, \sigma_e)$:

$$r_j^C \equiv \sigma_e\tilde{q}_j - \ln \tilde{Q}_j^M(\sigma_e) - \ln w_j - \ln L_j + \ln(1-\alpha) + \ln R_j$$

$$r_j^P \equiv \ln R_j - (1-\alpha)\ln(\tilde{Q}_j^M(\sigma_e) \cdot L_j)$$

$$r_j^S \equiv \sigma_e\tilde{q}_j - \alpha\ln(\tilde{Q}_j^M(\sigma_e) \cdot L_j) - \ln w_j + \ln(1-\alpha)$$

Define observed average skill proxy at firm $j$:
$$\bar{v}_j = \frac{1}{L_j}\sum_{i:y_i=j} v_i$$

### Moment choices for $(\alpha, \sigma_e)$

1. **Combined only, internal instruments:**
$$E\!\left[(1,\; L_j,\; R_j,\; w_j,\; \bar{v}_j,\; \tilde{q}_j)' \cdot r_j^C\right] = 0 \qquad \text{(2SLS weight matrix)}$$

2. **Combined only, amenity shock:**
$$E\!\left[(1,\; z_j^2)' \cdot r_j^C\right] = 0$$

3. **Combined and screening, amenity shock:**
$$E\!\left[r_j^C\right] = 0, \qquad E\!\left[\ddot{z}_j^2 \cdot r_j^S\right] = 0$$

4. **Combined and screening, internal instruments and amenity shock:**
$$E\!\left[(1,\; L_j,\; R_j,\; w_j,\; \bar{v}_j,\; \tilde{q}_j)' \cdot r_j^C\right] = 0, \qquad E\!\left[\ddot{z}_j^2 \cdot r_j^S\right] = 0 \qquad \text{(2SLS weight matrix)}$$

5. **Combined and production, amenity shock:**
$$E\!\left[r_j^C\right] = 0, \qquad E\!\left[\ddot{z}_j^2 \cdot r_j^P\right] = 0$$

6. **Combined and production, internal instruments and amenity shock:**
$$E\!\left[(1,\; L_j,\; R_j,\; w_j,\; \bar{v}_j,\; \tilde{q}_j)' \cdot r_j^C\right] = 0, \qquad E\!\left[\ddot{z}_j^2 \cdot r_j^P\right] = 0 \qquad \text{(2SLS weight matrix)}$$

### Routine for testing

1. Consider 2 setups: true $\tilde{q}_j$, solver initialized at true $(\alpha, \sigma_e)$; MLE-estimated $\tilde{q}_j$, solver initialized at naive-estimated $(\alpha, \sigma_e)$.
2. For each setup, estimate $\alpha$ and $\sigma_e$ using all moment options above.
3. Report in a table: solver time, iterations, convergence; resulting estimates vs true values; the overidentification statistic and $p$-value where appropriate.

<!-- TO DO: DECIDE ON MOMENTS USED IN FINAL ESTIMATION BELOW

**For estimation (2 moments, 2 parameters, just-identified):**

$$E[r_j^C] = 0 \quad \text{(combined} \times \text{constant — pins } \alpha \text{ from labor share level)}$$

$$E[\bar{v}_j \cdot r_j^C] = 0 \quad \text{(combined} \times \text{mean skill proxy — pins } \sigma_e \text{ from cross-firm variation)}$$

where $\bar{v}_j = \frac{1}{L_j}\sum_{i:y_i=j} v_i$ is the observed average skill proxy at firm $j$.

**For overidentification / specification test (1 additional moment):**

$$E[\ddot{z}_j^2 \cdot r_j^P] = 0 \quad \text{(demeaned amenity shock} \times \text{production residual)}$$

The combined equation identifies $(\alpha, \sigma_e)$ without instruments. The production equation with a demeaned amenity instrument provides an independent check.
-->