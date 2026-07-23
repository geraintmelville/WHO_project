# WHO_project
Build an app that predicts life expectancy, allowing users to choose between an ethical and a robust linear regression model by inputting their data accordingly.

# Overview
We were tasked with using WHO life expectancy data 

The dataset consists of 2865 records

# WHO Project

Predicting country-level life expectancy with two linear regression models, an
**ethical model** and a **robust model**, deployed in an
interactive Streamlit app that allows users to consent to the use of sensitive data and input their data accordingly.

---

## 1. Project Overview

Our team was asked to estimate life expectancy across countries using WHO
data (2000–2015, 183 countries per year). The brief raises an ethical question:
some countries are wary of sharing sensitive population data (e.g. health/medical
records) because it can be correlated back to quality-of-life measures with
financial or social consequences.

To respect that, we built **two models** rather than one:

| Model | Uses | When it's used |
|---|---|---|
| **Ethical model** | Only non-sensitive, freely disclosed features (Minimal Features)| Default — used unless the user actively consents to more |
| **Robust model** | Minimal features + additional/sensitive population statistics | Only after explicit user consent |

The app prompts:

> *"Do you consent to using advanced population data, which may include
> protected information, for better accuracy? (Y/N)"*

...and routes the prediction to the appropriate model based on the answer.

**Baseline to beat:** a comparison team produced a semi-robust model with an
**RMSE of 1.8**. Our target is to beat this with our robust model.

---

### Data dictionary

| Column | Description | Sensitivity |
|---|---|---|
| `Country`, `Region`, `Year` | Identifiers | Non-sensitive |
| `Infant_deaths`, `Under_five_deaths`, `Adult_mortality` | Mortality rates | Sensitive |
| `Alcohol_consumption` | Litres per capita | Sensitive |
| `Hepatitis_B`, `Polio`, `Diphtheria`, `Measles` | Immunisation/incidence rates (%) | Sensitive |
| `BMI` | Average population BMI | Sensitive |
| `Incidents_HIV` | HIV incidence rate | Sensitive |
| `GDP_per_capita`, `Population_mln` | Economic/demographic | Non-sensitive |
| `Thinness_ten_nineteen_years`, `Thinness_five_nine_years` | Malnutrition indicators | Sensitive |
| `Schooling` | Average years of schooling | Non-sensitive |
| `Economy_status_Developed` / `_Developing` | Development status (dummy-encoded) | Non-sensitive |


### Known data issues
Multicolinearity - Some features are highly correlated and therefore offer proportionally less predictive power than other features. We calculated VIF (Variance Inflation Factor) for all numeric features, to calculate vif, a linear regression model is fitted for each feature as the target variable, using only training data. We take the $0 \le R^2 \le 1$ value, which represents how much of the variation in the target variable is explained by the data, then, VIF $= \frac{1}{1-R^2}$. We found:
- `Infant_deaths` - $59.34$, `Under_five_deaths` - $62.38$, `Adult_mortality` - $23.79$, as we might expect.
- `Economy_status_Developed`/`Economy_status_Developing` - $\infty$ , they are complementary variables so $R^2 = 1$.
- `Polio` - $11.82$, `Diphtheria` - $12.68$, since these are vaccinations levels they are highly correlated.
- `Thinness_ten_nineteen_years` - $8.48$, `Thinness_five_nine_years` - $8.75$, again as we might expect.  

We have removed `Economy_status_Developing` since it is redundant. If we wanted to produce the most efficient and most interpretable model possible, we would also remove a feature from each of the highly correlated groups, or look at creating a new feature which aggregates each feature in the groups. However since the brief only defined RMSE as our benchmark metric, we have included these features so maximize our signal and predictive power.

---

## 2. Ethical Considerations

- **Data minimisation:** the ethical model should only use features a
  country would be comfortable disclosing without correlation risk
  (e.g. schooling, GDP, population) — explicitly excluding granular health
  statistics.
- **Informed consent:** the robust model is never used without consent; the
  app requires an explicit Y/N response before switching.
- **Feature justification:** for every feature in the robust model, we
  document *why* it improves prediction and *what it reveals* if it did
  correlate back to a country's population.
- **Transparency:** linear regression is used deliberately over black-box
  alternatives so coefficients remain interpretable and auditable by WHO
  or country stakeholders.
- **Intended use:** country-level trend estimation only — not for
  individual-level decisions, insurance, or resource-allocation cut-offs.

---

## 3. Methodology

1. **EDA** — distribution checks, missing data audit, correlation/VIF review.
2. **Feature split** — define which columns are "minimal" vs. "advanced" (see
   table above), justified against the ethical criteria in Section 2.
3. **Preprocessing** — One Hot Encoding (dummy encoding), scaling where needed using Robust scaling, train/test split.
4. **Modelling** — fit an OLS linear regression per model using
   `statsmodels`/`scikit-learn`.
5. **Evaluation** — RMSE, residual analysis; benchmark against the
   competitor's RMSE of 1.8.
6. **Deployment** — wrap both models in a single prediction function with a
   consent-based switch, then build the Streamlit front end around it.

---

## 4. Running the App

The app will:
1. Ask for consent to use advanced data (Y/N).
2. Collect the relevant inputs for whichever model applies.
3. Return a predicted life expectancy, along with the model used and its
   headline accuracy metric.

---

## 5. Model Comparison / Results

| Model | RMSE | Notes |
|---|---|---|
| Ethical | 4.354 | Privacy-preserving default |
| Advanced | 1.454 | Requires consent |
| Competitor baseline | 1.8 | Target to beat |

---

## Authors

Geraint, Zamzam, Sai
