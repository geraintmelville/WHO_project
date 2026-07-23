# WHO_project
Build an app that predicts life expectancy, allowing users to choose between an ethical and a robust linear regression model by inputting their data accordingly.

# Overview
We were tasked with using WHO life expectancy data 

The dataset consists of 2865 records

# WHO Project

Predicting country-level life expectancy with two linear regression models, an
**etchical model** and a **robust model**, deployed in an
interactive Streamlit app that allows users to consent to the use of sensitive data and input their data accordingly.

---

## 1. Project Overview

Our team was asked to estimate life expectancy across countries using WHO
data (2000–2015, 183 countries). The brief raises an ethical question:
some countries are wary of sharing sensitive population data (e.g. health/medical
records) because it can be correlated back to quality-of-life measures with
financial or social consequences.

To respect that, we built **two models** rather than one:

| Model | Uses | When it's used |
|---|---|---|
| **Ethical model** | Only non-sensitive, freely disclosed features | Default — used unless the user actively consents to more |
| **Robust model** | Minimal features + additional/sensitive population statistics | Only after explicit user consent |

The app prompts:

> *"Do you consent to using advanced population data, which may include
> protected information, for better accuracy? (Y/N)"*

...and routes the prediction to the appropriate model based on the answer.

**Baseline to beat:** a comparison team produced a semi-robust model with an
**RMSE of 1.8**. Our target is to beat this with our robust model.

---

## 2. Data

- **Source:** WHO life expectancy dataset, 2000–2015, 183 countries
  (`Life_Expectancy_Data.csv`)
- **Granularity:** one row per country per year
- **Rows:** ~2,865 (after accounting for missing years)
- **Target variable:** `Life_expectancy`

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

*(Fill in the final minimal vs. advanced feature split once decided — see
Section 4.)*

### Known data issues

- Missing values exist for some country-years; per the brief, these rows are
  dropped rather than imputed.
- `Infant_deaths` and `Under_five_deaths` are highly correlated — check VIF
  before including both in the same model.
- `Economy_status_Developed`/`Economy_status_Developing` are complementary
  dummies — only include one to avoid perfect multicollinearity.

---

## 3. Ethical Considerations

- **Data minimisation:** the minimal model should only use features a
  country would be comfortable disclosing without correlation risk
  (e.g. schooling, GDP, population) — explicitly excluding granular health
  statistics.
- **Informed consent:** the advanced model is never used silently; the
  app requires an explicit Y/N response before switching.
- **Feature justification:** for every feature in the advanced model, we
  document *why* it improves prediction and *what it reveals* if it did
  correlate back to a country's population.
- **Fairness check:** we test both models' residuals split by
  `Economy_status_Developed` vs. `Economy_status_Developing` to check
  neither model systematically under-predicts for developing nations.
- **Transparency:** linear regression is used deliberately over black-box
  alternatives so coefficients remain interpretable and auditable by WHO
  or country stakeholders.
- **Intended use:** country-level trend estimation only — not for
  individual-level decisions, insurance, or resource-allocation cut-offs.

---

## 4. Methodology

1. **EDA** — distribution checks, missing data audit, correlation/VIF review.
2. **Feature split** — define which columns are "minimal" vs. "advanced" (see
   table above), justified against the ethical criteria in Section 3.
3. **Preprocessing** — dummy encoding, scaling where needed, train/test split.
4. **Modelling** — fit an OLS linear regression per model using
   `statsmodels`/`scikit-learn`.
5. **Evaluation** — RMSE, R², residual analysis; benchmark against the
   competitor's RMSE of 1.8.
6. **Cross-validation** — k-fold CV to check robustness, not just a single
   train/test split.
7. **Deployment** — wrap both models in a single prediction function with a
   consent-based switch, then build the Streamlit front end around it.

---

## 5. Repository Structure

```
├── data/
│   └── Life_Expectancy_Data.csv
├── notebooks/
│   └── eda_and_modelling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_minimal_model.py
│   ├── train_advanced_model.py
│   └── predict.py          # single function, both models + consent logic
├── models/
│   ├── minimal_model.pkl
│   └── advanced_model.pkl
├── app/
│   └── app.py               # Streamlit app
├── requirements.txt
└── README.md
```

---

## 6. Installation & Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Requires Python 3.10+, `pandas`, `statsmodels`, `scikit-learn`, `streamlit`.

---

## 7. Running the App

```bash
streamlit run app/app.py
```

The app will:
1. Ask for consent to use advanced data (Y/N).
2. Collect the relevant inputs for whichever model applies.
3. Return a predicted life expectancy, along with the model used and its
   headline accuracy metric.

---

## 8. Using the Prediction Function Directly

For non-app use (per the "single file containing your function" deliverable):

```python
from src.predict import predict_life_expectancy

result = predict_life_expectancy(
    features={...},
    consent=True  # or False
)
```

---

## 9. Model Comparison / Results

| Model | Features used | RMSE | R² | Notes |
|---|---|---|---|---|
| Minimal | *(to fill in)* | *(to fill in)* | *(to fill in)* | Privacy-preserving default |
| Advanced | *(to fill in)* | *(to fill in)* | *(to fill in)* | Requires consent |
| Competitor baseline | — | 1.8 | — | Target to beat |

---

## 10. Limitations & Future Work

- Linear regression assumes linear relationships — a non-linear model may
  capture more variance in the advanced case.
- No uncertainty interval currently reported alongside point predictions.
- Dataset ends in 2015 — predictions don't reflect more recent global health
  shifts.

---

## 11. Deliverables Checklist

- [ ] Streamlit app file(s)
- [ ] Single file containing the prediction function
- [ ] (Optional) Link to live Streamlit deployment
- [ ] This README

---

## Authors

*(add team names here)*