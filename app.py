"""
WHO Life Expectancy Prediction — Streamlit App
================================================

Run locally with:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------

DATA_PATH = "Life_Expectancy_Data.csv"  # TODO: adjust path if needed
TARGET = "Life_expectancy"
BASELINE_RMSE = 1.8  # competitor benchmark from the brief

# TODO: this is YOUR ethical judgement call from the brief — adjust freely.
SENSITIVE_COLS = [
    "Adult_mortality",
    "Infant_deaths",
    "Under_five_deaths",
    "Incidents_HIV",
    "Hepatitis_B",
    "Measles",
    "Polio",
    "Diphtheria",
    "BMI",
    "Thinness_ten_nineteen_years",
    "Thinness_five_nine_years",
    "Alcohol_consumption",  # UPDATED: colleague's revised notebook now excludes this from the minimal/"Least information" model
]

ALWAYS_DROP = [
    "Country",  # used only to split on — see split_by_country()
    "Economy_status_Developing",  # perfect complement of Economy_status_Developed — keeping both
                                    # causes perfect collinearity (the "dummy variable trap");
                                    # dropping one doesn't change predictions, just makes the
                                    # remaining coefficient interpretable on its own.
]

WHO_BLUE = "#019CDE"  # sampled directly from the WHO logo file
WHO_NAVY = "#001450"  # the brief deck's darker navy accent (closing panel fill)

REGIONS = [
    "Africa", "Asia", "Central America and Caribbean", "European Union",
    "Middle East", "North America", "Oceania", "Rest of Europe", "South America",
]

# Readable labels for chart display only — underlying feature/column names are
# unchanged. Based on the dataset's actual field descriptions: Polio/Diphtheria/
# Hepatitis_B are immunization COVERAGE (%), not disease incidence, so they're
# labeled accordingly to avoid misreading the chart.
FEATURE_LABELS = {
    "Polio": "Polio Immunisation",
    "Diphtheria": "Diphtheria Immunisation",
    "Hepatitis_B": "Hepatitis B Immunisation",
    "Measles": "Measles Cases",
    "Incidents_HIV": "HIV Incidents",
    "BMI": "Average BMI",
    "Adult_mortality": "Adult Mortality",
    "Infant_deaths": "Infant Deaths",
    "Under_five_deaths": "Under-Five Deaths",
    "GDP_per_capita": "GDP per Capita",
    "Population_mln": "Population",
    "Alcohol_consumption": "Alcohol Consumption",
    "Thinness_ten_nineteen_years": "Thinness (10-19 yrs)",
    "Thinness_five_nine_years": "Thinness (5-9 yrs)",
    "Schooling": "Schooling",
    "Year": "Year",
}


# ---------------------------------------------------------------------------
# 2. DATA LOADING + COUNTRY-STRATIFIED SPLIT + MODEL TRAINING
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET])
    return df


def split_by_country(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified split at the COUNTRY level (not row level), so every row for
    a given country lands entirely in train or entirely in test. Country is
    only used here to decide split membership — never as a model feature."""
    country_region = df[["Country", "Region"]].drop_duplicates()
    train_countries, test_countries = train_test_split(
        country_region, test_size=test_size, stratify=country_region["Region"],
        random_state=random_state,
    )
    train_df = df[df["Country"].isin(train_countries["Country"])].copy()
    test_df = df[df["Country"].isin(test_countries["Country"])].copy()
    return train_df, test_df


def encode_region(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """One-hot encode Region with pd.get_dummies (drop_first=True), fit
    separately per split, then align columns in case of any mismatch."""
    train_enc = pd.get_dummies(train_df, columns=["Region"], drop_first=True)
    test_enc = pd.get_dummies(test_df, columns=["Region"], drop_first=True)

    region_cols = [c for c in train_enc.columns if c.startswith("Region_")]
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)

    train_enc[region_cols] = train_enc[region_cols].astype(int)
    test_enc[region_cols] = test_enc[region_cols].astype(int)
    return train_enc, test_enc


def fit_model(df: pd.DataFrame, drop_cols: list[str]):
    train_df, test_df = split_by_country(df)
    train_df, test_df = encode_region(train_df, test_df)

    cols_to_drop = [TARGET] + ALWAYS_DROP + drop_cols
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=cols_to_drop)
    y_test = test_df[TARGET]

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    train_rmse = root_mean_squared_error(y_train, model.predict(X_train_scaled))
    test_rmse = root_mean_squared_error(y_test, model.predict(X_test_scaled))

    return {
        "model": model,
        "scaler": scaler,
        "features": X_train.columns.tolist(),
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
    }


@st.cache_resource
def train_all_models():
    df = load_data()
    minimal = fit_model(df, drop_cols=SENSITIVE_COLS)
    advanced = fit_model(df, drop_cols=[])
    return minimal, advanced


# ---------------------------------------------------------------------------
# 3. THE CONSENT-BASED PREDICTION FUNCTION (the brief's core deliverable)
# ---------------------------------------------------------------------------

def predict_life_expectancy(input_features: dict, consent: bool, minimal_bundle, advanced_bundle):
    """Takes a dict of population statistics and a consent flag, and returns
    a life expectancy prediction using the appropriate model."""
    bundle = advanced_bundle if consent else minimal_bundle
    model, scaler, feature_list = bundle["model"], bundle["scaler"], bundle["features"]

    row = pd.DataFrame([{feat: input_features.get(feat, 0) for feat in feature_list}])
    row_scaled = pd.DataFrame(scaler.transform(row), columns=feature_list)
    prediction = model.predict(row_scaled)[0]
    return prediction


# ---------------------------------------------------------------------------
# 4. STREAMLIT UI — two-column layout, numeric inputs on the left
# ---------------------------------------------------------------------------

LOGO_PATH = Path(__file__).parent / "assets" / "who_logo.jpg"


def styled_header(text: str, level: str = "h2", align: str = "left", color: str = WHO_NAVY):
    """Render a styled header. st.title/st.subheader don't support alignment
    or custom color directly, so this uses a small HTML snippet via
    st.markdown instead."""
    st.markdown(
        f"<{level} style='text-align:{align}; color:{color};'>{text}</{level}>",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="WHO Life Expectancy Predictor", layout="wide")

    if LOGO_PATH.exists():
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.image(str(LOGO_PATH), use_container_width=True)

    styled_header("Life Expectancy Prediction", level="h2")
    st.caption(
        "Predicts average life expectancy from population statistics. "
        f"Competitor benchmark to beat: RMSE {BASELINE_RMSE}."
    )

    minimal_bundle, advanced_bundle = train_all_models()

    # ----------------------------- SIDEBAR: inputs (collapsible, native Streamlit sidebar) -----------------------------
    with st.sidebar:
        st.header("Input Features")

        st.markdown("**Data sharing consent**")
        consent_choice = st.radio(
            "Do you consent to using advanced population data, which may "
            "include protected information, for better accuracy?",
            options=["N", "Y"], horizontal=True,
        )
        consent = consent_choice == "Y"

        st.markdown("**Population statistics**")
        inputs = {}
        inputs["Year"] = st.number_input("Year", min_value=2000, max_value=2030, value=2015)
        inputs["GDP_per_capita"] = st.number_input("GDP per capita (USD)", min_value=0, value=5000)
        inputs["Population_mln"] = st.number_input("Population (millions)", min_value=0.0, value=10.0)
        inputs["Schooling"] = st.number_input("Average years of schooling", min_value=0.0, value=10.0)

        economy_status = st.selectbox("Economy status", ["Developing", "Developed"])
        inputs["Economy_status_Developed"] = 1 if economy_status == "Developed" else 0

        region = st.selectbox("Region", REGIONS)
        for r in REGIONS:
            col = f"Region_{r}"
            if col in advanced_bundle["features"]:  # baseline region has no dummy column
                inputs[col] = 1 if r == region else 0

        if consent:
            st.markdown("**Advanced / sensitive information** *(shared with consent)*")
            inputs["Adult_mortality"] = st.number_input("Adult mortality (per 1000)", min_value=0.0, value=150.0)
            inputs["Infant_deaths"] = st.number_input("Infant deaths (per 1000)", min_value=0.0, value=20.0)
            inputs["Under_five_deaths"] = st.number_input("Under-five deaths (per 1000)", min_value=0.0, value=25.0)
            inputs["Incidents_HIV"] = st.number_input("HIV incidents (per 1000)", min_value=0.0, value=0.1)
            inputs["Hepatitis_B"] = st.number_input("Hepatitis B immunization (%)", min_value=0, max_value=100, value=80)
            inputs["Measles"] = st.number_input("Measles cases (per 1000)", min_value=0, value=50)
            inputs["Polio"] = st.number_input("Polio immunization (%)", min_value=0, max_value=100, value=85)
            inputs["Diphtheria"] = st.number_input("Diphtheria immunization (%)", min_value=0, max_value=100, value=85)
            inputs["BMI"] = st.number_input("Average BMI", min_value=0.0, value=25.0)
            inputs["Thinness_ten_nineteen_years"] = st.number_input("Thinness, age 10-19 (%)", min_value=0.0, value=5.0)
            inputs["Thinness_five_nine_years"] = st.number_input("Thinness, age 5-9 (%)", min_value=0.0, value=5.0)
            inputs["Alcohol_consumption"] = st.number_input("Alcohol consumption (litres/capita)", min_value=0.0, value=5.0)
        else:
            st.info("Advanced/sensitive fields are hidden — consent not given. The minimal model will be used.")

        predict_clicked = st.button("Predict Life Expectancy", type="primary", use_container_width=True)

    # ----------------------------- MAIN AREA: results -----------------------------
    if predict_clicked:
        prediction = predict_life_expectancy(inputs, consent, minimal_bundle, advanced_bundle)
        model_used = "Advanced" if consent else "Minimal"
        st.success(f"Predictied Life Expectancy: **{prediction:.2f} years**")
        st.caption(f"Model used: {model_used} tier.")

        styled_header("Population Statistics Influence (% of Model Bias)", level="h2")
        bundle = advanced_bundle if consent else minimal_bundle

        # Coefficients from the fitted model, on RobustScaler-scaled units —
        # comparable across features regardless of original scale (e.g. Year
        # vs. a percentage vs. GDP in dollars), unlike raw input values.
        coefs = pd.Series(bundle["model"].coef_, index=bundle["features"])
        # keep the chart readable: show the non-region, non-binary features,
        # sorted by magnitude so the most influential features stand out
        plot_features = [f for f in coefs.index if not f.startswith("Region_")
                          and f != "Economy_status_Developed"]
        plot_coefs = coefs[plot_features].reindex(
            coefs[plot_features].abs().sort_values(ascending=False).index
        )
        # Normalize to a signed percentage of total influence: each bar shows
        # this feature's share of the combined |coefficient| total, keeping
        # its sign so direction (helps vs. hurts the prediction) is preserved.
        # Absolute values of the bars sum to 100%.
        total_abs = plot_coefs.abs().sum()
        plot_pct = plot_coefs / total_abs * 100

        fig = go.Figure(go.Bar(
            x=[FEATURE_LABELS.get(f, f) for f in plot_pct.index], y=plot_pct.values,
            marker_color=sign_colors(plot_pct.values),
        ))
        fig.update_layout(
            xaxis_title="Feature", yaxis_title="% Bias (share of total model influence)",
            yaxis_ticksuffix="%",
            height=560, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each bar shows this feature's share of the model's total influence (bars sum to "
            "100% in absolute terms). Blue bars pull predicted life expectancy up; orange bars "
            "pull it down. Based on the fitted model's coefficients on scaled features, so "
            "shares are comparable regardless of each feature's original units."
        )
    else:
        st.info("Fill in the inputs in the sidebar and click **Predict Life Expectancy** to see a result.")

    with st.expander("Model performance (train vs. test RMSE)", expanded=False):
        st.caption(
            "Country-stratified split — every row for a given country stays "
            "entirely in train or entirely in test."
        )
        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                "Minimal model — test RMSE",
                f"{minimal_bundle['test_rmse']:.3f}",
                delta=f"{minimal_bundle['test_rmse'] - minimal_bundle['train_rmse']:+.3f} vs train",
                delta_color="inverse",
            )
        with m2:
            st.metric(
                "Advanced model — test RMSE",
                f"{advanced_bundle['test_rmse']:.3f}",
                delta=f"{advanced_bundle['test_rmse'] - advanced_bundle['train_rmse']:+.3f} vs train",
                delta_color="inverse",
            )


def sign_colors(values, pos_color=WHO_BLUE, neg_color="#FF7043"):
    """Color bars by sign rather than cycling — positive % (pulls life
    expectancy up) in WHO blue, negative % (pulls it down) in a bright
    complementary orange, so direction is immediately visible at a glance."""
    return [pos_color if v >= 0 else neg_color for v in values]


def px_colors(n):
    """Bright, cycling palette anchored on the WHO logo's actual blue
    (#019CDE), paired with complementary greens/teals for contrast.
    Kept available for charts that aren't sign-based."""
    palette = [
        WHO_BLUE,   # WHO blue (sampled from the logo)
        "#00C853",  # bright green
        "#00BFA5",  # teal
        "#29B6F6",  # sky blue
        "#2ECC71",  # emerald
        "#1DE9B6",  # turquoise
        "#039BE5",  # deep sky blue
        "#66BB6A",  # medium green
        "#00E5FF",  # cyan
        "#43A047",  # forest green
    ]
    return [palette[i % len(palette)] for i in range(n)]


if __name__ == "__main__":
    main()