"""
WHO Life Expectancy Prediction — Streamlit App

Run locally with:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from functions import load_data, train_test_split_spec, encode_regions, fit_model, train_all_models, scale_data, predict_life_expectancy

WHO_BLUE = "#019CDE"  # sampled directly from the WHO logo file
WHO_NAVY = "#001450"  # the brief deck's darker navy accent (closing panel fill)

REGIONS = [
    "Africa", "Asia", "Central America and Caribbean", "European Union",
    "Middle East", "North America", "Oceania", "Rest of Europe", "South America"]

# ---------------------------------------------------------------------------
# STREAMLIT UI — two-column layout, numeric inputs on the left
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
        f"Competitor benchmark to beat: RMSE 1.8."
    )

    bundle_eth, bundle_rob, scaler = train_all_models()

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
            if col in bundle_rob["features"]:  # baseline region has no dummy column
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
        prediction = predict_life_expectancy(inputs, consent, bundle_eth, bundle_rob,scaler)
        model_used = "Robust" if consent else "Ethical"
        st.success(f"Predicted Life Expectancy: **{prediction:.2f} years**")
        st.caption(f"Model used: {model_used} tier.")

        styled_header("Population Statistics Influence (% of Model Bias)", level="h2")
        bundle = bundle_rob if consent else bundle_eth

        # Coefficients from the fitted model, on RobustScaler-scaled units —
        # comparable across features regardless of original scale (e.g. Year
        # vs. a percentage vs. GDP in dollars), unlike raw input values.
        coefs = pd.Series(bundle["model"].coef_, index=bundle["features"])
        
        # keep the chart readable: show the non-region, non-binary features,
        # sorted by magnitude so the most influential features stand out
        plot_features = [f for f in coefs.index if not f.startswith("Region_")
                          and f not in ("Economy_status_Developed", "Economy_status_Developing")]
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
            x=plot_pct.index, y=plot_pct.values,
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
                f"{bundle_eth['test_rmse']:.3f}",
                delta=f"{bundle_eth['test_rmse'] - bundle_eth['train_rmse']:+.3f} vs train",
                delta_color="inverse",
            )
        with m2:
            st.metric(
                "Advanced model — test RMSE",
                f"{bundle_rob['test_rmse']:.3f}",
                delta=f"{bundle_rob['test_rmse'] - bundle_rob['train_rmse']:+.3f} vs train",
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