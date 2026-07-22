"""
WHO Life Expectancy Prediction — Streamlit UI
================================================
!!! IMPORTANT — READ BEFORE RUNNING !!!
This file is named "streamlit.py" as requested, but that name COLLIDES with
the actual `streamlit` package this file imports (`import streamlit as st`).
Python will find THIS file before the real library and the import will
break. Rename this file (e.g. `streamlit_ui.py` or `app_ui.py`) before
running it — the code itself doesn't need to change, just the filename.

--------------------------------------------------------------------------
This file contains ONLY the Streamlit UI layer:
  - Page layout, sidebar inputs, results display, chart, styling.

It expects a `pipeline` module (pipeline.py, included alongside this file)
that exposes:
  - train_all_models() -> (minimal_bundle, advanced_bundle)
  - predict_life_expectancy(input_features, consent, minimal_bundle, advanced_bundle) -> float
  - REGIONS, BASELINE_RMSE

If your colleague's code replaces pipeline.py, as long as it exposes the
same functions/constants, this UI file shouldn't need any changes.

Run locally with:  streamlit run streamlit_ui.py   (after renaming, see above)
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import REGIONS, BASELINE_RMSE, train_all_models, predict_life_expectancy

# ---------------------------------------------------------------------------
# STYLING CONFIG
# ---------------------------------------------------------------------------

WHO_BLUE = "#019CDE"  # sampled directly from the WHO logo file
WHO_NAVY = "#001450"  # the brief deck's darker navy accent (closing panel fill)

LOGO_PATH = Path(__file__).parent / "assets" / "who_logo.jpg"


def styled_header(text: str, level: str = "h2", align: str = "left", color: str = WHO_NAVY):
    """Render a styled header. st.title/st.subheader don't support alignment
    or custom color directly, so this uses a small HTML snippet via
    st.markdown instead."""
    st.markdown(
        f"<{level} style='text-align:{align}; color:{color};'>{text}</{level}>",
        unsafe_allow_html=True,
    )


def px_colors(n):
    """Gradient from the brief's teal (#008080) to navy (#001450) accents,
    so the chart matches the deck's actual color scheme rather than a
    generic default palette."""
    teal = (0x00, 0x80, 0x80)
    navy = (0x00, 0x14, 0x50)
    if n <= 1:
        return ["#008080"]
    colors = []
    for i in range(n):
        t = i / (n - 1)
        r = round(teal[0] + (navy[0] - teal[0]) * t)
        g = round(teal[1] + (navy[1] - teal[1]) * t)
        b = round(teal[2] + (navy[2] - teal[2]) * t)
        colors.append(f"#{r:02X}{g:02X}{b:02X}")
    return colors


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

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
        inputs["Alcohol_consumption"] = st.number_input("Alcohol consumption (litres/capita)", min_value=0.0, value=5.0)
        inputs["Schooling"] = st.number_input("Average years of schooling", min_value=0.0, value=10.0)

        economy_status = st.selectbox("Economy status", ["Developing", "Developed"])
        inputs["Economy_status_Developed"] = 1 if economy_status == "Developed" else 0
        inputs["Economy_status_Developing"] = 1 if economy_status == "Developing" else 0

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
        else:
            st.info("Advanced/sensitive fields are hidden — consent not given. The minimal model will be used.")

        predict_clicked = st.button("Predict Life Expectancy", type="primary", use_container_width=True)

    # ----------------------------- MAIN AREA: results -----------------------------
    if predict_clicked:
        prediction = predict_life_expectancy(inputs, consent, minimal_bundle, advanced_bundle)
        model_used = "Advanced" if consent else "Minimal"
        st.success(f"Predicted Life Expectancy: **{prediction:.2f} years**")
        st.caption(f"Model used: {model_used} tier.")

        styled_header("Prediction Visualization", level="h2")
        bundle = advanced_bundle if consent else minimal_bundle
        feature_values = pd.Series({f: inputs.get(f, 0) for f in bundle["features"]})
        # keep the chart readable: show the non-region, non-binary inputs
        plot_features = [f for f in feature_values.index if not f.startswith("Region_")
                          and f not in ("Economy_status_Developed", "Economy_status_Developing")]
        fig = go.Figure(go.Bar(
            x=plot_features, y=feature_values[plot_features],
            marker_color=px_colors(len(plot_features)),
        ))
        fig.update_layout(
            xaxis_title="Feature", yaxis_title="Value",
            height=560, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Bars show the raw input values used for this prediction — features like "
            "GDP per capita naturally dominate the scale compared to percentages/rates."
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


if __name__ == "__main__":
    main()
