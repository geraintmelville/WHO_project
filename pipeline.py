"""
WHO Life Expectancy — Modeling Pipeline
=========================================
Data loading, country-stratified train/test split, one-hot encoding, scaling,
and model fitting for the minimal/advanced tiers.

This is what streamlit.py imports from. If your colleague is replacing this
with their own pipeline, the UI only needs these two functions to exist with
this exact interface:

    train_all_models() -> (minimal_bundle, advanced_bundle)
        Each bundle is a dict with keys: "model", "scaler", "features",
        "train_rmse", "test_rmse".

    predict_life_expectancy(input_features: dict, consent: bool,
                             minimal_bundle, advanced_bundle) -> float
        Returns a single predicted life expectancy value.

Everything else in this file (split_by_country, encode_region, fit_model) is
implementation detail behind those two functions — safe to replace entirely
as long as the interface above is preserved.
"""

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# CONFIG
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
]

ALWAYS_DROP = ["Country"]  # used only to split on — see split_by_country()

REGIONS = [
    "Africa", "Asia", "Central America and Caribbean", "European Union",
    "Middle East", "North America", "Oceania", "Rest of Europe", "South America",
]


# ---------------------------------------------------------------------------
# DATA LOADING + COUNTRY-STRATIFIED SPLIT + MODEL TRAINING
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
# THE CONSENT-BASED PREDICTION FUNCTION (the brief's core deliverable)
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
