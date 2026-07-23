from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DATA_PATH = "Life_Expectancy_Data.csv"  # TODO: adjust path if needed
TARGET = "Life_expectancy"

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
    "Alcohol_consumption",
    "Thinness_ten_nineteen_years",
    "Thinness_five_nine_years"
]

REGIONS = [
    "Africa", "Asia", "Central America and Caribbean", "European Union",
    "Middle East", "North America", "Oceania", "Rest of Europe", "South America"]

ALWAYS_DROP = ["Country"] 


class LifeExpectancyRow(BaseModel):
    """Schema for a single row of the raw life-expectancy dataset. Bounds are
    set from the observed data (e.g. Hepatitis_B/Polio/Diphtheria/Measles are
    0-100 immunisation coverage %), not arbitrary guesses."""

    Country: str = Field(min_length=1)
    Region: str
    Year: int = Field(ge=1990, le=2030)
    Infant_deaths: float = Field(ge=0, allow_inf_nan=False)
    Under_five_deaths: float = Field(ge=0, allow_inf_nan=False)
    Adult_mortality: float = Field(ge=0, allow_inf_nan=False)
    Alcohol_consumption: float = Field(ge=0, allow_inf_nan=False)
    Hepatitis_B: float = Field(ge=0, le=100, allow_inf_nan=False)
    Measles: float = Field(ge=0, le=100, allow_inf_nan=False)
    BMI: float = Field(ge=0, allow_inf_nan=False)
    Polio: float = Field(ge=0, le=100, allow_inf_nan=False)
    Diphtheria: float = Field(ge=0, le=100, allow_inf_nan=False)
    Incidents_HIV: float = Field(ge=0, allow_inf_nan=False)
    GDP_per_capita: float = Field(ge=0, allow_inf_nan=False)
    Population_mln: float = Field(ge=0, allow_inf_nan=False)
    Thinness_ten_nineteen_years: float = Field(ge=0, le=100, allow_inf_nan=False)
    Thinness_five_nine_years: float = Field(ge=0, le=100, allow_inf_nan=False)
    Schooling: float = Field(ge=0, allow_inf_nan=False)
    Economy_status_Developed: int = Field(ge=0, le=1)
    Economy_status_Developing: int = Field(ge=0, le=1)
    Life_expectancy: float = Field(ge=1, le=120, allow_inf_nan=False)

    @field_validator("Region")
    @classmethod
    def region_must_be_known(cls, v: str) -> str:
        if v not in REGIONS:
            raise ValueError(f"unrecognised region: {v!r}")
        return v

    @field_validator("Economy_status_Developing")
    @classmethod
    def economy_flags_are_complementary(cls, v: int, info) -> int:
        developed = info.data.get("Economy_status_Developed")
        if developed is not None and v == developed:
            raise ValueError(
                "Economy_status_Developed/Developing must be complementary "
                "(exactly one of the two should be 1)"
            )
        return v


_ROW_ADAPTER = TypeAdapter(list[LifeExpectancyRow])


def validate_data(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate every row of df against LifeExpectancyRow.

    strict=False (default): invalid rows are dropped and reported via the
    Streamlit UI, and the cleaned dataframe is returned.
    strict=True: raises ValueError listing every problem instead of dropping
    anything, useful when running outside Streamlit (e.g. in a notebook/CI).
    """
    records = df.to_dict(orient="records")

    try:
        _ROW_ADAPTER.validate_python(records)
        return df
    except ValidationError as exc:
        errors = exc.errors()
        bad_positions = sorted({err["loc"][0] for err in errors})

        if strict:
            details = "\n".join(
                f"row {err['loc'][0]} ({err['loc'][1] if len(err['loc']) > 1 else '?'}): {err['msg']}"
                for err in errors
            )
            raise ValueError(f"Data validation failed for {len(bad_positions)} row(s):\n{details}")

        st.warning(f"Dropped {len(bad_positions)} row(s) that failed data validation.")
        with st.expander("Show validation errors"):
            for err in errors:
                row_pos = err["loc"][0]
                field = err["loc"][1] if len(err["loc"]) > 1 else ""
                st.write(f"Row {row_pos} ({field}): {err['msg']}")

        bad_labels = df.index[bad_positions]
        return df.drop(index=bad_labels).reset_index(drop=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Loads data from csv, drops any nulls in the target variable and validates 
    against LifeExpectancyRow schema using validate_data function."""
    
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET])
    df = validate_data(df)
    return df

def train_test_split_spec(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified split at the COUNTRY level (not row level), so every row for
    a given country lands entirely in train or entirely in test. Country is
    only used here to decide split membership — never as a model feature. 
    Split ensures each region is represented fairly across the split (roughly same 
    proportion of each region in test and train sets."""
    
    country_region = df[["Country", "Region"]].drop_duplicates()
    train_countries, test_countries = train_test_split(
        country_region, test_size=test_size, stratify=country_region["Region"],
        random_state=random_state,
    )
    train_df = df[df["Country"].isin(train_countries["Country"])].copy()
    test_df = df[df["Country"].isin(test_countries["Country"])].copy()
    return train_df, test_df

def encode_regions(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """One-hot encode Region with pd.get_dummies (drop_first=True), fit
    separately per split, then align columns in case of any mismatch."""
    train_enc = pd.get_dummies(train_df, columns=["Region"], drop_first=True)
    test_enc = pd.get_dummies(test_df, columns=["Region"], drop_first=True)

    region_cols = [c for c in train_enc.columns if c.startswith("Region_")]
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)

    train_enc[region_cols] = train_enc[region_cols].astype(int)
    test_enc[region_cols] = test_enc[region_cols].astype(int)
    return train_enc, test_enc

def scale_data(X_train, X_test):
    """Scale data using RobustScaler() from sklearn library. Scaler is fitted on train set, 
    used to transform both train and test, and returned to be used to transform user inputs."""
    
    cols_to_scale = ['Year', 'Infant_deaths', 'Under_five_deaths',
            'Adult_mortality', 'Alcohol_consumption', 'Hepatitis_B', 'Measles',
            'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV', 'GDP_per_capita',
            'Population_mln', 'Thinness_ten_nineteen_years', 'Thinness_five_nine_years',
            'Schooling']
    ignore_cols = X_train.columns.difference(cols_to_scale)

    scaler = RobustScaler()
    scaler.fit(X_train[cols_to_scale])

    X_train_scaled = pd.DataFrame(scaler.transform(X_train[cols_to_scale]), columns=cols_to_scale, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[cols_to_scale]), columns=cols_to_scale, index=X_test.index)

    X_train_scaled[ignore_cols] = X_train[ignore_cols]
    X_test_scaled[ignore_cols] = X_test[ignore_cols]

    return X_train_scaled, X_test_scaled, scaler

def fit_model(df_train: pd.DataFrame, df_test: pd.DataFrame, drop_cols: list[str]):
    """General function which recieves a train and test set along with a set of columns to drop. 
    Fits a simple linear regression model using the training data set, makes predictions on the 
    test set and returns the model, list of features, the train and test RMSE as a 'bundle' (library)"""
    
    safe_drop_cols = [c for c in drop_cols if c in df_train.columns]
    cols_to_drop = [TARGET] + ALWAYS_DROP + safe_drop_cols
    X_train = df_train.drop(columns=cols_to_drop)
    y_train = df_train[TARGET]
    X_test = df_test.drop(columns=cols_to_drop)
    y_test = df_test[TARGET]

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_rmse = root_mean_squared_error(y_train, model.predict(X_train))
    test_rmse = root_mean_squared_error(y_test, model.predict(X_test))

    return {
        "model": model,
        "features": X_train.columns.tolist(),
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
    }

@st.cache_resource
def train_all_models():
    """Function that combines the functions defined above. When called, it will 
    run the pipeline end-to-end and returns our ethical and robust models and the scaler used."""
    
    # Load and validate data
    df = load_data()
    
    # Split data into train/test using our special method
    df_train, df_test = train_test_split_spec(df,test_size = 0.2,random_state = 42)
    
    # One-hot encode our only categorical column, Regions
    df_train_enc, df_test_enc = encode_regions(df_train, df_test)
    
    # Scale our remaining numerical columns
    df_train_scaled, df_test_scaled, scaler = scale_data(df_train_enc, df_test_enc)
    
    # Fit our ethical model
    bundle_eth = fit_model(df_train_scaled, df_test_scaled, drop_cols = SENSITIVE_COLS)
    
    # Fit our robust model
    bundle_rob = fit_model(df_train_scaled, df_test_scaled, drop_cols = [])
    
    return bundle_eth, bundle_rob, scaler

def predict_life_expectancy(inputs: dict, consent: bool, bundle_eth, bundle_rob, scaler):
    '''Takes user inputs, 'inputs' and 'consent', as well as our robust and ethical models as 
    'bundles', and our scaler. Then chooses which model to use based on consent, scales the
    user data and makes a prediction.'''
    
    bundle = bundle_rob if consent else bundle_eth
    model, features = bundle["model"], bundle["features"]

    scaled_cols = list(scaler.feature_names_in_)
    passthrough_cols = [f for f in features if f not in scaled_cols]

    row_scaled = pd.DataFrame(
        scaler.transform(pd.DataFrame([{c: inputs.get(c, 0) for c in scaled_cols}])),
        columns=scaled_cols,
    )
    row_passthrough = pd.DataFrame([{c: inputs.get(c, 0) for c in passthrough_cols}])

    row_full = pd.concat([row_scaled, row_passthrough], axis=1)
    prediction = model.predict(row_full[features])[0]
    return prediction