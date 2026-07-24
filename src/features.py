"""
Feature engineering shared by the risk-classification, claim-frequency and
claim-amount models, plus the Streamlit app.
"""
import numpy as np
import pandas as pd

from src import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features used across all three models:
      - Customer_Age: from BirthDate as of the reference snapshot date
      - Vehicle_Age: Car_Year -> years since manufacture
      - Age_Group: bucketed customer age (captures U-shaped risk threshold effects)
      - Income_Quartile: household income rank, 1 (lowest) - 4 (highest)
      - Has_Claim: binary flag, Claim_Freq > 0 (secondary model target)
    """
    df = df.copy()
    ref_date = pd.Timestamp(config.REFERENCE_DATE)

    df["Customer_Age"] = (
        (ref_date - df["BirthDate"]).dt.days / 365.25
    ).round(1)
    df["Vehicle_Age"] = ref_date.year - df["Car_Year"]

    df["Age_Group"] = pd.cut(
        df["Customer_Age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )

    df["Income_Quartile"] = pd.qcut(
        df["Household_Income"], 4, labels=[1, 2, 3, 4]
    ).astype(int)

    df["Has_Claim"] = (df["Claim_Freq"] > 0).astype(int)

    return df


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series * 0
    return (series - lo) / (hi - lo)


def compute_risk_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate engineered features into normalized [0, 1] risk components,
    one per factor in config.RISK_WEIGHTS. Higher = riskier. Directions and
    relative importance are documented, human-reviewable business rules
    (see README "How the risk label was built"), not a learned function.
    """
    comp = pd.DataFrame(index=df.index)

    # U-shaped age risk: young (<25) and elderly (>65) drivers riskier,
    # minimum risk around 45-50.
    age = df["Customer_Age"]
    comp["age_risk"] = _minmax(((age - 47.5) / 25) ** 2)

    comp["kids_driving_risk"] = _minmax(df["Kids_Driving"].clip(upper=3))

    # Inverse relationship: higher income -> lower risk
    comp["income_risk"] = 1 - _minmax(df["Household_Income"])

    comp["car_year_risk"] = 1 - _minmax(df["Car_Year"])  # older car -> higher risk
    comp["vehicle_age_risk"] = _minmax(df["Vehicle_Age"])

    comp["car_use_risk"] = (df["Car_Use"] == "Commercial").astype(float)

    # Parents of driving-age kids carry more household exposure
    comp["parent_risk"] = (df["Parent"] == "Yes").astype(float)

    zone_risk_map = {
        "Highly Urban": 1.0,
        "Urban": 0.75,
        "Suburban": 0.5,
        "Rural": 0.35,
        "Highly Rural": 0.25,
    }
    comp["coverage_zone_risk"] = df["Coverage_Zone"].map(zone_risk_map).fillna(0.5)

    edu_risk_map = {"High School": 1.0, "Bachelors": 0.66, "Masters": 0.33, "PhD": 0.0}
    comp["education_risk"] = df["Education"].map(edu_risk_map).fillna(0.5)

    comp["gender_risk"] = (df["Gender"] == "Male").astype(float) * 0.5 + 0.25

    return comp
