"""
Lightweight tests covering the data/feature/label pipeline. These don't
retrain models (that's a slow, separate step — see src/train_*.py) but they
do guard the parts most likely to silently break: cleaning, feature
engineering, and the risk-label bucketing.

Run with: pytest tests/ -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_prep import clean_data, load_raw_data
from src.features import compute_risk_components, engineer_features
from src.risk_labeling import assign_risk_category, compute_raw_risk_score


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture(scope="module")
def engineered_df(clean_df):
    return engineer_features(clean_df)


def test_raw_data_loads_with_expected_columns(raw_df):
    expected = {
        "ID", "BirthDate", "Car Color", "Car Make", "Car Model", "Car Use",
        "Car Year", "Coverage Zone", "Education", "Gender", "Marital Status",
        "Parent", "Claim Amount", "Claim Freq", "Household Income", "Kids Driving",
    }
    assert expected.issubset(set(raw_df.columns))
    assert len(raw_df) > 30000


def test_clean_data_has_no_duplicate_ids(clean_df):
    assert clean_df["ID"].is_unique


def test_clean_data_fixes_marital_status_typo(clean_df):
    assert "Seperated" not in clean_df["Marital_Status"].unique()


def test_clean_data_value_ranges(clean_df):
    assert clean_df["Household_Income"].min() > 0
    assert clean_df["Claim_Amount"].min() >= 0
    assert clean_df["Car_Year"].between(1950, 2026).all()


def test_engineered_features_present(engineered_df):
    for col in ["Customer_Age", "Vehicle_Age", "Age_Group", "Income_Quartile", "Has_Claim"]:
        assert col in engineered_df.columns


def test_customer_age_is_plausible(engineered_df):
    assert engineered_df["Customer_Age"].between(15, 90).all()


def test_has_claim_matches_claim_freq(engineered_df):
    assert (engineered_df["Has_Claim"] == (engineered_df["Claim_Freq"] > 0).astype(int)).all()


def test_risk_components_are_normalized(engineered_df):
    components = compute_risk_components(engineered_df)
    for col in components.columns:
        assert components[col].min() >= 0
        assert components[col].max() <= 1.0 + 1e-9


def test_risk_category_distribution_matches_target(engineered_df):
    category = assign_risk_category(engineered_df)
    props = category.value_counts(normalize=True)
    assert abs(props["Low"] - config.RISK_CLASS_PROPORTIONS["Low"]) < 0.01
    assert abs(props["Medium"] - config.RISK_CLASS_PROPORTIONS["Medium"]) < 0.01
    assert abs(props["High"] - config.RISK_CLASS_PROPORTIONS["High"]) < 0.01


def test_risk_category_is_deterministic_given_seed(engineered_df):
    cat1 = assign_risk_category(engineered_df, random_state=42)
    cat2 = assign_risk_category(engineered_df, random_state=42)
    assert (cat1 == cat2).all()


def test_risk_score_is_reproducible(engineered_df):
    score1 = compute_raw_risk_score(engineered_df)
    score2 = compute_raw_risk_score(engineered_df)
    pd.testing.assert_series_equal(score1, score2)
