"""
Builds the Risk_Category proxy label (Low / Medium / High).

IMPORTANT — read this before quoting model accuracy anywhere:
The raw dataset has no ground-truth underwriting risk decision. This module
converts documented, weighted risk factors (see config.RISK_WEIGHTS) into a
single score, adds calibrated noise so the label reflects real-world
unpredictability, and buckets the result into three tiers that mirror a
realistic insurance book of business (roughly 60-65% low risk, a quarter
medium, a little over a tenth high risk). Every downstream model is
therefore learning to recover a transparent proxy, not a real underwriting
outcome — call this out explicitly in interviews and in the README.
"""
import numpy as np
import pandas as pd

from src import config
from src.features import compute_risk_components


def compute_raw_risk_score(df: pd.DataFrame) -> pd.Series:
    components = compute_risk_components(df)
    weights = pd.Series(config.RISK_WEIGHTS)
    # align column order
    components = components[weights.index]
    score = components.mul(weights, axis=1).sum(axis=1)
    return score


def assign_risk_category(
    df: pd.DataFrame, random_state: int = config.RANDOM_STATE
) -> pd.Series:
    rng = np.random.default_rng(random_state)

    raw_score = compute_raw_risk_score(df)
    noise = rng.normal(
        loc=0.0,
        scale=raw_score.std() * config.RISK_LABEL_NOISE_FRACTION,
        size=len(raw_score),
    )
    noisy_score = raw_score + noise

    low_cut = noisy_score.quantile(config.RISK_CLASS_PROPORTIONS["Low"])
    med_cut = noisy_score.quantile(
        config.RISK_CLASS_PROPORTIONS["Low"] + config.RISK_CLASS_PROPORTIONS["Medium"]
    )

    category = pd.cut(
        noisy_score,
        bins=[-np.inf, low_cut, med_cut, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return category.astype(str)


if __name__ == "__main__":
    from src.data_prep import load_raw_data, clean_data
    from src.features import engineer_features

    df = engineer_features(clean_data(load_raw_data()))
    df[config.RISK_TARGET] = assign_risk_category(df)
    print(df[config.RISK_TARGET].value_counts(normalize=True).round(3))
