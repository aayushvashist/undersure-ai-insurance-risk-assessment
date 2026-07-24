"""
Generates the exploratory data analysis figures (01-10) referenced in the
README and notebooks. Run standalone: `python -m src.eda`
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.data_prep import clean_data, load_raw_data
from src.features import engineer_features
from src.risk_labeling import assign_risk_category
from src.utils import PALETTE, save_fig

sns.set_style("whitegrid")
RISK_ORDER = ["Low", "Medium", "High"]


def load_full_frame() -> pd.DataFrame:
    df = engineer_features(clean_data(load_raw_data()))
    df[config.RISK_TARGET] = assign_risk_category(df)
    return df


def fig01_risk_distribution(df):
    counts = df[config.RISK_TARGET].value_counts().reindex(RISK_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(
        counts,
        labels=[f"{k}\n{v:,} ({v/len(df)*100:.1f}%)" for k, v in counts.items()],
        colors=[PALETTE[k] for k in RISK_ORDER],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax.set_title(f"Overall Risk Distribution Across {len(df):,} Policies")
    save_fig(fig, config.FIGURES_DIR / "01_risk_distribution.png")


def fig02_age_pattern(df):
    ct = pd.crosstab(df["Age_Group"], df[config.RISK_TARGET], normalize="index")[RISK_ORDER] * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, color=[PALETTE[k] for k in RISK_ORDER], ax=ax)
    ax.set_ylabel("% of policies")
    ax.set_xlabel("Age Group")
    ax.set_title("Age Distribution by Risk Category")
    plt.xticks(rotation=0)
    ax.legend(title="Risk")
    save_fig(fig, config.FIGURES_DIR / "02_age_pattern.png")


def fig03_gender_marital(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["Gender", "Marital_Status"]):
        ct = pd.crosstab(df[col], df[config.RISK_TARGET], normalize="index")[RISK_ORDER] * 100
        ct.plot(kind="bar", stacked=True, color=[PALETTE[k] for k in RISK_ORDER], ax=ax, legend=False)
        ax.set_ylabel("% of policies")
        ax.set_xlabel(col.replace("_", " "))
        plt.setp(ax.get_xticklabels(), rotation=20)
    axes[1].legend(title="Risk", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Risk Distribution by Gender and Marital Status")
    save_fig(fig, config.FIGURES_DIR / "03_gender_marital.png")


def fig04_vehicle_age(df):
    bins = [0, 3, 6, 10, 15, 100]
    labels = ["0-3", "4-6", "7-10", "11-15", "15+"]
    df = df.copy()
    df["Vehicle_Age_Band"] = pd.cut(df["Vehicle_Age"], bins=bins, labels=labels)
    ct = pd.crosstab(df["Vehicle_Age_Band"], df[config.RISK_TARGET], normalize="index")[RISK_ORDER] * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ct.plot(kind="bar", stacked=True, color=[PALETTE[k] for k in RISK_ORDER], ax=ax)
    ax.set_ylabel("% of policies")
    ax.set_xlabel("Vehicle Age (years)")
    ax.set_title("Vehicle Age Distribution and Risk")
    plt.xticks(rotation=0)
    save_fig(fig, config.FIGURES_DIR / "04_vehicle_age.png")


def fig05_income(df):
    ct = pd.crosstab(df["Income_Quartile"], df[config.RISK_TARGET], normalize="index")[RISK_ORDER] * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ct.plot(kind="bar", stacked=True, color=[PALETTE[k] for k in RISK_ORDER], ax=ax)
    ax.set_ylabel("% of policies")
    ax.set_xlabel("Household Income Quartile (1=lowest, 4=highest)")
    ax.set_title("Income Impact on Risk")
    plt.xticks(rotation=0)
    save_fig(fig, config.FIGURES_DIR / "05_income.png")


def fig06_car_use(df):
    ct = pd.crosstab(df["Car_Use"], df[config.RISK_TARGET], normalize="index")[RISK_ORDER] * 100
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ct.plot(kind="bar", stacked=True, color=[PALETTE[k] for k in RISK_ORDER], ax=ax)
    ax.set_ylabel("% of policies")
    ax.set_title("Car Use: Commercial vs Private")
    plt.xticks(rotation=0)
    save_fig(fig, config.FIGURES_DIR / "06_car_use.png")


def fig07_coverage_zone(df):
    ct = pd.crosstab(df["Coverage_Zone"], df[config.RISK_TARGET], normalize="index")[RISK_ORDER] * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct.plot(kind="bar", stacked=True, color=[PALETTE[k] for k in RISK_ORDER], ax=ax)
    ax.set_ylabel("% of policies")
    ax.set_title("Geographic Risk Distribution Across Coverage Zones")
    plt.xticks(rotation=15)
    save_fig(fig, config.FIGURES_DIR / "07_coverage_zone.png")


def fig08_education(df):
    order = ["High School", "Bachelors", "Masters", "PhD"]
    ct = pd.crosstab(df["Education"], df[config.RISK_TARGET], normalize="index").reindex(order)[RISK_ORDER] * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ct.plot(kind="bar", stacked=True, color=[PALETTE[k] for k in RISK_ORDER], ax=ax)
    ax.set_ylabel("% of policies")
    ax.set_title("Education Level and Risk")
    plt.xticks(rotation=15)
    save_fig(fig, config.FIGURES_DIR / "08_education.png")


def fig09_kids_driving(df):
    ct = pd.crosstab(df["Kids_Driving"], df[config.RISK_TARGET], normalize="index")[RISK_ORDER] * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ct.plot(kind="bar", stacked=True, color=[PALETTE[k] for k in RISK_ORDER], ax=ax)
    ax.set_ylabel("% of policies")
    ax.set_xlabel("Number of Kids Driving in Household")
    ax.set_title("Kids Driving in Household — Powerful Risk Predictor")
    plt.xticks(rotation=0)
    save_fig(fig, config.FIGURES_DIR / "09_kids_driving.png")


def fig10_correlation_heatmap(df):
    numeric_cols = config.FEATURE_COLUMNS_NUMERIC + ["Claim_Freq", "Claim_Amount"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix")
    save_fig(fig, config.FIGURES_DIR / "10_correlation_heatmap.png")


def main():
    df = load_full_frame()
    fig01_risk_distribution(df)
    fig02_age_pattern(df)
    fig03_gender_marital(df)
    fig04_vehicle_age(df)
    fig05_income(df)
    fig06_car_use(df)
    fig07_coverage_zone(df)
    fig08_education(df)
    fig09_kids_driving(df)
    fig10_correlation_heatmap(df)
    print("\nAll 10 EDA figures saved to reports/figures/.")


if __name__ == "__main__":
    main()
