"""
Central configuration for the UnderSure.AI pipeline.
Keeping paths / constants here means every script and notebook stays in sync.
"""
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw" / "insurance_policies_data.xlsx"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"

for d in (DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Modeling constants -----------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Snapshot date used to compute ages / vehicle ages from the raw dataset.
# The dataset covers policies issued 2020-2025; this sits in the middle of
# that window and is the single source of truth for all age calculations.
REFERENCE_DATE = "2024-01-01"

# Risk-label engineering: the raw export has no ground-truth underwriting
# decision, so we build a documented, transparent proxy label
# ("Risk_Category") from known motor-insurance risk factors, weighted by
# how much each one plausibly drives risk (age and household exposure
# first, geography and education last). This is a modeling limitation and
# is called out explicitly in the README and notebooks rather than
# presented as ground truth.
RISK_WEIGHTS = {
    "age_risk": 0.262,
    "kids_driving_risk": 0.214,
    "income_risk": 0.128,
    "car_year_risk": 0.105,
    "vehicle_age_risk": 0.098,
    "car_use_risk": 0.076,
    "parent_risk": 0.052,
    "coverage_zone_risk": 0.031,
    "education_risk": 0.024,
    "gender_risk": 0.010,
}

# Target class split the label bucketing aims to reproduce
RISK_CLASS_PROPORTIONS = {"Low": 0.636, "Medium": 0.248, "High": 0.116}

# Std-dev (as a fraction of the raw score's std-dev) of Gaussian noise added
# before bucketing. This keeps the classification problem realistically
# hard (~75-80% accuracy ceiling) instead of a trivially separable rule.
RISK_LABEL_NOISE_FRACTION = 0.55

FEATURE_COLUMNS_NUMERIC = [
    "Customer_Age",
    "Vehicle_Age",
    "Car_Year",
    "Household_Income",
    "Kids_Driving",
]
FEATURE_COLUMNS_CATEGORICAL = [
    "Car_Use",
    "Coverage_Zone",
    "Education",
    "Gender",
    "Marital_Status",
    "Parent",
]

RISK_TARGET = "Risk_Category"
CLAIM_FREQ_TARGET = "Has_Claim"
CLAIM_AMOUNT_TARGET = "Claim_Amount"
