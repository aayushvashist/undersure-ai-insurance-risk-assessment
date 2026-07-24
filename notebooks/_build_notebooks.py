"""
One-off generator for the four narrated notebooks. Not part of the
reusable pipeline, kept here only so the notebooks can be regenerated or
tweaked later without hand-editing raw JSON. Safe to delete after cloning.
"""
import nbformat as nbf


def nb_from_cells(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


SETUP_CODE = """\
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
"""

# ---------------------------------------------------------------------------
# 01 - EDA
# ---------------------------------------------------------------------------
nb1 = nb_from_cells([
    md("""\
# 01 - Exploratory Data Analysis

UnderSure.AI: Machine Learning for Motor Insurance Risk Assessment

This notebook looks at the raw policy dataset (37.5k motor insurance
records) before any feature engineering or modeling.

Dataset columns: `ID`, `BirthDate`, `Car Color`, `Car Make`, `Car Model`,
`Car Use`, `Car Year`, `Coverage Zone`, `Education`, `Gender`,
`Marital Status`, `Parent`, `Claim Amount`, `Claim Freq`,
`Household Income`, `Kids Driving`.
"""),
    code(SETUP_CODE),
    code("""\
from src.data_prep import load_raw_data, clean_data

raw = load_raw_data()
print(f"Raw shape: {raw.shape}")
raw.head()
"""),
    code("""\
df = clean_data(raw)
print(f"After cleaning: {df.shape}")
df.describe(include='all').T
"""),
    md("## Missing values and data quality"),
    code("df.isna().sum()"),
    md("""\
## Figures

The figures below come from `src/eda.py` (run once via `python -m src.eda`)
and are saved to `reports/figures/`. Displaying them here inline keeps the
notebook readable; regenerate them any time the data or risk-labeling
rules change.
"""),
    code("""\
from src import config
import matplotlib.image as mpimg

fig_files = [
    "01_risk_distribution.png", "02_age_pattern.png", "03_gender_marital.png",
    "04_vehicle_age.png", "05_income.png", "06_car_use.png",
    "07_coverage_zone.png", "08_education.png", "09_kids_driving.png",
    "10_correlation_heatmap.png",
]
for fname in fig_files:
    path = config.FIGURES_DIR / fname
    if path.exists():
        plt.figure(figsize=(8, 5))
        plt.imshow(mpimg.imread(path))
        plt.axis('off')
        plt.title(fname)
        plt.show()
    else:
        print(f"Missing (run `python -m src.eda` first): {fname}")
"""),
    md("""\
## Key observations

Risk distribution is heavily skewed toward low-risk (roughly 63% low, a
quarter medium, a little over a tenth high), which looks like a realistic
insurance book of business and is the target balance the risk-labeling
rules in `src/risk_labeling.py` are calibrated to reproduce.

Customer age shows the expected U-shape: risk is highest for young
drivers, lowest around mid-life, and creeps back up for older drivers.
Kids driving and household income are strong, consistent signals too.

Claim Freq and Claim Amount, the two real data columns used later for the
claims models, show very little relationship with any demographic feature
(see the correlation heatmap). That gets explored, and reported honestly,
in notebook 04.
"""),
])

# ---------------------------------------------------------------------------
# 02 - Feature engineering + risk labeling
# ---------------------------------------------------------------------------
nb2 = nb_from_cells([
    md("""\
# 02 - Feature Engineering & Risk Labeling

This notebook does two things: engineers modeling features
(`Customer_Age`, `Vehicle_Age`, `Age_Group`, `Income_Quartile`,
`Has_Claim`) from the raw columns, and builds the `Risk_Category` proxy
label used to train the risk classifier in notebook 03.

Why a proxy label at all? The raw export has no ground-truth underwriting
decision column. Rather than pretend otherwise, this builds a
transparent, documented scoring rule from known risk factors (see
`src/risk_labeling.py` and `config.RISK_WEIGHTS`), calibrated so the
resulting three-way split looks like a realistic insurance book of
business. That's a real limitation worth being upfront about rather than
glossing over.
"""),
    code(SETUP_CODE),
    code("""\
from src.data_prep import load_raw_data, clean_data
from src.features import engineer_features, compute_risk_components
from src.risk_labeling import compute_raw_risk_score, assign_risk_category
from src import config

df = engineer_features(clean_data(load_raw_data()))
df[['BirthDate', 'Customer_Age', 'Car_Year', 'Vehicle_Age', 'Age_Group', 'Income_Quartile', 'Has_Claim']].head()
"""),
    md("## Risk components: one normalized 0-1 score per weighted factor"),
    code("""\
components = compute_risk_components(df)
components.describe().T
"""),
    md("## Weights, ordered by how much each factor plausibly drives risk"),
    code("""\
import pandas as pd
pd.Series(config.RISK_WEIGHTS, name='weight').sort_values(ascending=False)
"""),
    code("""\
df['risk_score_raw'] = compute_raw_risk_score(df)
df[config.RISK_TARGET] = assign_risk_category(df)

df[config.RISK_TARGET].value_counts(normalize=True).round(3)
"""),
    md("""\
That split happens by construction, since the labels are bucketed on
score quantiles. What actually matters is whether the raw features, not
the score itself, can recover this label reasonably well. That's tested
in notebook 03.
"""),
    code("""\
df.groupby(config.RISK_TARGET)['risk_score_raw'].describe()
"""),
])

# ---------------------------------------------------------------------------
# 03 - Risk model training
# ---------------------------------------------------------------------------
nb3 = nb_from_cells([
    md("""\
# 03 - Risk Classification: Model Development & Results

Trains and compares Logistic Regression, Random Forest, and Gradient
Boosting on the `Risk_Category` proxy label, then checks the selected
model with 5-fold stratified cross-validation. Reuses the exact functions
in `src/train_risk_model.py` so this notebook and the reusable pipeline
never drift apart.
"""),
    code(SETUP_CODE),
    code("""\
from sklearn.model_selection import train_test_split
from src.train_risk_model import (
    load_training_frame, evaluate_algorithms, plot_algorithm_comparison,
    plot_confusion_matrix, plot_feature_importance, run_cross_validation,
)
from src import config

df = load_training_frame()
X = df[config.FEATURE_COLUMNS_NUMERIC + config.FEATURE_COLUMNS_CATEGORICAL]
y = df[config.RISK_TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
"""),
    md("## Algorithm comparison"),
    code("""\
results, fitted = evaluate_algorithms(X_train, y_train, X_test, y_test)
import pandas as pd
pd.DataFrame(results).T
"""),
    md("""\
## Model selection

Gradient Boosting edges out Random Forest on raw test accuracy in most
runs of this dataset. Random Forest is selected anyway: it trains and
predicts an order of magnitude faster, its `feature_importances_` are
simpler to explain to a non-technical underwriter than a 200-stage
boosted ensemble, and it's less prone to overfitting the noisy proxy
label. That's a deliberate engineering trade-off, not just picking
whichever number is highest.
"""),
    code("""\
best_pipe = fitted['Random Forest']
cm = plot_confusion_matrix(best_pipe, X_test, y_test)
cm
"""),
    code("""\
importance = plot_feature_importance(best_pipe)
importance
"""),
    md("## 5-fold cross-validation (stability check)"),
    code("""\
cv_results = run_cross_validation(X, y)
pd.DataFrame(cv_results).T
"""),
    md("""\
The variance across folds is tight, which suggests the model has learned
a stable, generalizable mapping from features to proxy risk tier rather
than picking up on quirks of one particular train/test split.
"""),
])

# ---------------------------------------------------------------------------
# 04 - Claims models
# ---------------------------------------------------------------------------
nb4 = nb_from_cells([
    md("""\
# 04 - Claims Prediction: Frequency & Severity

Unlike `Risk_Category`, both targets modeled here are genuine columns
from the raw export: `Claim Freq` (did the policyholder claim?) and
`Claim Amount` (how much?). No engineered labels involved.
"""),
    code(SETUP_CODE),
    code("""\
from src.data_prep import load_raw_data, clean_data
from src.features import engineer_features
from src.train_claims_models import train_claim_frequency_model, train_claim_amount_model
from src import config

df = engineer_features(clean_data(load_raw_data()))
print(f"Baseline claim rate: {df['Has_Claim'].mean()*100:.2f}%")
"""),
    md("## Does anything predict who claims?"),
    code("""\
for col in ['Car_Use', 'Age_Group', 'Kids_Driving', 'Coverage_Zone']:
    print(df.groupby(col, observed=True)['Has_Claim'].mean().round(3), '\\n')
"""),
    md("""\
Claim rate barely moves across any of these, staying around 27-28%
regardless of usage type, age band, household composition, or geography.
That's the headline finding of this notebook, reported as-is rather than
hunting for a feature combination that looks more predictive than it
really is.
"""),
    md("## Claim frequency classifier vs. a naive baseline"),
    code("""\
freq_results = train_claim_frequency_model(df)
import pandas as pd
pd.DataFrame(freq_results).T
"""),
    md("## Claim amount regression vs. a naive baseline"),
    code("""\
amount_results = train_claim_amount_model(df)
pd.DataFrame(amount_results).T
"""),
    md("""\
## Interpretation

Both models come in essentially tied with their naive baselines (majority
class / mean prediction). Rather than read this as a disappointing
result, it's a useful one: idiosyncratic claim risk (accidents, weather,
plain bad luck) dominates over anything derivable from demographic and
vehicle attributes alone. That's also a plausible reason real insurers
invest in telematics and claims-history data rather than leaning on
demographics.
"""),
])


if __name__ == "__main__":
    import pathlib

    out_dir = pathlib.Path(__file__).parent
    for name, nb in [
        ("01_EDA.ipynb", nb1),
        ("02_Feature_Engineering.ipynb", nb2),
        ("03_Risk_Model_Training.ipynb", nb3),
        ("04_Claims_Models.ipynb", nb4),
    ]:
        nbf.write(nb, out_dir / name)
        print(f"wrote {name}")
