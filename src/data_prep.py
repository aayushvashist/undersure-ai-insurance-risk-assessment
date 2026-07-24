"""
Load and clean the raw insurance policy export.
"""
import pandas as pd

from src import config


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw Excel export exactly as provided."""
    path = path or config.DATA_RAW
    df = pd.read_excel(path, sheet_name="insurance_data_car")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: standardize column names, fix known typos, drop exact
    duplicates, and validate value ranges. The raw export turns out to be
    ~99% clean already, so this mostly guards against edge cases (bad years,
    negative income, etc.) rather than doing heavy imputation.
    """
    df = df.copy()

    # Standardize column names to snake/Pascal without spaces for downstream code
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Known typo in the source export
    if "Marital_Status" in df.columns:
        df["Marital_Status"] = df["Marital_Status"].replace(
            {"Seperated": "Separated"}
        )

    # Drop exact duplicate policy records, if any
    before = len(df)
    df = df.drop_duplicates(subset=["ID"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[data_prep] Dropped {dropped} duplicate policy rows.")

    # Guard against impossible values
    df = df[(df["Car_Year"] >= 1950) & (df["Car_Year"] <= 2026)]
    df = df[df["Household_Income"] > 0]
    df = df[df["Claim_Amount"] >= 0]
    df = df[df["Claim_Freq"] >= 0]

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw = load_raw_data()
    print(f"Loaded {len(raw):,} raw rows, {raw.shape[1]} columns")
    cleaned = clean_data(raw)
    print(f"After cleaning: {len(cleaned):,} rows")
    print(cleaned.dtypes)
