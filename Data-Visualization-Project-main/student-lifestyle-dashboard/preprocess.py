"""Data loading, cleaning and feature engineering for the Student Lifestyle dashboard.

Run directly to regenerate the cleaned dataset:
    python preprocess.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "student_lifestyle_raw.csv"
CLEAN_PATH = DATA_DIR / "student_lifestyle_clean.csv"

HOUR_COLS = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
]
NUMERIC_COLS = HOUR_COLS + ["GPA"]
STRESS_ORDER = ["Low", "Moderate", "High"]


def iqr_bounds(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def load_and_clean(path: Path = RAW_PATH):
    """Return (clean_df, quality_report)."""
    df = pd.read_csv(path)
    report = {"rows_raw": len(df), "columns_raw": df.shape[1]}

    # 1. Missing values: numeric -> median, categorical -> mode
    report["missing_before"] = int(df.isna().sum().sum())
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    df["Stress_Level"] = df["Stress_Level"].astype(str).str.strip().str.title()
    df.loc[~df["Stress_Level"].isin(STRESS_ORDER), "Stress_Level"] = df["Stress_Level"].mode()[0]
    report["missing_after"] = int(df.isna().sum().sum())

    # 2. Duplicates
    report["duplicate_rows"] = int(df.duplicated(subset="Student_ID").sum())
    df = df.drop_duplicates(subset="Student_ID").reset_index(drop=True)

    # 3. Logical consistency: a day has 24 hours; GPA is on a 0-4 scale
    total = df[HOUR_COLS].sum(axis=1)
    df["Total_Hours"] = total.round(1)
    report["days_not_24h"] = int((~np.isclose(total, 24, atol=0.15)).sum())
    report["gpa_out_of_range"] = int(((df["GPA"] < 0) | (df["GPA"] > 4)).sum())

    # 4. Outliers (IQR rule). Flagged, not deleted, so the user can toggle them in the dashboard.
    flags = pd.DataFrame(index=df.index)
    bounds = {}
    for c in NUMERIC_COLS:
        lo, hi = iqr_bounds(df[c])
        bounds[c] = (round(lo, 2), round(hi, 2))
        flags[c] = (df[c] < lo) | (df[c] > hi)
    df["Is_Outlier"] = flags.any(axis=1)
    report["outliers_per_column"] = flags.sum().astype(int).to_dict()
    report["outlier_rows"] = int(df["Is_Outlier"].sum())
    report["iqr_bounds"] = bounds

    # 5. Feature engineering (categorical views of numeric columns)
    df["Stress_Level"] = pd.Categorical(df["Stress_Level"], STRESS_ORDER, ordered=True)
    df["Stress_Code"] = df["Stress_Level"].cat.codes  # 0 Low, 1 Moderate, 2 High
    df["Study_Band"] = pd.cut(
        df["Study_Hours_Per_Day"], [0, 6, 8, 24], labels=["Low (<6h)", "Medium (6-8h)", "High (>8h)"], right=False
    )
    df["Sleep_Band"] = pd.cut(
        df["Sleep_Hours_Per_Day"], [0, 6, 8, 24], labels=["Short (<6h)", "Normal (6-8h)", "Long (>8h)"], right=False
    )
    df["GPA_Band"] = pd.cut(
        df["GPA"], [0, 3.0, 3.5, 4.01], labels=["Below 3.0", "3.0-3.49", "3.5+"], right=False
    )
    report["rows_clean"] = len(df)
    report["columns_clean"] = df.shape[1]
    return df, report


if __name__ == "__main__":
    clean, rep = load_and_clean()
    clean.to_csv(CLEAN_PATH, index=False)
    for k, v in rep.items():
        print(f"{k}: {v}")
    print(f"\nSaved {CLEAN_PATH}")
