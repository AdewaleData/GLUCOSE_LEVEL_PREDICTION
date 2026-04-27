import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path(r"c:\Users\555555\OneDrive\Desktop\GLUCOSE\clean-dataset.csv")
OUTPUT_CSV = Path(r"c:\Users\555555\OneDrive\Desktop\GLUCOSE\clean-dataset-relevant.csv")
REPORT_JSON = Path(r"c:\Users\555555\OneDrive\Desktop\GLUCOSE\clean-dataset-relevant-report.json")


def clip_range(df: pd.DataFrame, column: str, low: float, high: float) -> pd.Series:
    """Flag values outside expected ranges."""
    bad = (df[column] < low) | (df[column] > high)
    return bad


def robust_iqr_mask(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
    """Return boolean mask of outliers by robust IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - (multiplier * iqr)
    high = q3 + (multiplier * iqr)
    return (series < low) | (series > high)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    original_rows = len(df)

    # Remove known leakage/ordering columns from modeling surface.
    drop_cols = [c for c in ["index", "pl"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Remove exact duplicate rows first.
    full_dedup_rows_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    after_full_dedup_rows = len(df)

    # Basic physiologic filters (for true demographic/vital columns only).
    range_rules = {
        "Heart_Rate": (30, 220),
        "Age": (10, 100),
        "Height": (120, 230),  # cm
        "Weight": (30, 250),  # kg
        "Glucose_level": (40, 500),  # mg/dL
    }

    outlier_flags = {}
    for col, (low, high) in range_rules.items():
        if col in df.columns:
            outlier_flags[col] = int(clip_range(df, col, low, high).sum())
            df = df[(df[col] >= low) & (df[col] <= high)]

    # Peak columns are waveform amplitudes, not mmHg BP values.
    # Use robust IQR filtering to suppress extreme artifacts only.
    signal_cols = [c for c in ["PPG_Signal", "Systolic_Peak", "Diastolic_Peak", "Pulse_Area"] if c in df.columns]
    iqr_outlier_flags = {}
    iqr_removed_total = 0
    for col in signal_cols:
        bad = robust_iqr_mask(df[col], multiplier=3.0)
        iqr_outlier_flags[col] = int(bad.sum())
        iqr_removed_total += int(bad.sum())
        df = df[~bad]

    filtered_rows = len(df)

    # Resolve conflicting labels: same physiological state must map to one glucose value.
    # Keep Patient_Id inside the state so we do not merge different people into one record.
    state_cols = [
        "Patient_Id",
        "PPG_Signal",
        "Heart_Rate",
        "Systolic_Peak",
        "Diastolic_Peak",
        "Pulse_Area",
        "Gender",
        "Age",
        "Height",
        "Weight",
    ]
    state_cols = [c for c in state_cols if c in df.columns]

    state_group = df.groupby(state_cols, as_index=False)["Glucose_level"]
    conflict_stats = state_group.agg(["nunique", "count"]).reset_index()
    conflicting_states = int((conflict_stats["nunique"] > 1).sum())

    # Median is robust against occasional abnormal glucose labels.
    df = (
        state_group.median()
        .rename(columns={"Glucose_level": "Glucose_level"})
        .copy()
    )
    after_conflict_resolution_rows = len(df)

    # Convert binary gender to 0/1 consistently.
    if "Gender" in df.columns:
        gmap = {1: 0, 2: 1}
        df["Gender_Binary"] = df["Gender"].map(gmap).fillna(0).astype(int)

    # Clinical/physiological engineered features.
    df["Height_m"] = df["Height"] / 100.0
    df["BMI"] = df["Weight"] / (df["Height_m"] ** 2)
    df["BSA_Mosteller"] = np.sqrt((df["Height"] * df["Weight"]) / 3600.0)

    # Pulse/pressure interactions.
    df["PulsePressure"] = df["Systolic_Peak"] - df["Diastolic_Peak"]
    df["MAP"] = df["Diastolic_Peak"] + (df["PulsePressure"] / 3.0)
    df["RPP"] = df["Heart_Rate"] * df["Systolic_Peak"]
    df["ShockIndex"] = df["Heart_Rate"] / df["Systolic_Peak"].replace(0, np.nan)

    # Morphology normalization features.
    df["PulseArea_per_HR"] = df["Pulse_Area"] / df["Heart_Rate"].replace(0, np.nan)
    df["PulseArea_per_PP"] = df["Pulse_Area"] / df["PulsePressure"].replace(0, np.nan)

    # Keep only interpretable engineered features (avoid feature explosion).
    keep_cols = [
        "Patient_Id",
        "PPG_Signal",
        "Heart_Rate",
        "Systolic_Peak",
        "Diastolic_Peak",
        "Pulse_Area",
        "Gender",
        "Gender_Binary",
        "Age",
        "Height",
        "Weight",
        "Height_m",
        "BMI",
        "BSA_Mosteller",
        "PulsePressure",
        "MAP",
        "RPP",
        "ShockIndex",
        "PulseArea_per_HR",
        "PulseArea_per_PP",
        "Glucose_level",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Final cleanup for model-ready table.
    before_nan_drop = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    nan_inf_rows_removed = before_nan_drop - len(df)

    # Final hard guarantee: one row per unique physiological state.
    df = df.drop_duplicates(subset=state_cols, keep="first").reset_index(drop=True)
    final_rows = len(df)

    # Save outputs.
    df.to_csv(OUTPUT_CSV, index=False)

    report = {
        "input_file": str(INPUT_CSV),
        "output_file": str(OUTPUT_CSV),
        "original_rows": original_rows,
        "full_duplicate_rows_removed": full_dedup_rows_removed,
        "after_full_dedup_rows": after_full_dedup_rows,
        "after_physiology_filter_rows": filtered_rows,
        "iqr_rows_removed_total": iqr_removed_total,
        "conflicting_states_resolved_by_median": conflicting_states,
        "after_conflict_resolution_rows": after_conflict_resolution_rows,
        "nan_inf_rows_removed": nan_inf_rows_removed,
        "final_rows": final_rows,
        "dropped_columns": drop_cols,
        "state_columns_for_uniqueness": state_cols,
        "outlier_rows_detected_by_column": outlier_flags,
        "iqr_outlier_rows_detected_by_signal_column": iqr_outlier_flags,
        "final_columns": list(df.columns),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
