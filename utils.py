"""
utils.py
--------
Shared constants, logging configuration, data validation, and file-loading
helpers for the AI Vendor Selection Assistant.
"""

from __future__ import annotations

import io
import logging
from typing import List, Tuple

import pandas as pd

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str = "vendor_assistant") -> logging.Logger:
    """Return a configured module-level logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
REQUIRED_COLUMNS: List[str] = [
    "Vendor",
    "Unit Price (INR)",
    "Quantity",
    "Total Price (INR)",
    "Warranty (Years)",
    "Delivery (Days)",
    "Quality Score",
    "Vendor Risk",
    "Payment Terms (Days)",
    "Past Performance (/5)",
    "Support SLA (Hours)",
    "Replacement Policy",
    "Compliance Score",
    "MSME Certified",
    "OEM Authorized",
    "Location",
]

NUMERIC_COLUMNS: List[str] = [
    "Unit Price (INR)",
    "Quantity",
    "Total Price (INR)",
    "Warranty (Years)",
    "Delivery (Days)",
    "Quality Score",
    "Payment Terms (Days)",
    "Past Performance (/5)",
    "Support SLA (Hours)",
    "Compliance Score",
]

YES_NO_COLUMNS: List[str] = ["MSME Certified", "OEM Authorized", "Replacement Policy"]

RISK_MAP = {"Low": 100, "Medium": 60, "High": 20}

# Criteria that participate in the weighted scoring model, mapped to the
# dataframe column that holds their *normalized* (0-100) value after
# processing. Order matches the default weight breakdown.
SCORING_CRITERIA = [
    "Price",
    "Quality",
    "Warranty",
    "Vendor Risk",
    "Delivery",
    "Past Performance",
    "Compliance",
    "Support SLA",
]

# Raw source column for each scoring criterion
CRITERIA_SOURCE_COLUMN = {
    "Price": "Total Price (INR)",
    "Quality": "Quality Score",
    "Warranty": "Warranty (Years)",
    "Vendor Risk": "Vendor Risk",
    "Delivery": "Delivery (Days)",
    "Past Performance": "Past Performance (/5)",
    "Compliance": "Compliance Score",
    "Support SLA": "Support SLA (Hours)",
}

# Criteria where a LOWER raw value is better (must be inverted on normalize)
LOWER_IS_BETTER = {"Price", "Delivery", "Support SLA"}

DEFAULT_WEIGHTS = {
    "Price": 30,
    "Quality": 20,
    "Warranty": 15,
    "Vendor Risk": 10,
    "Delivery": 10,
    "Past Performance": 5,
    "Compliance": 5,
    "Support SLA": 5,
}


# --------------------------------------------------------------------------
# File loading
# --------------------------------------------------------------------------
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load a Streamlit UploadedFile (csv or xlsx) into a DataFrame.
    Raises ValueError with a human-readable message on failure.
    """
    name = getattr(uploaded_file, "name", "uploaded_file")
    try:
        if name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError(
                f"Unsupported file type for '{name}'. Please upload a .csv or .xlsx file."
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse uploaded file %s", name)
        raise ValueError(f"Could not read '{name}': {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_csv_bytes(data: bytes) -> pd.DataFrame:
    """Load a dataframe from raw CSV bytes (used for the bundled sample data)."""
    df = pd.read_csv(io.BytesIO(data))
    df.columns = [str(c).strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the uploaded dataframe has the required schema and
    reasonable data quality. Returns (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    if df is None or df.empty:
        return False, ["The uploaded file is empty."]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(
            "Missing required column(s): " + ", ".join(f"'{c}'" for c in missing_cols)
        )
        # Can't validate further meaningfully without the columns
        return False, errors

    if df["Vendor"].isna().any() or (df["Vendor"].astype(str).str.strip() == "").any():
        errors.append("One or more rows have a blank 'Vendor' name.")

    if df["Vendor"].duplicated().any():
        dupes = df.loc[df["Vendor"].duplicated(), "Vendor"].tolist()
        errors.append(f"Duplicate vendor name(s) found: {', '.join(map(str, dupes))}")

    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad_rows = df.loc[coerced.isna() & df[col].notna()]
        if not bad_rows.empty:
            errors.append(
                f"Column '{col}' contains non-numeric value(s) in row(s): "
                f"{', '.join(str(i + 2) for i in bad_rows.index)}"
            )
        if coerced.isna().any():
            errors.append(f"Column '{col}' has missing value(s).")

    valid_risk = set(RISK_MAP.keys())
    bad_risk = set(df["Vendor Risk"].dropna().astype(str).str.strip()) - valid_risk
    if bad_risk:
        errors.append(
            f"Column 'Vendor Risk' must be one of {sorted(valid_risk)}. "
            f"Found invalid value(s): {sorted(bad_risk)}"
        )

    for col in YES_NO_COLUMNS:
        valid_yn = {"Yes", "No"}
        bad_yn = set(df[col].dropna().astype(str).str.strip().str.title()) - valid_yn
        if bad_yn:
            errors.append(
                f"Column '{col}' must be 'Yes' or 'No'. Found invalid value(s): {sorted(bad_yn)}"
            )

    if len(df) < 2:
        errors.append("At least two vendors are required to run a comparison.")

    return (len(errors) == 0), errors


def yes_no_to_numeric(series: pd.Series) -> pd.Series:
    """Convert Yes/No text values to 1/0."""
    return series.astype(str).str.strip().str.title().map({"Yes": 1, "No": 0})


def risk_to_numeric(series: pd.Series) -> pd.Series:
    """Convert Low/Medium/High vendor risk into a numeric score (higher = safer)."""
    return series.astype(str).str.strip().map(RISK_MAP)
