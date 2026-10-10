from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "equalai_employee_data.csv"
REQUIRED_COLUMNS = {
    "gender",
    "age",
    "department",
    "experience_years",
    "salary",
    "promotion_status",
}
NUMERIC_LIMITS = {
    "age": (16, 100),
    "experience_years": (0, 80),
    "salary": (0, None),
}


def _read_csv(uploaded_file: BinaryIO | None, csv_path: Path) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            frame = pd.read_csv(uploaded_file)
        except (UnicodeError, pd.errors.ParserError, ValueError) as error:
            raise ValueError(f"Uploaded file is not a readable CSV: {error}") from error
        return frame, "Uploaded CSV — provenance not independently verified"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except (UnicodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"Could not parse {csv_path.name}: {error}") from error
    source = (
        "Bundled synthetic demonstration dataset"
        if csv_path.name == "equalai_employee_data.csv"
        else f"Local CSV: {csv_path.name} — provenance not independently verified"
    )
    return frame, source


def _normalize_gender(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Other/Unknown"
    label = str(value).strip().casefold()
    known = {
        "f": "Female",
        "female": "Female",
        "woman": "Female",
        "women": "Female",
        "m": "Male",
        "male": "Male",
        "man": "Male",
        "men": "Male",
    }
    return known.get(label, "Other/Unknown")


def _normalize_promotion(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Unknown"
    label = str(value).strip().casefold()
    yes_labels = {"yes", "y", "true", "1", "1.0", "promoted"}
    no_labels = {"no", "n", "false", "0", "0.0", "not promoted"}
    if label in yes_labels:
        return "Yes"
    if label in no_labels:
        return "No"
    return "Unknown"


def _make_synthetic_demo() -> pd.DataFrame:
    departments = ["Engineering", "Sales", "HR", "Finance", "Product", "Operations"]
    rows: list[dict[str, Any]] = []
    for index in range(60):
        female = index % 2 == 0
        department = departments[(index // 2) % len(departments)]
        experience = 1 + (index * 7) % 20
        rows.append(
            {
                "gender": "Female" if female else "Male",
                "age": 22 + (index * 3) % 29,
                "department": department,
                "experience_years": experience,
                "salary": 34000 + experience * 3200 + (1200 if not female else 0),
                "promotion_status": "Yes" if index % 4 else "No",
            }
        )
    return pd.DataFrame(rows)


def load_employee_data(
    uploaded_file: BinaryIO | None = None,
    csv_path: Path = DATA_PATH,
    use_synthetic_demo: bool = False,
) -> tuple[pd.DataFrame, dict[str, int], str]:
    """Load and clean employee data while preserving a compact quality summary."""
    if uploaded_file is None and not csv_path.exists() and use_synthetic_demo:
        frame = _make_synthetic_demo()
        source = "Generated synthetic demo dataset — not real-world evidence"
    else:
        frame, source = _read_csv(uploaded_file, csv_path)

    frame.columns = [str(column).strip().casefold() for column in frame.columns]
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing_columns)
            + ". Required fields are gender, age, department, experience_years, salary, promotion_status."
        )

    rows_loaded = len(frame)
    duplicates_removed = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    missing_values = int(frame[list(REQUIRED_COLUMNS)].isna().sum().sum())

    gender_raw = frame["gender"]
    expected_gender = {
        "f", "female", "woman", "women", "m", "male", "man", "men",
    }
    unexpected_gender = int(
        (~gender_raw.map(lambda value: pd.isna(value) or str(value).strip().casefold() in expected_gender)).sum()
    )
    promotion_raw = frame["promotion_status"]
    expected_promotion = {
        "yes", "y", "true", "1", "1.0", "promoted",
        "no", "n", "false", "0", "0.0", "not promoted",
    }
    unexpected_promotion = int(
        (~promotion_raw.map(lambda value: pd.isna(value) or str(value).strip().casefold() in expected_promotion)).sum()
    )

    invalid_numeric = 0
    for column, (minimum, maximum) in NUMERIC_LIMITS.items():
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if minimum is not None:
            invalid |= converted.notna() & (converted < minimum)
        if maximum is not None:
            invalid |= converted.notna() & (converted > maximum)
        invalid_numeric += int(invalid.sum())
        converted = converted.mask(invalid)
        frame[column] = converted.astype(float)

    frame["gender"] = frame["gender"].map(_normalize_gender)
    frame["promotion_status"] = frame["promotion_status"].map(_normalize_promotion)
    frame["department"] = (
        frame["department"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )
    frame = frame.reset_index(drop=True)
    quality = {
        "rows_loaded": int(rows_loaded),
        "rows_after_cleaning": int(len(frame)),
        "duplicates_removed": duplicates_removed,
        "missing_values": missing_values,
        "invalid_numeric_values": invalid_numeric,
        "unexpected_gender_labels": unexpected_gender,
        "unexpected_promotion_labels": unexpected_promotion,
    }
    return frame, quality, source
