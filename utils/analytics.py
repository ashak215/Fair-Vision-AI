from __future__ import annotations

import pandas as pd


def filter_employees(
    df: pd.DataFrame,
    departments: list[str] | None = None,
    experience_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    if departments is not None:
        filtered = filtered[filtered["department"].isin(departments)]
    if experience_range is not None:
        low, high = experience_range
        valid = filtered["experience_years"].isna() | filtered["experience_years"].between(low, high)
        filtered = filtered[valid]
    return filtered.copy()


def build_gender_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count each normalized gender group within each department."""
    if df.empty:
        return pd.DataFrame(columns=["department", "gender", "employees"])
    return (
        df.groupby(["department", "gender"], dropna=False, observed=True)
        .size()
        .rename("employees")
        .reset_index()
        .sort_values(["department", "gender"])
        .reset_index(drop=True)
    )


def get_salary_summary(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["salary"])
    if valid.empty:
        return pd.DataFrame(
            columns=["gender", "employees_with_salary", "average_salary", "median_salary"]
        )
    return (
        valid.groupby("gender", dropna=False, observed=True)
        .agg(
            employees_with_salary=("salary", "size"),
            average_salary=("salary", "mean"),
            median_salary=("salary", "median"),
        )
        .reset_index()
        .sort_values("gender")
        .reset_index(drop=True)
    )


def get_promotion_summary(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["promotion_status"].isin(["Yes", "No"])]
    if valid.empty:
        return pd.DataFrame(
            columns=["gender", "employees_with_recorded_outcome", "promoted", "promotion_rate"]
        )
    result = (
        valid.assign(promoted=valid["promotion_status"].eq("Yes"))
        .groupby("gender", dropna=False, observed=True)
        .agg(
            employees_with_recorded_outcome=("promoted", "size"),
            promoted=("promoted", "sum"),
            promotion_rate=("promoted", "mean"),
        )
        .reset_index()
        .sort_values("gender")
        .reset_index(drop=True)
    )
    result["promotion_rate"] = result["promotion_rate"] * 100
    return result


def build_detailed_gender_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize headcount, salary, experience, and recorded outcomes per department/gender."""
    columns = [
        "department",
        "gender",
        "employee_count",
        "average_salary",
        "median_salary",
        "average_experience_years",
        "recorded_promotion_count",
        "promoted_count",
        "promotion_rate",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    with_outcome = df.assign(
        promotion_binary=df["promotion_status"].map({"Yes": 1.0, "No": 0.0})
    )
    summary = (
        with_outcome.groupby(["department", "gender"], dropna=False, observed=True)
        .agg(
            employee_count=("gender", "size"),
            average_salary=("salary", "mean"),
            median_salary=("salary", "median"),
            average_experience_years=("experience_years", "mean"),
            recorded_promotion_count=("promotion_binary", "count"),
            promoted_count=("promotion_binary", "sum"),
            promotion_rate=("promotion_binary", "mean"),
        )
        .reset_index()
        .sort_values(["department", "gender"])
        .reset_index(drop=True)
    )
    summary["promotion_rate"] = summary["promotion_rate"] * 100
    return summary[columns]
