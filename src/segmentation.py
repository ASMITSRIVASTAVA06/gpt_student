"""Interpretable student segmentation."""

from __future__ import annotations

import pandas as pd


SEGMENT_DESCRIPTIONS = {
    "High performers": "Strong scores across academics, engagement, readiness, and skills with no urgent rule flags.",
    "Consistent but under-engaged students": "Solid academic and attendance profile, but low engagement activity.",
    "Academically at-risk students": "Academic score or backlog pattern indicates academic recovery support is needed.",
    "Placement preparation needed": "Placement-readiness indicators are weaker than the student's broader profile.",
    "Attendance support needed": "Attendance is the primary support concern.",
    "Developing students": "Mixed profile without one dominant critical issue; needs balanced support.",
}


def segment_for_row(row: pd.Series) -> str:
    flags = set(row.get("warning_flag_codes", []))
    if row.get("success_score", 0) >= 85 and not flags:
        return "High performers"
    if row.get("academic_score", 100) < 60 or row.get("backlogs", 0) >= 2:
        return "Academically at-risk students"
    if row.get("attendance_score", 100) < 65:
        return "Attendance support needed"
    if row.get("placement_score", 100) < 60 and row.get("year", 0) >= 3:
        return "Placement preparation needed"
    if row.get("academic_score", 0) >= 75 and row.get("attendance_score", 0) >= 75 and row.get("engagement_score", 100) < 55:
        return "Consistent but under-engaged students"
    return "Developing students"


def apply_segments(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["segment"] = result.apply(segment_for_row, axis=1)
    result["segment_description"] = result["segment"].map(SEGMENT_DESCRIPTIONS)
    return result


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "segment" not in df.columns:
        return pd.DataFrame(columns=["Segment", "Students", "Description"])
    summary = (
        df.groupby("segment", dropna=False)
        .size()
        .reset_index(name="Students")
        .rename(columns={"segment": "Segment"})
        .sort_values("Students", ascending=False)
    )
    summary["Description"] = summary["Segment"].map(SEGMENT_DESCRIPTIONS)
    return summary.reset_index(drop=True)

