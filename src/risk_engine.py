"""Rule-based risk classification and warning flags."""

from __future__ import annotations

import pandas as pd

from .scoring import ScoreSettings


WARNING_RULES = {
    "critical_attendance": "Critically low attendance",
    "low_attendance": "Low attendance",
    "multiple_backlogs": "Multiple academic backlogs",
    "low_assignment_completion": "Low assignment completion",
    "poor_placement_readiness": "Poor placement readiness",
    "low_technical_skills": "Low technical skills",
    "low_feedback": "Low feedback or support sentiment",
}

CRITICAL_RULES = {"critical_attendance", "multiple_backlogs"}


def warning_flags_for_row(row: pd.Series) -> list[str]:
    flags: list[str] = []
    attendance = row.get("overall_attendance_pct", row.get("attendance_score", 100))
    backlogs = row.get("backlogs", 0)
    assignment = row.get("assignment_completion_pct", row.get("lms_score", 100))
    placement = row.get("placement_score", 100)
    technical = row.get("technical_skill_score", row.get("skills_score", 100))
    feedback = row.get("feedback_score", 100)

    if pd.notna(attendance) and float(attendance) < 50:
        flags.append("critical_attendance")
    elif pd.notna(attendance) and float(attendance) < 65:
        flags.append("low_attendance")
    if pd.notna(backlogs) and float(backlogs) >= 2:
        flags.append("multiple_backlogs")
    if pd.notna(assignment) and float(assignment) < 50:
        flags.append("low_assignment_completion")
    if pd.notna(placement) and float(placement) < 50:
        flags.append("poor_placement_readiness")
    if pd.notna(technical) and float(technical) < 50:
        flags.append("low_technical_skills")
    if pd.notna(feedback) and float(feedback) < 50:
        flags.append("low_feedback")
    return flags


def flag_labels(flags: list[str]) -> list[str]:
    return [WARNING_RULES.get(flag, flag.replace("_", " ").title()) for flag in flags]


def final_risk_level(score_band: str, flags: list[str]) -> tuple[str, bool]:
    """Escalate score-only risk when critical rule flags are present."""
    has_critical = any(flag in CRITICAL_RULES for flag in flags)
    if score_band == "High risk":
        return "High risk", has_critical
    if has_critical:
        return "High risk", True
    if score_band == "Low risk" and flags:
        return "Medium risk", False
    return score_band, False


def apply_risk_flags(df: pd.DataFrame, settings: ScoreSettings | None = None) -> pd.DataFrame:
    """Add rule-based warning flags, final risk level, and review indicators."""
    result = df.copy()
    result["warning_flag_codes"] = result.apply(warning_flags_for_row, axis=1)
    result["warning_flags"] = result["warning_flag_codes"].apply(flag_labels)
    risk_pairs = result.apply(
        lambda row: final_risk_level(str(row.get("score_band", "Low risk")), row["warning_flag_codes"]),
        axis=1,
    )
    result["risk_level"] = [pair[0] for pair in risk_pairs]
    result["risk_escalated_by_flags"] = [pair[1] for pair in risk_pairs]
    result["urgent_review"] = (
        (result["risk_level"] == "High risk")
        | result["warning_flag_codes"].apply(lambda flags: any(flag in CRITICAL_RULES for flag in flags))
    )
    result["primary_reasons"] = result.apply(primary_reasons, axis=1)
    return result


def primary_reasons(row: pd.Series, max_reasons: int = 3) -> str:
    reasons = flag_labels(row.get("warning_flag_codes", []))
    weak_components = []
    for component in ["academic", "attendance", "lms", "placement", "skills", "feedback", "engagement"]:
        value = row.get(f"{component}_score")
        if pd.notna(value) and float(value) < 60:
            weak_components.append(f"{component.title()} score {float(value):.0f}")
    combined = reasons + weak_components
    if not combined:
        return "Score and rule checks show no urgent weakness."
    return "; ".join(combined[:max_reasons])

