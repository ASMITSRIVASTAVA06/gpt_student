"""Explainable intervention recommendations based on observed indicators."""

from __future__ import annotations

import pandas as pd


def recommendations_for_row(row: pd.Series) -> list[str]:
    """Generate specific support actions from the student's actual data."""
    recommendations: list[str] = []
    flags = set(row.get("warning_flag_codes", []))

    if "critical_attendance" in flags or "low_attendance" in flags:
        attendance = row.get("overall_attendance_pct")
        detail = f"current attendance {attendance:.1f}%" if pd.notna(attendance) else "attendance below target"
        recommendations.append(f"Schedule faculty mentoring and attendance recovery support ({detail}).")

    if "multiple_backlogs" in flags or row.get("academic_score", 100) < 60:
        backlogs = row.get("backlogs", 0)
        recommendations.append(
            f"Create a subject-wise academic recovery plan and weekly backlog review ({int(backlogs) if pd.notna(backlogs) else 0} backlogs recorded)."
        )

    if "low_assignment_completion" in flags or row.get("lms_score", 100) < 60:
        assignment = row.get("assignment_completion_pct")
        detail = f"{assignment:.1f}% assignments completed" if pd.notna(assignment) else "low LMS completion"
        recommendations.append(f"Provide assignment catch-up support and LMS nudges ({detail}).")

    if "poor_placement_readiness" in flags or row.get("placement_score", 100) < 60:
        recommendations.append("Add aptitude, coding practice, resume review, and mock interview sessions.")

    if "low_technical_skills" in flags or row.get("skills_score", 100) < 60:
        recommendations.append("Assign a technical-skills learning path with peer code review and soft-skills practice.")

    if row.get("engagement_score", 100) < 55:
        recommendations.append("Suggest suitable clubs, peer-learning groups, or hackathon participation to improve engagement.")

    if "low_feedback" in flags:
        recommendations.append("Arrange a faculty check-in to understand satisfaction, sentiment, and support-request patterns.")

    if not recommendations:
        recommendations.append("Maintain current plan and offer enrichment opportunities such as advanced projects or mentoring roles.")

    return recommendations


def apply_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["recommendations"] = result.apply(recommendations_for_row, axis=1)
    result["top_recommendation"] = result["recommendations"].apply(lambda items: items[0] if items else "")
    return result

