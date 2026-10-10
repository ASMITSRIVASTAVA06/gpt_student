"""Transparent Student Success Score calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


COMPONENTS = [
    "academic",
    "attendance",
    "lms",
    "engagement",
    "placement",
    "skills",
    "feedback",
]

COMPONENT_LABELS = {
    "academic": "Academic performance",
    "attendance": "Attendance",
    "lms": "LMS completion",
    "engagement": "Student engagement",
    "placement": "Placement readiness",
    "skills": "Technical and soft skills",
    "feedback": "Feedback",
}

DEFAULT_WEIGHTS = {
    "academic": 30.0,
    "attendance": 20.0,
    "lms": 10.0,
    "engagement": 10.0,
    "placement": 15.0,
    "skills": 10.0,
    "feedback": 5.0,
}

DEFAULT_THRESHOLDS = {
    "high_risk_below": 60.0,
    "medium_risk_below": 80.0,
}


@dataclass(frozen=True)
class ScoreSettings:
    """Configurable score weights and risk thresholds."""

    weights: dict[str, float]
    high_risk_below: float = DEFAULT_THRESHOLDS["high_risk_below"]
    medium_risk_below: float = DEFAULT_THRESHOLDS["medium_risk_below"]

    @classmethod
    def defaults(cls) -> "ScoreSettings":
        return cls(weights=DEFAULT_WEIGHTS.copy())


def clip_0_100(value: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(value, 0, 100)


def validate_weights(weights: dict[str, float]) -> dict[str, float]:
    """Validate that weights cover every component and total 100."""
    missing = set(COMPONENTS) - set(weights)
    extra = set(weights) - set(COMPONENTS)
    if missing:
        raise ValueError(f"Missing weights for: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"Unknown weight components: {', '.join(sorted(extra))}")
    total = sum(float(value) for value in weights.values())
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("Weights cannot be negative.")
    if not np.isclose(total, 100.0, atol=0.01):
        raise ValueError(f"Weights must total 100, not {total:.2f}.")
    return {key: float(weights[key]) for key in COMPONENTS}


def row_mean(df: pd.DataFrame, columns: Iterable[str], default: float = 50.0) -> pd.Series:
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(default, index=df.index, dtype="float64")
    return df[available].mean(axis=1, skipna=True).fillna(default)


def normalize_count(series: pd.Series, target: float) -> pd.Series:
    return clip_0_100((pd.to_numeric(series, errors="coerce").fillna(0) / target) * 100)


def calculate_component_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 0-100 component scores before weighting."""
    scores = pd.DataFrame(index=df.index)

    cgpa_score = clip_0_100(pd.to_numeric(df.get("cgpa", 5), errors="coerce").fillna(5) * 10)
    academic_base = pd.concat(
        [
            cgpa_score.rename("cgpa_score"),
            row_mean(df, ["semester_marks_pct"], 50).rename("semester_marks_pct"),
            row_mean(df, ["core_subject_score", "elective_subject_score"], 50).rename("subject_score"),
        ],
        axis=1,
    ).mean(axis=1)
    backlogs = pd.to_numeric(df.get("backlogs", 0), errors="coerce").fillna(0)
    scores["academic_score"] = clip_0_100(academic_base - np.minimum(backlogs * 8, 35))

    overall_attendance = row_mean(df, ["overall_attendance_pct"], np.nan)
    subject_attendance = row_mean(
        df,
        ["math_attendance_pct", "programming_attendance_pct", "lab_attendance_pct"],
        np.nan,
    )
    attendance = overall_attendance.fillna(subject_attendance)
    attendance = attendance.fillna(50)
    scores["attendance_score"] = clip_0_100((attendance * 0.7) + (subject_attendance.fillna(attendance) * 0.3))

    login_score = normalize_count(df.get("lms_logins_30d", pd.Series(0, index=df.index)), 45)
    scores["lms_score"] = clip_0_100(
        row_mean(df, ["assignment_completion_pct"], 50) * 0.4
        + row_mean(df, ["content_completion_pct"], 50) * 0.25
        + row_mean(df, ["quiz_average_pct"], 50) * 0.2
        + login_score * 0.15
    )

    scores["engagement_score"] = clip_0_100(
        normalize_count(df.get("clubs_count", pd.Series(0, index=df.index)), 3) * 0.18
        + normalize_count(df.get("events_attended", pd.Series(0, index=df.index)), 14) * 0.22
        + normalize_count(df.get("hackathons", pd.Series(0, index=df.index)), 3) * 0.22
        + normalize_count(df.get("certifications", pd.Series(0, index=df.index)), 3) * 0.22
        + normalize_count(df.get("peer_learning_hours", pd.Series(0, index=df.index)), 20) * 0.16
    )

    scores["placement_score"] = clip_0_100(
        row_mean(df, ["aptitude_score", "coding_score", "mock_interview_score", "resume_score"], 50)
    )

    scores["skills_score"] = clip_0_100(
        row_mean(df, ["technical_skill_score"], 50) * 0.4
        + row_mean(df, ["soft_skill_score"], 50) * 0.25
        + row_mean(df, ["communication_score"], 50) * 0.2
        + row_mean(df, ["teamwork_score"], 50) * 0.15
    )

    support_requests = pd.to_numeric(df.get("support_requests", 0), errors="coerce").fillna(0)
    feedback_base = row_mean(
        df,
        ["student_satisfaction_score", "faculty_feedback_score", "sentiment_score"],
        50,
    )
    scores["feedback_score"] = clip_0_100(feedback_base - np.minimum(support_requests * 2.0, 20))

    return scores.round(2)


def classify_score_band(score: float, settings: ScoreSettings) -> str:
    """Classify score by the configured score-only thresholds."""
    if score < settings.high_risk_below:
        return "High risk"
    if score < settings.medium_risk_below:
        return "Medium risk"
    return "Low risk"


def apply_success_scores(df: pd.DataFrame, settings: ScoreSettings | None = None) -> pd.DataFrame:
    """Add component scores, weighted contributions, success score, and score band."""
    if settings is None:
        settings = ScoreSettings.defaults()
    weights = validate_weights(settings.weights)

    result = df.copy()
    component_scores = calculate_component_scores(result)
    result = pd.concat([result, component_scores], axis=1)

    weighted_columns = []
    for component in COMPONENTS:
        score_column = f"{component}_score"
        contribution_column = f"{component}_contribution"
        result[contribution_column] = (result[score_column] * weights[component] / 100).round(2)
        weighted_columns.append(contribution_column)

    result["success_score"] = clip_0_100(result[weighted_columns].sum(axis=1)).round(2)
    result["score_band"] = result["success_score"].apply(lambda score: classify_score_band(float(score), settings))
    result["weight_total"] = sum(weights.values())
    return result


def score_formula_text(weights: dict[str, float] | None = None) -> str:
    """Return a readable weighted-score formula."""
    weights = weights or DEFAULT_WEIGHTS
    parts = [f"{COMPONENT_LABELS[key]} x {weights[key]:.0f}%" for key in COMPONENTS]
    return "Success Score = " + " + ".join(parts) + ". Each component is normalized to 0-100 before weighting."

