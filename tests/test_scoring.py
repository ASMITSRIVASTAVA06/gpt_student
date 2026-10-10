from __future__ import annotations

import pandas as pd
import pytest

from src.recommendations import apply_recommendations
from src.risk_engine import apply_risk_flags
from src.scoring import DEFAULT_WEIGHTS, ScoreSettings, apply_success_scores, classify_score_band, validate_weights
from src.segmentation import apply_segments


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["S1", "S2", "S3"],
            "student_name": ["Strong", "Critical attendance", "Backlog"],
            "department": ["CS", "CS", "IT"],
            "year": [4, 3, 2],
            "cgpa": [9.2, 9.1, 5.0],
            "semester_marks_pct": [92, 91, 48],
            "backlogs": [0, 0, 3],
            "core_subject_score": [94, 92, 44],
            "elective_subject_score": [90, 89, 50],
            "overall_attendance_pct": [94, 42, 72],
            "assignment_completion_pct": [96, 94, 45],
            "content_completion_pct": [93, 92, 50],
            "quiz_average_pct": [91, 90, 40],
            "lms_logins_30d": [42, 40, 8],
            "clubs_count": [3, 3, 0],
            "events_attended": [12, 11, 2],
            "hackathons": [3, 3, 0],
            "certifications": [3, 3, 0],
            "peer_learning_hours": [18, 17, 2],
            "aptitude_score": [88, 88, 45],
            "coding_score": [91, 92, 40],
            "mock_interview_score": [86, 87, 42],
            "resume_score": [90, 91, 46],
            "technical_skill_score": [92, 93, 42],
            "soft_skill_score": [88, 87, 55],
            "communication_score": [90, 88, 50],
            "teamwork_score": [91, 90, 54],
            "student_satisfaction_score": [86, 85, 50],
            "faculty_feedback_score": [88, 88, 52],
            "support_requests": [0, 1, 5],
            "sentiment_score": [84, 84, 48],
        }
    )


def test_success_scores_are_bounded_and_weighted() -> None:
    scored = apply_success_scores(sample_frame())
    assert scored["success_score"].between(0, 100).all()
    assert scored.loc[0, "success_score"] > 85
    assert scored.loc[2, "success_score"] < 60


def test_risk_classification_threshold_boundaries() -> None:
    settings = ScoreSettings.defaults()
    assert classify_score_band(59.99, settings) == "High risk"
    assert classify_score_band(60.00, settings) == "Medium risk"
    assert classify_score_band(79.99, settings) == "Medium risk"
    assert classify_score_band(80.00, settings) == "Low risk"


def test_invalid_weight_total_is_rejected() -> None:
    weights = DEFAULT_WEIGHTS.copy()
    weights["feedback"] = 10
    with pytest.raises(ValueError):
        validate_weights(weights)


def test_critical_warning_flag_escalates_high_scoring_student() -> None:
    scored = apply_success_scores(sample_frame())
    flagged = apply_risk_flags(scored)

    critical = flagged[flagged["student_id"] == "S2"].iloc[0]
    assert "critical_attendance" in critical["warning_flag_codes"]
    assert critical["risk_level"] == "High risk"
    assert bool(critical["risk_escalated_by_flags"]) is True


def test_recommendations_are_specific_to_observed_risks() -> None:
    scored = apply_success_scores(sample_frame())
    flagged = apply_risk_flags(scored)
    recommended = apply_recommendations(flagged)

    attendance_recs = " ".join(recommended.loc[recommended["student_id"] == "S2", "recommendations"].iloc[0])
    backlog_recs = " ".join(recommended.loc[recommended["student_id"] == "S3", "recommendations"].iloc[0])
    assert "attendance recovery" in attendance_recs
    assert "backlog" in backlog_recs


def test_segments_are_rule_based_labels() -> None:
    scored = apply_success_scores(sample_frame())
    flagged = apply_risk_flags(scored)
    segmented = apply_segments(flagged)

    assert "segment" in segmented.columns
    assert segmented.loc[segmented["student_id"] == "S1", "segment"].iloc[0] == "High performers"

