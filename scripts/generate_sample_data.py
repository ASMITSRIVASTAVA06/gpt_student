"""Generate reproducible synthetic CSV sources for the demo dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIXTURE_DIR = DATA_DIR / "quality_fixtures"
RNG_SEED = 20261010
N_STUDENTS = 560

DEPARTMENTS = [
    "Computer Science",
    "Information Technology",
    "Electronics",
    "Mechanical",
    "Civil",
    "Data Science",
]
YEARS = [1, 2, 3, 4]
PERSONAS = [
    "high_performer",
    "developing",
    "low_attendance",
    "academic_recovery",
    "placement_prep_needed",
    "under_engaged",
]
PERSONA_WEIGHTS = [0.20, 0.27, 0.13, 0.16, 0.14, 0.10]


def clamp(values: np.ndarray | float, low: float, high: float) -> np.ndarray | float:
    """Clamp numeric values to a closed range."""
    return np.clip(values, low, high)


def score_by_persona(
    rng: np.random.Generator,
    persona: str,
    centers: dict[str, float],
    spread: float,
) -> float:
    value = rng.normal(centers[persona], spread)
    return float(clamp(value, 0, 100))


def build_students(rng: np.random.Generator) -> pd.DataFrame:
    ids = [f"STU{i:04d}" for i in range(1, N_STUDENTS + 1)]
    personas = rng.choice(PERSONAS, size=N_STUDENTS, p=PERSONA_WEIGHTS)
    departments = rng.choice(DEPARTMENTS, size=N_STUDENTS)
    years = rng.choice(YEARS, size=N_STUDENTS, p=[0.24, 0.26, 0.27, 0.23])

    # Synthetic labels avoid implying these are actual student identities.
    students = pd.DataFrame(
        {
            "student_id": ids,
            "student_name": [f"Synthetic Student {i:04d}" for i in range(1, N_STUDENTS + 1)],
            "department": departments,
            "year": years,
            "section": rng.choice(["A", "B", "C", "D"], size=N_STUDENTS),
            "advisor": [f"Faculty Mentor {i}" for i in rng.integers(1, 19, size=N_STUDENTS)],
            "synthetic_demo_record": True,
            "persona_seed": personas,
        }
    )
    return students


def build_sources(students: pd.DataFrame, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    personas = students["persona_seed"].to_numpy()
    ids = students["student_id"].to_numpy()
    year = students["year"].to_numpy()

    academic_center = {
        "high_performer": 88,
        "developing": 72,
        "low_attendance": 80,
        "academic_recovery": 55,
        "placement_prep_needed": 76,
        "under_engaged": 74,
    }
    attendance_center = {
        "high_performer": 92,
        "developing": 78,
        "low_attendance": 49,
        "academic_recovery": 68,
        "placement_prep_needed": 80,
        "under_engaged": 76,
    }
    lms_center = {
        "high_performer": 90,
        "developing": 70,
        "low_attendance": 72,
        "academic_recovery": 58,
        "placement_prep_needed": 73,
        "under_engaged": 44,
    }
    placement_center = {
        "high_performer": 86,
        "developing": 66,
        "low_attendance": 70,
        "academic_recovery": 55,
        "placement_prep_needed": 42,
        "under_engaged": 61,
    }
    skill_center = {
        "high_performer": 88,
        "developing": 68,
        "low_attendance": 72,
        "academic_recovery": 57,
        "placement_prep_needed": 61,
        "under_engaged": 63,
    }
    feedback_center = {
        "high_performer": 84,
        "developing": 72,
        "low_attendance": 68,
        "academic_recovery": 63,
        "placement_prep_needed": 70,
        "under_engaged": 58,
    }

    academic_base = np.array([score_by_persona(rng, p, academic_center, 8) for p in personas])
    attendance_base = np.array([score_by_persona(rng, p, attendance_center, 10) for p in personas])
    lms_base = np.array([score_by_persona(rng, p, lms_center, 11) for p in personas])
    placement_base = np.array([score_by_persona(rng, p, placement_center, 12) for p in personas])
    skill_base = np.array([score_by_persona(rng, p, skill_center, 10) for p in personas])
    feedback_base = np.array([score_by_persona(rng, p, feedback_center, 9) for p in personas])

    backlogs = np.where(
        personas == "academic_recovery",
        rng.choice([1, 2, 3, 4], size=N_STUDENTS, p=[0.28, 0.36, 0.24, 0.12]),
        rng.choice([0, 1, 2], size=N_STUDENTS, p=[0.84, 0.13, 0.03]),
    )
    backlogs = np.where(personas == "high_performer", rng.choice([0, 1], size=N_STUDENTS, p=[0.97, 0.03]), backlogs)
    cgpa = clamp((academic_base / 10) + rng.normal(0, 0.35, N_STUDENTS) - (backlogs * 0.18), 4.0, 10.0)

    academics = pd.DataFrame(
        {
            "student_id": ids,
            "cgpa": np.round(cgpa, 2),
            "semester_marks_pct": np.round(clamp(academic_base + rng.normal(0, 5, N_STUDENTS), 0, 100), 1),
            "backlogs": backlogs,
            "core_subject_score": np.round(clamp(academic_base + rng.normal(0, 7, N_STUDENTS), 0, 100), 1),
            "elective_subject_score": np.round(clamp(academic_base + rng.normal(0, 7, N_STUDENTS), 0, 100), 1),
        }
    )

    attendance = pd.DataFrame(
        {
            "student_id": ids,
            "overall_attendance_pct": np.round(attendance_base, 1),
            "math_attendance_pct": np.round(clamp(attendance_base + rng.normal(0, 6, N_STUDENTS), 0, 100), 1),
            "programming_attendance_pct": np.round(clamp(attendance_base + rng.normal(0, 7, N_STUDENTS), 0, 100), 1),
            "lab_attendance_pct": np.round(clamp(attendance_base + rng.normal(0, 6, N_STUDENTS), 0, 100), 1),
        }
    )

    lms = pd.DataFrame(
        {
            "student_id": ids,
            "lms_logins_30d": np.round(clamp((lms_base / 100) * 42 + rng.normal(0, 5, N_STUDENTS), 0, 60)).astype(int),
            "assignment_completion_pct": np.round(clamp(lms_base + rng.normal(0, 8, N_STUDENTS), 0, 100), 1),
            "quiz_average_pct": np.round(clamp(lms_base + rng.normal(0, 9, N_STUDENTS), 0, 100), 1),
            "content_completion_pct": np.round(clamp(lms_base + rng.normal(0, 8, N_STUDENTS), 0, 100), 1),
        }
    )

    engagement = pd.DataFrame(
        {
            "student_id": ids,
            "clubs_count": clamp(np.round((lms_base + skill_base) / 55 + rng.normal(0, 0.7, N_STUDENTS)), 0, 5).astype(int),
            "events_attended": clamp(np.round((lms_base + placement_base) / 11 + rng.normal(0, 2.4, N_STUDENTS)), 0, 25).astype(int),
            "hackathons": clamp(np.round((skill_base / 35) + rng.normal(0, 0.9, N_STUDENTS)), 0, 6).astype(int),
            "certifications": clamp(np.round((skill_base + placement_base) / 50 + rng.normal(0, 0.8, N_STUDENTS)), 0, 6).astype(int),
            "peer_learning_hours": np.round(clamp((lms_base / 100) * 18 + rng.normal(0, 4, N_STUDENTS), 0, 35), 1),
        }
    )

    placement = pd.DataFrame(
        {
            "student_id": ids,
            "aptitude_score": np.round(clamp(placement_base + rng.normal(0, 8, N_STUDENTS), 0, 100), 1),
            "coding_score": np.round(clamp(placement_base + rng.normal(0, 10, N_STUDENTS) + (year - 2) * 2, 0, 100), 1),
            "mock_interview_score": np.round(clamp(placement_base + rng.normal(0, 9, N_STUDENTS), 0, 100), 1),
            "resume_score": np.round(clamp(placement_base + rng.normal(0, 7, N_STUDENTS), 0, 100), 1),
        }
    )

    skills = pd.DataFrame(
        {
            "student_id": ids,
            "technical_skill_score": np.round(clamp(skill_base + rng.normal(0, 9, N_STUDENTS), 0, 100), 1),
            "soft_skill_score": np.round(clamp(skill_base + rng.normal(0, 8, N_STUDENTS), 0, 100), 1),
            "communication_score": np.round(clamp(skill_base + rng.normal(0, 8, N_STUDENTS), 0, 100), 1),
            "teamwork_score": np.round(clamp(skill_base + rng.normal(0, 7, N_STUDENTS), 0, 100), 1),
        }
    )

    feedback = pd.DataFrame(
        {
            "student_id": ids,
            "student_satisfaction_score": np.round(clamp(feedback_base + rng.normal(0, 8, N_STUDENTS), 0, 100), 1),
            "faculty_feedback_score": np.round(clamp(feedback_base + rng.normal(0, 8, N_STUDENTS), 0, 100), 1),
            "support_requests": clamp(np.round((100 - feedback_base) / 18 + rng.normal(0, 1.1, N_STUDENTS)), 0, 8).astype(int),
            "sentiment_score": np.round(clamp(feedback_base + rng.normal(0, 10, N_STUDENTS), 0, 100), 1),
        }
    )

    # Guaranteed showcase cases for demos and tests.
    attendance.loc[attendance["student_id"] == "STU0007", "overall_attendance_pct"] = 42.0
    attendance.loc[attendance["student_id"] == "STU0007", ["math_attendance_pct", "programming_attendance_pct", "lab_attendance_pct"]] = [45.0, 39.0, 43.0]
    academics.loc[academics["student_id"] == "STU0007", ["cgpa", "semester_marks_pct", "backlogs", "core_subject_score", "elective_subject_score"]] = [9.2, 91.0, 0, 94.0, 90.0]
    lms.loc[lms["student_id"] == "STU0007", ["lms_logins_30d", "assignment_completion_pct", "quiz_average_pct", "content_completion_pct"]] = [47, 93.0, 89.0, 95.0]
    placement.loc[placement["student_id"] == "STU0007", ["aptitude_score", "coding_score", "mock_interview_score", "resume_score"]] = [90.0, 92.0, 87.0, 91.0]
    skills.loc[skills["student_id"] == "STU0007", ["technical_skill_score", "soft_skill_score", "communication_score", "teamwork_score"]] = [93.0, 87.0, 88.0, 90.0]
    feedback.loc[feedback["student_id"] == "STU0007", ["student_satisfaction_score", "faculty_feedback_score", "support_requests", "sentiment_score"]] = [86.0, 88.0, 1, 84.0]

    academics.loc[academics["student_id"] == "STU0013", ["cgpa", "semester_marks_pct", "backlogs", "core_subject_score", "elective_subject_score"]] = [5.1, 49.0, 4, 45.0, 50.0]
    placement.loc[placement["student_id"] == "STU0021", ["aptitude_score", "coding_score", "mock_interview_score", "resume_score"]] = [38.0, 31.0, 42.0, 44.0]
    lms.loc[lms["student_id"] == "STU0034", ["lms_logins_30d", "assignment_completion_pct", "quiz_average_pct", "content_completion_pct"]] = [4, 28.0, 41.0, 32.0]

    # Realistic missingness for normal demo data.
    for frame, columns in [
        (academics, ["semester_marks_pct", "core_subject_score"]),
        (attendance, ["math_attendance_pct"]),
        (lms, ["quiz_average_pct"]),
        (placement, ["mock_interview_score"]),
        (skills, ["communication_score"]),
        (feedback, ["student_satisfaction_score"]),
    ]:
        mask = rng.random(N_STUDENTS) < 0.015
        frame.loc[mask, columns[0]] = np.nan

    students = students.drop(columns=["persona_seed"])
    return {
        "sample_students.csv": students,
        "sample_academics.csv": academics,
        "sample_attendance.csv": attendance,
        "sample_lms.csv": lms,
        "sample_engagement.csv": engagement,
        "sample_placement.csv": placement,
        "sample_skills.csv": skills,
        "sample_feedback.csv": feedback,
    }


def write_quality_fixtures(sources: dict[str, pd.DataFrame]) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    dirty_academics = pd.concat(
        [
            sources["sample_academics.csv"].head(8),
            sources["sample_academics.csv"].iloc[[0]],
            pd.DataFrame(
                [
                    {
                        "student_id": "STU9999",
                        "cgpa": 12.7,
                        "semester_marks_pct": -8,
                        "backlogs": "many",
                        "core_subject_score": 140,
                        "elective_subject_score": None,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    dirty_attendance = pd.concat(
        [
            sources["sample_attendance.csv"].head(8),
            pd.DataFrame(
                [
                    {
                        "student_id": "STU8888",
                        "overall_attendance_pct": 152,
                        "math_attendance_pct": "absent",
                        "programming_attendance_pct": -5,
                        "lab_attendance_pct": 60,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    dirty_lms = pd.concat(
        [
            sources["sample_lms.csv"].head(8),
            pd.DataFrame(
                [
                    {
                        "student_id": "STU7777",
                        "lms_logins_30d": -2,
                        "assignment_completion_pct": 130,
                        "quiz_average_pct": "missing",
                        "content_completion_pct": 55,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    dirty_academics.to_csv(FIXTURE_DIR / "dirty_academics.csv", index=False)
    dirty_attendance.to_csv(FIXTURE_DIR / "dirty_attendance.csv", index=False)
    dirty_lms.to_csv(FIXTURE_DIR / "dirty_lms.csv", index=False)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    students = build_students(rng)
    sources = build_sources(students, rng)

    for filename, frame in sources.items():
        frame.to_csv(DATA_DIR / filename, index=False)

    write_quality_fixtures(sources)

    summary = pd.DataFrame(
        {
            "file": list(sources.keys()),
            "records": [len(frame) for frame in sources.values()],
            "columns": [len(frame.columns) for frame in sources.values()],
        }
    )
    summary.to_csv(DATA_DIR / "sample_data_manifest.csv", index=False)
    print(f"Generated {len(students)} synthetic students across {len(sources)} CSV sources in {DATA_DIR}")


if __name__ == "__main__":
    main()

