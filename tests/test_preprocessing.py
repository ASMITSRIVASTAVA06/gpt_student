from __future__ import annotations

import pandas as pd

from src.preprocessing import clean_source_frame, filter_students, merge_sources


def test_clean_source_handles_duplicates_and_invalid_values() -> None:
    raw = pd.DataFrame(
        {
            "Student ID": ["S1", "S1", "S2", ""],
            "CGPA": ["8.0", "9.0", "bad", "7.0"],
            "Semester Marks %": [80, 90, 110, 70],
            "Backlogs": [0, 0, "many", 1],
        }
    )

    cleaned, stats = clean_source_frame("academics", raw, "dirty.csv")

    assert len(cleaned) == 2
    assert stats.duplicate_id_rows == 2
    assert stats.invalid_values == 3
    assert cleaned.loc[cleaned["student_id"] == "S1", "cgpa"].iloc[0] == 8.5


def test_merge_sources_tracks_unmatched_records_without_row_multiplication() -> None:
    students = pd.DataFrame(
        {
            "student_id": ["S1", "S2"],
            "student_name": ["One", "Two"],
            "department": ["CS", "IT"],
            "year": [1, 2],
            "section": ["A", "B"],
            "advisor": ["A1", "A2"],
        }
    )
    attendance = pd.DataFrame(
        {
            "student_id": ["S1", "S3"],
            "overall_attendance_pct": [92, 81],
        }
    )

    merged, quality = merge_sources({"students": students, "attendance": attendance})

    assert len(merged) == 3
    assert merged["student_id"].is_unique
    attendance_stats = next(item for item in quality.source_stats if item.source == "attendance")
    assert attendance_stats.unmatched_records == 1
    assert attendance_stats.missing_from_source == 1


def test_filter_students_handles_empty_data() -> None:
    empty = pd.DataFrame()
    assert filter_students(empty, departments=["CS"]).empty

