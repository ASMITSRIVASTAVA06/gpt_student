"""Data validation, cleaning, and source integration utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np
import pandas as pd


STUDENT_ID = "student_id"


@dataclass(frozen=True)
class SourceConfig:
    """Schema and validation rules for a source CSV."""

    key: str
    filename: str
    display_name: str
    required_columns: tuple[str, ...]
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class SourceQuality:
    """Quality statistics captured while processing one source."""

    source: str
    display_name: str
    file_name: str
    input_rows: int
    processed_rows: int = 0
    missing_values: int = 0
    duplicate_id_rows: int = 0
    invalid_values: int = 0
    unmatched_records: int = 0
    missing_from_source: int = 0
    warnings: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass
class IntegrationQuality:
    """Complete quality report for an integration run."""

    source_stats: list[SourceQuality] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    missing_by_column: dict[str, int] = field(default_factory=dict)
    input_records_total: int = 0
    processed_records: int = 0
    uploaded_file_names: list[str] = field(default_factory=list)
    synthetic_demo_data: bool = False

    def source_stats_frame(self) -> pd.DataFrame:
        rows = [
            {
                "Source": item.display_name,
                "File": item.file_name,
                "Input rows": item.input_rows,
                "Processed rows": item.processed_rows,
                "Duplicate ID rows": item.duplicate_id_rows,
                "Invalid values": item.invalid_values,
                "Missing values": item.missing_values,
                "Unmatched records": item.unmatched_records,
                "Students missing source": item.missing_from_source,
            }
            for item in self.source_stats
        ]
        return pd.DataFrame(rows)

    def missing_frame(self) -> pd.DataFrame:
        if not self.missing_by_column:
            return pd.DataFrame(columns=["Column", "Missing values"])
        return (
            pd.DataFrame(
                [{"Column": column, "Missing values": count} for column, count in self.missing_by_column.items()]
            )
            .sort_values(["Missing values", "Column"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def warning_frame(self) -> pd.DataFrame:
        rows = [{"Warning": warning} for warning in self.warnings]
        return pd.DataFrame(rows, columns=["Warning"])


SOURCE_CONFIGS: dict[str, SourceConfig] = {
    "students": SourceConfig(
        key="students",
        filename="sample_students.csv",
        display_name="Student master",
        required_columns=(
            STUDENT_ID,
            "student_name",
            "department",
            "year",
            "section",
            "advisor",
        ),
        numeric_ranges={"year": (1, 6)},
        aliases={"id": STUDENT_ID, "studentid": STUDENT_ID, "student": STUDENT_ID},
    ),
    "academics": SourceConfig(
        key="academics",
        filename="sample_academics.csv",
        display_name="Academic records",
        required_columns=(STUDENT_ID, "cgpa", "semester_marks_pct", "backlogs"),
        numeric_ranges={
            "cgpa": (0, 10),
            "semester_marks_pct": (0, 100),
            "backlogs": (0, 20),
            "core_subject_score": (0, 100),
            "elective_subject_score": (0, 100),
        },
        aliases={"marks": "semester_marks_pct", "semester_marks": "semester_marks_pct"},
    ),
    "attendance": SourceConfig(
        key="attendance",
        filename="sample_attendance.csv",
        display_name="Attendance",
        required_columns=(STUDENT_ID, "overall_attendance_pct"),
        numeric_ranges={
            "overall_attendance_pct": (0, 100),
            "math_attendance_pct": (0, 100),
            "programming_attendance_pct": (0, 100),
            "lab_attendance_pct": (0, 100),
        },
        aliases={"attendance": "overall_attendance_pct", "overall_attendance": "overall_attendance_pct"},
    ),
    "lms": SourceConfig(
        key="lms",
        filename="sample_lms.csv",
        display_name="Learning management system",
        required_columns=(STUDENT_ID, "assignment_completion_pct"),
        numeric_ranges={
            "lms_logins_30d": (0, 200),
            "assignment_completion_pct": (0, 100),
            "quiz_average_pct": (0, 100),
            "content_completion_pct": (0, 100),
        },
    ),
    "engagement": SourceConfig(
        key="engagement",
        filename="sample_engagement.csv",
        display_name="Engagement",
        required_columns=(STUDENT_ID,),
        numeric_ranges={
            "clubs_count": (0, 20),
            "events_attended": (0, 100),
            "hackathons": (0, 30),
            "certifications": (0, 40),
            "peer_learning_hours": (0, 200),
        },
    ),
    "placement": SourceConfig(
        key="placement",
        filename="sample_placement.csv",
        display_name="Placement preparation",
        required_columns=(STUDENT_ID,),
        numeric_ranges={
            "aptitude_score": (0, 100),
            "coding_score": (0, 100),
            "mock_interview_score": (0, 100),
            "resume_score": (0, 100),
        },
    ),
    "skills": SourceConfig(
        key="skills",
        filename="sample_skills.csv",
        display_name="Skills",
        required_columns=(STUDENT_ID,),
        numeric_ranges={
            "technical_skill_score": (0, 100),
            "soft_skill_score": (0, 100),
            "communication_score": (0, 100),
            "teamwork_score": (0, 100),
        },
    ),
    "feedback": SourceConfig(
        key="feedback",
        filename="sample_feedback.csv",
        display_name="Feedback",
        required_columns=(STUDENT_ID,),
        numeric_ranges={
            "student_satisfaction_score": (0, 100),
            "faculty_feedback_score": (0, 100),
            "support_requests": (0, 50),
            "sentiment_score": (0, 100),
        },
    ),
}


def normalize_column_name(column: Any) -> str:
    """Convert a source column into a predictable snake_case name."""
    value = str(column).strip().lower()
    value = re.sub(r"[%]+", " pct ", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def normalize_columns(df: pd.DataFrame, config: SourceConfig) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [normalize_column_name(col) for col in normalized.columns]
    rename_map = {normalize_column_name(key): value for key, value in config.aliases.items()}
    normalized = normalized.rename(columns=rename_map)
    return normalized


def _aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, str] = {}
    for column in df.columns:
        if column == STUDENT_ID:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            aggregations[column] = "mean"
        else:
            aggregations[column] = "first"
    return df.groupby(STUDENT_ID, as_index=False).agg(aggregations)


def clean_source_frame(
    source_key: str,
    df: pd.DataFrame,
    file_name: str | None = None,
) -> tuple[pd.DataFrame, SourceQuality]:
    """Normalize, validate, clean, and deduplicate a single source frame."""
    if source_key not in SOURCE_CONFIGS:
        raise KeyError(f"Unknown source: {source_key}")

    config = SOURCE_CONFIGS[source_key]
    cleaned = normalize_columns(df, config)
    stats = SourceQuality(
        source=source_key,
        display_name=config.display_name,
        file_name=file_name or config.filename,
        input_rows=len(cleaned),
    )

    if cleaned.empty:
        stats.warnings.append(f"{config.display_name} is empty.")
        return cleaned, stats

    missing_columns = [column for column in config.required_columns if column not in cleaned.columns]
    if missing_columns:
        stats.warnings.append(
            f"{config.display_name} is missing required columns: {', '.join(missing_columns)}."
        )
        for column in missing_columns:
            cleaned[column] = np.nan

    if STUDENT_ID not in cleaned.columns:
        stats.warnings.append(f"{config.display_name} cannot be merged because student_id is absent.")
        return pd.DataFrame(columns=[STUDENT_ID]), stats

    cleaned[STUDENT_ID] = cleaned[STUDENT_ID].astype("string").str.strip().str.upper()
    blank_ids = int(cleaned[STUDENT_ID].isna().sum() + (cleaned[STUDENT_ID] == "").sum())
    if blank_ids:
        stats.warnings.append(f"{blank_ids} rows in {config.display_name} had blank student IDs and were removed.")
        stats.actions.append("Removed rows with blank student IDs.")
        cleaned = cleaned[cleaned[STUDENT_ID].notna() & (cleaned[STUDENT_ID] != "")]

    for column, (minimum, maximum) in config.numeric_ranges.items():
        if column not in cleaned.columns:
            continue
        before_non_null = cleaned[column].notna().sum()
        numeric = pd.to_numeric(cleaned[column], errors="coerce")
        invalid_type_count = int(before_non_null - numeric.notna().sum())
        out_of_range = numeric.notna() & ((numeric < minimum) | (numeric > maximum))
        out_of_range_count = int(out_of_range.sum())
        if invalid_type_count:
            stats.warnings.append(
                f"{invalid_type_count} non-numeric values in {config.display_name}.{column} were converted to missing."
            )
        if out_of_range_count:
            stats.warnings.append(
                f"{out_of_range_count} out-of-range values in {config.display_name}.{column} were converted to missing."
            )
        stats.invalid_values += invalid_type_count + out_of_range_count
        numeric.loc[out_of_range] = np.nan
        cleaned[column] = numeric

    duplicate_mask = cleaned.duplicated(STUDENT_ID, keep=False)
    if duplicate_mask.any():
        duplicate_rows = int(duplicate_mask.sum())
        stats.duplicate_id_rows = duplicate_rows
        stats.warnings.append(
            f"{duplicate_rows} duplicate student ID rows found in {config.display_name}; values were aggregated by ID."
        )
        stats.actions.append("Aggregated duplicate student IDs using numeric means and first categorical values.")
        cleaned = _aggregate_duplicates(cleaned)

    stats.processed_rows = len(cleaned)
    stats.missing_values = int(cleaned.isna().sum().sum())
    return cleaned.reset_index(drop=True), stats


def _presence_column(source_key: str) -> str:
    return f"has_{source_key}_source"


def merge_sources(sources: dict[str, pd.DataFrame], file_names: dict[str, str] | None = None) -> tuple[pd.DataFrame, IntegrationQuality]:
    """Clean and outer-merge source dataframes on student_id without row multiplication."""
    file_names = file_names or {}
    quality = IntegrationQuality(uploaded_file_names=list(file_names.values()))

    cleaned_sources: dict[str, pd.DataFrame] = {}
    for source_key, frame in sources.items():
        cleaned, stats = clean_source_frame(source_key, frame, file_names.get(source_key))
        cleaned[_presence_column(source_key)] = True
        cleaned_sources[source_key] = cleaned
        quality.source_stats.append(stats)
        quality.warnings.extend(stats.warnings)
        quality.actions.extend(stats.actions)
        quality.input_records_total += stats.input_rows

    if not cleaned_sources:
        quality.warnings.append("No source files were available for integration.")
        return pd.DataFrame(), quality

    source_order = [key for key in SOURCE_CONFIGS if key in cleaned_sources]
    merged = cleaned_sources[source_order[0]].copy()
    base_ids = set(cleaned_sources.get("students", merged)[STUDENT_ID].dropna().astype(str))

    for source_key in source_order[1:]:
        source = cleaned_sources[source_key]
        if source_key != "students":
            source_ids = set(source[STUDENT_ID].dropna().astype(str))
            unmatched = source_ids - base_ids if base_ids else set()
            missing_from_source = base_ids - source_ids if base_ids else set()
            stat = next(item for item in quality.source_stats if item.source == source_key)
            stat.unmatched_records = len(unmatched)
            stat.missing_from_source = len(missing_from_source)
            if unmatched:
                message = f"{SOURCE_CONFIGS[source_key].display_name} has {len(unmatched)} IDs not found in the student master."
                quality.warnings.append(message)
                stat.warnings.append(message)
            if missing_from_source:
                stat.actions.append(
                    f"{len(missing_from_source)} student master IDs had no {SOURCE_CONFIGS[source_key].display_name} record."
                )
        merged = merged.merge(source, on=STUDENT_ID, how="outer", validate="one_to_one")

    for source_key in source_order:
        column = _presence_column(source_key)
        if column not in merged.columns:
            merged[column] = False
        merged[column] = merged[column].fillna(False).astype(bool)

    if "student_name" not in merged.columns:
        merged["student_name"] = merged[STUDENT_ID]
    else:
        merged["student_name"] = merged["student_name"].fillna(merged[STUDENT_ID])

    for column in ("department", "section", "advisor"):
        if column not in merged.columns:
            merged[column] = "Unknown"
        merged[column] = merged[column].fillna("Unknown")

    if "year" not in merged.columns:
        merged["year"] = np.nan

    if "synthetic_demo_record" in merged.columns:
        merged["synthetic_demo_record"] = merged["synthetic_demo_record"].fillna(True).astype(bool)
        quality.synthetic_demo_data = bool(merged["synthetic_demo_record"].all())
    else:
        merged["synthetic_demo_record"] = False

    quality.processed_records = len(merged)
    quality.missing_by_column = {
        column: int(count)
        for column, count in merged.isna().sum().items()
        if int(count) > 0 and not column.startswith("has_")
    }
    return merged.sort_values(STUDENT_ID).reset_index(drop=True), quality


def filter_students(
    df: pd.DataFrame,
    departments: list[str] | None = None,
    years: list[int] | None = None,
    risk_levels: list[str] | None = None,
    search_text: str = "",
) -> pd.DataFrame:
    """Apply common dashboard filters while safely handling empty inputs."""
    if df.empty:
        return df.copy()

    filtered = df.copy()
    if departments:
        filtered = filtered[filtered["department"].isin(departments)]
    if years:
        filtered = filtered[filtered["year"].isin(years)]
    if risk_levels and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"].isin(risk_levels)]
    if search_text:
        query = search_text.strip().lower()
        if query:
            searchable = (
                filtered[STUDENT_ID].astype(str).str.lower()
                + " "
                + filtered["student_name"].astype(str).str.lower()
                + " "
                + filtered["department"].astype(str).str.lower()
            )
            filtered = filtered[searchable.str.contains(re.escape(query), na=False)]
    return filtered.reset_index(drop=True)

