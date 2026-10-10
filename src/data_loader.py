"""CSV loading helpers for sample data and Streamlit uploads."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .preprocessing import SOURCE_CONFIGS, merge_sources


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def identify_source_from_filename(filename: str) -> str | None:
    """Infer a known source key from a CSV file name."""
    normalized = filename.lower().replace("-", "_").replace(" ", "_")
    for source_key in SOURCE_CONFIGS:
        if source_key in normalized:
            return source_key
    if "student" in normalized or "master" in normalized:
        return "students"
    return None


def load_sample_sources(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the bundled synthetic source CSVs."""
    sources: dict[str, pd.DataFrame] = {}
    for source_key, config in SOURCE_CONFIGS.items():
        path = data_dir / config.filename
        if path.exists():
            sources[source_key] = pd.read_csv(path)
    return sources


def load_uploaded_sources(uploaded_files: list[BinaryIO]) -> tuple[dict[str, pd.DataFrame], dict[str, str], list[str]]:
    """Load CSV uploads and return known sources, file names, and ignored files."""
    sources: dict[str, pd.DataFrame] = {}
    file_names: dict[str, str] = {}
    ignored: list[str] = []

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "uploaded.csv")
        source_key = identify_source_from_filename(filename)
        if source_key is None:
            ignored.append(filename)
            continue
        frame = pd.read_csv(uploaded_file)
        sources[source_key] = frame
        file_names[source_key] = filename

    return sources, file_names, ignored


def load_integrated_sample() -> tuple[pd.DataFrame, object]:
    """Convenience loader for the bundled demo dataset."""
    sources = load_sample_sources()
    file_names = {key: SOURCE_CONFIGS[key].filename for key in sources}
    return merge_sources(sources, file_names=file_names)

