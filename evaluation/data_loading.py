"""Data loading utilities for evaluation."""

import yaml
import csv
from typing import Any, Dict


def load_params() -> Dict[str, Any]:
    """Load parameters from params.yaml."""
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_text(path: str) -> str:
    """Load text content from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_ground_truth_csv(path: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth CSV into a dict keyed by Mapped_ID."""
    gt: Dict[str, Dict[str, Any]] = {}

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mapped_id = row.get("Mapped_ID") or row.get("Mapped ID")
                    if not mapped_id:
                        continue
                    gt[mapped_id] = row
            return gt
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode CSV file: {path}")
