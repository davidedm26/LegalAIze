"""Data loading utilities for evaluation."""

import yaml
import csv
from typing import Any, Dict
import os
import fitz  # PyMuPDF


def load_params() -> Dict[str, Any]:
    """Load parameters from params.yaml."""
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def load_text(path: str) -> str:
    """Load text content from a file. Supports .txt and .pdf files using PyMuPDF."""
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".pdf":
        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def load_ground_truth_csv(path: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth CSV into a dict keyed by Requirement_ID."""
    gt: Dict[str, Dict[str, Any]] = {}

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    requirement_id = row.get("Requirement_ID") or row.get("Mapped ID")
                    if not requirement_id:
                        continue
                    gt[requirement_id] = row
            return gt
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode CSV file: {path}")
