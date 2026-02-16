"""General utilities for evaluation."""

import re
from typing import Any, Dict, List, Optional


def slugify_case_name(name: str) -> str:
    """Convert a case name to a slug format (URL-friendly string)."""
    base = name or "case"
    slug = re.sub(r"[^a-z0-9_-]+", "-", base.lower()).strip("-")
    return slug or "case"


def normalize_case_selector(selector: Any) -> Optional[List[int]]:
    """Normalize case selector to a list of integers."""
    if selector is None:
        return None
    if isinstance(selector, int):
        return [selector]
    if isinstance(selector, list):
        normalized: List[int] = []
        for item in selector:
            try:
                normalized.append(int(item))
            except (TypeError, ValueError):
                print(f"⚠ Ignoring invalid case selector entry : {item}")
        return normalized or None
    try:
        value = int(selector)
        return [value]
    except (TypeError, ValueError):
        print(f"⚠ Unsupported case selector type: {selector}")
        return None


def select_cases(gt_cases: List[Dict[str, Any]], selector: Optional[List[int]]) -> List[Dict[str, Any]]:
    """Select specific test cases based on selector indices."""
    if not selector:
        return gt_cases
    selected: List[Dict[str, Any]] = []
    total = len(gt_cases)
    for idx in selector:
        if idx < 1 or idx > total:
            print(f"⚠ Case index {idx} is out of range (1-{total}). Skipping.")
            continue
        selected.append(gt_cases[idx - 1])
    return selected
