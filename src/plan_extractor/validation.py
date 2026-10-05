from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .models import EvaluationMetrics


@dataclass(frozen=True)
class ExpectedMeasurement:
    type: str
    value_mm: float
    secondary_value_mm: float | None
    element_name: str | None


def load_ground_truth(path: str | Path) -> list[ExpectedMeasurement]:
    """Load expected measurements from a ground-truth JSON file.
    Each record becomes a typed value used for extraction accuracy evaluation.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for item in data.get("measurements", []):
        out.append(
            ExpectedMeasurement(
                type=item.get("type", "dimension"),
                value_mm=float(item["value_mm"]),
                secondary_value_mm=(
                    float(item["secondary_value_mm"])
                    if item.get("secondary_value_mm") is not None
                    else None
                ),
                element_name=item.get("element_name"),
            )
        )
    return out


def _close(a: float | None, b: float | None, tolerance: float) -> bool:
    """Return whether two optional numeric values are within a tolerance."""
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tolerance


# Common unit-conversion factors. If a value is off by roughly one of
# these ratios, it's more likely a unit-conversion bug than a bad reading.
_UNIT_RATIOS = [1000.0, 1 / 1000.0, 25.4, 1 / 25.4, 304.8, 1 / 304.8]


def _looks_like_unit_mismatch(
    expected_val: float | None,
    actual_val: float | None,
    ratio_tolerance: float = 0.03,
) -> bool:
    """Detect whether two values differ by a common unit-conversion ratio."""
    if expected_val is None or actual_val is None or actual_val == 0:
        return False
    ratio = expected_val / actual_val
    for r in _UNIT_RATIOS:
        if abs(ratio - r) / r <= ratio_tolerance:
            return True
    return False


def evaluate_measurements(
    expected,
    actual,
    tolerance_mm: float = 5.0,
    duplicate_warning_count: int = 0,
) -> EvaluationMetrics:
    """Compare extracted measurements with expected values and calculate metrics."""
    used = set()
    correct = 0
    incorrect_value = 0
    incorrect_unit = 0
    missed = 0

    for exp in expected:
        candidate_idx = None
        # Prefer matching by element name first — it's a more reliable
        # identity than the value when two rooms share dimensions.
        if exp.element_name:
            named_candidates = [
                (i, act)
                for i, act in enumerate(actual)
                if i not in used
                and act.associated_element.name
                and act.associated_element.name.lower() == exp.element_name.lower()
            ]

            def distance(candidate):
                """Score a candidate by how closely it matches the expected room measurement."""
                _, act = candidate
                primary_distance = (
                    abs(exp.value_mm - act.normalized_value_mm)
                    if act.normalized_value_mm is not None
                    else float("inf")
                )
                secondary_distance = (
                    abs(exp.secondary_value_mm - act.normalized_secondary_value_mm)
                    if exp.secondary_value_mm is not None
                    and act.normalized_secondary_value_mm is not None
                    else 0.0
                    if exp.secondary_value_mm is None
                    and act.normalized_secondary_value_mm is None
                    else float("inf")
                )
                return primary_distance + secondary_distance

            if named_candidates:
                candidate_idx, _ = min(named_candidates, key=distance)

        # Fall back to value-based matching only when no element name
        # was given in the ground truth.
        if candidate_idx is None and not exp.element_name:
            for i, act in enumerate(actual):
                if i in used:
                    continue
                if act.normalized_value_mm is None:
                    continue
                if _close(exp.value_mm, act.normalized_value_mm, tolerance_mm) and _close(
                    exp.secondary_value_mm, act.normalized_secondary_value_mm, tolerance_mm
                ):
                    candidate_idx = i
                    break

        if candidate_idx is None:
            missed += 1
            continue

        used.add(candidate_idx)
        act = actual[candidate_idx]
        same_primary = _close(exp.value_mm, act.normalized_value_mm, tolerance_mm)
        same_secondary = _close(
            exp.secondary_value_mm, act.normalized_secondary_value_mm, tolerance_mm
        )

        if same_primary and same_secondary:
            correct += 1
        elif _looks_like_unit_mismatch(exp.value_mm, act.normalized_value_mm):
            incorrect_unit += 1
        else:
            incorrect_value += 1

    expected_count = len(expected)
    actual_count = len(actual)
    unexpected = max(0, actual_count - len(used))

    precision = correct / actual_count if actual_count else 0.0
    recall = correct / expected_count if expected_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    denom = correct + incorrect_value + incorrect_unit + missed + unexpected
    accuracy = correct / denom if denom else 0.0

    return EvaluationMetrics(
        expected=expected_count,
        matched=correct,
        incorrect_value=incorrect_value,
        incorrect_unit=incorrect_unit,
        missed=missed,
        unexpected=unexpected,
        duplicate_count=duplicate_warning_count,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
    )

 #   - If the ground truth says “Bedroom” but OCR associates the dimension with no room—or a differently named room—the code won’t fall back to matching the values. It can report a miss even if the numbers are right.
#- When several extracted measurements have the same room name, it picks the closest values, even if they’re outside the 5 mm tolerance; it then classifies that chosen result as wrong.
#- If there’s no room name in the ground truth, it takes the first value match within tolerance. Results can depend on the order of the extracted measurements.
#- The matching doesn’t compare the measurement type, and unit-mismatch detection checks the primary value only.
#- Its accuracy formula is the project’s chosen definition; it should be described as a custom evaluation metric, not assumed to cover every standard meaning of “accuracy.”
#So it’s fine as an initial way to compare these sample plans, as long as you explain the limits. It does not prove the extractor handles all dimensions or all possible matching cases.