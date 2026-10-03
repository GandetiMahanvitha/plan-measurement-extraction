from __future__ import annotations

from dataclasses import dataclass
import re

# do we need this because whatever is the room name we want the same name if 
#it is bedroom1 i want it to be seen as bedroom1 in json
# this list is limited moving forward we can add more keywords to this list as we encounter more room types in the plans
ROOM_KEYWORDS = {
    "bedroom": "bedroom",
    "master bedroom": "bedroom",
    "guest room": "bedroom",
    "living": "living",
    "living room": "living",
    "living / dining": "living_dining",
    "living/dining": "living_dining",
    "dining": "dining",
    "dining room": "dining",
    "kitchen": "kitchen",
    "bath": "bathroom",
    "bathroom": "bathroom",
    "wc": "bathroom",
    "study": "study",
    "office": "study",
    "utility": "utility",
    "laundry": "utility",
    "store": "storage",
    "storage": "storage",
    "garage": "garage",
    "hall": "hall",
    "corridor": "corridor",
    "foyer": "foyer",
}


@dataclass(frozen=True)
class ParsedMeasurement:
    kind: str
    raw_text: str
    value: float | None = None
    secondary_value: float | None = None
    unit: str | None = None
    ratio: float | None = None


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR spacing and multiplication symbols for parser matching.
    This gives equivalent OCR variants a consistent form before regex parsing.
    """
    text = text.strip()
    text = text.replace("×", "x")
    text = re.sub(r"\s+", " ", text)
    return text

#we don't need this
def classify_room_label(text: str) -> str | None:
    """Map recognized room words to a normalized room category.
    It returns None for text that does not identify a supported room type.
    """
    lower = normalize_ocr_text(text).lower()
    # Remove simple room numbers, but keep words.
    lower_words = re.sub(r"\b\d+\b", "", lower).strip(" -/")
    for keyword, category in sorted(ROOM_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if keyword in lower_words:
            return category
    return None


def parse_scale(text: str) -> ParsedMeasurement | None:
    """Parse scale labels such as `SCALE 1:100` into a numeric ratio.
    Non-scale text and non-positive ratios are rejected with None.
    """
    clean = normalize_ocr_text(text)
    clean = re.sub(r"^[^A-Za-z0-9]*", "", clean)
    m = re.search(r"(?:scale\s*)?1\s*[:/]\s*(\d{1,4})", clean, re.I)
    if not m:
        return None
    ratio = float(m.group(1))
    if ratio <= 0:
        return None
    return ParsedMeasurement(kind="scale", raw_text=text, ratio=ratio)


def _metric_to_mm(value: float, unit: str) -> float:
    """Convert a metric value expressed in mm, cm, or m to millimeters."""
    unit = unit.lower()
    if unit == "mm":
        return value
    if unit == "cm":
        return value * 10.0
    if unit == "m":
        return value * 1000.0
    raise ValueError(unit)


def parse_dimension_pair(text: str) -> ParsedMeasurement | None:
    """Parse paired metric dimensions such as `3600 x 3300` into millimeters.
    It accepts optional metric units and returns None for invalid or implausible pairs.
    """
    clean = normalize_ocr_text(text)
    # Examples: 3600 x 3300, 3.6m x 3.3m, 3600x3300 mm
    # so mm/cm/m can be only there so there won't be any units
    m = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m)?\s*[xX]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)?",
        clean,
        re.I,
    )
    if not m:
        return None

    v1 = float(m.group(1))
    u1 = (m.group(2) or "").lower()
    v2 = float(m.group(3))
    u2 = (m.group(4) or u1 or "").lower()

    unit = u1 or u2 or None
    if unit:
        # Normalize both if units are present.
        return ParsedMeasurement(
            kind="dimension_pair",
            raw_text=text,
            value=_metric_to_mm(v1, u1 or unit),
            secondary_value=_metric_to_mm(v2, u2 or unit),
            unit="mm",
        )
#how can I assume if no unit present it is mm
  #makes an assumption like no unit present it is mm
    if 100 <= v1 <= 100000 and 100 <= v2 <= 100000:
        return ParsedMeasurement(
            kind="dimension_pair",
            raw_text=text,
            value=v1,
            secondary_value=v2,
            unit="mm",
        )
    return None

#chatgpt says this has an edge case
def _parse_fraction(whole: str, numerator: str, denominator: str) -> float:
    """Convert a whole-number fraction expression into a floating-point value."""
    try:
        whole_num = float(whole) if whole else 0.0
        return whole_num + (float(numerator) / float(denominator))
    except (ValueError, ZeroDivisionError):
        return 0.0


#not supported
#12 1/2' — fractional feet
#12.5' — decimal feet
#12'-1/2" — fractional inches without a whole-inch number
#12'-6.5" — decimal inches
#12′-6″ — typographic prime symbols instead of straight ' and "
#12 ft 6 in — spelled-out units
def parse_imperial(text: str) -> ParsedMeasurement | None:
    """Parse feet-and-inch notation and convert it to millimeters.

    Whole inches and optional fractional inches are supported after the feet marker.
    """
    clean = normalize_ocr_text(text)
    # Examples: 12'-6", 12' 6", 12', 12'-6 1/2"
    # why i want to convert all into mm
    m = re.fullmatch(
        r"(\d+)\s*'\s*-?\s*(?:(\d+)(?:\s+(\d+)\s*/\s*(\d+))?\s*\")?",
        clean,
    )
    if not m:
        return None
    feet = int(m.group(1))
    if m.group(3) and m.group(4):
        inches = _parse_fraction(m.group(2), m.group(3), m.group(4))
    else:
        inches = float(m.group(2) or 0)
    total_mm = (feet * 12 + inches) * 25.4
    return ParsedMeasurement(
        kind="dimension",
        raw_text=text,
        value=round(total_mm, 2),
        unit="mm",
    )

# 9.5 not handled
def parse_inches_only(text: str) -> ParsedMeasurement | None:
    """Parse bare inch values, including optional fractional inches, to millimeters."""
    clean = normalize_ocr_text(text)
    # Examples: 34", 34 1/2"  (bare inches, no feet)
    m = re.fullmatch(r"(\d+)(?:\s+(\d+)\s*/\s*(\d+))?\s*\"", clean)
    if not m:
        return None
    if m.group(2) and m.group(3):
        inches = _parse_fraction(m.group(1), m.group(2), m.group(3))
    else:
        inches = float(m.group(1))
    total_mm = inches * 25.4
    return ParsedMeasurement(
        kind="dimension",
        raw_text=text,
        value=round(total_mm, 2),
        unit="mm",
    )


def parse_single_dimension(text: str) -> ParsedMeasurement | None:
    """Parse one dimension in imperial, metric, or conservative bare-number form.
    All accepted values are normalized to millimeters for downstream comparison.
    """
    clean = normalize_ocr_text(text)

    imperial = parse_imperial(clean)
    if imperial:
        return imperial

    inches_only = parse_inches_only(clean)
    if inches_only:
        return inches_only

    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)", clean, re.I)
    if m:
        value = float(m.group(1))
        unit = m.group(2).lower()
        return ParsedMeasurement(
            kind="dimension",
            raw_text=text,
            value=_metric_to_mm(value, unit),
            unit="mm",
        )

    # Bare dimension numbers: deliberately conservative.
    if re.fullmatch(r"\d{3,6}", clean):
        value = float(clean)
        if 100 <= value <= 100000:
            return ParsedMeasurement(
                kind="dimension",
                raw_text=text,
                value=value,
                unit="mm",
            )
    return None


def parse_measurement(text: str) -> ParsedMeasurement | None:
    """Try scale, paired-dimension, and single-dimension parsers in priority order.
    Returning the first match gives the pipeline one uniform parsed representation.
    """
    return (
        parse_scale(text)
        or parse_dimension_pair(text)
        or parse_single_dimension(text)
    )
#Examples it doesn’t currently handle include fractional feet (12 1/2'), decimal inches (9.5"), '
#'or an inch fraction without a whole number (1/2"). So it handles a specific set of patterns, not all formats.