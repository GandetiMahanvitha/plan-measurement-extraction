from plan_extractor.parsing import (
    is_plausible_room_label,
    parse_dimension_pair,
    parse_imperial_pair,
    parse_measurement,
    parse_scale,
    parse_single_dimension,
)


def test_metric_dimension_pair_without_unit_defaults_to_mm():
    parsed = parse_dimension_pair("3600 x 3300")
    assert parsed is not None
    assert parsed.value == 3600
    assert parsed.secondary_value == 3300
    assert parsed.unit == "mm"


def test_metric_pair_in_metres_is_normalized():
    parsed = parse_dimension_pair("3.6m x 3.3m")
    assert parsed is not None
    assert parsed.value == 3600
    assert parsed.secondary_value == 3300


def test_scale():
    parsed = parse_scale("SCALE 1:100")
    assert parsed is not None
    assert parsed.ratio == 100


def test_scale_with_period_separator():
    parsed = parse_scale("SCALE 1.100")
    assert parsed is not None
    assert parsed.ratio == 100


def test_scale_with_leading_ocr_punctuation():
    parsed = parse_scale("~SCALE 1:100")
    assert parsed is not None
    assert parsed.ratio == 100


def test_imperial_dimension():
    parsed = parse_single_dimension("12'-6\"")
    assert parsed is not None
    assert round(parsed.value, 1) == 3810.0


def test_imperial_dimension_pairs():
    cases = [
        ("16'-0\" X 13'-1\"", 4876.8, 3987.8),
        ("9'-0\"x 16'-0\"", 2743.2, 4876.8),
        ("9'-7\" x 12'-5\"", 2921.0, 3784.6),
        ("12'-7\" X 10'-0\"", 3835.4, 3048.0),
        ("11'-2\" x 13'-0\"", 3403.6, 3962.4),
        ("11'-2\" X 16'-3\"", 3403.6, 4953.0),
        ("18'-0\" X 20'-0\"", 5486.4, 6096.0),
    ]

    for text, first_mm, second_mm in cases:
        parsed = parse_imperial_pair(text)
        assert parsed is not None
        assert parsed.kind == "dimension_pair"
        assert parsed.value == first_mm
        assert parsed.secondary_value == second_mm
        assert parsed.unit == "mm"
        assert parse_measurement(text) == parsed


def test_imperial_pair_normalizes_ocr_quote_variants():
    foot_marks = ["’", "‘", "`", "´", "′"]
    inch_marks = ["”", "“", "″", "''"]
    pairs = [(foot, '"') for foot in foot_marks]
    pairs.extend([("'", inch) for inch in inch_marks])

    for foot_mark, inch_mark in pairs:
        parsed = parse_imperial_pair(
            f"16{foot_mark}-0{inch_mark} X 13'-1\""
        )
        assert parsed is not None
        assert parsed.value == 4876.8
        assert parsed.secondary_value == 3987.8


def test_parenthesized_imperial_value_is_contextual():
    parsed = parse_measurement("(14'-6\")")

    assert parsed is not None
    assert parsed.kind == "context_dimension"
    assert parsed.value == 4419.6
    assert parsed.secondary_value is None
    assert parsed.unit == "mm"


def test_plausible_room_labels():
    assert is_plausible_room_label("BEDROOM 1")
    assert is_plausible_room_label("PORCH")
    assert is_plausible_room_label("LIVING / DINING")
    assert is_plausible_room_label("LAUNDRY /")


def test_implausible_room_labels():
    assert not is_plausible_room_label("T")
    assert not is_plausible_room_label("Sample Plan 2")
    assert not is_plausible_room_label("SCALE 1:100")
    assert not is_plausible_room_label("3000 x 2800")
    assert not is_plausible_room_label("A-102")
