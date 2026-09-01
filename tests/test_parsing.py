from plan_extractor.parsing import (
    classify_room_label,
    parse_dimension_pair,
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


def test_scale_with_leading_ocr_punctuation():
    parsed = parse_scale("~SCALE 1:100")
    assert parsed is not None
    assert parsed.ratio == 100


def test_imperial_dimension():
    parsed = parse_single_dimension("12'-6\"")
    assert parsed is not None
    assert round(parsed.value, 1) == 3810.0


def test_room_label():
    assert classify_room_label("BEDROOM 1") == "bedroom"
    assert classify_room_label("LIVING / DINING") == "living_dining"
    assert classify_room_label("A-102") is None
