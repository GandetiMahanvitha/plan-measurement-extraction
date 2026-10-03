from plan_extractor.models import (
    AssociatedElement,
    BoundingBox,
    ConfidenceBreakdown,
    Measurement,
    Space,
    SourceReference,
)
from plan_extractor.pipeline import _merge_room_dimensions


def make_measurement(measurement_id, value, secondary=None, measurement_type="dimension"):
    return Measurement(
        measurement_id=measurement_id,
        type=measurement_type,
        raw_text=str(value),
        value=value,
        secondary_value=secondary,
        unit="mm",
        normalized_value_mm=value,
        normalized_secondary_value_mm=secondary,
        associated_element=AssociatedElement(
            element_id="SPACE-001",
            element_type="space",
            name="BEDROOM 1",
        ),
        source=SourceReference(
            raw_text=str(value),
            bounding_box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
            extraction_pass="test",
        ),
        confidence=ConfidenceBreakdown(
            ocr=0.8,
            format_validation=1,
            association=0.9,
            overall=0.85,
        ),
    )


def test_matching_dimension_lines_are_absorbed_by_room_pair():
    pair = make_measurement("DIM-0003", 3600, 3300, "room_dimension_pair")
    width = make_measurement("DIM-0001", 3600)
    height = make_measurement("DIM-0002", 3300)
    space = Space(
        space_id="SPACE-001",
        name="BEDROOM 1",
        source=pair.source,
        dimensions=["DIM-0001", "DIM-0002", "DIM-0003"],
    )

    merged = _merge_room_dimensions([width, height, pair], [space], tolerance=5)

    assert [measurement.measurement_id for measurement in merged] == ["DIM-0003"]
    assert merged[0].corroborated_by_dimension_line is True
    assert space.dimensions == ["DIM-0003"]


def test_disagreeing_dimension_line_is_retained_and_flagged():
    pair = make_measurement("DIM-0003", 3600, 3300, "room_dimension_pair")
    conflict = make_measurement("DIM-0001", 3700)

    merged = _merge_room_dimensions([pair, conflict], [], tolerance=5)

    assert {measurement.measurement_id for measurement in merged} == {"DIM-0003", "DIM-0001"}
    assert "disagrees" in (merged[0].ambiguity or merged[1].ambiguity or "")
    assert all(measurement.review_required for measurement in merged)


def test_missing_pair_is_inferred_from_horizontal_and_vertical_singles():
    width = make_measurement("DIM-0001", 5000)
    width.orientation = "horizontal"
    height = make_measurement("DIM-0002", 4200)
    height.orientation = "vertical"
    space = Space(
        space_id="SPACE-001",
        name="BEDROOM 1",
        source=width.source,
        dimensions=["DIM-0001", "DIM-0002"],
    )

    merged = _merge_room_dimensions([width, height], [space], tolerance=5)

    assert len(merged) == 1
    assert merged[0].type == "room_dimension_pair"
    assert (merged[0].value, merged[0].secondary_value) == (5000, 4200)
    assert merged[0].corroborated_by_dimension_line is True


def test_missing_leading_one_is_corrected_from_matching_pair():
    pair = make_measurement("DIM-0003", 1800, 1600, "room_dimension_pair")
    corrupted = make_measurement("DIM-0001", 600)
    corrupted.orientation = "vertical"
    space = Space(
        space_id="SPACE-001",
        name="UTILITY",
        source=corrupted.source,
        dimensions=["DIM-0001", "DIM-0003"],
    )

    merged = _merge_room_dimensions([corrupted, pair], [space], tolerance=5)

    assert len(merged) == 1
    assert merged[0].type == "room_dimension_pair"
    assert merged[0].corroborated_by_dimension_line is True
    assert merged[0].ambiguity is None


def test_single_dimension_without_pair_is_flagged_for_review():
    single = make_measurement("DIM-0001", 2400)
    single.orientation = "horizontal"
    space = Space(
        space_id="SPACE-001",
        name="BATH",
        source=single.source,
        dimensions=["DIM-0001"],
    )

    merged = _merge_room_dimensions([single], [space], tolerance=5)

    assert len(merged) == 1
    assert merged[0].type == "dimension"
    assert merged[0].review_required is True
    assert "Only one dimension found" in (merged[0].ambiguity or "")