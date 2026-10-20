from plan_extractor.models import (
    AssociatedElement,
    BoundingBox,
    ConfidenceBreakdown,
    Measurement,
    Space,
    SourceReference,
    WarningItem,
)
from plan_extractor.pipeline import (
    _apply_review_flags,
    _unparsed_dimension_warnings,
    _merge_room_dimensions,
    _result_status,
    _rooms_without_dimensions_warning,
    _summary_review_required_count,
)
from plan_extractor.ocr_engine import OCRDetection


def make_measurement(
    measurement_id,
    value,
    secondary=None,
    measurement_type="dimension",
    orientation="horizontal",
    raw_text=None,
    box=None,
):
    bounding_box = box or (
        BoundingBox(x_min=0, y_min=0, x_max=2, y_max=1)
        if orientation == "horizontal"
        else BoundingBox(x_min=0, y_min=0, x_max=1, y_max=2)
    )
    return Measurement(
        measurement_id=measurement_id,
        type=measurement_type,
        raw_text=raw_text or str(value),
        value=value,
        secondary_value=secondary,
        unit="mm",
        associated_element=AssociatedElement(
            element_id="SPACE-001",
            name="BEDROOM 1",
        ),
        source=SourceReference(
            raw_text=raw_text or str(value),
            bounding_box=bounding_box,
        ),
        confidence=ConfidenceBreakdown(
            ocr=0.8,
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
    assert merged[0].confidence.overall == 0.9
    assert space.dimensions == ["DIM-0003"]


def test_disagreeing_dimension_line_is_retained_and_flagged():
    pair = make_measurement("DIM-0003", 3600, 3300, "room_dimension_pair")
    conflict = make_measurement("DIM-0001", 3700)

    merged = _merge_room_dimensions([pair, conflict], [], tolerance=5)

    assert {measurement.measurement_id for measurement in merged} == {"DIM-0003", "DIM-0001"}
    assert "disagrees" in (merged[0].ambiguity or merged[1].ambiguity or "")
    assert all(measurement.review_required for measurement in merged)


def test_missing_pair_is_inferred_from_horizontal_and_vertical_singles():
    width = make_measurement(
        "DIM-0001",
        5000,
        orientation="horizontal",
        raw_text="5000",
        box=BoundingBox(x_min=10, y_min=20, x_max=30, y_max=25),
    )
    height = make_measurement(
        "DIM-0002",
        4200,
        orientation="vertical",
        raw_text="4200",
        box=BoundingBox(x_min=15, y_min=5, x_max=20, y_max=40),
    )
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
    assert (
        merged[0].ambiguity
        == "The printed pair was not read. Width and height were taken from two separate dimension-line numbers (5000 and 4200). Please check."
    )
    assert merged[0].review_required is True
    assert merged[0].raw_text == "5000 + 4200"
    assert merged[0].source.raw_text == "5000 + 4200"
    assert merged[0].source.bounding_box == BoundingBox(
        x_min=10, y_min=5, x_max=30, y_max=40
    )
    assert [
        (source.raw_text, source.bounding_box, source.ocr)
        for source in merged[0].component_sources
    ] == [
        (
            "5000",
            BoundingBox(x_min=10, y_min=20, x_max=30, y_max=25),
            0.8,
        ),
        (
            "4200",
            BoundingBox(x_min=15, y_min=5, x_max=20, y_max=40),
            0.8,
        ),
    ]


def test_rebuilt_pair_uses_honest_confidence_and_component_ocr_values():
    width = make_measurement(
        "DIM-0001",
        5000,
        orientation="horizontal",
        raw_text="5000",
        box=BoundingBox(x_min=10, y_min=20, x_max=30, y_max=25),
    )
    width.confidence = ConfidenceBreakdown(ocr=0.9, association=0.9, overall=0.9)
    height = make_measurement(
        "DIM-0002",
        4200,
        orientation="vertical",
        raw_text="4200",
        box=BoundingBox(x_min=15, y_min=5, x_max=20, y_max=40),
    )
    height.confidence = ConfidenceBreakdown(ocr=0.6, association=0.8, overall=0.7)
    space = Space(
        space_id="SPACE-001",
        name="BEDROOM 1",
        source=width.source,
        dimensions=["DIM-0001", "DIM-0002"],
    )

    merged = _merge_room_dimensions([width, height], [space], tolerance=5)

    assert len(merged) == 1
    assert merged[0].confidence.ocr == 0.75
    assert merged[0].confidence.association == 0.85
    assert merged[0].confidence.overall == 0.79
    assert merged[0].component_sources[0].ocr == 0.9
    assert merged[0].component_sources[1].ocr == 0.6
    assert merged[0].ambiguity == (
        "The printed pair was not read. Width and height were taken from two separate dimension-line numbers "
        "(5000 and 4200). Please check."
    )


def test_printed_pair_has_no_component_sources():
    pair = make_measurement(
        "DIM-0003",
        3600,
        3300,
        "room_dimension_pair",
        raw_text="3600 x 3300",
    )
    width = make_measurement("DIM-0001", 3600)
    height = make_measurement("DIM-0002", 3300, orientation="vertical")

    merged = _merge_room_dimensions([width, height, pair], [], tolerance=5)

    assert merged == [pair]
    assert pair.component_sources == []


def test_context_dimension_does_not_conflict_with_main_pair():
    pair = make_measurement("DIM-0003", 3900, 3600, "room_dimension_pair")
    context = make_measurement(
        "DIM-0004",
        4419.6,
        measurement_type="context_dimension",
        raw_text="(14'-6\")",
    )
    context.review_required = True
    context.ambiguity = "Parenthesized value; its meaning is unclear."

    merged = _merge_room_dimensions([pair, context], [], tolerance=5)

    assert merged == [pair, context]
    assert pair.ambiguity is None
    assert context.review_required is True


def test_empty_extraction_warns_for_undimensioned_rooms_and_counts_review():
    source = SourceReference(
        raw_text="KITCHEN",
        bounding_box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=5),
    )
    spaces = [
        Space(space_id="SPACE-001", name="KITCHEN", source=source),
        Space(space_id="SPACE-002", name="PANTRY", source=source),
    ]

    warning = _rooms_without_dimensions_warning(spaces)

    assert warning is not None
    assert warning.code == "ROOMS_WITHOUT_DIMENSIONS"
    assert "KITCHEN, PANTRY" in warning.message
    assert "Likely reason:" in warning.message
    assert _summary_review_required_count([], spaces) == 2


def test_unparsed_dimension_text_warnings_include_ocr_evidence():
    detections = [
        OCRDetection(
            text='614\'-6"1',
            confidence=0.68,
            bbox=((10, 20), (110, 20), (110, 40), (10, 40)),
            extraction_pass="dimension",
        ),
        OCRDetection(
            text="12'-6",
            confidence=0.72,
            bbox=((120, 20), (180, 20), (180, 40), (120, 40)),
            extraction_pass="dimension",
        ),
        OCRDetection(
            text="1",
            confidence=0.90,
            bbox=((200, 20), (210, 20), (210, 40), (200, 40)),
            extraction_pass="dimension",
        ),
        OCRDetection(
            text="8",
            confidence=0.90,
            bbox=((220, 20), (230, 20), (230, 40), (220, 40)),
            extraction_pass="dimension",
        ),
        OCRDetection(
            text="UP",
            confidence=0.90,
            bbox=((240, 20), (270, 20), (270, 40), (240, 40)),
            extraction_pass="general",
        ),
        OCRDetection(
            text="DN",
            confidence=0.90,
            bbox=((280, 20), (310, 20), (310, 40), (280, 40)),
            extraction_pass="general",
        ),
        OCRDetection(
            text='16\'-0" X 13\'-1"',
            confidence=0.91,
            bbox=((320, 20), (420, 20), (420, 40), (320, 40)),
            extraction_pass="dimension",
        ),
    ]

    warnings = _unparsed_dimension_warnings(detections, {})

    assert len(warnings) == 2
    assert all(warning.code == "UNPARSED_DIMENSION_TEXT" for warning in warnings)
    assert "614'-6\"1" in warnings[0].message
    assert "bounding box=" in warnings[0].message
    assert "OCR confidence=0.6800" in warnings[0].message
    assert "12'-6" in warnings[1].message


def test_unparsed_dimension_warning_suppressed_when_overlapping_accepted_value():
    garbled_reading = OCRDetection(
        text='614\'-6"1',
        confidence=0.68,
        bbox=((131, 860), (221, 860), (221, 892), (131, 892)),
        extraction_pass="dimension",
    )
    accepted = make_measurement(
        "DIM-0005",
        4419.6,
        measurement_type="context_dimension",
        raw_text='(14\'-6")',
        box=BoundingBox(x_min=129, y_min=859, x_max=223, y_max=895),
    )

    warnings = _unparsed_dimension_warnings(
        [garbled_reading],
        {"warnings": {"suppress_overlap_threshold": 0.5}},
        [accepted],
    )

    assert warnings == []


def test_unparsed_dimension_warning_remains_when_not_overlapping():
    garbled_reading = OCRDetection(
        text='614\'-6"1',
        confidence=0.68,
        bbox=((300, 400), (390, 400), (390, 432), (300, 432)),
        extraction_pass="dimension",
    )
    accepted = make_measurement(
        "DIM-0005",
        4419.6,
        measurement_type="context_dimension",
        raw_text='(14\'-6")',
        box=BoundingBox(x_min=129, y_min=859, x_max=223, y_max=895),
    )

    warnings = _unparsed_dimension_warnings(
        [garbled_reading],
        {"warnings": {"suppress_overlap_threshold": 0.5}},
        [accepted],
    )

    assert len(warnings) == 1
    assert warnings[0].code == "UNPARSED_DIMENSION_TEXT"
    assert '614\'-6"1' in warnings[0].message


def test_unparsed_imperial_value_warning_remains_without_accepted_overlap():
    detection = OCRDetection(
        text="12'-6",
        confidence=0.72,
        bbox=((10, 20), (70, 20), (70, 40), (10, 40)),
        extraction_pass="dimension",
    )

    warnings = _unparsed_dimension_warnings([detection], {})

    assert len(warnings) == 1
    assert warnings[0].code == "UNPARSED_DIMENSION_TEXT"


def test_unparsed_dimension_warnings_are_counted_for_review():
    warning = WarningItem(
        code="UNPARSED_DIMENSION_TEXT",
        message="OCR dimension candidate needs review.",
    )

    assert _summary_review_required_count([], [], [warning]) == 1
    assert _result_status([], [warning]) == "partial"
    assert _result_status([], []) == "failed"


def test_missing_leading_one_is_corrected_from_matching_pair():
    pair = make_measurement("DIM-0003", 1800, 1600, "room_dimension_pair")
    corrupted = make_measurement("DIM-0001", 600, orientation="vertical")
    space = Space(
        space_id="SPACE-001",
        name="UTILITY",
        source=corrupted.source,
        dimensions=["DIM-0001", "DIM-0003"],
    )

    merged = _merge_room_dimensions([corrupted, pair], [space], tolerance=5)

    assert len(merged) == 1
    assert merged[0].type == "room_dimension_pair"
    assert merged[0].value == 1800
    assert merged[0].ambiguity is None


def test_single_dimension_without_pair_is_flagged_for_review():
    single = make_measurement("DIM-0001", 2400, orientation="horizontal")
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


def test_conflicting_room_pairs_flag_both_with_pair_values():
    first = make_measurement("DIM-1", 3300, 2800, "room_dimension_pair")
    second = make_measurement("DIM-2", 3300, 2000, "room_dimension_pair")

    _apply_review_flags(
        [first, second],
        [Space(space_id="SPACE-001", name="KITCHEN", source=first.source)],
        {"confidence": {"ocr_review_threshold": 0.50, "association_review_threshold": 0.60}},
        tolerance=5,
    )

    assert first.review_required is True
    assert second.review_required is True
    assert first.ambiguity == "Room has conflicting dimension pairs: 3300 x 2800 vs 3300 x 2000"
    assert second.ambiguity == "Room has conflicting dimension pairs: 3300 x 2000 vs 3300 x 2800"


def test_duplicate_room_pairs_are_not_flagged_as_conflicts():
    first = make_measurement("DIM-1", 3300, 2800, "room_dimension_pair")
    second = make_measurement("DIM-2", 3302, 2803, "room_dimension_pair")

    _apply_review_flags(
        [first, second],
        [Space(space_id="SPACE-001", name="KITCHEN", source=first.source)],
        {"confidence": {"ocr_review_threshold": 0.50, "association_review_threshold": 0.60}},
        tolerance=5,
    )

    assert first.review_required is False
    assert second.review_required is False


def test_ocr_and_association_thresholds_flag_independently():
    accepted = make_measurement("DIM-1", 3000)
    accepted.confidence.ocr = 0.55
    accepted.confidence.association = 0.96
    low_ocr = make_measurement("DIM-2", 2800)
    low_ocr.confidence.ocr = 0.45
    low_association = make_measurement("DIM-3", 2500)
    low_association.confidence.association = 0.50
    space = Space(space_id="SPACE-001", name="ROOM", source=accepted.source)

    _apply_review_flags(
        [accepted, low_ocr, low_association],
        [space],
        {"confidence": {"ocr_review_threshold": 0.50, "association_review_threshold": 0.60}},
        tolerance=5,
    )

    assert accepted.review_required is False
    assert accepted.ambiguity is None
    assert low_ocr.review_required is True
    assert low_ocr.ambiguity == "OCR confidence below 0.50"
    assert low_association.review_required is True
    assert low_association.ambiguity == "Association to the room is weak"


def test_unassociated_measurement_is_flagged_with_reason():
    measurement = make_measurement("DIM-1", 3000)
    measurement.associated_element = AssociatedElement()

    _apply_review_flags(
        [measurement],
        [],
        {"confidence": {"ocr_review_threshold": 0.50, "association_review_threshold": 0.60}},
        tolerance=5,
    )

    assert measurement.review_required is True
    assert measurement.ambiguity == "Measurement has no associated room"