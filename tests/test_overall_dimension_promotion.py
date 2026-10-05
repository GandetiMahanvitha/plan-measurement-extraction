from plan_extractor.ocr_engine import OCRDetection
from plan_extractor.parsing import ParsedMeasurement
from plan_extractor.pipeline import _promote_overall_dimensions


def make_candidate(
    value,
    orientation="horizontal",
    *,
    secondary_value=None,
    kind="dimension",
    text=None,
):
    if orientation == "horizontal":
        bbox = ((0, 0), (20, 0), (20, 2), (0, 2))
    else:
        bbox = ((0, 0), (2, 0), (2, 20), (0, 20))

    detection = OCRDetection(
        text=text or str(value),
        confidence=0.9,
        bbox=bbox,
        extraction_pass="test",
    )
    parsed = ParsedMeasurement(
        kind=kind,
        raw_text=detection.text,
        value=value,
        secondary_value=secondary_value,
        unit="mm",
    )
    return detection, parsed


def promote(candidates):
    overall_width_candidates = []
    overall_height_candidates = []
    _promote_overall_dimensions(
        candidates,
        overall_width_candidates,
        overall_height_candidates,
    )
    return overall_width_candidates, overall_height_candidates


def test_promotes_largest_horizontal_single_and_removes_it_from_room_candidates():
    overall = make_candidate(12600)
    room_dimension = make_candidate(8000)
    candidates = [overall, room_dimension]

    width_candidates, height_candidates = promote(candidates)

    assert len(width_candidates) == 1
    assert width_candidates[0][0] == 12600
    assert height_candidates == []
    assert candidates == [room_dimension]


def test_does_not_promote_single_equal_to_a_dimension_pair_value():
    pair = make_candidate(
        4800,
        secondary_value=4200,
        kind="dimension_pair",
        text="4800 x 4200",
    )
    matching_single = make_candidate(4800)
    smaller_single = make_candidate(3000)
    candidates = [pair, matching_single, smaller_single]

    width_candidates, height_candidates = promote(candidates)

    assert width_candidates == []
    assert height_candidates == []
    assert candidates == [pair, matching_single, smaller_single]


def test_does_not_promote_when_largest_is_less_than_one_and_a_half_times_next():
    candidates = [make_candidate(9000), make_candidate(7000)]

    width_candidates, height_candidates = promote(candidates)

    assert width_candidates == []
    assert height_candidates == []
    assert len(candidates) == 2


def test_removes_duplicate_reads_of_promoted_overall_value():
    overall = make_candidate(12600, text="12600")
    duplicate = make_candidate(12603, text="12603")
    room_dimension = make_candidate(8000)
    candidates = [overall, duplicate, room_dimension]

    width_candidates, height_candidates = promote(candidates)

    assert len(width_candidates) == 1
    assert height_candidates == []
    assert candidates == [room_dimension]


def test_leaves_candidates_and_overall_lists_unchanged_when_nothing_qualifies():
    candidates = [make_candidate(9000), make_candidate(7000)]
    original_candidates = candidates.copy()
    width_candidates = []
    height_candidates = []

    _promote_overall_dimensions(
        candidates,
        width_candidates,
        height_candidates,
    )

    assert candidates == original_candidates
    assert width_candidates == []
    assert height_candidates == []
