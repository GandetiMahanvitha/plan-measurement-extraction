from plan_extractor.ocr_engine import (
    OCRDetection,
    filter_detections_by_text_shape,
)
from plan_extractor.pipeline import _needs_targeted_ocr_crop


def make_detection(text, confidence=1.0, bbox=None):
    return OCRDetection(
        text=text,
        confidence=confidence,
        bbox=bbox or ((0, 0), (1, 0), (1, 1), (0, 1)),
        extraction_pass="test",
    )


def test_filter_drops_empty_text_and_noise_shapes():
    detections = [
        make_detection(""),
        make_detection("   "),
        make_detection("T"),
        make_detection("0006"),
        make_detection("0091 X 0081"),
        make_detection("8FINICM 1", 0.40),
        make_detection("CIIINC / NININC", 0.43),
        make_detection("NCICF 1:100", 0.30),
    ]

    assert filter_detections_by_text_shape(detections, 0.75) == []


def test_filter_keeps_dimension_shapes_regardless_of_confidence():
    detections = [
        make_detection("2400", 0.10),
        make_detection("11400", 0.10),
        make_detection("3000 x 2800", 0.10),
        make_detection("3000X2800", 0.10),
        make_detection("3000 × 2800", 0.10),
        make_detection("5200 x 4200", 0.46),
        make_detection("16'-0\" X 13'-1\"", 0.10),
        make_detection("9'-0\"x 16'-0\"", 0.10),
        make_detection("(14'-6\")", 0.10),
        make_detection("12'", 0.10),
        make_detection("34\"", 0.10),
        make_detection("12'-6 1/2\"", 0.10),
    ]

    assert filter_detections_by_text_shape(detections, 0.75) == detections


def test_filter_keeps_imperial_values_but_rejects_unmarked_digits():
    detections = [
        make_detection("1", 0.10),
        make_detection("8", 0.10),
        make_detection("614'-6\"1", 0.10),
    ]

    assert filter_detections_by_text_shape(detections, 0.75) == []


def test_filter_keeps_supported_scale_and_confident_room_labels():
    detections = [
        make_detection("SCALE 1:100", 0.10),
        make_detection("SCALE 1.100", 0.10),
        make_detection("BEDROOM 1", 0.75, bbox=((0, 0), (50, 0), (50, 10), (0, 10))),
        make_detection("LIVING / DINING", 0.90, bbox=((100, 0), (160, 0), (160, 10), (100, 10))),
        make_detection("LAUNDRY /", 0.90, bbox=((200, 0), (250, 0), (250, 10), (200, 10))),
        make_detection("BEDROOM 2", 0.74, bbox=((300, 0), (350, 0), (350, 10), (300, 10))),
    ]

    assert filter_detections_by_text_shape(detections, 0.75) == detections[:-1]


def test_filter_merges_adjacent_labels_and_excludes_non_room_vocabulary():
    detections = [
        make_detection(
            "OPEN TO",
            bbox=((0, 0), (100, 0), (100, 20), (0, 20)),
        ),
        make_detection(
            "ABOVE",
            bbox=((20, 18), (100, 18), (100, 38), (20, 38)),
        ),
        make_detection(
            "SUNKEN",
            bbox=((0, 100), (80, 100), (80, 120), (0, 120)),
        ),
        make_detection(
            "LAUNDRY",
            bbox=((2, 119), (100, 119), (100, 139), (2, 139)),
        ),
        make_detection(
            "OPT",
            bbox=((300, 0), (340, 0), (340, 20), (300, 20)),
        ),
        make_detection(
            "DOOR",
            bbox=((305, 19), (360, 19), (360, 39), (305, 39)),
        ),
        make_detection(
            "KITCHEN",
            bbox=((500, 200), (600, 200), (600, 220), (500, 220)),
        ),
    ]

    filtered = filter_detections_by_text_shape(
        detections,
        0.75,
        ["above", "open to", "door", "opt", "dn", "up"],
    )

    assert {detection.text for detection in filtered} == {
        "SUNKEN LAUNDRY",
        "KITCHEN",
    }


def test_overlapping_duplicate_label_fragments_are_removed_before_merging():
    detections = [
        make_detection(
            "STORE",
            0.9996,
            bbox=((0, 0), (100, 0), (100, 20), (0, 20)),
        ),
        make_detection(
            "SIORE /",
            0.5648,
            bbox=((2, 0), (100, 0), (100, 20), (2, 20)),
        ),
        make_detection(
            "UTILITY",
            0.7905,
            bbox=((0, 25), (100, 25), (100, 45), (0, 45)),
        ),
        make_detection(
            "UTILI",
            0.2089,
            bbox=((2, 25), (100, 25), (100, 45), (2, 45)),
        ),
    ]

    filtered = filter_detections_by_text_shape(
        detections,
        0.75,
        known_room_words=["store"],
        known_room_min_confidence=0.55,
        duplicate_overlap_threshold=0.5,
    )

    assert len(filtered) == 1
    assert filtered[0].text == "STORE UTILITY"
    assert "SIORE" not in filtered[0].text
    assert filtered[0].text.split() == ["STORE", "UTILITY"]
    assert filtered[0].confidence == 0.7905


def test_room_label_duplicate_overlap_threshold_is_configurable():
    detections = [
        make_detection(
            "STORE",
            0.99,
            bbox=((0, 0), (100, 0), (100, 20), (0, 20)),
        ),
        make_detection(
            "SIORE /",
            0.80,
            bbox=((40, 0), (140, 0), (140, 20), (40, 20)),
        ),
    ]

    filtered_at_default = filter_detections_by_text_shape(
        detections,
        0.75,
        known_room_words=["store"],
        known_room_min_confidence=0.55,
        duplicate_overlap_threshold=0.5,
    )
    filtered_at_higher_threshold = filter_detections_by_text_shape(
        detections,
        0.75,
        known_room_words=["store"],
        known_room_min_confidence=0.55,
        duplicate_overlap_threshold=0.75,
    )

    assert len(filtered_at_default) == 1
    assert len(filtered_at_higher_threshold) == 2
    assert any("SIORE" in detection.text for detection in filtered_at_higher_threshold)


def test_filter_keeps_recognized_breakfast_and_normalizes_missed_room_labels():
    detections = [
        make_detection("BREAKFAST", 0.99),
        make_detection("IPANTRY", 0.97, bbox=((10, 0), (50, 0), (50, 10), (10, 10))),
        make_detection("WJ.C:", 0.63, bbox=((100, 0), (140, 0), (140, 10), (100, 10))),
    ]

    filtered = filter_detections_by_text_shape(
        detections,
        0.75,
        room_label_aliases={
            "IPANTRY": "PANTRY",
            "WJ.C:": "W.I.C.",
        },
        alias_min_confidence=0.60,
    )

    assert [detection.text for detection in filtered] == [
        "BREAKFAST",
        "PANTRY",
        "W.I.C.",
    ]


def test_known_room_words_use_lower_confidence_threshold():
    detections = [
        make_detection("BREAKFAST", 0.68),
        make_detection("UNKNOWNWORD", 0.68, bbox=((100, 0), (180, 0), (180, 10), (100, 10))),
        make_detection("KITCHEN", 0.90, bbox=((200, 0), (250, 0), (250, 10), (200, 10))),
    ]

    filtered = filter_detections_by_text_shape(
        detections,
        0.75,
        known_room_words=["kitchen", "breakfast"],
        known_room_min_confidence=0.55,
    )

    assert filtered == [detections[0], detections[2]]


def test_room_label_debug_trace_explains_breakfast_confidence_decision(capsys):
    detection = make_detection("BREAKFAST", 0.6785)

    filtered = filter_detections_by_text_shape(
        [detection],
        0.75,
        known_room_words=["breakfast"],
        known_room_min_confidence=0.55,
        debug_raw_detections=True,
    )

    assert filtered == [detection]
    output = capsys.readouterr().out
    assert "raw='BREAKFAST' confidence=0.6785" in output
    assert "merged_candidate='BREAKFAST' confidence=0.6785" in output
    assert "text_shape=pass" in output
    assert "known_room_word=breakfast" in output
    assert "confidence_threshold=0.55 (pass)" in output
    assert "final=keep" in output


def test_targeted_ocr_crop_selects_unrecognized_label_shape_only():
    broad_digit = make_detection(
        "1",
        bbox=((0, 0), (100, 0), (100, 20), (0, 20)),
    )
    recognized_pair = make_detection(
        "16'-0\" X 13'-1\"",
        bbox=((0, 0), (100, 0), (100, 20), (0, 20)),
    )
    normal_label = make_detection(
        "PANTRY",
        bbox=((0, 0), (60, 0), (60, 20), (0, 20)),
    )

    assert not _needs_targeted_ocr_crop(broad_digit, [recognized_pair])
    assert _needs_targeted_ocr_crop(
        broad_digit,
        [make_detection("16'-0\" X 13'-1\"", bbox=((200, 0), (300, 0), (300, 20), (200, 20)))],
    )
    assert not _needs_targeted_ocr_crop(normal_label, [])
