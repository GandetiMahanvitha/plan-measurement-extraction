from plan_extractor.ocr_engine import (
    OCRDetection,
    filter_detections_by_text_shape,
)


def make_detection(text, confidence=1.0):
    return OCRDetection(
        text=text,
        confidence=confidence,
        bbox=((0, 0), (1, 0), (1, 1), (0, 1)),
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
    ]

    assert filter_detections_by_text_shape(detections, 0.75) == detections


def test_filter_keeps_supported_scale_and_confident_room_labels():
    detections = [
        make_detection("SCALE 1:100", 0.10),
        make_detection("SCALE 1.100", 0.10),
        make_detection("BEDROOM 1", 0.75),
        make_detection("LIVING / DINING", 0.90),
        make_detection("LAUNDRY /", 0.90),
        make_detection("BEDROOM 2", 0.74),
    ]

    assert filter_detections_by_text_shape(detections, 0.75) == detections[:-1]
