from plan_extractor.ocr_engine import OCRDetection
from plan_extractor.pipeline import _merge_general_ocr_passes


def detection(text, confidence):
    return OCRDetection(
        text=text,
        confidence=confidence,
        bbox=((0, 0), (20, 0), (20, 10), (0, 10)),
        extraction_pass="general",
    )


def test_general_pass_merge_accepts_two_passes_and_keeps_best_agreeing_read():
    first = detection("BEDROOM 1", 0.80)
    second = detection("BEDROOM 1", 0.95)

    merged = _merge_general_ocr_passes([[first], [second]])

    assert merged == [second]
