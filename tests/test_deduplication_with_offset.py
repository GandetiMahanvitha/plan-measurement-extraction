"""
Test to demonstrate the deduplication fix for edge-padded OCR detections.

This test creates two detection objects that represent the SAME physical text
at the SAME location, but with one having coordinates offset by the padding
amount (simulating what happens when one is captured from a padded OCR pass
and one from the normal pass). It verifies that after coordinate correction,
they are correctly identified as duplicates and merged, keeping the
higher-confidence one.
"""

from plan_extractor.ocr_engine import OCRDetection, deduplicate


def test_deduplicate_with_padding_offset():
    """
    Simulate two detections of the same text:
    - One from normal thresholded pass at (100, 100) to (150, 150)
    - One from padded pass at (200, 200) to (250, 250) [offset by padding=100]

    After coordinate correction (remove offset, apply scale_factor=2.0),
    both should map to the same normalized location and be deduped.
    """
    padding = 100
    scale_factor = 2.0

    # Detection from normal (unpadded) OCR pass - lower confidence
    normal_det = OCRDetection(
        text="3600",
        confidence=0.95,
        bbox=((100.0, 100.0), (150.0, 100.0), (150.0, 150.0), (100.0, 150.0)),
        extraction_pass="dimension",
    )

    # Detection from padded OCR pass - higher confidence, but offset by padding
    edge_det = OCRDetection(
        text="3600",
        confidence=0.99,  # Higher confidence
        bbox=((200.0, 200.0), (250.0, 200.0), (250.0, 250.0), (200.0, 250.0)),
        extraction_pass="dimension",
    )

    # Apply coordinate correction to both (simulating apply_scaling from pipeline)
    def apply_scaling(det, offset=0):
        points = tuple(
            ((x - offset) / scale_factor, (y - offset) / scale_factor)
            for x, y in det.bbox
        )
        return OCRDetection(
            text=det.text,
            confidence=det.confidence,
            bbox=points,
            extraction_pass=det.extraction_pass,
        )

    normal_corrected = apply_scaling(normal_det, offset=0)
    edge_corrected = apply_scaling(edge_det, offset=padding)

    # After correction, both should have the same coordinates (within scale factor)
    assert normal_corrected.x_min == edge_corrected.x_min
    assert normal_corrected.y_min == edge_corrected.y_min
    assert normal_corrected.x_max == edge_corrected.x_max
    assert normal_corrected.y_max == edge_corrected.y_max

    # Now deduplicate: with the fix, these should be recognized as duplicates
    deduped = deduplicate([normal_corrected, edge_corrected])

    # Should keep only 1 (the higher-confidence edge_det)
    assert len(deduped) == 1
    assert deduped[0].confidence == 0.99
    assert deduped[0].text == "3600"


def test_deduplicate_without_coordinate_correction_fails():
    """
    Demonstrate that WITHOUT coordinate correction before deduplication,
    the same two detections would NOT be recognized as duplicates.
    This shows the bug: if deduplicate is called before scaling is applied,
    the offset coordinates prevent IOU overlap detection.
    """
    padding = 100

    # Detection from normal pass
    normal_det = OCRDetection(
        text="3600",
        confidence=0.95,
        bbox=((100.0, 100.0), (150.0, 100.0), (150.0, 150.0), (100.0, 150.0)),
        extraction_pass="dimension",
    )

    # Detection from padded pass (still offset, not corrected)
    edge_det_uncorrected = OCRDetection(
        text="3600",
        confidence=0.99,
        bbox=((200.0, 200.0), (250.0, 200.0), (250.0, 250.0), (200.0, 250.0)),
        extraction_pass="dimension",
    )

    # If we deduplicate WITHOUT correcting coordinates first,
    # they won't be recognized as duplicates (demonstrating the bug)
    deduped = deduplicate([normal_det, edge_det_uncorrected])

    # Bug: keeps both (IOU overlap is 0 because offset prevents overlap detection)
    assert len(deduped) == 2
    print("Bug demonstration: without coordinate correction before dedup, got 2 detections instead of 1")


if __name__ == "__main__":
    test_deduplicate_with_padding_offset()
    print("✓ Deduplication with coordinate correction works correctly")
    test_deduplicate_without_coordinate_correction_fails()
    print("✓ Bug is confirmed: without correction, duplicates are not merged")
