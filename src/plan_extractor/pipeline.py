from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2

from .association import infer_orientation, nearest_space, normalized_center_distance
from .models import (
    AssociatedElement,
    BoundingBox,
    ComponentSource,
    ConfidenceBreakdown,
    ExtractionSummary,
    Measurement,
    OverallDimensionSource,
    OverallDimensions,
    PlanResult,
    ScaleInfo,
    SourceReference,
    Space,
    WarningItem,
)
from .ocr_engine import (
    OCRDetection,
    OCREngine,
    create_ocr_engine,
    deduplicate,
    filter_detections_by_text_shape,
)
from .parsing import (
    ParsedMeasurement,
    is_plausible_room_label,
    normalize_ocr_text,
    parse_measurement,
)
from .preprocessing import build_variants, load_image
from .validation import evaluate_measurements, load_ground_truth


OVERALL_DIMENSION_REGION_RATIO = 0.12
OVERALL_DIMENSION_DUPLICATE_TOLERANCE_MM = 5.0
DEFAULT_OCR_CONFIDENCE_WEIGHT = 0.6
DEFAULT_ASSOCIATION_CONFIDENCE_WEIGHT = 0.4


def _bbox(det) -> BoundingBox:
    """Convert an OCR detection's extents into the output bounding-box model."""
    return BoundingBox(
        x_min=det.x_min,
        y_min=det.y_min,
        x_max=det.x_max,
        y_max=det.y_max,
    )


def _source(det) -> SourceReference:
    """Build location-specific source evidence for an OCR detection."""
    return SourceReference(
        raw_text=det.text,
        bounding_box=_bbox(det),
    )


def _promote_overall_dimensions(
    parsed_candidates: list[tuple[OCRDetection, ParsedMeasurement]],
    overall_width_candidates: list[tuple[float | None, float, OverallDimensionSource]],
    overall_height_candidates: list[tuple[float | None, float, OverallDimensionSource]],
) -> None:
    pair_values = [
        value
        for _, parsed in parsed_candidates
        if parsed.kind == "dimension_pair"
        for value in (parsed.value, parsed.secondary_value)
        if value is not None
    ]
    largest_pair_value = max(pair_values, default=None)
    promoted_orientations: set[str] = set()
    promoted_values: dict[str, float] = {}

    for orientation, overall_candidates in (
        ("horizontal", overall_width_candidates),
        ("vertical", overall_height_candidates),
    ):
        singles = [
            (det, parsed)
            for det, parsed in parsed_candidates
            if parsed.kind == "dimension"
            and parsed.value is not None
            and infer_orientation(_bbox(det)) == orientation
        ]
        singles.sort(key=lambda candidate: candidate[1].value, reverse=True)

        distinct_value_representatives: list[float] = []
        for _, parsed in singles:
            value = parsed.value
            if not distinct_value_representatives or (
                distinct_value_representatives[-1] - value
                > OVERALL_DIMENSION_DUPLICATE_TOLERANCE_MM
            ):
                distinct_value_representatives.append(value)

        if len(distinct_value_representatives) < 2:
            continue
        largest_value, next_largest_value = distinct_value_representatives[:2]
        if largest_pair_value is not None and largest_value <= largest_pair_value:
            continue
        if largest_value < 1.5 * next_largest_value:
            continue

        promoted = next(
            parsed
            for _, parsed in singles
            if parsed.value == largest_value
        )
        det = next(
            det
            for det, parsed in singles
            if parsed is promoted
        )
        box = _bbox(det)
        source = OverallDimensionSource(
            raw_text=det.text,
            bounding_box=box,
            confidence=ConfidenceBreakdown(
                ocr=round(det.confidence, 4),
                association=1.0,
                overall=round(
                    DEFAULT_OCR_CONFIDENCE_WEIGHT * det.confidence
                    + DEFAULT_ASSOCIATION_CONFIDENCE_WEIGHT,
                    4,
                ),
            ),
        )
        overall_candidates.append((promoted.value, det.confidence, source))
        promoted_orientations.add(orientation)
        promoted_values[orientation] = promoted.value

    if not promoted_orientations:
        return

    parsed_candidates[:] = [
        (det, parsed)
        for det, parsed in parsed_candidates
        if not (
            parsed.kind != "dimension_pair"
            and parsed.value is not None
            and infer_orientation(_bbox(det)) in promoted_orientations
            and abs(
                parsed.value - promoted_values[infer_orientation(_bbox(det))]
            )
            <= OVERALL_DIMENSION_DUPLICATE_TOLERANCE_MM
        )
    ]


def _merge_room_dimensions(
    measurements: list[Measurement],
    spaces: list[Space],
    tolerance: float,
    config: dict | None = None,
) -> list[Measurement]:
    """Keep room pairs canonical while preserving genuine conflicts."""
    removed_ids = set()
    cfg = (config or {}).get("confidence", {}) if config else {}
    ocr_weight = float(cfg.get("ocr_weight", DEFAULT_OCR_CONFIDENCE_WEIGHT))
    association_weight = float(
        cfg.get("association_weight", DEFAULT_ASSOCIATION_CONFIDENCE_WEIGHT)
    )

    space_by_id = {space.space_id: space for space in spaces}
    space_ids = set(space_by_id)
    space_ids.update(
        measurement.associated_element.element_id
        for measurement in measurements
        if measurement.associated_element.element_id
    )

    for space_id in space_ids:
        space = space_by_id.get(space_id)
        room_measurements = [
            measurement
            for measurement in measurements
            if measurement.associated_element.element_id == space_id
        ]
        pairs = [measurement for measurement in room_measurements if measurement.type == "room_dimension_pair"]
        singles = [
            measurement
            for measurement in room_measurements
            if measurement.type == "dimension" and measurement.secondary_value is None
        ]

        pair = pairs[0] if pairs else None
        if pair is None:
            clusters: list[list[Measurement]] = []
            for single in sorted(singles, key=lambda item: item.confidence.ocr, reverse=True):
                matching_cluster = next(
                    (
                        cluster
                        for cluster in clusters
                        if abs(cluster[0].value - single.value) <= tolerance
                    ),
                    None,
                )
                if matching_cluster is None:
                    clusters.append([single])
                else:
                    matching_cluster.append(single)

            if len(clusters) >= 2:
                representatives = [cluster[0] for cluster in clusters]
                horizontal = next(
                    (
                        item
                        for item in representatives
                        if infer_orientation(item.source.bounding_box) == "horizontal"
                    ),
                    None,
                )
                vertical = next(
                    (
                        item
                        for item in representatives
                        if infer_orientation(item.source.bounding_box) == "vertical"
                    ),
                    None,
                )
                if horizontal and vertical:
                    combined_raw_text = (
                        f"{horizontal.source.raw_text} + {vertical.source.raw_text}"
                    )
                    horizontal_box = horizontal.source.bounding_box
                    vertical_box = vertical.source.bounding_box
                    combined_box = BoundingBox(
                        x_min=min(horizontal_box.x_min, vertical_box.x_min),
                        y_min=min(horizontal_box.y_min, vertical_box.y_min),
                        x_max=max(horizontal_box.x_max, vertical_box.x_max),
                        y_max=max(horizontal_box.y_max, vertical_box.y_max),
                    )
                    rebuilt_ocr = round(
                        (horizontal.confidence.ocr + vertical.confidence.ocr) / 2.0,
                        4,
                    )
                    rebuilt_association = round(
                        (
                            horizontal.confidence.association
                            + vertical.confidence.association
                        )
                        / 2.0,
                        4,
                    )
                    rebuilt_overall = round(
                        _application_confidence(
                            rebuilt_ocr,
                            rebuilt_association,
                            {"confidence": {"ocr_weight": ocr_weight, "association_weight": association_weight}},
                        ),
                        4,
                    )
                    ambiguity = (
                        "The printed pair was not read. "
                        "Width and height were taken from two separate dimension-line numbers "
                        f"({horizontal.value:g} and {vertical.value:g}). Please check."
                    )
                    pair = horizontal.model_copy(
                        update={
                            "measurement_id": f"DIM-PAIR-{space.space_id}",
                            "type": "room_dimension_pair",
                            "raw_text": combined_raw_text,
                            "secondary_value": vertical.value,
                            "source": SourceReference(
                                raw_text=combined_raw_text,
                                bounding_box=combined_box,
                            ),
                            "component_sources": [
                                ComponentSource(
                                    raw_text=horizontal.source.raw_text,
                                    bounding_box=horizontal_box,
                                    ocr=round(horizontal.confidence.ocr, 4),
                                ),
                                ComponentSource(
                                    raw_text=vertical.source.raw_text,
                                    bounding_box=vertical_box,
                                    ocr=round(vertical.confidence.ocr, 4),
                                ),
                            ],
                            "confidence": ConfidenceBreakdown(
                                ocr=rebuilt_ocr,
                                association=rebuilt_association,
                                overall=rebuilt_overall,
                            ),
                            "ambiguity": ambiguity,
                            "review_required": True,
                        }
                    )
                    object.__setattr__(pair, "_is_rebuilt", True)
                    measurements.append(pair)
                    if space:
                        space.dimensions.append(pair.measurement_id)

        if pair is None:
            if len(singles) == 1:
                single = singles[0]
                single.ambiguity = (
                    "Only one dimension found for this room; a second dimension (width or height) may be missing"
                )
                single.review_required = True
            continue

        pair_values = [
            value
            for value in (pair.value, pair.secondary_value)
            if value is not None
        ]

        matched = 0
        conflicts = []
        for single in singles:
            corrected_leading_one = next(
                (
                    value
                    for value in pair_values
                    if single.value is not None
                    and 1000 + single.value == value
                    and single.value < 1000
                ),
                None,
            )
            if corrected_leading_one is not None:
                single.value = corrected_leading_one

            if any(
                single.value is not None and abs(single.value - value) <= tolerance
                for value in pair_values
            ):
                removed_ids.add(single.measurement_id)
                matched += 1
            else:
                conflicts.append(single)

        if matched and not getattr(pair, "_is_rebuilt", False):
            pair.confidence.overall = round(min(1.0, pair.confidence.overall + 0.05), 4)

        if conflicts:
            conflict_text = "Dimension-line reading disagrees with the room dimension pair."
            _append_ambiguity(pair, conflict_text)
            pair.review_required = True
            for single in conflicts:
                _append_ambiguity(single, conflict_text)
                single.review_required = True

    if removed_ids:
        for space in spaces:
            space.dimensions = [
                measurement_id
                for measurement_id in space.dimensions
                if measurement_id not in removed_ids
            ]

    return [measurement for measurement in measurements if measurement.measurement_id not in removed_ids]


def _append_ambiguity(measurement: Measurement, reason: str) -> None:
    existing = measurement.ambiguity.split("; ") if measurement.ambiguity else []
    if reason not in existing:
        existing.append(reason)
    measurement.ambiguity = "; ".join(existing)


def _apply_review_flags(
    measurements: list[Measurement],
    spaces: list[Space],
    config: dict,
    tolerance: float,
) -> None:
    """Apply review rules based on explicit extraction and association risks."""
    confidence_config = config["confidence"]
    ocr_threshold = float(confidence_config.get("ocr_review_threshold", 0.50))
    association_threshold = float(
        confidence_config.get("association_review_threshold", 0.60)
    )
    known_space_ids = {space.space_id for space in spaces}

    pairs_by_space: dict[str, list[Measurement]] = {}
    for measurement in measurements:
        associated = measurement.associated_element
        if associated.element_id and associated.element_id in known_space_ids:
            if measurement.type == "room_dimension_pair":
                pairs_by_space.setdefault(associated.element_id, []).append(measurement)
        else:
            measurement.review_required = True
            _append_ambiguity(measurement, "Measurement has no associated room")

        if measurement.confidence.ocr < ocr_threshold:
            measurement.review_required = True
            _append_ambiguity(measurement, "OCR confidence below 0.50")
        if measurement.confidence.association < association_threshold:
            measurement.review_required = True
            _append_ambiguity(measurement, "Association to the room is weak")

    for pairs in pairs_by_space.values():
        for index, pair in enumerate(pairs):
            pair_values = (pair.value, pair.secondary_value)
            conflicting_pairs = []
            for other_index, other in enumerate(pairs):
                if index == other_index:
                    continue
                other_values = (other.value, other.secondary_value)
                if any(
                    left is None or right is None or abs(left - right) > tolerance
                    for left, right in zip(pair_values, other_values)
                ):
                    conflicting_pairs.append(other)

            for other in conflicting_pairs:
                pair.review_required = True
                _append_ambiguity(
                    pair,
                    "Room has conflicting dimension pairs: "
                    f"{_format_dimension_pair(pair)} vs "
                    f"{_format_dimension_pair(other)}",
                )


def _format_dimension_pair(measurement: Measurement) -> str:
    primary = f"{measurement.value:g}" if measurement.value is not None else "?"
    secondary = (
        f"{measurement.secondary_value:g}"
        if measurement.secondary_value is not None
        else "?"
    )
    return f"{primary} x {secondary}"


def _rooms_without_dimensions_warning(
    spaces: list[Space],
) -> WarningItem | None:
    """Describe rooms needing review when OCR yielded no dimensions."""
    missing_names = [space.name for space in spaces if not space.dimensions]
    if not missing_names:
        return None
    return WarningItem(
        code="ROOMS_WITHOUT_DIMENSIONS",
        message=(
            "No dimensions were parsed for these rooms: "
            f"{', '.join(missing_names)}. Likely reason: OCR did not recognize "
            "the dimension text or the parser does not support its notation."
        ),
    )


def _looks_like_dimension_text(text: str) -> bool:
    """Identify numeric OCR text that resembles a dimension notation."""
    normalized = normalize_ocr_text(text)
    return bool(
        re.search(r"\d", normalized)
        and (
            "'" in normalized
            or '"' in normalized
            or re.search(r"\d\s*[xX]\s*\d", normalized)
            or re.search(r"\d\s*-\s*\d", normalized)
        )
    )


def _parsed_dimension_is_plausible(parsed: ParsedMeasurement | None) -> bool:
    """Check parsed measurement values against the supported millimeter range."""
    if parsed is None:
        return False
    values = [parsed.value]
    if parsed.secondary_value is not None:
        values.append(parsed.secondary_value)
    return all(value is not None and 100 <= value <= 100000 for value in values)


def _box_overlap_over_smaller_area(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    """Return intersection area divided by the smaller bounding-box area."""
    intersection_width = max(
        0.0,
        min(first.x_max, second.x_max) - max(first.x_min, second.x_min),
    )
    intersection_height = max(
        0.0,
        min(first.y_max, second.y_max) - max(first.y_min, second.y_min),
    )
    first_area = max(0.0, first.x_max - first.x_min) * max(
        0.0, first.y_max - first.y_min
    )
    second_area = max(0.0, second.x_max - second.x_min) * max(
        0.0, second.y_max - second.y_min
    )
    smaller_area = min(first_area, second_area)
    if smaller_area == 0.0:
        return 0.0
    return intersection_width * intersection_height / smaller_area


def _unparsed_dimension_warnings(
    detections: list[OCRDetection],
    config: dict,
    accepted_measurements: list[Measurement] | None = None,
) -> list[WarningItem]:
    """Report dimension-like OCR detections rejected by filtering or parsing."""
    confidence_config = config.get("confidence", {})
    room_label_config = config.get("room_labels", {})
    room_label_threshold = float(
        confidence_config.get("room_label_min_confidence", 0.75)
    )
    warnings_config = config.get("warnings", {})
    suppress_overlap_threshold = float(
        warnings_config.get("suppress_overlap_threshold", 0.5)
    )
    accepted_boxes = [
        source.bounding_box
        for measurement in (accepted_measurements or [])
        for source in [measurement.source, *measurement.component_sources]
    ]
    warning_by_detection: dict[tuple, WarningItem] = {}
    for detection in detections:
        if not _looks_like_dimension_text(detection.text):
            continue

        detection_box = _bbox(detection)
        if any(
            _box_overlap_over_smaller_area(detection_box, accepted_box)
            >= suppress_overlap_threshold
            for accepted_box in accepted_boxes
        ):
            continue

        parsed = parse_measurement(detection.text)
        survives_filter = bool(
            filter_detections_by_text_shape(
                [detection],
                room_label_threshold,
                room_label_config.get("non_room_vocabulary", []),
                room_label_config.get("ocr_aliases", {}),
                float(room_label_config.get("alias_min_confidence", 0.60)),
                room_label_config.get("known_room_words", []),
                float(room_label_config.get("known_room_min_confidence", 0.55)),
                duplicate_overlap_threshold=float(
                    room_label_config.get("duplicate_overlap_threshold", 0.5)
                ),
            )
        )
        if survives_filter and _parsed_dimension_is_plausible(parsed):
            continue

        key = (
            detection.text,
            tuple(
                round(coordinate, 2)
                for point in detection.bbox
                for coordinate in point
            ),
            round(detection.confidence, 4),
        )
        warning_by_detection[key] = WarningItem(
            code="UNPARSED_DIMENSION_TEXT",
            message=(
                f"Unparsed dimension-like OCR text: {detection.text}; "
                f"bounding box={detection.bbox}; "
                f"OCR confidence={detection.confidence:.4f}."
            ),
        )
    return list(warning_by_detection.values())


def _needs_targeted_ocr_crop(
    detection: OCRDetection,
    dimension_pairs: list[OCRDetection],
) -> bool:
    """Identify broad one-character OCR results not covered by recognized pairs."""
    text = detection.text.strip()
    box_width = detection.x_max - detection.x_min
    box_height = detection.y_max - detection.y_min
    if (
        not text.isdigit()
        or len(text) != 1
        or box_height <= 0
        or box_width < 30
        or box_width / box_height < 2.5
    ):
        return False

    return not any(
        _compute_iou(
            (
                detection.x_min,
                detection.y_min,
                detection.x_max,
                detection.y_max,
            ),
            (pair.x_min, pair.y_min, pair.x_max, pair.y_max),
        )
        >= 0.15
        for pair in dimension_pairs
    )


def _summary_review_required_count(
    measurements: list[Measurement],
    spaces: list[Space],
    warnings: list[WarningItem] | None = None,
) -> int:
    """Count flagged measurements, undimensioned rooms, and unparsed dimensions."""
    count = sum(1 for measurement in measurements if measurement.review_required)
    if not measurements:
        count += sum(1 for space in spaces if not space.dimensions)
    count += sum(
        1
        for warning in (warnings or [])
        if warning.code == "UNPARSED_DIMENSION_TEXT"
    )
    return count


def _result_status(
    measurements: list[Measurement],
    warnings: list[WarningItem],
) -> str:
    """Choose a status while keeping unparsed dimension evidence reviewable."""
    if not measurements:
        return (
            "partial"
            if any(
                warning.code == "UNPARSED_DIMENSION_TEXT"
                for warning in warnings
            )
            else "failed"
        )
    return "partial" if warnings else "success"


def _application_confidence(
    ocr_conf: float,
    association_conf: float,
    config: dict,
) -> float:
    """Combine OCR and association scores using configured weights."""
    cfg = config["confidence"]
    score = (
        float(cfg.get("ocr_weight", 0.6)) * ocr_conf
        + float(cfg.get("association_weight", 0.4)) * association_conf
    )
    return max(0.0, min(1.0, score))


def _compute_iou(bbox1, bbox2) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.
    Each bbox is (x_min, y_min, x_max, y_max).
    """
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    
    # Intersection
    ix_min = max(x_min1, x_min2)
    iy_min = max(y_min1, y_min2)
    ix_max = min(x_max1, x_max2)
    iy_max = min(y_max1, y_max2)
    
    if ix_min >= ix_max or iy_min >= iy_max:
        return 0.0
    
    intersection = (ix_max - ix_min) * (iy_max - iy_min)
    
    # Union
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def _merge_general_ocr_passes(passes, iou_threshold=0.70) -> list:
    """Merge general OCR detections across passes using voting and confidence.
    Groups detections whose bounding boxes overlap significantly (>IoU threshold).
    For each group:
    - If 2+ runs agree on exact same text: use that text with highest confidence among agreeing detections.
    - If all 3 disagree: use highest confidence detection.
    - If detection only in 1 run: keep it but mark as single-run detection.
    """
    # All detections across passes, tagged with pass number
    tagged_detections = []
    for pass_num, detections in enumerate(passes):
        for det in detections:
            tagged_detections.append((det, pass_num))
    
    # Track which detections have been grouped
    grouped = set()
    merged_detections = []
    
    # Group detections by spatial overlap
    for i, (det_i, pass_i) in enumerate(tagged_detections):
        if i in grouped:
            continue
        
        # Find all detections that overlap significantly with this one
        group = [(det_i, pass_i, i)]
        bbox_i = (det_i.x_min, det_i.y_min, det_i.x_max, det_i.y_max)
        
        for j, (det_j, pass_j) in enumerate(tagged_detections):
            if i >= j or j in grouped:
                continue
            
            bbox_j = (det_j.x_min, det_j.y_min, det_j.x_max, det_j.y_max)
            iou = _compute_iou(bbox_i, bbox_j)
            
            if iou > iou_threshold:
                group.append((det_j, pass_j, j))
        
        # Mark all detections in this group as processed
        for _, _, idx in group:
            grouped.add(idx)
        
        # Apply voting logic
        if len(group) == 1:
            # Single detection across all 3 passes
            det, pass_num, _ = group[0]
            merged_detections.append(det)
        else:
            # Multiple detections in this group
            text_votes = {}  # text -> list of (detection, pass_num, confidence)
            for det, pass_num, _ in group:
                if det.text not in text_votes:
                    text_votes[det.text] = []
                text_votes[det.text].append((det, pass_num, det.confidence))
            
            # Find text with most votes (majority voting)
            best_text = max(text_votes.keys(), key=lambda t: len(text_votes[t]))
            agreeing_detections = text_votes[best_text]
            
            if len(agreeing_detections) >= 2:
                # Majority agreement: use best text with highest confidence
                best_det = max(agreeing_detections, key=lambda x: x[2])[0]
                merged_detections.append(best_det)
            else:
                # No majority: use highest confidence detection from entire group
                best_det = max(group, key=lambda x: x[0].confidence)[0]
                merged_detections.append(best_det)
    
    return merged_detections


def extract_plan(
    image_path: str | Path,
    config: dict,
    ground_truth_path: str | Path | None = None,
    engine: OCREngine | None = None,
) -> PlanResult:
    """Extract spaces, dimensions, evidence, and validation data from a plan image."""
    total_started = perf_counter()
    preprocessing_started = perf_counter()
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    image = load_image(path)
    # how we are sure the the first and second are height , width 
    # 3000x2000 how can i know 3000 is height or 2000 is height
    height, width = image.shape[:2]
    variants = build_variants(image, config)
    scale_factor = float(config["preprocessing"].get("upscale_factor", 2.0))
    preprocessing_seconds = perf_counter() - preprocessing_started
    print(f"Timing [{path.name}] preprocessing: {preprocessing_seconds:.3f}s")

    engine_init_seconds = 0.0
    if engine is None:
        engine_started = perf_counter()
        engine = create_ocr_engine(config)
        engine_init_seconds = perf_counter() - engine_started
        print(f"Timing [{path.name}] OCR engine initialization: {engine_init_seconds:.3f}s")

    ocr_seconds = 0.0

    def timed_ocr(label: str, reader, *args):
        nonlocal ocr_seconds
        started = perf_counter()
        detections = reader(*args)
        elapsed = perf_counter() - started
        ocr_seconds += elapsed
        print(f"Timing [{path.name}] OCR {label}: {elapsed:.3f}s")
        return detections

    general_pass_count = int(config["ocr"].get("general_passes", 3))
    if general_pass_count < 1:
        raise ValueError("ocr.general_passes must be at least 1")
    general_passes = [
        timed_ocr(
            f"general_full_pass_{pass_num + 1}",
            engine.read_general,
            variants["upscaled_gray"],
        )
        for pass_num in range(general_pass_count)
    ]
    general = (
        _merge_general_ocr_passes(general_passes, iou_threshold=0.70)
        if general_pass_count > 1
        else general_passes[0]
    )
    
    padding = 100
    padded_thresholded = cv2.copyMakeBorder(
        variants["thresholded"],
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    dimensions = timed_ocr(
        "dimensions", engine.read_dimensions, variants["thresholded"]
    )
    edge_dimensions = timed_ocr(
        "edge_dimensions",
        engine.read_dimensions,
        padded_thresholded,
        "dimension_edge",
    )
    # what is this logic
    edge_dimensions = [
        det
        for det in edge_dimensions
        if det.x_min <= padding + 40
        and det.y_min >= padding
        and det.y_max <= padding + height
        and len(det.text.strip()) >= 3
    ]

    print("=" * 60)
    print("RAW OCR OUTPUT — nothing filtered or parsed yet")
    print("=" * 60)
    print(f"\n--- 'general' pass ({len(general)} detections) ---")
    for det in general:
        print(f"  text={det.text!r}, confidence={det.confidence:.4f}")
    print(f"\n--- 'dimensions' pass ({len(dimensions)} detections) ---")
    for det in dimensions:
        print(f"  text={det.text!r}, confidence={det.confidence:.4f}")
    print(f"\n--- 'edge_dimensions' pass ({len(edge_dimensions)} detections) ---")
    for det in edge_dimensions:
        print(f"  text={det.text!r}, confidence={det.confidence:.4f}")
    print("=" * 60)


    # Because OCR was run on the upscaled variants, normalize boxes back to
    # original image coordinates. This must happen BEFORE deduplication so that
    # detections from different OCR passes can be correctly compared for overlap.
    def apply_scaling(det, offset=0):
        """Undo OCR-image padding and scaling so detections share image coordinates."""
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

    # Apply coordinate correction to all detections from all OCR passes.
    general = [apply_scaling(det, offset=0) for det in general]
    dimensions = [apply_scaling(det, offset=0) for det in dimensions]
    edge_dimensions = [apply_scaling(det, offset=padding) for det in edge_dimensions]

    # A long, one-character OCR box often means a room label was reduced to
    # noise (for example, PANTRY read as "1"). Re-read only those regions enlarged.
    label_crop_detections: list[OCRDetection] = []
    crop_zoom = 3.0
    crop_padding = 12
    upscaled_height, upscaled_width = variants["upscaled_gray"].shape[:2]
    known_pair_detections = [
        detection
        for detection in dimensions + edge_dimensions
        if (parsed := parse_measurement(detection.text)) is not None
        and parsed.kind == "dimension_pair"
    ]
    for det in general:
        if not _needs_targeted_ocr_crop(det, known_pair_detections):
            continue

        left = max(0, int(det.x_min) - crop_padding)
        top = max(0, int(det.y_min) - crop_padding)
        right = min(width, int(det.x_max) + crop_padding)
        bottom = min(height, int(det.y_max) + crop_padding)
        crop = variants["upscaled_gray"][
            int(top * scale_factor) : min(
                upscaled_height, int(bottom * scale_factor)
            ),
            int(left * scale_factor) : min(
                upscaled_width, int(right * scale_factor)
            ),
        ]
        if crop.size == 0:
            continue

        enlarged_crop = cv2.resize(
            crop,
            None,
            fx=crop_zoom,
            fy=crop_zoom,
            interpolation=cv2.INTER_CUBIC,
        )
        for crop_det in timed_ocr(
            "room_label_crop",
            engine.read_general,
            enlarged_crop,
        ):
            label_crop_detections.append(
                OCRDetection(
                    text=crop_det.text,
                    confidence=crop_det.confidence,
                    bbox=tuple(
                        (
                            left + x / (scale_factor * crop_zoom),
                            top + y / (scale_factor * crop_zoom),
                        )
                        for x, y in crop_det.bbox
                    ),
                    extraction_pass="room_label_crop",
                )
            )

    raw_dimension_detections = (
        general + dimensions + edge_dimensions + label_crop_detections
    )

    # Now deduplicate with all detections in the same coordinate space.
    room_label_min_confidence = float(
        config["confidence"].get("room_label_min_confidence", 0.75)
    )
    room_label_config = config.get("room_labels", {})
    non_room_vocabulary = room_label_config.get("non_room_vocabulary", [])
    detections = deduplicate(
        filter_detections_by_text_shape(
            general + dimensions + edge_dimensions + label_crop_detections,
            room_label_min_confidence,
            non_room_vocabulary,
            room_label_config.get("ocr_aliases", {}),
            float(room_label_config.get("alias_min_confidence", 0.60)),
            room_label_config.get("known_room_words", []),
            float(room_label_config.get("known_room_min_confidence", 0.55)),
            bool(config["ocr"].get("debug_raw_detections", False)),
            float(room_label_config.get("duplicate_overlap_threshold", 0.5)),
        )
    )
    # for det in general:
    #     print(f"  text={det.text!r}, confidence={det.confidence:.4f}, bbox={det.bbox}")

    spaces: list[Space] = []

    for det in detections:
        if not is_plausible_room_label(det.text, non_room_vocabulary):
            continue
        spaces.append(
            Space(
                space_id=f"SPACE-{len(spaces)+1:03d}",
                name=det.text.strip(),
                source=_source(det),
            )
        )

    # OCR can split a combined room label across two stacked lines
    # (e.g. "STORE" on one line, "/" or "UTILITY" directly below it)
    # instead of one row. Search ALL spaces for a vertically-stacked
    # match, since spaces aren't ordered by page position.
    merge_target: dict[str, str] = {}
    for space in spaces:
        if not space.name.rstrip().endswith("/"):
            continue
        box_a = space.source.bounding_box
        best_match = None
        best_gap = None
        for other in spaces:
            if other.space_id == space.space_id:
                continue
            box_b = other.source.bounding_box
            gap = box_b.y_min - box_a.y_max
            horizontally_aligned = abs(box_a.x_min - box_b.x_min) <= 30
            if -5 <= gap <= 25 and horizontally_aligned:
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_match = other.space_id
        if best_match:
            merge_target[space.space_id] = best_match

    skip_ids: set[str] = set()
    for space in spaces:
        if space.space_id in merge_target:
            target_id = merge_target[space.space_id]
            target = next(s for s in spaces if s.space_id == target_id)
            space.name = f"{space.name} {target.name}".strip()
            space.dimensions.extend(target.dimensions)
            skip_ids.add(target_id)

    spaces = [s for s in spaces if s.space_id not in skip_ids]
    scale_info = ScaleInfo()
    overall_dimensions = OverallDimensions()
    parsed_candidates = []
    overall_width_candidates = []
    overall_height_candidates = []
    scale_boxes: list[BoundingBox] = []

    # Full-plan OCR can miss the small scale label, so inspect its usual
    # bottom-region location separately before classifying measurements.
    scale_crop_top = int(variants["upscaled_gray"].shape[0] * 0.80)
    scale_crop = variants["upscaled_gray"][scale_crop_top:, :]
    scale_crop_detections = timed_ocr(
        "scale_crop", engine.read_general, scale_crop
    )
    raw_dimension_detections.extend(
        OCRDetection(
            text=det.text,
            confidence=det.confidence,
            bbox=tuple(
                (x / scale_factor, (y + scale_crop_top) / scale_factor)
                for x, y in det.bbox
            ),
            extraction_pass="scale_crop",
        )
        for det in scale_crop_detections
    )
    scale_crop_detections = filter_detections_by_text_shape(
        scale_crop_detections,
        room_label_min_confidence,
        non_room_vocabulary,
        known_room_words=room_label_config.get("known_room_words", []),
        known_room_min_confidence=float(
            room_label_config.get("known_room_min_confidence", 0.55)
        ),
        duplicate_overlap_threshold=float(
            room_label_config.get("duplicate_overlap_threshold", 0.5)
        ),
    )
    for det in scale_crop_detections:
        if not parse_measurement(det.text) or parse_measurement(det.text).kind != "scale":
            continue
        detections.append(
            OCRDetection(
                text=det.text,
                confidence=det.confidence,
                bbox=tuple(
                    (x / scale_factor, (y + scale_crop_top) / scale_factor)
                    for x, y in det.bbox
                ),
                extraction_pass="scale_crop",
            )
        )

    if config["ocr"].get("debug_raw_detections", False):
        print("OVERALL REGION OCR TEXT:")
        for det in detections:
            center = _bbox(det).center
            if center.y <= height * OVERALL_DIMENSION_REGION_RATIO or center.x <= width * OVERALL_DIMENSION_REGION_RATIO:
                print(f"  text={det.text!r}, confidence={det.confidence:.4f}")

    # First pass: find the scale label(s) ONLY, so we know where any
    # nearby scale-bar legend numbers (e.g. "5m" at the end of a graphic
    # scale bar) are likely to be, before deciding what counts as a
    # real, standalone dimension.
    for det in detections:
        parsed = parse_measurement(det.text)
        if parsed and parsed.kind == "scale":
            scale_boxes.append(_bbox(det))
            if not scale_info.detected or det.confidence > (scale_info.confidence or 0):
                scale_info = ScaleInfo(
                    detected=True,
                    raw_text=det.text,
                    ratio=parsed.ratio,
                    confidence=round(det.confidence, 4),
                )

    scale_bar_proximity = float(
        config["association"].get("scale_bar_proximity", 0.06)
    )

    for det in detections:
        parsed = parse_measurement(det.text)
        if not parsed:
            continue
        if parsed.kind == "scale":
            continue

        box = _bbox(det)

        # Skip anything sitting right next to a detected scale label —
        # likely a legend number on a graphic scale bar (e.g. the "5m"
        # at the end of "0 1 2 3 4 5m"), not a real room dimension.
        near_scale_bar = any(
            abs(box.center.y - scale_box.center.y) / max(height, 1) <= scale_bar_proximity
            for scale_box in scale_boxes
        )
        if near_scale_bar:
            continue

        center = box.center
        is_overall_width = (
            infer_orientation(box) == "horizontal"
            and center.y <= height * OVERALL_DIMENSION_REGION_RATIO
        )
        is_overall_height = (
            infer_orientation(box) == "vertical"
            and center.x <= width * OVERALL_DIMENSION_REGION_RATIO
        )
        if is_overall_width or is_overall_height:
            source = OverallDimensionSource(
                raw_text=det.text,
                bounding_box=box,
                confidence=ConfidenceBreakdown(
                    ocr=round(det.confidence, 4),
                    association=1.0,
                    overall=round(
                        _application_confidence(
                            det.confidence,
                            1.0,
                            config,
                        ),
                        4,
                    ),
                ),
            )
            if is_overall_width:
                overall_width_candidates.append((parsed.value, det.confidence, source))
            elif is_overall_height:
                overall_height_candidates.append((parsed.value, det.confidence, source))
            continue
        parsed_candidates.append((det, parsed))

    _promote_overall_dimensions(
        parsed_candidates,
        overall_width_candidates,
        overall_height_candidates,
    )

    # Fallback: if overall width was not detected in the main pass, crop the top
    # region and retry. This follows the same pattern as the scale detection fallback.
    if not overall_width_candidates:
        width_crop_height = int(variants["upscaled_gray"].shape[0] * 0.15)
        width_crop = variants["upscaled_gray"][:width_crop_height, :]
        width_crop_detections = timed_ocr(
            "width_fallback_crop", engine.read_general, width_crop
        )

        raw_dimension_detections.extend(
            OCRDetection(
                text=det.text,
                confidence=det.confidence,
                bbox=tuple(
                    (x / scale_factor, y / scale_factor)
                    for x, y in det.bbox
                ),
                extraction_pass="width_fallback_crop",
            )
            for det in width_crop_detections
        )
        
        width_crop_detections = filter_detections_by_text_shape(
            width_crop_detections,
            room_label_min_confidence,
            non_room_vocabulary,
            known_room_words=room_label_config.get("known_room_words", []),
            known_room_min_confidence=float(
                room_label_config.get("known_room_min_confidence", 0.55)
            ),
            duplicate_overlap_threshold=float(
                room_label_config.get("duplicate_overlap_threshold", 0.5)
            ),
        )
        for det in width_crop_detections:
            parsed = parse_measurement(det.text)
            if not parsed or parsed.kind == "scale":
                continue
            
            # Scale coordinates back from crop space to image space.
            det_scaled = OCRDetection(
                text=det.text,
                confidence=det.confidence,
                bbox=tuple(
                    (x / scale_factor, y / scale_factor)
                    for x, y in det.bbox
                ),
                extraction_pass="width_fallback_crop",
            )
            
            box = _bbox(det_scaled)
            
            # Look for horizontal dimension in the top region.
            if infer_orientation(box) == "horizontal":
                source = OverallDimensionSource(
                    raw_text=det_scaled.text,
                    bounding_box=box,
                    confidence=ConfidenceBreakdown(
                        ocr=round(det_scaled.confidence, 4),
                        association=1.0,
                        overall=round(
                            _application_confidence(
                                det_scaled.confidence,
                                1.0,
                                config,
                            ),
                            4,
                        ),
                    ),
                )
                overall_width_candidates.append((parsed.value, det_scaled.confidence, source))
    
    if overall_width_candidates:
        value, _, source = max(overall_width_candidates, key=lambda item: item[0] or 0)
        overall_dimensions.width_mm = value
        overall_dimensions.width_source = source
    if overall_height_candidates:
        value, _, source = max(overall_height_candidates, key=lambda item: item[0] or 0)
        overall_dimensions.height_mm = value
        overall_dimensions.height_source = source

    measurement_list: list[Measurement] = []
    max_distance = float(config["association"].get("max_normalized_distance", 0.35))

    for det, parsed in parsed_candidates:
        box = _bbox(det)
        space, assoc_conf, distance = nearest_space(
            box,
            spaces,
            width,
            height,
            max_distance,
        )

        if space:
            associated = AssociatedElement(
                element_id=space.space_id,
                name=space.name,
            )
        else:
            associated = AssociatedElement(
                name=None,
            )

        overall = _application_confidence(
            det.confidence,
            assoc_conf,
            config,
        )

        measurement_id = f"DIM-{len(measurement_list)+1:04d}"

        m = Measurement(
            measurement_id=measurement_id,
            type=(
                "room_dimension_pair"
                if parsed.kind == "dimension_pair" and space
                else "context_dimension"
                if parsed.kind == "context_dimension"
                else "dimension"
            ),
            raw_text=det.text,
            value=parsed.value,
            secondary_value=parsed.secondary_value,
            unit="mm" if parsed.value is not None else None,
            associated_element=associated,
            source=_source(det),
            confidence=ConfidenceBreakdown(
                ocr=round(det.confidence, 4),
                association=round(assoc_conf, 4),
                overall=round(overall, 4),
            ),
            ambiguity=(
                "Parenthesized value; its meaning is unclear."
                if parsed.kind == "context_dimension"
                else None
            ),
            review_required=parsed.kind == "context_dimension",
        )
        measurement_list.append(m)
        if space:
            space.dimensions.append(measurement_id)

    tolerance = float(config["validation"].get("numeric_tolerance_mm", 5.0))
    measurement_list = _merge_room_dimensions(
        measurement_list,
        spaces,
        tolerance,
        config=config,
    )
    _apply_review_flags(measurement_list, spaces, config, tolerance)
    warnings = _unparsed_dimension_warnings(
        raw_dimension_detections,
        config,
        measurement_list,
    )

    for space in spaces:
        if not space.dimensions:
            warnings.append(
                WarningItem(
                    code="ROOM_WITHOUT_DIMENSIONS",
                    message=f"Room {space.name!r} has no associated dimensions.",
                )
            )

    if not measurement_list:
        missing_dimensions_warning = _rooms_without_dimensions_warning(spaces)
        if missing_dimensions_warning is not None:
            warnings.append(missing_dimensions_warning)

    for measurement in measurement_list:
        if measurement.review_required:
            warnings.append(
                WarningItem(
                    code="REVIEW_REQUIRED",
                    message=measurement.ambiguity or "Measurement requires review.",
                    measurement_id=measurement.measurement_id,
                )
            )

    # Duplicate warning: same normalized value and nearly identical box/pass combinations
    seen_values = {}
    for m in measurement_list:
        key = (
            m.value,
            m.secondary_value,
            m.associated_element.name,
        )
        if key in seen_values:
            warnings.append(
                WarningItem(
                    code="POTENTIAL_DUPLICATE",
                    message=(
                        f"Measurement {m.measurement_id} may duplicate "
                        f"{seen_values[key]}."
                    ),
                    measurement_id=m.measurement_id,
                )
            )
        else:
            seen_values[key] = m.measurement_id

    high = sum(1 for m in measurement_list if m.confidence.overall >= 0.90)
    medium = sum(1 for m in measurement_list if 0.75 <= m.confidence.overall < 0.90)
    low = sum(1 for m in measurement_list if m.confidence.overall < 0.75)

    status = _result_status(measurement_list, warnings)
    if not measurement_list:
        warnings.append(
            WarningItem(
                code="NO_MEASUREMENTS",
                message="No measurement candidates were extracted from the image.",
            )
        )

    result = PlanResult(
        plan_id=f"{path.stem.upper()}-{uuid4().hex[:8].upper()}",
        status=status,
        source_file=path.name,
        input_type=path.suffix.lower().lstrip("."),
        image_width_px=width,
        image_height_px=height,
        primary_unit="mm",
        scale=scale_info,
        overall_dimensions=overall_dimensions,
        spaces=spaces,
        measurements=measurement_list,
        warnings=warnings,
        summary=ExtractionSummary(
            ocr_detections=len(detections),
            spaces_detected=len(spaces),
            measurements_detected=len(measurement_list),
            review_required=_summary_review_required_count(
                measurement_list,
                spaces,
                warnings,
            ),
            high_confidence=high,
            medium_confidence=medium,
            low_confidence=low,
        ),
    )

    if ground_truth_path:
        expected = load_ground_truth(ground_truth_path)
        tolerance = float(config["validation"].get("numeric_tolerance_mm", 5.0))
        dup_count = sum(1 for w in warnings if w.code == "POTENTIAL_DUPLICATE")
        result.evaluation = evaluate_measurements(
            expected,
            result.measurements,
            tolerance_mm=tolerance,
            duplicate_warning_count=dup_count,
        )

    total_seconds = perf_counter() - total_started
    rest_seconds = max(
        0.0,
        total_seconds
        - preprocessing_seconds
        - engine_init_seconds
        - ocr_seconds,
    )
    print(f"Timing [{path.name}] rest: {rest_seconds:.3f}s")
    print(f"Timing [{path.name}] total: {total_seconds:.3f}s")

    return result
