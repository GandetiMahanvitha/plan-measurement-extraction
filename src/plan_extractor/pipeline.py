from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2

from .association import infer_orientation, nearest_space, normalized_center_distance
from .models import (
    AssociatedElement,
    BoundingBox,
    ConfidenceBreakdown,
    DocumentSource,
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
from .ocr_engine import EasyOCREngine, deduplicate
from .parsing import classify_room_label, parse_measurement
from .preprocessing import build_variants, load_image
from .validation import evaluate_measurements, load_ground_truth


OVERALL_DIMENSION_REGION_RATIO = 0.12


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
        extraction_pass=det.extraction_pass,
    )


def _merge_room_dimensions(
    measurements: list[Measurement],
    spaces: list[Space],
    tolerance: float,
) -> list[Measurement]:
    """Keep room pairs canonical while preserving genuine conflicts."""
    removed_ids = set()

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
            if measurement.type == "dimension" and measurement.normalized_secondary_value_mm is None
        ]

        pair = pairs[0] if pairs else None
        if pair is None:
            clusters: list[list[Measurement]] = []
            for single in sorted(singles, key=lambda item: item.confidence.ocr, reverse=True):
                matching_cluster = next(
                    (
                        cluster
                        for cluster in clusters
                        if abs(cluster[0].normalized_value_mm - single.normalized_value_mm) <= tolerance
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
                    (item for item in representatives if item.orientation == "horizontal"),
                    None,
                )
                vertical = next(
                    (item for item in representatives if item.orientation == "vertical"),
                    None,
                )
                if horizontal and vertical:
                    pair = horizontal.model_copy(
                        update={
                            "measurement_id": f"DIM-PAIR-{space.space_id}",
                            "type": "room_dimension_pair",
                            "raw_text": f"{horizontal.value:g} x {vertical.value:g}",
                            "secondary_value": vertical.value,
                            "normalized_secondary_value_mm": vertical.normalized_value_mm,
                            "source_type": "dimension_pair_text",
                            "source": horizontal.source.model_copy(
                                update={"raw_text": f"{horizontal.value:g} x {vertical.value:g}"}
                            ),
                            "confidence": horizontal.confidence.model_copy(
                                update={"overall": max(horizontal.confidence.overall, vertical.confidence.overall)}
                            ),
                            "corroborated_by_dimension_line": True,
                            "ambiguity": "Pair text was not detected; inferred from dimension-line readings.",
                            "review_required": True,
                        }
                    )
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

        pair_values = [value for value in (pair.normalized_value_mm, pair.normalized_secondary_value_mm) if value is not None]

        matched = 0
        conflicts = []
        for single in singles:
            corrected_leading_one = next(
                (
                    value
                    for value in pair_values
                    if single.normalized_value_mm is not None
                    and 1000 + single.normalized_value_mm == value
                    and single.normalized_value_mm < 1000
                ),
                None,
            )
            if corrected_leading_one is not None:
                single.value = corrected_leading_one
                single.normalized_value_mm = corrected_leading_one

            if any(
                abs(single.normalized_value_mm - value) <= tolerance
                for value in pair_values
            ):
                removed_ids.add(single.measurement_id)
                matched += 1
            else:
                conflicts.append(single)

        if matched:
            pair.corroborated_by_dimension_line = True
            pair.confidence.overall = round(min(1.0, pair.confidence.overall + 0.05), 4)

        if conflicts:
            conflict_text = "Dimension-line reading disagrees with the room dimension pair."
            pair.ambiguity = f"{pair.ambiguity}; {conflict_text}" if pair.ambiguity else conflict_text
            pair.review_required = True
            for single in conflicts:
                single.ambiguity = conflict_text
                single.review_required = True

    if removed_ids:
        for space in spaces:
            space.dimensions = [
                measurement_id
                for measurement_id in space.dimensions
                if measurement_id not in removed_ids
            ]

    return [measurement for measurement in measurements if measurement.measurement_id not in removed_ids]


def _application_confidence(
    ocr_conf: float,
    format_conf: float,
    association_conf: float,
    config: dict,
) -> float:
    """Combine OCR, format, and association scores using configured weights."""
    cfg = config["confidence"]
    score = (
        float(cfg.get("ocr_weight", 0.45)) * ocr_conf
        + float(cfg.get("format_weight", 0.25)) * format_conf
        + float(cfg.get("association_weight", 0.30)) * association_conf
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


def _merge_general_ocr_passes(pass1, pass2, pass3, iou_threshold=0.70) -> list:
    """Merge 3 OCR detection passes using majority voting on text and highest confidence.
    Groups detections across the 3 runs whose bounding boxes overlap significantly (>IoU threshold).
    For each group:
    - If 2+ runs agree on exact same text: use that text with highest confidence among agreeing detections.
    - If all 3 disagree: use highest confidence detection.
    - If detection only in 1 run: keep it but mark as single-run detection.
    """
    from .ocr_engine import OCRDetection
    
    # All detections across all 3 passes, tagged with pass number
    tagged_detections = []
    for det in pass1:
        tagged_detections.append((det, 0))
    for det in pass2:
        tagged_detections.append((det, 1))
    for det in pass3:
        tagged_detections.append((det, 2))
    
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
) -> PlanResult:
    """Extract spaces, dimensions, evidence, and validation data from a plan image."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    image = load_image(path)
    # how we are sure the the first and second are height , width 
    # 3000x2000 how can i know 3000 is height or 2000 is height
    height, width = image.shape[:2]
    variants = build_variants(image, config)
    scale_factor = float(config["preprocessing"].get("upscale_factor", 2.0))

    engine = EasyOCREngine(config)

    # Run general OCR pass 3 times and merge results using voting to reduce non-determinism.
    general_pass1 = engine.read_general(variants["upscaled_gray"])
    general_pass2 = engine.read_general(variants["upscaled_gray"])
    general_pass3 = engine.read_general(variants["upscaled_gray"])
    general = _merge_general_ocr_passes(general_pass1, general_pass2, general_pass3, iou_threshold=0.70)
    
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
    dimensions = engine.read_dimensions(variants["thresholded"])
    edge_dimensions = engine.read_dimensions(padded_thresholded, "dimension_edge")
    # what is this logic
    edge_dimensions = [
        det
        for det in edge_dimensions
        if det.x_min <= padding + 40
        and det.y_min >= padding
        and det.y_max <= padding + height
        and len(det.text.strip()) >= 3
    ]

    # Because OCR was run on the upscaled variants, normalize boxes back to
    # original image coordinates. This must happen BEFORE deduplication so that
    # detections from different OCR passes can be correctly compared for overlap.
    def apply_scaling(det, offset=0):
        """Undo OCR-image padding and scaling so detections share image coordinates."""
        from .ocr_engine import OCRDetection
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

    # Now deduplicate with all detections in the same coordinate space.
    detections = deduplicate(general + dimensions + edge_dimensions)
    # for det in general:
    #     print(f"  text={det.text!r}, confidence={det.confidence:.4f}, bbox={det.bbox}")

    spaces: list[Space] = []
    warnings: list[WarningItem] = []

    for det in detections:
        category = classify_room_label(det.text)
        if not category:
            continue
        spaces.append(
            Space(
                space_id=f"SPACE-{len(spaces)+1:03d}",
                name=det.text.strip(),
                category=category,
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
    scale_crop_detections = engine.read_general(scale_crop)
    from .ocr_engine import OCRDetection
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
                extraction_pass=det.extraction_pass,
                confidence=ConfidenceBreakdown(
                    ocr=round(det.confidence, 4),
                    format_validation=1.0,
                    association=1.0,
                    overall=round(
                        _application_confidence(
                            det.confidence,
                            1.0,
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

    # Fallback: if overall width was not detected in the main pass, crop the top
    # region and retry. This follows the same pattern as the scale detection fallback.
    if not overall_width_candidates:
        width_crop_height = int(variants["upscaled_gray"].shape[0] * 0.15)
        width_crop = variants["upscaled_gray"][:width_crop_height, :]
        width_crop_detections = engine.read_general(width_crop)
        
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
                    extraction_pass=det_scaled.extraction_pass,
                    confidence=ConfidenceBreakdown(
                        ocr=round(det_scaled.confidence, 4),
                        format_validation=1.0,
                        association=1.0,
                        overall=round(
                            _application_confidence(
                                det_scaled.confidence,
                                1.0,
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
    review_threshold = float(config["confidence"].get("review_threshold", 0.75))
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
                element_type="space",
                name=space.name,
            )
        else:
            associated = AssociatedElement(
                element_type="unresolved",
                name=None,
            )

        format_conf = 1.0
        overall = _application_confidence(
            det.confidence,
            format_conf,
            assoc_conf,
            config,
        )

        measurement_id = f"DIM-{len(measurement_list)+1:04d}"
        review_required = overall < review_threshold or space is None

        ambiguity = None
        if space is None:
            ambiguity = "Measurement could not be confidently associated with a space."
        elif distance is not None and distance > max_distance * 0.75:
            ambiguity = "Measurement is relatively far from its associated space label."

        m = Measurement(
            measurement_id=measurement_id,
            type=(
                "room_dimension_pair"
                if parsed.kind == "dimension_pair" and space
                else "dimension"
            ),
            raw_text=det.text,
            value=parsed.value,
            secondary_value=parsed.secondary_value,
            unit="mm" if parsed.value is not None else None,
            normalized_value_mm=parsed.value,
            normalized_secondary_value_mm=parsed.secondary_value,
            orientation=infer_orientation(box),
            source_type=(
                "dimension_pair_text"
                if parsed.kind == "dimension_pair"
                else "explicit_text"
            ),
            associated_element=associated,
            source=_source(det),
            confidence=ConfidenceBreakdown(
                ocr=round(det.confidence, 4),
                format_validation=format_conf,
                association=round(assoc_conf, 4),
                overall=round(overall, 4),
            ),
            ambiguity=ambiguity,
            review_required=review_required,
        )
        measurement_list.append(m)
        if space:
            space.dimensions.append(measurement_id)

    tolerance = float(config["validation"].get("numeric_tolerance_mm", 5.0))
    measurement_list = _merge_room_dimensions(measurement_list, spaces, tolerance)

    for measurement in measurement_list:
        if measurement.review_required:
            warnings.append(
                WarningItem(
                    code="REVIEW_REQUIRED",
                    message=measurement.ambiguity or "Measurement confidence is below the configured threshold.",
                    measurement_id=measurement.measurement_id,
                )
            )

    # Duplicate warning: same normalized value and nearly identical box/pass combinations
    seen_values = {}
    for m in measurement_list:
        key = (
            m.normalized_value_mm,
            m.normalized_secondary_value_mm,
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

    status = "success"
    if warnings:
        status = "partial"
    if not measurement_list:
        status = "failed"
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
        document_source=DocumentSource(file_name=path.name, page_number=1),
        scale=scale_info,
        overall_dimensions=overall_dimensions,
        spaces=spaces,
        measurements=measurement_list,
        warnings=warnings,
        summary=ExtractionSummary(
            ocr_detections=len(detections),
            spaces_detected=len(spaces),
            measurements_detected=len(measurement_list),
            review_required=sum(1 for m in measurement_list if m.review_required),
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

    return result
