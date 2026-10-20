from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

import numpy as np

from .parsing import is_plausible_room_label, parse_measurement


_SINGLE_DIMENSION_PATTERN = re.compile(r"[1-9]\d{2,4}")
_DIMENSION_PAIR_PATTERN = re.compile(r"[1-9]\d{2,4}\s*[xX×]\s*[1-9]\d{2,4}")
_IMPERIAL_VALUE_TEXT = (
    r"\d+\s*(?:'\s*(?:-?\s*\d+(?:\s+\d+\s*/\s*\d+)?\s*\"?)?"
    r"|\"\s*(?:\d+\s*/\s*\d+)?)"
)
_IMPERIAL_VALUE_PATTERN = re.compile(
    rf"(?:\(\s*{_IMPERIAL_VALUE_TEXT}\s*\)|{_IMPERIAL_VALUE_TEXT})"
)
_IMPERIAL_PAIR_PATTERN = re.compile(
    rf"{_IMPERIAL_VALUE_TEXT}\s*[xX×]\s*{_IMPERIAL_VALUE_TEXT}"
)
_SCALE_PATTERN = re.compile(r"(?:SCALE\s*)?1\s*[:.]\s*\d{1,4}", re.IGNORECASE)
_LABEL_TEXT_PATTERN = re.compile(r"[A-Za-z][A-Za-z./ ]*")


@dataclass(frozen=True)
class OCRDetection:
    text: str
    confidence: float
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    extraction_pass: str

    @property
    def x_min(self) -> float:
        """Return the smallest x coordinate in the detected quadrilateral."""
        return min(p[0] for p in self.bbox)

    @property
    def y_min(self) -> float:
        """Return the smallest y coordinate in the detected quadrilateral."""
        return min(p[1] for p in self.bbox)

    @property
    def x_max(self) -> float:
        """Return the largest x coordinate in the detected quadrilateral."""
        return max(p[0] for p in self.bbox)

    @property
    def y_max(self) -> float:
        """Return the largest y coordinate in the detected quadrilateral."""
        return max(p[1] for p in self.bbox)


def filter_detections_by_text_shape(
    detections: list[OCRDetection],
    room_label_min_confidence: float,
    non_room_vocabulary: list[str] | None = None,
    room_label_aliases: dict[str, str] | None = None,
    alias_min_confidence: float = 0.0,
    known_room_words: list[str] | None = None,
    known_room_min_confidence: float = 0.55,
    debug_raw_detections: bool = False,
    duplicate_overlap_threshold: float = 0.5,
) -> list[OCRDetection]:
    """Keep dimension-like text and confident, supported room labels."""
    aliases = {
        source.strip().casefold(): target.strip()
        for source, target in (room_label_aliases or {}).items()
    }
    alias_targets = {target.casefold() for target in aliases.values()}
    normalized_detections = [
        OCRDetection(
            text=aliases.get(detection.text.strip().casefold(), detection.text),
            confidence=detection.confidence,
            bbox=detection.bbox,
            extraction_pass=detection.extraction_pass,
        )
        for detection in detections
    ]
    merged_detections = merge_adjacent_label_detections(
        normalized_detections,
        duplicate_overlap_threshold,
    )
    if debug_raw_detections:
        _print_room_label_trace(
            normalized_detections,
            merged_detections,
            known_room_words or [],
            non_room_vocabulary or [],
            room_label_min_confidence,
            known_room_min_confidence,
            alias_min_confidence,
            alias_targets,
        )

    filtered = []
    for detection in merged_detections:
        text = detection.text.strip()
        if not text:
            continue
        parsed = parse_measurement(text)
        is_alias = text.casefold() in alias_targets
        known_word = _matching_vocabulary_word(text, known_room_words or [])
        minimum_confidence = (
            alias_min_confidence
            if is_alias
            else known_room_min_confidence
            if known_word
            else room_label_min_confidence
        )
        if (
            _SINGLE_DIMENSION_PATTERN.fullmatch(text)
            or _DIMENSION_PAIR_PATTERN.fullmatch(text)
            or _IMPERIAL_VALUE_PATTERN.fullmatch(text)
            or _IMPERIAL_PAIR_PATTERN.fullmatch(text)
            or (
                parsed is not None
                and parsed.kind in {"dimension_pair", "context_dimension"}
            )
            or _SCALE_PATTERN.fullmatch(text)
        ):
            filtered.append(detection)
            continue
        if (
            is_plausible_room_label(text, non_room_vocabulary)
            and detection.confidence >= minimum_confidence
        ):
            filtered.append(detection)
    return filtered


def _matching_vocabulary_word(text: str, vocabulary: list[str]) -> str | None:
    """Return the first vocabulary entry present as a whole word or phrase."""
    normalized = text.casefold()
    for term in vocabulary:
        term = term.strip().casefold()
        if term and re.search(
            rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
            normalized,
        ):
            return term
    return None


def _print_room_label_trace(
    raw_detections: list[OCRDetection],
    merged_detections: list[OCRDetection],
    known_room_words: list[str],
    non_room_vocabulary: list[str],
    room_label_min_confidence: float,
    known_room_min_confidence: float,
    alias_min_confidence: float,
    alias_targets: set[str],
) -> None:
    """Print room-label filtering stages when OCR debugging is enabled."""
    print("ROOM LABEL TRACE")
    for raw in raw_detections:
        raw_text = raw.text.strip()
        merged_match = next(
            (
                candidate
                for candidate in merged_detections
                if raw_text.casefold() in candidate.text.casefold()
                and candidate.x_min <= raw.x_min
                and candidate.y_min <= raw.y_min
                and candidate.x_max >= raw.x_max
                and candidate.y_max >= raw.y_max
            ),
            None,
        )
        candidate = merged_match or raw
        text = candidate.text.strip()
        known_word = _matching_vocabulary_word(text, known_room_words)
        is_alias = text.casefold() in alias_targets
        minimum_confidence = (
            alias_min_confidence
            if is_alias
            else known_room_min_confidence
            if known_word
            else room_label_min_confidence
        )
        shape_ok = bool(_LABEL_TEXT_PATTERN.fullmatch(text)) and (
            is_plausible_room_label(text, None)
            or is_plausible_room_label(text, non_room_vocabulary)
        )
        excluded_word = _matching_vocabulary_word(text, non_room_vocabulary)
        confidence_ok = candidate.confidence >= minimum_confidence
        plausible = is_plausible_room_label(text, non_room_vocabulary)
        print(
            f"  raw={raw.text!r} confidence={raw.confidence:.4f}; "
            f"merged_candidate={candidate.text!r} "
            f"confidence={candidate.confidence:.4f}; "
            f"text_shape={'pass' if shape_ok else 'reject'}; "
            f"known_room_word={known_word or 'no'}; "
            f"confidence_threshold={minimum_confidence:.2f} "
            f"({'pass' if confidence_ok else 'reject'}); "
            f"non_room_vocabulary={excluded_word or 'no'}; "
            f"final={'keep' if plausible and confidence_ok else 'drop'}"
        )


def merge_adjacent_label_detections(
    detections: list[OCRDetection],
    duplicate_overlap_threshold: float = 0.5,
) -> list[OCRDetection]:
    """Remove overlapping duplicate labels, then join nearby label fragments."""
    labels = [
        (index, detection)
        for index, detection in enumerate(detections)
        if _LABEL_TEXT_PATTERN.fullmatch(detection.text.strip())
    ]
    retained_labels: list[tuple[int, OCRDetection]] = []
    duplicate_indices: set[int] = set()
    for index, detection in sorted(
        labels,
        key=lambda item: item[1].confidence,
        reverse=True,
    ):
        if any(
            _overlap_over_smaller_area(detection, retained)
            >= duplicate_overlap_threshold
            for _, retained in retained_labels
        ):
            duplicate_indices.add(index)
        else:
            retained_labels.append((index, detection))
    labels = sorted(retained_labels, key=lambda item: item[0])

    neighbors: dict[int, set[int]] = {index: set() for index, _ in labels}
    for label_index, (index_a, detection_a) in enumerate(labels):
        for index_b, detection_b in labels[label_index + 1 :]:
            if _adjacent_label_boxes(detection_a, detection_b):
                neighbors[index_a].add(index_b)
                neighbors[index_b].add(index_a)

    merged_by_index: dict[int, OCRDetection] = {}
    consumed: set[int] = set()
    for index, detection in labels:
        if index in consumed or not neighbors[index]:
            continue
        group = {index}
        pending = [index]
        while pending:
            current = pending.pop()
            for neighbor in neighbors[current] - group:
                group.add(neighbor)
                pending.append(neighbor)

        if len(group) < 2:
            continue
        group_detections = [
            item for original_index, item in labels if original_index in group
        ]
        group_detections.sort(key=lambda item: (item.y_min, item.x_min))
        merged_text = " ".join(item.text.strip() for item in group_detections)
        x_min = min(item.x_min for item in group_detections)
        y_min = min(item.y_min for item in group_detections)
        x_max = max(item.x_max for item in group_detections)
        y_max = max(item.y_max for item in group_detections)
        merged_by_index[min(group)] = OCRDetection(
            text=merged_text,
            confidence=min(item.confidence for item in group_detections),
            bbox=(
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ),
            extraction_pass=group_detections[0].extraction_pass,
        )
        consumed.update(group)

    result = []
    for index, detection in enumerate(detections):
        if index in merged_by_index:
            result.append(merged_by_index[index])
        if index not in consumed and index not in duplicate_indices:
            result.append(detection)
    return result


def _overlap_over_smaller_area(
    first: OCRDetection,
    second: OCRDetection,
) -> float:
    """Return intersection area divided by the smaller label-box area."""
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


def _adjacent_label_boxes(first: OCRDetection, second: OCRDetection) -> bool:
    """Return whether two label boxes touch on a line or on neighboring lines."""
    overlap_width = max(
        0.0,
        min(first.x_max, second.x_max) - max(first.x_min, second.x_min),
    )
    overlap_height = max(
        0.0,
        min(first.y_max, second.y_max) - max(first.y_min, second.y_min),
    )
    overlap_area = overlap_width * overlap_height
    first_area = (first.x_max - first.x_min) * (first.y_max - first.y_min)
    second_area = (second.x_max - second.x_min) * (second.y_max - second.y_min)
    if overlap_area >= 0.5 * min(first_area, second_area):
        return False

    max_height = max(first.y_max - first.y_min, second.y_max - second.y_min)
    vertical_gap = max(
        0.0,
        first.y_min - second.y_max,
        second.y_min - first.y_max,
    )
    horizontal_gap = max(
        0.0,
        first.x_min - second.x_max,
        second.x_min - first.x_max,
    )
    same_line = (
        abs(
            (first.y_min + first.y_max) / 2
            - (second.y_min + second.y_max) / 2
        )
        <= max_height * 0.55
        and horizontal_gap <= max_height * 0.75
    )
    first_center_x = (first.x_min + first.x_max) / 2
    second_center_x = (second.x_min + second.x_max) / 2
    stacked_lines = (
        abs(first_center_x - second_center_x) <= max(20.0, max_height * 0.55)
        and vertical_gap <= max_height * 0.35
    )
    return same_line or stacked_lines


class EasyOCREngine:
    def __init__(self, config: dict):
        """Initialize EasyOCR with the project's language and runtime settings."""
        import easyocr

        ocr_cfg = config["ocr"]
        self.reader = easyocr.Reader(
            ocr_cfg.get("languages", ["en"]),
            gpu=ocr_cfg.get("gpu", False),
        )
        self.cfg = ocr_cfg

    def _debug_raw(self, detections: list[OCRDetection], extraction_pass: str) -> None:
        """Print raw detections when OCR debugging is enabled in configuration."""
        if not self.cfg.get("debug_raw_detections", False):
            return
        print(f"RAW OCR [{extraction_pass}] detections: {len(detections)}")
        for index, detection in enumerate(detections, start=1):
            print(
                f"  {index:03d}: text={detection.text!r}, "
                f"confidence={detection.confidence:.4f}, "
                f"bbox={detection.bbox}"
            )

    def _convert(self, rows: Iterable, extraction_pass: str) -> list[OCRDetection]:
        """Convert EasyOCR rows into validated OCRDetection records."""
        result = []
        for row in rows:
            if len(row) != 3:
                continue
            bbox, text, confidence = row
            try:
                points = tuple((float(p[0]), float(p[1])) for p in bbox)
                if len(points) != 4:
                    continue
                result.append(
                    OCRDetection(
                        text=str(text).strip(),
                        confidence=float(confidence),
                        bbox=points,
                        extraction_pass=extraction_pass,
                    )
                )
            except (TypeError, ValueError, IndexError):
                continue
        return result

    def read_general(self, image: np.ndarray) -> list[OCRDetection]:
        """Run unrestricted OCR to capture room labels and general plan text."""
        rows = self.reader.readtext(
            image,
            detail=1,
            decoder="greedy",
            rotation_info=self.cfg.get("rotation_info", [90, 180, 270]),
            min_size=self.cfg.get("min_size", 3),
            text_threshold=self.cfg.get("text_threshold", 0.45),
            low_text=self.cfg.get("low_text", 0.25),
            link_threshold=self.cfg.get("link_threshold", 0.20),
            canvas_size=self.cfg.get("canvas_size", 3840),
            mag_ratio=self.cfg.get("mag_ratio", 1.5),
            contrast_ths=self.cfg.get("contrast_ths", 0.20),
            adjust_contrast=self.cfg.get("adjust_contrast", 0.70),
        )
        detections = self._convert(rows, "general")
        self._debug_raw(detections, "general")
        return detections

    def read_dimensions(
        self,
        image: np.ndarray,
        extraction_pass: str = "dimension",
    ) -> list[OCRDetection]:
        """Run allowlisted OCR for numeric dimensions in a prepared image variant."""
        rows = self.reader.readtext(
            image,
            detail=1,
            decoder="beamsearch",
            # so this reduces chances of ocr reading other characters
            allowlist=self.cfg.get(
                "dimension_allowlist",
                "0123456789xX.:/-'\"mMftincFTINC ",
            ),
            rotation_info=self.cfg.get("rotation_info", [90, 180, 270]),
            min_size=self.cfg.get("min_size", 3),
            text_threshold=self.cfg.get("text_threshold", 0.45),
            low_text=self.cfg.get("low_text", 0.25),
            link_threshold=self.cfg.get("link_threshold", 0.20),
            canvas_size=self.cfg.get("canvas_size", 3840),
            mag_ratio=self.cfg.get("mag_ratio", 1.5),
            contrast_ths=self.cfg.get("contrast_ths", 0.20),
            adjust_contrast=self.cfg.get("adjust_contrast", 0.70),
        )
        detections = self._convert(rows, extraction_pass)
        self._debug_raw(detections, extraction_pass)
        return detections


def _iou(a: OCRDetection, b: OCRDetection) -> float:
    """Calculate intersection-over-union for two OCR bounding boxes."""
    x1 = max(a.x_min, b.x_min)
    y1 = max(a.y_min, b.y_min)
    x2 = min(a.x_max, b.x_max)
    y2 = min(a.y_max, b.y_max)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a.x_max - a.x_min) * max(0.0, a.y_max - a.y_min)
    area_b = max(0.0, b.x_max - b.x_min) * max(0.0, b.y_max - b.y_min)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def deduplicate(detections: list[OCRDetection]) -> list[OCRDetection]:
    """Keep the strongest OCR result when boxes substantially overlap."""
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[OCRDetection] = []
    for item in ordered:
        duplicate = False
        for existing in kept:
            same_text = item.text.lower() == existing.text.lower()
            if _iou(item, existing) >= 0.50 and same_text:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept

    #check if i can avoid deduplication and IOU beacuse i need to run image at once 
