from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

import numpy as np

from .parsing import is_plausible_room_label


_SINGLE_DIMENSION_PATTERN = re.compile(r"[1-9]\d{2,4}")
_DIMENSION_PAIR_PATTERN = re.compile(r"[1-9]\d{2,4}\s*[xX×]\s*[1-9]\d{2,4}")
_SCALE_PATTERN = re.compile(r"(?:SCALE\s*)?1\s*[:.]\s*\d{1,4}", re.IGNORECASE)


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
) -> list[OCRDetection]:
    """Keep dimension-like text and confident, supported room labels."""
    filtered = []
    for detection in detections:
        text = detection.text.strip()
        if not text:
            continue
        if (
            _SINGLE_DIMENSION_PATTERN.fullmatch(text)
            or _DIMENSION_PAIR_PATTERN.fullmatch(text)
            or _SCALE_PATTERN.fullmatch(text)
        ):
            filtered.append(detection)
            continue
        if (
            is_plausible_room_label(text)
            and detection.confidence >= room_label_min_confidence
        ):
            filtered.append(detection)
    return filtered


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
