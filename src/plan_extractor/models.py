from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def center(self) -> Point:
        """Return the geometric center of the bounding box.

        The center is used for spatial association and overall-dimension classification.
        """
        return Point(
            x=(self.x_min + self.x_max) / 2.0,
            y=(self.y_min + self.y_max) / 2.0,
        )


class ConfidenceBreakdown(BaseModel):
    ocr: float = Field(ge=0, le=1)
    association: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)


class DocumentSource(BaseModel):
    file_name: str
    page_number: int = 1


class SourceReference(BaseModel):
    raw_text: str
    bounding_box: BoundingBox
    extraction_pass: str


class AssociatedElement(BaseModel):
    element_id: Optional[str] = None
    element_type: str = "unresolved"
    name: Optional[str] = None


class Measurement(BaseModel):
    measurement_id: str
    type: str
    raw_text: str
    value: Optional[float] = None
    secondary_value: Optional[float] = None
    unit: Optional[str] = None
    #everytime mm cannot be a unit in there or generally we are giving mm
    normalized_value_mm: Optional[float] = None
    normalized_secondary_value_mm: Optional[float] = None
    orientation: Literal["horizontal", "vertical", "unknown"] = "unknown"
    source_type: Literal[
        "explicit_text",
        "dimension_pair_text",
        "scale_calculated",
        "native_vector",
        "unknown",
    ] = "explicit_text"
    associated_element: AssociatedElement
    source: SourceReference
    confidence: ConfidenceBreakdown
    corroborated_by_dimension_line: bool = False
    ambiguity: Optional[str] = None
    review_required: bool = False


class Space(BaseModel):
    space_id: str
    name: str
    category: str
    source: SourceReference
    dimensions: list[str] = Field(default_factory=list)


class ScaleInfo(BaseModel):
    detected: bool = False
    raw_text: Optional[str] = None
    ratio: Optional[float] = None
    confidence: Optional[float] = None


class OverallDimensionSource(SourceReference):
    confidence: ConfidenceBreakdown


class OverallDimensions(BaseModel):
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    width_source: Optional[OverallDimensionSource] = None
    height_source: Optional[OverallDimensionSource] = None


class WarningItem(BaseModel):
    code: str
    message: str
    measurement_id: Optional[str] = None


class ExtractionSummary(BaseModel):
    ocr_detections: int
    spaces_detected: int
    measurements_detected: int
    review_required: int
    high_confidence: int
    medium_confidence: int
    low_confidence: int


class EvaluationMetrics(BaseModel):
    expected: int
    matched: int
    incorrect_value: int = 0
    incorrect_unit: int = 0
    missed: int
    unexpected: int
    duplicate_count: int = 0
    precision: float
    recall: float
    f1: float
    accuracy: float = 0.0


class PlanResult(BaseModel):
    schema_version: str = "1.0"
    plan_id: str
    status: Literal["success", "partial", "failed"]
    source_file: str
    input_type: str
    image_width_px: int
    image_height_px: int
    primary_unit: str = "mm"
    document_source: DocumentSource
    scale: ScaleInfo
    overall_dimensions: OverallDimensions
    spaces: list[Space]
    measurements: list[Measurement]
    warnings: list[WarningItem]
    summary: ExtractionSummary
    evaluation: Optional[EvaluationMetrics] = None
