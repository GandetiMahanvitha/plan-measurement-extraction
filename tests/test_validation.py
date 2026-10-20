from plan_extractor.models import (
    AssociatedElement,
    BoundingBox,
    ConfidenceBreakdown,
    Measurement,
    SourceReference,
)
from plan_extractor.validation import ExpectedMeasurement, evaluate_measurements


def make_measurement(value, secondary=None, name=None):
    return Measurement(
        measurement_id="DIM-1",
        type="dimension",
        raw_text=str(value),
        value=value,
        secondary_value=secondary,
        unit="mm",
        associated_element=AssociatedElement(
            name=name,
        ),
        source=SourceReference(
            raw_text=str(value),
            bounding_box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
        ),
        confidence=ConfidenceBreakdown(
            ocr=1,
            association=1,
            overall=1,
        ),
    )


def test_measurement_output_contains_only_retained_fields():
    measurement = make_measurement(3600, 3300)
    output = measurement.model_dump(mode="json")

    assert output["associated_element"] == {"element_id": None, "name": None}
    assert output["value"] == 3600
    assert output["secondary_value"] == 3300
    assert output["unit"] == "mm"
    assert output["source"] == {
        "raw_text": "3600",
        "bounding_box": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
    }
    assert not {
        "element_type",
        "orientation",
        "corroborated_by_dimension_line",
        "normalized_value_mm",
        "normalized_secondary_value_mm",
        "source_type",
        "extraction_pass",
    }.intersection(output)


def test_evaluation_perfect_match():
    expected = [
        ExpectedMeasurement(
            type="dimension",
            value_mm=3600,
            secondary_value_mm=3300,
            element_name="BEDROOM 1",
        )
    ]
    actual = [make_measurement(3600, 3300, "BEDROOM 1")]
    metrics = evaluate_measurements(expected, actual)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_evaluation_prefers_closest_same_name_measurement():
    expected = [
        ExpectedMeasurement(
            type="room_dimension_pair",
            value_mm=3600,
            secondary_value_mm=3300,
            element_name="BEDROOM 1",
        )
    ]
    actual = [
        make_measurement(2500, 2500, "BEDROOM 1"),
        make_measurement(3600, 3300, "BEDROOM 1"),
    ]

    metrics = evaluate_measurements(expected, actual)

    assert metrics.matched == 1
    assert metrics.incorrect_value == 0
    assert metrics.unexpected == 1
