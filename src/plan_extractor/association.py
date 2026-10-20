from __future__ import annotations

from math import hypot

from .models import BoundingBox



def infer_orientation(box: BoundingBox) -> str:
    """Classify a bounding box as horizontal, vertical, or unknown.

    This supports orientation-aware association of dimension labels with spaces.
    """
    width = box.x_max - box.x_min
    height = box.y_max - box.y_min
    if width > height * 1.5:
        return "horizontal"
    if height > width * 1.5:
        return "vertical"
    return "unknown"


def normalized_center_distance(
    a: BoundingBox,
    b: BoundingBox,
    image_width: int,
    image_height: int,
) -> float:
    """Measure the centers' distance relative to the image dimensions.

    Normalization makes spatial comparisons consistent across differently sized plans.
    """
    ac = a.center
    bc = b.center
    dx = (ac.x - bc.x) / max(image_width, 1)
    dy = (ac.y - bc.y) / max(image_height, 1)
    return hypot(dx, dy)


def axis_alignment_score(
    measurement_box: BoundingBox,
    space_box: BoundingBox,
    image_width: int,
    image_height: int,
) -> float:
    """How well a measurement lines up with a room, independent of raw
    distance. A measurement drawn along a room's edge usually shares
    roughly the same x-range (for a vertical/height dimension) or the
    same y-range (for a horizontal/width dimension) as that room's own
    label box. Returns 0.0 (no alignment) to 1.0 (perfect alignment)."""
    m_center = measurement_box.center
    s_center = space_box.center

    x_gap = abs(m_center.x - s_center.x) / max(image_width, 1)
    y_gap = abs(m_center.y - s_center.y) / max(image_height, 1)

    # Best-case alignment is whichever axis the measurement lines up
    # on more closely — horizontal dimensions align in y, vertical
    # dimensions align in x.
    best_gap = min(x_gap, y_gap)
    return max(0.0, 1.0 - best_gap * 4.0)  # 0.25 normalized gap -> score 0


def nearest_space(
    measurement_box: BoundingBox,
    spaces,
    image_width: int,
    image_height: int,
    max_distance: float,
    alignment_weight: float = 0.4,
):
    """Choose the best nearby space using distance and axis alignment.

    It returns the space, confidence score, and raw normalized distance for review logic.
    """
    if not spaces:
        return None, 0.0, None

    best_space = None
    best_score = None
    best_distance = None

    for space in spaces:
        distance = normalized_center_distance(
            measurement_box,
            space.source.bounding_box,
            image_width,
            image_height,
        )
        alignment = axis_alignment_score(
            measurement_box,
            space.source.bounding_box,
            image_width,
            image_height,
        )
        # Combine: a close-but-misaligned label and a slightly-farther
        # but well-aligned label are now both considered, instead of
        # raw distance alone always winning.
        distance_score = max(0.0, 1.0 - (distance / max_distance)) if max_distance else 0.0
        combined_score = (1 - alignment_weight) * distance_score + alignment_weight * alignment

        if best_score is None or combined_score > best_score:
            best_space = space
            best_score = combined_score
            best_distance = distance

    if best_distance is None or best_distance > max_distance:
        return None, 0.0, best_distance

    return best_space, round(max(0.0, min(1.0, best_score)), 4), best_distance