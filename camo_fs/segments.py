"""Validated COCO polygon conversion for Ultralytics segmentation labels."""

from __future__ import annotations

import math
from typing import Any


class DataIntegrityError(ValueError):
    """Raised when a COCO annotation cannot safely become a YOLO segment."""


def annotation_to_yolo(
    annotation: dict[str, Any],
    image: dict[str, Any],
    category_to_index: dict[int, int],
) -> tuple[str, bool]:
    """Convert one validated polygon instance into one normalized YOLO label line."""
    width, height = _image_dimensions(image)
    category_id = annotation.get("category_id")
    if not isinstance(category_id, int) or category_id not in category_to_index:
        raise DataIntegrityError("Annotation category is absent from the canonical taxonomy")
    _validate_bbox(annotation.get("bbox"), width, height)
    components = _validate_polygons(annotation.get("segmentation"), width, height)

    merged = _merge_components(components)
    values = [coordinate for point in merged for coordinate in (point[0] / width, point[1] / height)]
    return " ".join([str(category_to_index[category_id]), *(_format(value) for value in values)]), len(components) > 1


def _image_dimensions(image: dict[str, Any]) -> tuple[float, float]:
    width = image.get("width")
    height = image.get("height")
    if not _is_finite_number(width) or not _is_finite_number(height) or width <= 0 or height <= 0:
        raise DataIntegrityError("Image width and height must be positive finite numbers")
    return float(width), float(height)


def _validate_bbox(bbox: object, width: float, height: float) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_finite_number(value) for value in bbox):
        raise DataIntegrityError("Bounding box must contain four finite numbers")
    x, y, box_width, box_height = (float(value) for value in bbox)
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0 or x + box_width > width or y + box_height > height:
        raise DataIntegrityError("Bounding box is outside image bounds")


def _validate_polygons(segmentation: object, width: float, height: float) -> list[list[tuple[float, float]]]:
    if isinstance(segmentation, dict):
        raise DataIntegrityError("RLE segmentations are unsupported; polygon data is required")
    if not isinstance(segmentation, list) or not segmentation:
        raise DataIntegrityError("Segmentation must be a non-empty list of polygons")

    components: list[list[tuple[float, float]]] = []
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            raise DataIntegrityError("Each polygon needs at least three x/y point pairs")
        if not all(_is_finite_number(value) for value in polygon):
            raise DataIntegrityError("Polygon coordinates must be finite numbers")
        points = [(float(polygon[index]), float(polygon[index + 1])) for index in range(0, len(polygon), 2)]
        if any(x < 0 or y < 0 or x > width or y > height for x, y in points):
            raise DataIntegrityError("Polygon coordinates are outside image bounds")
        components.append(points)
    return components


def _merge_components(components: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    merged = components[0]
    for component in components[1:]:
        left_index, right_index = min(
            (
                (left_index, right_index)
                for left_index in range(len(merged))
                for right_index in range(len(component))
            ),
            key=lambda indices: (
                _squared_distance(merged[indices[0]], component[indices[1]]),
                indices[0],
                indices[1],
            ),
        )
        # YOLO stores one polygon per instance. Rotating at nearest boundaries keeps
        # every COCO component but introduces connecting edges between disconnected parts.
        merged = _rotate(merged, left_index) + _rotate(component, right_index)
    return merged


def _rotate(points: list[tuple[float, float]], index: int) -> list[tuple[float, float]]:
    return points[index:] + points[:index]


def _squared_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _format(value: float) -> str:
    return format(value, ".12g")
