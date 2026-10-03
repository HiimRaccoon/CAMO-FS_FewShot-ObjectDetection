import math

import pytest

from camo_fs.segments import DataIntegrityError, annotation_to_yolo


IMAGE = {"id": 1, "file_name": "fixture.png", "width": 100, "height": 50}
CATEGORY_TO_INDEX = {5: 0, 20: 1}


def _annotation(
    segmentation: object = None,
    *,
    category_id: int = 20,
    bbox: object = None,
) -> dict:
    return {
        "id": 7,
        "image_id": 1,
        "category_id": category_id,
        "bbox": [0, 0, 50, 25] if bbox is None else bbox,
        "segmentation": [[0, 0, 50, 0, 50, 25]] if segmentation is None else segmentation,
    }


def _coordinates(line: str) -> list[float]:
    return [float(value) for value in line.split()[1:]]


def test_annotation_to_yolo_normalizes_simple_polygon_with_contiguous_class_index() -> None:
    line, multi_polygon = annotation_to_yolo(_annotation(), IMAGE, CATEGORY_TO_INDEX)

    assert line == "1 0 0 0.5 0 0.5 0.5"
    assert multi_polygon is False


def test_annotation_to_yolo_keeps_vertices_from_every_disconnected_polygon() -> None:
    annotation = _annotation(
        [
            [0, 0, 10, 0, 10, 10],
            [80, 30, 90, 30, 90, 40],
        ],
        bbox=[0, 0, 90, 40],
    )

    line, multi_polygon = annotation_to_yolo(annotation, IMAGE, CATEGORY_TO_INDEX)

    coordinates = _coordinates(line)
    assert multi_polygon is True
    assert len(coordinates) == 12
    assert {0.0, 0.1, 0.2, 0.6, 0.8, 0.9} <= set(coordinates)


def test_multi_polygon_yolo_output_is_renderable_from_all_source_vertices() -> None:
    line, _ = annotation_to_yolo(
        _annotation(
            [
                [0, 0, 10, 0, 10, 10],
                [80, 30, 90, 30, 90, 40],
            ],
            bbox=[0, 0, 90, 40],
        ),
        IMAGE,
        CATEGORY_TO_INDEX,
    )

    coordinates = _coordinates(line)
    rendered_points = list(zip(coordinates[::2], coordinates[1::2], strict=True))
    assert {(0.0, 0.0), (0.1, 0.0), (0.1, 0.2)} <= set(rendered_points)
    assert {(0.8, 0.6), (0.9, 0.6), (0.9, 0.8)} <= set(rendered_points)


@pytest.mark.parametrize(
    ("annotation", "image", "mapping"),
    [
        (_annotation(segmentation={"counts": "abc", "size": [50, 100]}), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(segmentation=[[0, 0, 10]]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(segmentation=[[0, 0, 10, 0]]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(segmentation=[[0, 0, math.nan, 0, 10, 10]]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(segmentation=[[0, 0, 101, 0, 10, 10]]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(bbox=[0, 0, 101, 1]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(bbox=[0, 0, -1, 1]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(), {**IMAGE, "width": 0}, CATEGORY_TO_INDEX),
        (_annotation(category_id=999), IMAGE, CATEGORY_TO_INDEX),
    ],
)
def test_annotation_to_yolo_rejects_malformed_input(
    annotation: dict,
    image: dict,
    mapping: dict[int, int],
) -> None:
    with pytest.raises(DataIntegrityError):
        annotation_to_yolo(annotation, image, mapping)


def test_annotation_to_yolo_rejects_nonfinite_bbox() -> None:
    with pytest.raises(DataIntegrityError):
        annotation_to_yolo(_annotation(bbox=[0, 0, math.inf, 1]), IMAGE, CATEGORY_TO_INDEX)
