import json
import struct
from pathlib import Path

from camo_fs.annotations import Taxonomy, audit_shot
from camo_fs.paths import DatasetPaths


CATEGORIES = [
    {"id": 20, "name": "Fox"},
    {"id": 5, "name": "Bat"},
]


def _write_png(path: Path, width: int = 2, height: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    )


def _annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    x: int = 0,
) -> dict:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, 0, 1, 1],
        "segmentation": [[x, 0, x + 1, 0, x + 1, 1]],
    }


def _image(image_id: int, filename: str, width: int = 2, height: int = 2) -> dict:
    return {"id": image_id, "file_name": filename, "width": width, "height": height}


def _write_json(path: Path, images: list[dict], annotations: list[dict], categories=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": CATEGORIES if categories is None else categories,
            }
        ),
        encoding="utf-8",
    )


def _paths_and_taxonomy(tmp_path: Path) -> tuple[DatasetPaths, Taxonomy]:
    paths = DatasetPaths.from_root(tmp_path / "input", tmp_path / "work")
    test_image = _image(900, "test.png")
    _write_png(paths.images_dir / test_image["file_name"])
    _write_json(paths.test_json, [test_image], [_annotation(900, 900, 5)])
    return paths, Taxonomy.from_test_json(paths.test_json)


def _write_shot_file(
    paths: DatasetPaths,
    class_name: str,
    images: list[dict],
    annotations: list[dict],
    categories=None,
) -> None:
    for image in images:
        image_path = paths.images_dir / image["file_name"]
        if not image_path.exists():
            _write_png(image_path, image["width"], image["height"])
    _write_json(
        paths.few_shot_dir / f"camo5_{class_name}_1shot_split1.json",
        images,
        annotations,
        categories,
    )


def _write_complete_shot(paths: DatasetPaths) -> None:
    _write_shot_file(
        paths,
        "Bat",
        [_image(1, "bat.png")],
        [_annotation(1, 1, 5)],
    )
    _write_shot_file(
        paths,
        "Fox",
        [_image(2, "fox.png")],
        [_annotation(2, 2, 20)],
    )


def test_taxonomy_sorts_noncontiguous_category_ids_into_yolo_indices(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)

    assert taxonomy.names == ["Bat", "Fox"]
    assert taxonomy.category_to_index == {5: 0, 20: 1}
    assert taxonomy.category_pairs == ((5, "Bat"), (20, "Fox"))


def test_audit_rejects_taxonomy_mismatch(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)
    _write_complete_shot(paths)
    _write_shot_file(
        paths,
        "Bat",
        [_image(1, "bat.png")],
        [_annotation(1, 1, 5)],
        categories=[{"id": 5, "name": "Bat"}, {"id": 20, "name": "Wrong Fox"}],
    )

    report = audit_shot(1, paths, taxonomy)

    assert "taxonomy_mismatch" in {issue.code for issue in report.errors}


def test_audit_requires_exact_official_shot_file_set(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)
    _write_shot_file(
        paths,
        "Bat",
        [_image(1, "bat.png")],
        [_annotation(1, 1, 5)],
    )
    _write_shot_file(
        paths,
        "Extra",
        [_image(3, "extra.png")],
        [_annotation(3, 3, 20)],
    )

    report = audit_shot(1, paths, taxonomy)

    codes = {issue.code for issue in report.errors}
    assert "missing_shot_file" in codes
    assert "unexpected_shot_file" in codes


def test_reused_id_in_different_image_is_retained(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)
    _write_shot_file(
        paths,
        "Bat",
        [_image(1, "bat.png")],
        [_annotation(7, 1, 5)],
    )
    _write_shot_file(
        paths,
        "Fox",
        [_image(2, "fox.png")],
        [_annotation(7, 2, 20)],
    )

    report = audit_shot(1, paths, taxonomy)

    assert report.annotation_count == 2
    assert not report.errors
    assert "reused_annotation_id" in {issue.code for issue in report.warnings}


def test_reused_id_tracks_all_image_class_contexts(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)
    _write_shot_file(
        paths,
        "Bat",
        [_image(1, "bat.png")],
        [_annotation(826, 1, 5, x=0)],
    )
    _write_shot_file(
        paths,
        "Fox",
        [_image(2, "fox.png")],
        [_annotation(826, 2, 20, x=0), _annotation(826, 2, 20, x=1)],
    )

    report = audit_shot(1, paths, taxonomy)

    assert "reused_annotation_id" in {issue.code for issue in report.warnings}
    assert "conflicting_annotation_id" in {issue.code for issue in report.errors}


def test_audit_deduplicates_exact_annotation_content_and_reports_it(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)
    _write_complete_shot(paths)
    duplicate = _annotation(8, 1, 5)
    _write_shot_file(
        paths,
        "Bat",
        [_image(1, "bat.png")],
        [duplicate, duplicate.copy()],
    )

    report = audit_shot(1, paths, taxonomy)

    assert report.annotation_count == 2
    assert "duplicate_annotation" in {issue.code for issue in report.errors}


def test_nonfinite_annotation_geometry_is_reported_not_crashed(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)
    _write_complete_shot(paths)
    malformed = _annotation(8, 1, 5)
    malformed["bbox"] = [float("nan"), 0, 1, 1]
    _write_shot_file(paths, "Bat", [_image(1, "bat.png")], [malformed])

    report = audit_shot(1, paths, taxonomy)

    assert "invalid_annotation_geometry" in {issue.code for issue in report.errors}


def test_audit_rejects_conflicting_image_metadata(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)
    _write_shot_file(
        paths,
        "Bat",
        [_image(1, "shared.png", width=2)],
        [_annotation(1, 1, 5)],
    )
    _write_shot_file(
        paths,
        "Fox",
        [_image(1, "shared.png", width=3)],
        [_annotation(2, 1, 20)],
    )

    report = audit_shot(1, paths, taxonomy)

    assert "conflicting_image_metadata" in {issue.code for issue in report.errors}


def test_audit_rejects_conflicting_geometry_for_same_annotation_id(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)
    _write_complete_shot(paths)
    _write_shot_file(
        paths,
        "Bat",
        [_image(1, "bat.png")],
        [_annotation(8, 1, 5, x=0), _annotation(8, 1, 5, x=1)],
    )

    report = audit_shot(1, paths, taxonomy)

    assert "conflicting_annotation_id" in {issue.code for issue in report.errors}


def test_audit_rejects_annotation_that_references_missing_image(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)
    _write_shot_file(
        paths,
        "Bat",
        [_image(1, "bat.png")],
        [_annotation(1, 99, 5)],
    )
    _write_shot_file(
        paths,
        "Fox",
        [_image(2, "fox.png")],
        [_annotation(2, 2, 20)],
    )

    report = audit_shot(1, paths, taxonomy)

    assert "annotation_image_missing" in {issue.code for issue in report.errors}


def test_audit_rejects_declared_image_dimensions_that_do_not_match_source(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)
    _write_complete_shot(paths)
    _write_shot_file(
        paths,
        "Bat",
        [_image(1, "bat.png", width=3)],
        [_annotation(1, 1, 5)],
    )

    report = audit_shot(1, paths, taxonomy)

    assert "image_dimension_mismatch" in {issue.code for issue in report.errors}


def test_audit_rejects_train_test_image_overlap(tmp_path: Path) -> None:
    paths, taxonomy = _paths_and_taxonomy(tmp_path)
    _write_shot_file(
        paths,
        "Bat",
        [_image(900, "test.png")],
        [_annotation(1, 900, 5)],
    )
    _write_shot_file(
        paths,
        "Fox",
        [_image(2, "fox.png")],
        [_annotation(2, 2, 20)],
    )

    report = audit_shot(1, paths, taxonomy)

    assert "train_test_image_id_overlap" in {issue.code for issue in report.errors}
    assert "train_test_filename_overlap" in {issue.code for issue in report.errors}
