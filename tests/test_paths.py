from pathlib import Path

import pytest

from camo_fs.paths import DatasetPaths


def test_default_kaggle_paths() -> None:
    paths = DatasetPaths.from_root()

    assert paths.data_root == Path("/kaggle/input/datasets/danhnt/camo-fs-dataset")
    assert paths.work_root == Path("/kaggle/working")
    assert paths.images_dir == Path(
        "/kaggle/input/datasets/danhnt/camo-fs-dataset/images/images"
    )
    assert paths.annotations_root == Path(
        "/kaggle/input/datasets/danhnt/camo-fs-dataset/few-shot-annotations"
    )
    assert paths.few_shot_dir == Path(
        "/kaggle/input/datasets/danhnt/camo-fs-dataset/"
        "few-shot-annotations/subsplit/split1"
    )
    assert paths.test_json == Path(
        "/kaggle/input/datasets/danhnt/camo-fs-dataset/"
        "few-shot-annotations/camo5_test_split1.json"
    )
    assert paths.prepared_root == Path("/kaggle/working/camo_fs_yolo")
    assert paths.runs_root == Path("/kaggle/working/runs")
    assert paths.results_root == Path("/kaggle/working/results")


def test_custom_roots_for_fixture(tmp_path: Path) -> None:
    data_root = tmp_path / "fixture-data"
    work_root = tmp_path / "fixture-work"

    paths = DatasetPaths.from_root(data_root=data_root, work_root=work_root)

    assert paths.data_root == data_root
    assert paths.work_root == work_root
    assert paths.images_dir == data_root / "images" / "images"
    assert paths.annotations_root == data_root / "few-shot-annotations"
    assert paths.few_shot_dir == data_root / "few-shot-annotations/subsplit/split1"
    assert paths.test_json == data_root / "few-shot-annotations/camo5_test_split1.json"
    assert paths.prepared_root == work_root / "camo_fs_yolo"
    assert paths.runs_root == work_root / "runs"
    assert paths.results_root == work_root / "results"


def test_rejects_output_inside_input(tmp_path: Path) -> None:
    data_root = tmp_path / "fixture-data"

    with pytest.raises(ValueError, match="work_root.*data_root"):
        DatasetPaths.from_root(data_root=data_root, work_root=data_root / "generated")


def test_rejects_output_anywhere_under_kaggle_input() -> None:
    with pytest.raises(ValueError, match="work_root.*kaggle/input"):
        DatasetPaths.from_root(
            work_root=Path("/kaggle/input/another-dataset/generated")
        )
