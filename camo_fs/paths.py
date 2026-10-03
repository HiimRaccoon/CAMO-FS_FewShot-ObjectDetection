"""Dataset input and generated-artifact path contracts."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    """Resolved paths for one CAMO-FS dataset input and Kaggle work area."""

    data_root: Path
    work_root: Path
    images_dir: Path
    annotations_root: Path
    few_shot_dir: Path
    test_json: Path
    prepared_root: Path
    runs_root: Path
    results_root: Path

    DEFAULT_DATA_ROOT: ClassVar[Path] = Path(
        "/kaggle/input/datasets/danhnt/camo-fs-dataset"
    )
    DEFAULT_WORK_ROOT: ClassVar[Path] = Path("/kaggle/working")

    @classmethod
    def from_root(
        cls,
        data_root: Path = DEFAULT_DATA_ROOT,
        work_root: Path = DEFAULT_WORK_ROOT,
    ) -> "DatasetPaths":
        """Build paths and prevent generated artifacts from entering an input tree."""
        resolved_data_root = Path(data_root)
        resolved_work_root = Path(work_root)

        if _is_within(resolved_work_root, Path("/kaggle/input")):
            raise ValueError("work_root must not be inside /kaggle/input")
        if _is_within(resolved_work_root, resolved_data_root):
            raise ValueError("work_root must not be inside data_root")

        annotations_root = resolved_data_root / "few-shot-annotations"
        return cls(
            data_root=resolved_data_root,
            work_root=resolved_work_root,
            images_dir=resolved_data_root / "images" / "images",
            annotations_root=annotations_root,
            few_shot_dir=annotations_root / "subsplit" / "split1",
            test_json=annotations_root / "camo5_test_split1.json",
            prepared_root=resolved_work_root / "camo_fs_yolo",
            runs_root=resolved_work_root / "runs",
            results_root=resolved_work_root / "results",
        )


def _is_within(candidate: Path, parent: Path) -> bool:
    """Return whether ``candidate`` resolves to ``parent`` or one of its children."""
    try:
        candidate.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True
