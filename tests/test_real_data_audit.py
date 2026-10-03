import os
from pathlib import Path

import pytest

from camo_fs.annotations import Taxonomy, audit_shot
from camo_fs.paths import DatasetPaths


pytestmark = pytest.mark.real_data


@pytest.mark.real_data
def test_official_splits_match_audited_counts(tmp_path: Path) -> None:
    data_root_value = os.environ.get("CAMO_FS_DATA_ROOT")
    if data_root_value is None:
        pytest.skip("Set CAMO_FS_DATA_ROOT to run the read-only official-data audit")

    paths = DatasetPaths.from_root(Path(data_root_value), tmp_path / "work")
    taxonomy = Taxonomy.from_test_json(paths.test_json)

    expected_counts = {1: (47, 47), 2: (94, 94), 3: (141, 141), 5: (197, 235)}
    for shot, (image_count, annotation_count) in expected_counts.items():
        report = audit_shot(shot, paths, taxonomy)
        assert not report.errors
        assert (report.image_count, report.annotation_count) == (image_count, annotation_count)
