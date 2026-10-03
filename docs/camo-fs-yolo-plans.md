# CAMO-FS YOLO Baseline and FG/BG Triplet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task by task. Track execution in [tasks](camo-fs-yolo-tasks.md). Write a failing test before production code for each behavioral task.

**Goal:** Produce a Kaggle-runnable comparison of native YOLO11n-Seg and YOLO11n-Seg with class-agnostic FG/BG Triplet Loss on the official CAMO-FS 1/2/3/5-shot splits.

**Architecture:** A small Python package owns COCO auditing/conversion, prepared dataset creation, run identity, triplet sampling, and reporting. Four thin CLI scripts call that package. The enhanced path extends the installed Ultralytics segmentation trainer/model in project code, captures the verified first P3/8 `Segment` head input, and adds a bounded auxiliary loss to the native loss. The baseline uses the native trainer. Both read the same prepared dataset.

**Tech Stack:** Python 3, Ultralytics YOLO11n-Seg, PyTorch, NumPy, Pillow/OpenCV, PyYAML, pytest; Kaggle single GPU for enhanced training. Pin Ultralytics exactly only after the integration test and enhanced smoke run succeed on Kaggle.

**Spec:** [Feature specification](camo-fs-yolo-spec.md); [design contract](camo-fs-yolo-design-decisions.md).

## Global constraints

- Read official data from `/kaggle/input/datasets/danhnt/camo-fs-dataset`; write generated artifacts only under `/kaggle/working`.
- Training uses the exact official `*_Kshot_split1.json` files; final evaluation uses `camo5_test_split1.json`. No random split, no test-as-validation.
- Taxonomy is 47 classes from the test JSON `categories`, sorted by category ID; 5-shot must retain 197 images and 235 instances.
- The test set is shared once; each shot has its own train tree. Never globally deduplicate by annotation ID.
- Both methods load the same original `yolo11n-seg.pt`, use `val=False`, `overlap_mask=False`, `mosaic=0.0`, `mixup=0.0`, `copy_paste=0.0`, and the same remaining augmentation preset.
- Initial geometric preset for both methods: normal resize/letterbox and horizontal flip; set degrees, translate, scale, shear, perspective, vertical flip, and other geometry-changing transforms to zero until their valid-region propagation is verified. HSV/color augmentation may remain. Record every effective augmentation value in manifests and hashes. This is a conservative implementation choice permitted by the spec.
- The enhanced method uses transformed GT instance masks from the current training batch, never original polygons or predicted masks; negatives come from valid unpadded background in the same transformed image.
- Enhanced defaults: weight `0.1`, margin `0.3`, at most `16` triplets per valid instance, seed `2024`, single GPU `device=0`. They are engineering defaults, not claimed optima.
- `--overwrite` starts fresh; `--resume` continues only the matching interrupted run. The complete run identity includes method and a configuration fingerprint.
- No numeric `model.model[16]` assumption. Locate the `Segment` head semantically and validate its first spatial input as P3/8 on the installed version.
- Do not pin an unverified Ultralytics version. Installation, target-version compatibility, checkpoint availability, and training execution are explicit Kaggle gates.
- Work in the current repository without adding dataset images, prepared datasets, checkpoints, or result artifacts to Git. Preserve existing untracked `data/`, notebook, `.agents/`, and docs.

## File map and ownership

| Path | Responsibility |
| --- | --- |
| `pyproject.toml`, `requirements.txt`, `.gitignore` | Editable package, test/dependency setup, generated-artifact exclusion |
| `camo_fs/paths.py` | Kaggle defaults and validated input/output path resolution |
| `camo_fs/annotations.py` | COCO loading, canonical taxonomy, merge, deduplication, integrity audit |
| `camo_fs/segments.py` | Validated COCO polygons to YOLO instance polygon labels |
| `camo_fs/prepare.py` | Audit-first materialization, shared test, per-shot train trees, YAML/manifest |
| `camo_fs/runs.py` | Typed run config, stable fingerprint, run paths, manifest, overwrite/resume guard, summary status/upsert |
| `camo_fs/valid_region.py` | Version-independent valid-region geometry and conservative feature-cell reduction from explicit inputs |
| `camo_fs/triplet.py` | Per-instance GT mask projection, bounded sampling, normalized margin loss |
| `camo_fs/ultralytics_ext.py` | Target-version geometry adapter, verified P3 capture, and smallest project-owned trainer/model extension |
| `camo_fs/train.py` | Baseline/enhanced dispatch, config parity, seed, training lifecycle |
| `camo_fs/evaluate.py` | `last.pt` validation on `split="test"`, six AP metrics, summary upsert |
| `camo_fs/visualize.py` | Deterministic prediction renders and optional sampler debug view |
| `scripts/{prepare_dataset,train_yolo,evaluate_yolo,visualize_predictions}.py` | Argument parsing and package calls only |
| `tests/test_*.py` | Dataset-independent synthetic behavior tests and target-version integration marker |
| `README.md`, optional `notebooks/kaggle_entrypoint.ipynb` | Kaggle setup, execution, limits, and result presentation |

`pyproject.toml` should support `pip install -e .`, making direct `python scripts/...py` invocations import the package without script-local path tricks. `requirements.txt` is provisional until the Kaggle version gate; never present a placeholder version as pinned.

## Interface contract for tasks

- `camo_fs.paths.DatasetPaths.from_root(data_root: Path, work_root: Path) -> DatasetPaths`: exposes `images_dir`, `few_shot_dir`, `test_json`, `prepared_root`, `runs_root`, and `results_root`; rejects output paths under input root.
- `camo_fs.annotations.Taxonomy.from_test_json(path: Path) -> Taxonomy`: exposes ordered `names: list[str]` and `category_to_index: dict[int, int]`.
- `camo_fs.annotations.audit_shot(shot: int, paths: DatasetPaths, taxonomy: Taxonomy) -> AuditReport`: contains deduplicated image/annotation records, source files, errors, warnings, and counts; no dataset writes.
- `camo_fs.segments.annotation_to_yolo(annotation: dict, image: dict, category_to_index: dict[int,int]) -> tuple[str, bool]`: label line plus multi-polygon flag; invalid input raises `DataIntegrityError`.
- `camo_fs.prepare.prepare_selected(shots: list[int], paths: DatasetPaths, overwrite: bool, continue_on_error: bool) -> list[PreparedShot]`: always audit the shared test first. With `continue_on_error=False`, audit **all** selected training shots and the shared test before creating any prepared dataset artifact; if any audit fails, write audit/error reports only and materialize nothing. With `continue_on_error=True`, audit and materialize each shot independently after the shared test audit passes, record each failed shot, and copy the common test tree only once. A failed shared-test audit prevents every shot from materializing.
- `camo_fs.runs.RunConfig` is immutable and includes `run_kind` (`benchmark` or `smoke`); `fingerprint(config: RunConfig, weights_sha256: str, data_sha256: str) -> str`, `run_path(config: RunConfig, runs_root: Path) -> Path`, `verify_resume(manifest: dict, config: RunConfig, weights_sha256: str, data_sha256: str) -> None`, and `upsert_summary(results_dir: Path, row: dict) -> None` share one canonical identity serialization. `run_kind` also enters the fingerprint and summary so smoke rows cannot overwrite benchmark rows.
- `camo_fs.valid_region.valid_letterbox_mask(source_hw: tuple[int,int], input_hw: tuple[int,int], ratio_pad: tuple | None = None) -> torch.BoolTensor`: version-independent geometry from explicit inputs, producing one mask per image at network-input resolution. A P3 cell is valid background only if **every input pixel in its corresponding spatial cell** is inside the valid transformed-image region (conservative all-valid reduction); any partly padded cell is invalid. Do not use nearest-neighbor or any-valid downsampling for this validity mask. T09 owns the target-version adapter that extracts and verifies actual preprocessing geometry before calling this function; T07 must not guess Ultralytics-internal metadata.
- `camo_fs.triplet.sample_and_loss(features: Tensor, masks: Tensor, batch_idx: Tensor, valid: Tensor, count: int, margin: float, generator: torch.Generator) -> TripletResult`: returns differentiable scalar, sample/skip counts, and optional sampled positions for debug visualization.
- `camo_fs.ultralytics_ext.P3Capture` manages one forward's head input and lifecycle; `FGSegmentationTrainer` creates the extended model. The extension's model `loss(batch, preds=None)` delegates to native loss and adds the auxiliary term to the loss scalar while preserving the **exact native return structure and loss-item semantics** expected by the verified trainer. Triplet metrics use a separate logging channel, never an added return element.
- `camo_fs.train.train_one(config: RunConfig, paths: DatasetPaths, resume: bool, overwrite: bool) -> Path`: returns its `last.pt` path only after the requested run completes.
- `camo_fs.evaluate.evaluate_one(run_dir: Path, data_yaml: Path, results_dir: Path) -> dict`: loads `last.pt`, calls official test split, and upserts one row keyed by full identity.
- `camo_fs.visualize.visualize_one(run_dir: Path, data_yaml: Path, num_images: int = 20, conf: float = 0.25, seed: int = 2024) -> list[Path]`.

If an installed Ultralytics version makes an interface above impossible, the implementation task must stop at its compatibility gate, record the exact incompatible signature, and revise this plan/spec deliberately. It must not silently bypass triplet training.

## Review focus

These six failure modes are easy to miss; their owning tasks below must include the named test.

1. Distinct 5-shot objects sharing annotation IDs `826`, `386`, `387` must survive merge; `tests/test_annotations.py::test_reused_id_in_different_image_is_retained` (Task 2).
2. A generated YAML must never map `val` to official test, including fallback parser mode; `tests/test_prepare.py::test_val_never_points_to_test` (Task 4).
3. Background sampling must reject letterbox padding and partially padded feature cells; `tests/test_triplet.py::test_negative_excludes_padding` (Task 8).
4. A failed/multiple P3 hook capture must not reuse the prior batch tensor; `tests/test_ultralytics_ext.py::test_capture_rejects_stale_or_multiple_forward` (Task 9).
5. Two 5-shot/seed-2024 runs with different epochs or triplet weight must have different paths and summary keys; `tests/test_runs.py::test_training_config_changes_identity` (Task 5) and `tests/test_evaluate.py::test_upsert_preserves_distinct_config_hashes` (Task 11).
6. Enhanced loss must produce a nonzero gradient on the captured P3 tensor while baseline must neither instantiate nor execute triplet sampling; `tests/test_ultralytics_ext.py::test_enhanced_p3_gradient_and_baseline_bypasses_triplet` (Task 9).

## Execution sequence

| Phase | Tasks | Exit gate |
| --- | --- | --- |
| A: Correct data | 1–4 | Synthetic prep tests pass; real local annotation audit confirms official counts without writing to `data/` |
| B: Reproducible baseline | 5–6 | Identity/resume tests pass; baseline train command can complete on Kaggle with `last.pt` |
| C: Auxiliary method | 7–9 | Synthetic mask/sampler/hook tests pass; installed-version integration verifies geometry metadata, gradient, feature contract, and a prepared multi-polygon label through the real segmentation dataloader |
| D: Runtime proof | 10 | One enhanced smoke run completes on Kaggle, has nonzero triplet signal, and its reloaded `last.pt` yields at least one paired box/mask prediction on prepared training imagery |
| E: Results and handoff | 11–13 | Six AP fields, statuses, deterministic images, README and Kaggle commands verified; full eight-run comparison can be executed |

The detailed red/green steps and dependency graph are in [tasks](camo-fs-yolo-tasks.md). A task is complete only after its focused test and the full available local suite pass. Kaggle-only gates must be marked **not verified** until actually run; do not infer their success from synthetic tests.

## Verification matrix

| Evidence | Command or artifact | Required observation |
| --- | --- | --- |
| Synthetic suite | `pytest -q` | No failing dataset or triplet tests; no CAMO-FS path required |
| Local read-only source audit | `pytest -q -m real_data` where local CAMO-FS files are available | 47/94/141/197 unique train images and 47/94/141/235 instances; multi-polygon counts 5/10/16/38; 2,655 test images, 3,108 instances; no source writes |
| Target library probe | `pytest -q -m integration` on Kaggle | Verified preprocessing geometry adapter, correct `Segment` head input, tensor shape/stride, native loss signature, nonzero auxiliary gradient, and one prepared merged multi-polygon instance accepted without corruption/drop |
| Enhanced smoke | `--method fgbg-triplet --shot 1 --epochs 1 --device 0` after preparation | Current-batch P3 capture, nonzero triplet loss in some batch, finite training, then reload `last.pt` through the evaluation/visualization inference path and obtain at least one paired box/mask prediction on prepared **train** imagery; no official-test inference |
| Baseline parity | Baseline and enhanced manifests for same shot | Same data checksum, base weights checksum, seed, image size, batch, augmentation; only method/triplet fields differ |
| Final evaluation | Both methods × four shots | Eight independent rows with six AP values or explicit failure statuses; no test-based checkpoint choice |

The optional local source audit is read-only; dataset materialization and training run on Kaggle. The Kaggle smoke and full runs are release gates, not claims that can be verified in this Windows environment.

## Spec coverage map

| Spec requirement | Owning task(s) |
| --- | --- |
| FR-01–FR-04 (official data, taxonomy, merge, audit) | T02, T04 |
| FR-05 (segmentation conversion and sanity views) | T03, T12 |
| FR-06–FR-07 (prepared layout and overwrite) | T04 |
| FR-08–FR-09 (methods and shared training protocol) | T05, T06, T09 |
| FR-10 (verified P3 source) | T09 |
| FR-11 (transformed GT mask, same-image samples, padding) | T07, T08 |
| FR-12–FR-14 (loss, hook, single GPU) | T08–T10 |
| FR-15–FR-17 (identity, recovery, batch failure) | T05, T06 |
| FR-18–FR-19 (test evaluation and summary) | T11 |
| FR-20 (provenance) | T04, T05, T10 |
| FR-21 (prediction views) | T12 |
| QG-01–QG-02 (synthetic and integration tests) | T02–T09, T11–T12 |
| QG-03 (smoke), QG-04 (verified pin), QG-05 (docs) | T10, T13 |

## Scope boundaries

Do not add memory banks, hard-negative mining, multi-scale metric learning, cross-shot transfer, DDP, a custom YOLO reimplementation, a random split, or a research hyperparameter search. Do not add another model variant or report enhanced gains without measured results. Optional Kaggle notebook and sampler debug plots come after the CLI and core smoke gate.
