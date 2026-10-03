# CAMO-FS Few-Shot Detection and Segmentation Specification

**Status:** Ready for implementation planning  
**Source of decisions:** [CAMO-FS YOLO Design Decisions](camo-fs-yolo-design-decisions.md)  
**Audience:** Project maintainer and implementation contributors

## Purpose

Provide a Kaggle-runnable, reproducible comparison of two methods for
few-shot camouflaged object detection and instance segmentation:

1. **Baseline:** COCO-pretrained YOLO11n-Seg with its native losses.
2. **Enhanced:** the same checkpoint and native losses, with an auxiliary
   class-agnostic foreground/background (FG/BG) triplet loss.

Both methods use the same official CAMO-FS 1-, 2-, 3-, and 5-shot training
splits and the same official test set. The deliverable is a working personal
GitHub project with auditable data preparation, independent runs, six reported
AP metrics per run, visualizations, and clear instructions for Kaggle. The
project does not claim a performance gain before experiments are run.

## User scenarios and acceptance

### 1. Prepare an official shot split

The maintainer selects one shot or `all`. The pipeline reads the official
class-specific annotations, verifies their integrity, converts instance
polygons to segmentation labels, and writes a self-contained dataset under
`/kaggle/working`.

**Acceptance:** A successful 5-shot preparation retains 197 unique training
images and all 235 official instances. The common test dataset is prepared
once. A failed audit produces a report and prevents that shot from being used
for training.

### 2. Run a fair baseline/enhanced comparison

The maintainer starts either method for one shot or runs all shots. Each run
starts from the original `yolo11n-seg.pt` checkpoint, records its configuration,
and writes to a unique run directory. Paired baseline and enhanced runs use
the same prepared data and training settings; the auxiliary objective is their
intended difference.

**Acceptance:** Running `--shot all` once for each method yields eight
independent runs (two methods × four shots). No run initializes from a
different shot or method. The baseline does not execute auxiliary-loss code.

### 3. Recover an interrupted run

The maintainer explicitly resumes a compatible interrupted run or explicitly
overwrites an existing run to start again from the original checkpoint.

**Acceptance:** Resume rejects a mismatch in method, shot, base weights, or
training configuration. Overwrite starts fresh. Neither operation silently
uses another run's checkpoint.

### 4. Evaluate and inspect results

The maintainer evaluates `last.pt` from each completed run on the shared
official test set, compares detection and segmentation AP, and views
deterministically chosen prediction images.

**Acceptance:** Results distinguish method, shot, seed, and configuration
fingerprint. A failed run remains visible with its error. Re-running final
evaluation updates its existing row rather than duplicating it.

## Functional requirements

### Official data and preparation

- **FR-01 — Inputs:** Use the official 1-, 2-, 3-, and 5-shot class-specific
  COCO JSON files under
  `/kaggle/input/datasets/danhnt/camo-fs-dataset/few-shot-annotations/subsplit/split1`
  for training. Use only `camo5_test_split1.json` from the same dataset for
  final evaluation. Never create a random train/test split or alter the
  membership of an official shot split.
- **FR-02 — Taxonomy:** Derive one 47-class COCO-category-to-contiguous-index
  mapping from the official test JSON, sorted by category ID. Save it as
  `/kaggle/working/camo_fs_yolo/category_mapping.json` and use it for every
  method, shot, test result, and visualization. Reject a training JSON whose
  `(category_id, category_name)` taxonomy differs.
- **FR-03 — Merge and deduplication:** Merge every class-specific file for the
  selected shot. Deduplicate images by identity and annotations by actual
  image/category/geometry content. Log annotation-ID reuse across distinct
  objects; never globally deduplicate by annotation ID. Detect repeated
  geometry in the same image/class as an integrity error.
- **FR-04 — Audit gate:** Before materializing a dataset, reject missing image
  files, invalid dimensions or bounding boxes, invalid or unsupported
  segmentations, malformed or out-of-range polygons, unmapped categories,
  conflicting image metadata, and train/test image overlap. Write an audit
  report under `/kaggle/working/results/` with shot, counts, error types, and
  relevant image/annotation IDs and filenames. Do not silently skip records.
- **FR-05 — Segmentation conversion:** Produce normalized YOLO segmentation
  labels, preserving each object polygon. Convert multi-polygon instances with
  the agreed Ultralytics-compatible merge, log them, and provide sample
  ground-truth visualizations. Document that connecting disconnected polygons
  may lose COCO topology. Do not create detection-only labels.
- **FR-06 — Prepared layout:** Copy only the images needed for each selected
  train split into its own `/kaggle/working/camo_fs_yolo/shot_K/train` tree.
  Copy the shared official test images and labels once into
  `/kaggle/working/camo_fs_yolo/test`. Each shot receives a `data.yaml` with
  its own `train` and the shared `test` split. Never point `val` at official
  test; if the selected Ultralytics version requires `val` in the YAML, point
  it to the train split as a technical placeholder and keep training
  validation disabled.
- **FR-07 — Preparation overwrite:** Refuse to reuse or replace an existing
  prepared target by default. With explicit `--overwrite`, rebuild all
  affected images, labels, YAML, and manifest without retaining stale files.

### Training methods

- **FR-08 — Method selection:** Support `--method baseline` and
  `--method fgbg-triplet`, plus `--shot {1,2,3,5,all}`. Each method/shot
  combination starts independently from the same COCO-pretrained
  `yolo11n-seg.pt` checkpoint. Accept `--weights` so an Internet-off Kaggle
  session can use an attached local checkpoint; fail clearly when unavailable.
- **FR-09 — Shared training protocol:** Expose epochs, image size, batch,
  device, and seed (default `2024`). Train for fixed epochs without using
  official test for validation, early stopping, checkpoint selection, or
  tuning. Use `val=False` where supported and report `last.pt`. Both methods
  use per-instance training masks (`overlap_mask=False`) and matching
  augmentation settings. Initially set `mosaic=0.0`, `mixup=0.0`, and
  `copy_paste=0.0` for both; other single-image augmentation may remain.
- **FR-10 — FG/BG feature source:** Target one intermediate spatial feature
  level: the P3/8 tensor provided as the first spatial feature input to the
  `Segment` head in the verified target Ultralytics version. Locate and
  verify this semantic head input rather than assuming a fixed numeric layer
  index. Validate the selected head, expected tensor shape and stride, and
  gradient path against the installed version before training. A missing or
  incompatible feature source is a clear failure, never a silent fallback
  to baseline training.
- **FR-11 — GT triplets:** For each valid GT instance, sample a foreground
  anchor and a distinct foreground positive from that instance, plus a
  background negative outside the union of GT objects in the same transformed
  image. Exclude letterbox padding and other invalid image regions.
  The sampler must use GT masks after the same geometric preprocessing and
  augmentation applied to the training image. It must not independently
  resize the original COCO polygon and assume alignment with the feature
  tensor. Project these transformed instance masks to feature-map resolution.
  Sampling is bounded and reproducible under the run seed where practical.
  Skip and count instances with fewer than two usable foreground locations
  or no valid background; an empty batch contributes finite zero auxiliary
  loss.
- **FR-12 — Auxiliary objective:** L2-normalize sampled features and apply a
  Euclidean triplet-margin loss. Defaults are `--triplet-weight 0.1`,
  `--triplet-margin 0.3`, and `--triplets-per-instance 16`. These are initial
  engineering defaults for the first implementation; they are not claimed
  to be optimal for CAMO-FS. The enhanced total objective is native YOLO loss
  plus weighted triplet loss. It must propagate gradients into the feature
  extractor. The baseline uses only native loss.
  Log native losses, raw and weighted triplet loss, combined loss where
  available, sampled-triplet count, and skipped-instance count. Fail on a
  non-finite auxiliary loss.
- **FR-13 — Hook lifecycle:** Before each forward, clear any prior feature
  capture. Capture exactly one current P3 tensor without detaching it, assert
  batch/spatial/channel dimensions, and release the reference after the step.
  Missing, stale, or multiple unexpected captures are errors.
- **FR-14 — Device scope:** The initial enhanced method supports one GPU at
  `device=0`. Multi-GPU/DDP is outside this version. Use the same device
  setting for paired baseline/enhanced comparisons.

### Runs, results, and recovery

- **FR-15 — Run identity:** Include model, method, shot, seed, and a
  deterministic configuration fingerprint in the run identity. Fingerprint
  base weights, method, shot, seed, epochs, image size, batch, augmentation
  settings, and the applicable triplet parameters. Keep method-separated,
  readable run paths under `/kaggle/working/runs/yolo11n-seg/`.
- **FR-16 — Run safety:** An existing run path causes failure by default.
  `--resume` is explicit and works only with a matching manifest and the
  interrupted run's own checkpoint. `--overwrite` is explicit and starts a
  fresh run from the original pretrained checkpoint. No cross-method or
  cross-shot resume is allowed.
- **FR-17 — Batch failure behavior:** `--shot all` processes shots in order and
  stops at the first failure by default. Optional `--continue-on-error`
  records the failed shot and proceeds. The final summary makes success and
  failure of every attempted run visible.
- **FR-18 — Final evaluation:** Evaluate each reported `last.pt` using the
  shared official test split with `split="test"`. Report Box AP, AP50, AP75
  and Mask AP, AP50, AP75. Do not use test metrics to choose checkpoint,
  training duration, or triplet parameters. A repeat evaluation may
  regenerate result artifacts.
- **FR-19 — Summary:** Upsert
  `/kaggle/working/results/summary.csv` by the complete run identity. Each
  row includes model, method, shot, seed, epochs, image size, batch, applicable
  triplet settings, six AP values, weights path, `status`, and
  `error_message`. Failed rows retain an error and have empty metrics.
- **FR-20 — Provenance:** Prepared datasets and training runs write manifests
  with arguments, source JSON paths/checksums, dataset counts, mapping path,
  output path, timestamp, seed, weights path/checksum when local, actual
  Python/library versions, and method. Enhanced runs also record feature
  source, triplet settings, triplet count, and skipped-instance count.
  Record warnings when full CUDA determinism cannot be guaranteed.
- **FR-21 — Predictions:** By default, save visualizations for 20 official
  test images selected deterministically by seed `2024` at confidence `0.25`.
  Include box, predicted mask, class, and confidence. Allow image count,
  confidence, and seed to be changed. Separate outputs by run identity.
  Enhanced training may additionally save a small debug view of GT mask and
  sampled foreground/background points.

## Quality and release gates

- **QG-01 — Local tests:** `pytest -q` works without CAMO-FS data or Kaggle.
  Synthetic COCO fixtures test mapping, normalization, invalid data,
  multi-polygon conversion, deduplication with reused IDs, overlap detection,
  and preparation failure. Synthetic tensor tests cover mask projection,
  same-instance and same-image sampling, determinism, padding exclusion,
  finite/zero loss, gradient propagation, skipped instances, method switching,
  and hook lifecycle.
- **QG-02 — Integration test:** On the actual target environment, an
  integration test confirms the installed Ultralytics trainer/model/loss
  interfaces, P3/8 capture, checkpoint loading, and nonzero auxiliary
  gradient. It fails with a diagnostic when incompatible.
- **QG-03 — Enhanced smoke run:** Before the full eight-run comparison, one
  small enhanced training run on an official training split demonstrates
  current-batch P3 capture, nonzero triplet loss on at least some batches,
  auxiliary gradients, completion without NaN/Inf, and successful checkpoint
  save/load. It must not use official test metrics to select training settings.
- **QG-04 — Dependency record:** After enhanced integration and a training
  run work on Kaggle, `requirements.txt` pins that verified Ultralytics
  version exactly. Torch/CUDA may use Kaggle's working runtime; their actual
  versions are captured in manifests.
- **QG-05 — Documentation:** README explains dependency setup, Internet-on
  and Internet-off checkpoint use, preparation, independent training,
  resume/overwrite, final evaluation, summary fields, and known
  multi-polygon conversion limits. A thin Kaggle notebook may orchestrate
  the CLI scripts but contains no duplicate pipeline logic.

## Success criteria

1. Official 1/2/3/5-shot preparation passes audit without losing valid
   annotations; the 5-shot output retains all 235 instances and the shared
   official test output contains 2,655 images and 3,108 annotations.
2. Before full experiments, one enhanced smoke run demonstrates P3 capture,
   nonzero triplet loss on at least some batches, auxiliary gradients, finite
   training losses, and working checkpoint save/load.
3. A maintainer can run each of the eight method/shot combinations from the
   original checkpoint and identify its data, configuration, weights, and
   outcome from the saved artifacts.
4. The result table supports paired baseline/enhanced comparison at every
   shot using all six requested AP metrics. No minimum AP or improvement is
   assumed in advance.
5. A failed audit, incompatible integration, missing checkpoint, invalid
   resume, or non-finite auxiliary loss ends with a clear diagnostic and a
   visible failure record where a run was attempted.
6. The synthetic test suite runs locally without the actual dataset; the
   target-environment integration test passes before enhanced Kaggle runs.

## Assumptions and boundaries

- The supplied official annotation files and image files are available at
  the Kaggle paths recorded in the design contract. Dataset preparation,
  training, and final evaluation execute on Kaggle; local development can
  use synthetic fixtures.
- The development machine currently lacks Ultralytics and PyTorch. The exact
  compatible Ultralytics version and whether a `val` key is required with
  `val=False` must be verified on the target environment before a version is
  pinned or enhanced training begins.
- The project excludes random splitting, cross-shot fine-tuning, predicted
  masks for triplet construction, memory banks, hard-negative mining,
  multi-scale triplet fusion, class-aware metric learning, and multi-GPU/DDP
  training for the initial enhanced method.
