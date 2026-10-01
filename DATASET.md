# CAMO-FS Dataset

## 1. Overview

**CAMO-FS (Camouflaged Few-Shot Dataset)** is a Computer Vision dataset designed for **camouflaged object detection and instance segmentation**.

The dataset contains objects, mainly animals, whose colors, textures, and patterns are visually similar to their surrounding environments.

This makes CAMO-FS more challenging than conventional object detection datasets because the foreground object may be difficult to distinguish from the background.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total images | 2,852 |
| Total instances | 3,342 |
| Fine-grained classes | 47 |
| Coarse classes | 10 |
| Tasks | Object Detection, Instance Segmentation |
| Annotation format | COCO JSON |
| Few-shot settings | 1-shot, 2-shot, 3-shot, 5-shot |

---

## 2. Example Images

CAMO-FS contains camouflaged objects in different natural environments such as forests, rocks, grass, trees, and underwater scenes.

### Example 1

<!-- Add an example image here -->

![Example 1](docs/images/example_1.jpg)

### Example 2

<!-- Add an example image here -->

![Example 2](docs/images/example_2.jpg)

### Example 3

<!-- Add an example image here -->

![Example 3](docs/images/example_3.jpg)

---

## 3. Tasks

CAMO-FS can be used for two main Computer Vision tasks.

### 3.1. Object Detection

The model must determine:

- where the camouflaged object is located;
- its bounding box;
- its object class.

```text
Input Image
    |
    v
Object Detector
    |
    +----> Bounding Box
    |
    +----> Object Class
```

### 3.2. Instance Segmentation

Instance segmentation additionally predicts the pixels that belong to each detected object.

```text
Input Image
    |
    v
Instance Segmentation Model
    |
    +----> Bounding Box
    |
    +----> Object Class
    |
    +----> Segmentation Mask
```

This task is especially difficult for camouflaged objects because the boundary between the object and the background may be unclear.

---

## 4. Few-Shot Learning

The most important characteristic of CAMO-FS is that it is designed for **Few-Shot Learning**.

In normal deep learning datasets, a model may be trained using hundreds or thousands of examples for every class.

CAMO-FS intentionally does the opposite.

Only a very small number of labeled object instances are selected for training.

The supported experimental settings are:

```text
1-shot -> 1 training instance per class
2-shot -> 2 training instances per class
3-shot -> 3 training instances per class
5-shot -> 5 training instances per class
```

CAMO-FS contains **47 fine-grained classes**.

For example, in the 5-shot setting:

```text
47 classes
    x
5 instances per class
    =
235 training instances
```

These 235 training instances are contained in approximately **197 images**.

Therefore:

> Although CAMO-FS contains 2,852 images in total, only a very small part of the dataset is used for few-shot training.

The majority of the images are reserved for evaluation.

---

## 5. Train/Test Split

CAMO-FS does **not** use a conventional random split such as:

```text
80% Training
20% Testing
```

Instead, the authors provide predefined few-shot annotations.

For the 5-shot experiment:

| Split | Images | Instances |
|---|---:|---:|
| Few-shot Train | ~197 | 235 |
| Test | 2,655 | 3,107 |

The important point is:

> The training set is intentionally extremely small.

This allows the experiment to measure whether the model can generalize to unseen camouflaged objects after learning from only a few examples.

Conceptually:

```text
                 CAMO-FS
               2,852 images
                    |
          +---------+---------+
          |                   |
          v                   v
 Few-Shot Training          Testing
  very few samples       2,655 images
          |
     +----+----+----+----+
     |    |    |    |
     v    v    v    v
   1-shot 2-shot 3-shot 5-shot
```

---

## 6. Dataset Structure

The few-shot annotations are structured as follows:

```text
data/
│
├── [image directory]/
│   ├── camouflage_XXXXXXXX.jpg
│   ├── camouflage_XXXXXXXX.jpg
│   ├── camouflage_XXXXXXXX.jpg
│   └── ...
│
└── few-shot-annotations/
    │
    ├── camo5_test_split1.json
    │
    └── subsplit/
        └── split1/
            │
            ├── camo5_Bat_1shot_split1.json
            ├── camo5_Bat_2shot_split1.json
            ├── camo5_Bat_3shot_split1.json
            ├── camo5_Bat_5shot_split1.json
            │
            ├── camo5_Bear_1shot_split1.json
            ├── camo5_Bear_2shot_split1.json
            ├── camo5_Bear_3shot_split1.json
            ├── camo5_Bear_5shot_split1.json
            │
            ├── camo5_Bird_1shot_split1.json
            ├── camo5_Bird_2shot_split1.json
            ├── camo5_Bird_3shot_split1.json
            ├── camo5_Bird_5shot_split1.json
            │
            ├── ...
            │
            ├── camo5_Turtle_1shot_split1.json
            ├── camo5_Turtle_2shot_split1.json
            ├── camo5_Turtle_3shot_split1.json
            ├── camo5_Turtle_5shot_split1.json
            │
            └── ...
```

### Important

The files are organized by:

```text
Class + Number of shots + Split
```

For example:

```text
camo5_Bat_1shot_split1.json

Class       : Bat
Shot        : 1-shot
Split       : split1
```

and:

```text
camo5_Bat_5shot_split1.json

Class       : Bat
Shot        : 5-shot
Split       : split1
```

---

## 7. Few-Shot Training Annotations

Inside:

```text
data/few-shot-annotations/subsplit/split1/
```

each class has its own annotation file for each few-shot setting.

For example:

```text
Bat
├── camo5_Bat_1shot_split1.json
├── camo5_Bat_2shot_split1.json
├── camo5_Bat_3shot_split1.json
└── camo5_Bat_5shot_split1.json
```

Similarly:

```text
Bear
├── camo5_Bear_1shot_split1.json
├── camo5_Bear_2shot_split1.json
├── camo5_Bear_3shot_split1.json
└── camo5_Bear_5shot_split1.json
```

This pattern is repeated for the dataset classes.

Therefore, a 5-shot experiment uses the corresponding:

```text
*_5shot_split1.json
```

annotations for the classes.

A 1-shot experiment uses:

```text
*_1shot_split1.json
```

annotations.

---

## 8. Test Annotation

The test annotation is stored separately at:

```text
data/few-shot-annotations/camo5_test_split1.json
```

Unlike the training annotations, which are divided by class and shot setting, the test annotation describes the common test set.

```text
few-shot-annotations/
│
├── camo5_test_split1.json       <- TEST
│
└── subsplit/
    └── split1/
        ├── *_1shot_split1.json  <- TRAIN
        ├── *_2shot_split1.json  <- TRAIN
        ├── *_3shot_split1.json  <- TRAIN
        └── *_5shot_split1.json  <- TRAIN
```

All few-shot configurations are evaluated using the same test set.

Conceptually:

```text
1-shot annotations ──┐
                     |
2-shot annotations ──|
                     |
3-shot annotations ──+──> Model ──> camo5_test_split1.json
                     |
5-shot annotations ──|
```

This makes the experiments directly comparable.

---

## 9. Why CAMO-FS is Challenging

In camouflaged scenes:

```text
Foreground color   ≈ Background color

Foreground texture ≈ Background texture

Object boundary    ≈ Background boundary
```

The object may visually blend into its surroundings.

At the same time, the model receives only a few training examples.

Therefore, the problem combines two challenges:

```text
Camouflaged Object Recognition
              +
        Few-Shot Learning
              |
              v
          CAMO-FS
```

- use the provided few-shot annotation files for training;
- use `camo5_test_split1.json` for evaluation.
