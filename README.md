# AI-Based Early Detection of Plant Health Issues Using Drone Imagery

CNN-based diagnostic model for cabbage pest and disease classification
using EfficientNet-B4, developed as the core computer vision engine for
**Tlatfarm**, a drone-assisted smart farming platform for crop health
monitoring.

## Overview

This project develops a deep learning pipeline for the early detection
of cabbage pests and diseases from drone-captured field imagery. The
primary objective is to support timely and consistent crop diagnosis by
replacing labor-intensive manual inspection with automated image
classification.

The repository covers the complete workflow, including exploratory data
analysis, data engineering, model development, evaluation, model
interpretability, and an exploratory multitask extension. Beyond model
training, considerable effort was devoted to handling inconsistent
annotation structures, validating dataset quality, and identifying
evaluation pitfalls that commonly arise in real-world computer vision
projects.

**Status:** Primary single-task model completed and validated
(July--August 2025). Multitask extension remains experimental and is
documented for future work.

**Contributors:** Dongju Park, Hanbeen Lim

**Internship:** Tlatfarm (Turbine Crew)

## Problem Context

-   Manual crop diagnosis is time-consuming and requires agronomic
    expertise.
-   Delayed identification of pests and diseases can significantly
    reduce crop yield.
-   Public agricultural statistics rarely provide field-level diagnostic
    information that individual farmers can directly utilize.
-   Drone imagery combined with computer vision enables scalable and
    timely crop monitoring.

## Objectives

-   Develop a CNN-based classifier for cabbage pest and disease
    diagnosis.
-   Build a production-oriented diagnostic engine for integration into
    the Tlatfarm platform.
-   Achieve robust multi-class classification performance.
-   Explore multitask learning to jointly predict diagnostic class and
    contextual agricultural attributes.

## Data

  ----------------------------------- -----------------------------------
  **Source**                          AI Hub (Korea) --- *Open-Field Crop
                                      Integrated Diagnosis Images*

  **Image Type**                      High-resolution cabbage field
                                      photographs

  **Annotation**                      JSON metadata with nested and
                                      inconsistent structures

  **Primary Split**                   44,130 training / 7,277 validation

  **Multitask Split**                 8,052 training / 2,695 validation
  ----------------------------------- -----------------------------------

**Diagnostic classes (6)**

-   Normal
-   Cabbage Sclerotinia Rot
-   Cabbage Soft Rot
-   Cabbage White Butterfly
-   Flea Beetle
-   Brown Marmorated Stink Bug

## Methodology

### 1. Exploratory Data Analysis & Data Engineering

-   Audited annotation consistency and image quality.
-   Implemented recursive JSON parsing for inconsistent nested
    annotations.
-   Investigated class imbalance and dataset integrity.

### 2. Primary Model

-   Backbone: EfficientNet-B4 (ImageNet pretrained)
-   Input size: 380×380
-   Loss: CrossEntropyLoss
-   Optimizer: Adam (3e-4)
-   Batch size: 32

### 3. Model Interpretability

Grad-CAM was applied to visualize prediction regions and support
qualitative model validation.

### 4. Multitask Extension

A shared EfficientNet-B4 backbone with multiple classification heads was
developed to jointly predict diagnosis, crop type, affected area, risk
level, and growth stage. This extension is included as exploratory work
because its evaluation pipeline requires further validation.

## Results

  Metric                      Value
  --------------------- -----------
  Validation Accuracy     **95.8%**
  Weighted F1-score        **0.96**
  Validation Images           7,277

The primary model achieved strong classification performance across six
diagnostic classes and serves as the intended inference engine for the
Tlatfarm smart farming pipeline.

## Troubleshooting

### Recursive Label Extraction

The annotation files contained inconsistent nested JSON structures using
different keys (`class` and `disease`). A recursive parser was
implemented to reliably traverse dictionaries and lists, standardizing
labels into a unified training format.

### Validation Set Construction

An exploratory attempt to balance the validation dataset by copying
training samples was identified as a data leakage risk. Those results
are intentionally excluded from the reported performance. Future work
should instead construct balanced validation data through stratified
splitting before model training.

## Repository Structure

``` text
plant-health-detection/
├── notebooks/
│   ├── 01_eda_label_structure_exploration.py
│   ├── 02_eda_class_distribution_and_image_quality.py
│   ├── 03_model_singletask_efficientnet_b4_train_eval.py
│   ├── 04_gradcam_interpretability.py
│   ├── 05_extension_validation_set_construction.py
│   └── 06_extension_multitask_efficientnet_b4.py
├── README.md
└── .gitignore
```

## Tech Stack

Python · PyTorch · torchvision · EfficientNet-B4 · scikit-learn ·
Grad-CAM · matplotlib · Pillow · tqdm

## Limitations & Future Work

-   Complete validation of the multitask learning pipeline.
-   Extend Grad-CAM analysis to multitask prediction heads.
-   Expand to multiple crop species.
-   Explore object detection and segmentation for large-scale drone
    imagery.
-   Evaluate training from scratch versus ImageNet pretraining for
    deployment scenarios.

## About

**Dongju Park** --- Data Science with a background in international
development.

Developed with **Hanbeen Lim** during an internship at **Tlatfarm
(Turbine Crew)**.
