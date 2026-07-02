AI-Based Early Detection of Plant Health Issues Using Drone Imagery
CNN-based diagnostic model for cabbage pest and disease classification (EfficientNet-B4), developed as the core computer vision engine for Tlatfarm, a smart-farm system using drone imagery for crop health monitoring.

Overview
Cabbage farming is highly vulnerable to pests and diseases that can cause significant crop loss, yet traditional diagnosis relies on manual inspection and expert knowledge — a process that is slow, inconsistent, and inaccessible to many smallholder farmers. Delayed detection often means the difference between a treatable issue and a lost harvest.

This project developed a convolutional neural network to automatically classify cabbage field images into healthy or diseased/pest-affected categories. The model is designed to be integrated into Tlatfarm, where drone-captured field imagery is analyzed to give farmers early, actionable diagnostic feedback — enabling proactive rather than reactive crop management. A secondary line of work explored extending the diagnosis to a multitask setup that jointly predicts disease class alongside contextual attributes (affected area, risk level, growth stage).

Status: Primary model completed and validated (July–August 2025); multitask extension developed but not yet fully validated (see Limitations) Contributors: Dongju Park, Hanbeen Lim (co-developers) Internship: Tlatfarm (Turbine Crew)

Problem Context
National- and regional-level agricultural statistics do not capture field-level pest/disease incidence, leaving individual farmers without timely diagnostic support.
Manual diagnosis requires agronomic expertise that is often unavailable in rural or resource-constrained farming communities.
Early detection materially affects treatment outcomes and yield — a diagnostic delay of days can turn a manageable outbreak into total crop loss.
Objectives
Develop a CNN-based image classification model to distinguish cabbage pest/disease categories and healthy plants from field images.
Achieve high, reliable classification accuracy suitable for real-world deployment.
Build a foundation diagnostic engine deployable within a drone-based smart-farm pipeline (Tlatfarm).
(Extension) Explore whether jointly predicting auxiliary attributes (affected area, risk level, growth stage) alongside diagnosis improves the model's usefulness for farmers.
Data
Source	AI Hub (Korea) — 노지 작물 통합 진단 이미지 (Open-Field Crop Integrated Diagnosis Images), public dataset
Images	Cabbage field photographs, JPEG, high resolution (predominantly 3024×3024 / 4032×3024)
Labels	Per-image JSON metadata with nested, inconsistently-keyed annotation structures
Split (primary model)	44,130 train / 7,277 validation
Split (multitask extension)	8,052 train / 2,695 validation (filtered subset)
Diagnostic classes (6): Normal, Cabbage Sclerotinia Rot, Cabbage Soft Rot, Cabbage White Butterfly, Flea Beetle, Brown Marmorated Stink Bug

Auxiliary attributes (multitask extension only): crop type, affected area, risk level, growth stage — annotated per image alongside the diagnostic class.

Methodology
1. Exploratory Data Analysis & Data Engineering
Audited label JSON structure, image resolution/format consistency, and corrupt or malformed files across train and validation sets.
Diagnosed and resolved a label-extraction bug caused by inconsistent JSON nesting (see Troubleshooting).
Identified a severe class imbalance in the original validation split (see Troubleshooting).
2. Primary Model: Single-Task Classification
Backbone: EfficientNet-B4, ImageNet-pretrained, fine-tuned end-to-end.
Classification head: Dropout (0.4) → fully connected layer → 6-class logits.
Input resolution: 380×380 px.
Loss: CrossEntropyLoss.
Training: Adam optimizer (lr = 3e-4), batch size 32, trained on the full 44,130/7,277 train/validation split.
3. Model Interpretability
Applied Grad-CAM to the trained primary model to visualize which image regions drove each diagnostic prediction, supporting model trust and error analysis.
4. Extension: Multitask Classification (in progress)
Architecture: Shared EfficientNet-B4 backbone with five parallel classification heads — diagnostic class, crop type, affected area, risk level, and growth stage.
Motivation: A single diagnosis label tells a farmer what is wrong; predicting area/risk/growth stage alongside it was intended to give more actionable context (e.g., how severe, what growth stage was affected).
Status: Training pipeline complete; evaluation pipeline currently unreliable due to a validation set construction issue (see Troubleshooting). Treated as in-progress exploratory work rather than a finished result.
Results (Primary Model)
Validated on the full, unmodified 7,277-image validation split (all 6 classes naturally represented, no data leakage):

Class	Precision	Recall	F1-score	Support
Normal	0.96	1.00	0.98	3,678
Sclerotinia Rot	0.98	0.94	0.96	752
Soft Rot	0.85	1.00	0.92	352
White Butterfly	1.00	0.88	0.94	2,115
Flea Beetle	0.76	0.98	0.86	180
Stink Bug	0.87	1.00	0.93	200
Overall accuracy			0.958	7,277
Validation accuracy: 95.8%, weighted F1: 0.96.
Deployed as the core diagnostic engine for the Tlatfarm smart-farm pipeline.
Flea Beetle shows the weakest precision (0.76), likely due to having the smallest support (180 samples) — a natural target for future data collection.
Troubleshooting
Recursive label extraction from inconsistent JSON structures
Problem: Label JSON files stored classification information under different keys (class vs. disease), nested at varying depths, causing naive parsing logic to silently miss labels.

Solution: Implemented a recursive parser (isinstance checks over dict/list) to traverse arbitrarily nested JSON and extract all classification-relevant keys regardless of depth, then harmonized disease/class fields into a single standardized label.

Result: Complete, consistently-labeled dataset for training — and a broader lesson in validating real-world data structures rather than assuming a clean, uniform schema.

Validation set class imbalance and a data-leakage pitfall
Problem: The original validation split for the underlying dataset was extremely imbalanced — 98% of validation labels were Normal, with the main disease/pest classes almost entirely absent. This made meaningful multi-class evaluation impossible on that split.

First attempted fix: Wrote a script to copy additional training images for the under-represented classes into the validation folder to balance it.

Issue identified: This approach introduces train/validation leakage — because it copies training samples directly into the validation set, any accuracy measured on that augmented split does not reflect true generalization. We caught this before treating the resulting numbers (from the multitask extension's evaluation) as valid, and are reporting the primary model's results from the original, unmodified split instead.

Correct approach (for future work): Perform stratified sampling from the original pool before any train/validation split is finalized, rather than balancing after the fact by duplicating training data into validation.

Repository Structure
├── notebooks/
│   ├── 01_eda_label_structure_exploration.ipynb          # JSON structure audit, recursive label parsing
│   ├── 02_eda_class_distribution_and_image_quality.ipynb # Class balance, image resolution/corruption checks
│   ├── 03_model_singletask_efficientnet_b4_train_eval.ipynb  # Primary model: training + evaluation
│   ├── 04_gradcam_interpretability.ipynb                 # Grad-CAM visualization for the primary model
│   ├── 05_extension_validation_set_construction.ipynb    # Multitask extension: validation augmentation (see Troubleshooting re: leakage)
│   └── 06_extension_multitask_efficientnet_b4.ipynb      # Multitask extension: training + (unreliable) evaluation
├── checkpoints/
│   └── simple_classifier_checkpoint.pth                  # Primary model weights (211MB — see note below)
└── README.md
Note on the checkpoint file: simple_classifier_checkpoint.pth is 211MB, exceeding GitHub's 100MB default limit. It is tracked via [Git LFS] / hosted externally at [link] — update this line with your chosen approach before publishing.

Tech Stack
Python · PyTorch · torchvision (EfficientNet-B4) · scikit-learn (evaluation metrics) · Grad-CAM · matplotlib · Pillow · tqdm

Limitations & Future Work
Multitask extension is not yet reliably validated. Its evaluation pipeline was affected by the validation-leakage issue described above; the reported training-time accuracy (~87–92%) should not be treated as a final, trustworthy metric.
The crop prediction head is currently non-informative: this dataset contains only cabbage, so the crop-classification task has a single possible answer. Extending the multitask model to multiple crop types would make this head meaningful.
Grad-CAM interpretability was only applied to the primary single-task model; extending it to the multitask heads would help validate whether auxiliary predictions (area, risk, growth stage) rely on sensible image regions.
The model classifies full, single-frame images rather than localizing affected regions. For drone-based aerial imagery covering a wide field of view, extending this to object detection or segmentation would likely be more directly useful for farmers.
ImageNet-pretrained weights were used for the primary model; a fully from-scratch variant was explored separately to sidestep potential ImageNet licensing constraints for commercial deployment, with an expected accuracy trade-off that has not yet been quantified.
About
Dongju Park — Data Science, background in international development. Co-developed with Hanbeen Lim during an internship at Tlatfarm (Turbine Crew), July–August 2025.
