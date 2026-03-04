# Mars Landslide Segmentation Challenge Solution

This repository contains a deep learning pipeline for **binary semantic segmentation of Martian landslides** for the **1st Mars Landslide Segmentation (Mars-LS) Challenge**. The goal is to predict pixel-wise binary masks that identify landslide regions from multi-band planetary image patches.

## Dataset

This solution is built for the **PBVS 1st Mars Landslide (Mars-LS)** dataset.

- **Kaggle dataset:** [PBVS 1st Mars Landslide (Mars-LS)](https://www.kaggle.com/datasets/shradhanjali15/pbvs-1st-mars-landslide-mars-ls)
- The dataset can also be downloaded from the challenge files under **`Mars_LSc_2025_dataset_1st_phase`**.

### Dataset overview

The challenge is based on the **Multi-modal Martian LandSlide (MMLS)** dataset, with the updated **MMLSv2** release focused on the **Valles Marineris (VM)** region on Mars. This region is a tectonic trough system with widespread landslide activity, making it highly relevant for landslide segmentation research.

The dataset is designed for **binary semantic segmentation**, where:
- **Foreground (1):** landslide pixels
- **Background (0):** non-landslide pixels

### Data sources and modalities

The seven-band input is a unified multi-modal composite created from multiple co-registered planetary data sources:

1. **Thermal inertia** from THEMIS observations *(band 1)*
2. **Slope map** derived from DEM *(band 2)*
3. **Digital Elevation Model (DEM)** *(band 3)*
4. **Grayscale imagery** *(band 4)*
5. **RGB imagery** from Viking mission data *(bands 5, 6, 7)*

Additional supporting sources used in creating the dataset include:
- THEMIS nighttime infrared imagery and thermal inertia products
- CTX imagery from Mars Reconnaissance Orbiter (MRO)
- MOLA DEM blended with HRSC data
- Slope derived in ESRI ArcGIS
- Viking colorized global mosaic

All datasets were co-registered and analyzed in **ESRI ArcGIS**, and landslides were manually digitized as polygons using morphological criteria that capture both depletion and run-out areas.

### Patch format

- Each sample is a **128 × 128** image patch
- Inputs are stored as **multi-band TIFF files**
- Masks are stored as **binary TIFF files** with the same spatial resolution

### Dataset splits

- **Training:** 465 image-mask pairs
- **Validation:** 66 image-mask pairs
- **Testing:** 133 images (no ground truth released)

Ground-truth masks are provided only for the training and validation sets. Test masks are withheld for official evaluation on the challenge platform.

## Approach

The solution uses a **3-model ensemble** designed to balance strong feature extraction, architectural diversity, and stable generalization on multi-modal planetary imagery.

### Core strategy

- Train multiple segmentation models with different backbones and architectures
- Normalize each spectral/modal band independently using training-set statistics
- Use robust binary loss tailored for segmentation
- Average model probabilities at inference time
- Apply a tuned decision threshold to generate final binary masks

### Model ensemble

Three complementary models are used:

1. **U-Net++ + EfficientNet-B4**
2. **U-Net + ResNet50**
3. **DeepLabV3+ + ResNet34**

All models are configured with:
- **7 input channels**
- **1 output channel**
- **No activation in the final layer** (logits output)

### Preprocessing

A robust per-band normalization scheme is used:

- Compute **mean** and **standard deviation** for each of the 7 channels using the **training set only**
- Replace any `NaN` or infinite values with `0`
- Apply z-score normalization independently per band
- Clip normalized values to `[-5, 5]` for stability

This is especially important because the input is multi-modal scientific raster data, not natural RGB imagery.

### Data augmentation

The training pipeline uses **safe geometric augmentations** that preserve spatial structure while avoiding unstable transforms on normalized scientific data.

Used augmentations include:
- Horizontal flip
- Vertical flip
- Random 90° rotations
- Transpose
- Elastic transform / grid distortion (light use)
- Light Gaussian noise
- Coarse dropout

**Important implementation note:** intensity transforms such as `RandomGamma` and `RandomBrightnessContrast` were removed because they can produce unstable values after z-score normalization and may introduce training-time `NaN`s on non-RGB scientific inputs.

### Loss function

A hybrid segmentation loss is used:

- **Soft Dice loss**
- **Weighted Binary Cross-Entropy (BCE) with logits**

The final loss is:

`Loss = dice_weight * DiceLoss + bce_weight * BCEWithLogitsLoss`

With the current configuration:
- `dice_weight = 0.6`
- `bce_weight = 0.4`

A foreground **positive class weight** is computed from the training masks to better handle class imbalance.

### Optimization and training settings

- **Optimizer:** AdamW
- **Learning rate:** `3e-4`
- **Weight decay:** `1e-4`
- **Scheduler:** CosineAnnealingWarmRestarts
- **Gradient clipping:** `max_norm = 1.0`
- **AMP:** disabled for stability in the current setup
- **Early stopping:** enabled with patience-based stopping

### Inference and ensembling

At inference time:

1. Each model predicts logits for the input patch
2. Logits are converted to probabilities with `sigmoid`
3. Probabilities are averaged across all ensemble members
4. A threshold is applied to generate binary masks

A validation-based threshold sweep was used to select the best cutoff.

### Best local threshold

From the reported validation sweep, the best ensemble threshold was:

- **Best threshold:** `0.60`
- **Best local validation mIoU:** `0.8251`

## Reported local validation results

### Individual model best scores

- **Model 1 (U-Net++ + EfficientNet-B4):** `0.7931` mIoU
- **Model 2 (U-Net + ResNet50):** `0.8082` mIoU
- **Model 3 (DeepLabV3+ + ResNet34):** `0.7999` mIoU

### Ensemble best score

- **3-model ensemble:** `0.8251` mIoU (local validation)

This indicates that the ensemble performs better than any single model and benefits from the diversity of the three architectures.

## Submission format

Predictions are saved as:
- **TIFF (`.tif`) files**
- **Binary masks** with values `{0, 1}`
- **Shape:** `128 × 128`
- Each output file must have the **exact same filename** as its corresponding test image

The final submission should be:
- Packaged as a single **`submission.zip`**
- Prediction TIFF files must be placed directly in the **root** of the ZIP
- No subfolders or extra files should be included

## Notes and practical guidance

- Use only the officially released challenge data
- Do not use hidden test data for training
- For phase-based evaluations, make sure predictions are generated for the **correct test split** for the active phase
- Always verify that the prediction filenames exactly match the reference test filenames before submission

## Summary

This solution combines:
- strong **multi-architecture ensembling**,
- **robust per-band normalization** for scientific raster data,
- **stable augmentations** suited to multi-modal imagery,
- **Dice + weighted BCE** optimization,
- and **threshold-tuned ensemble inference**.

It is designed to be a strong, reproducible baseline for the Mars-LS challenge and provides a solid foundation for further improvements such as weighted ensembling, test-time augmentation (TTA), or cross-validation.
