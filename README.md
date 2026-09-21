# Brain Tumor Classification & Segmentation

An end-to-end deep learning pipeline for automated brain tumor analysis on MRI images, covering both **4-class tumor classification** and **binary tumor segmentation** using the BRISC2025 dataset.

---

## Project Overview

Brain tumor diagnosis from MRI scans is a time-consuming process that requires expert radiologists and is subject to inter-observer variability. This project builds a deep learning system that automates two key diagnostic tasks:

- **Classification** — Given an MRI scan, identify the tumor type: Glioma, Meningioma, Pituitary Tumor, or No Tumor
- **Segmentation** — Given an MRI scan, draw a precise pixel-level boundary around the tumor region

The system is intended as a decision-support tool for radiologists, not a replacement.

---

## Dataset

**BRISC2025** — Brain tumor Image Segmentation and Classification dataset
- 6,000 T1-weighted MRI slices (axial, coronal, sagittal planes)
- Expert-annotated tumor masks for segmentation
- 4 classes: Glioma, Meningioma, Pituitary Tumor, No Tumor
- Source: [Kaggle — briscdataset/brisc2025](https://www.kaggle.com/datasets/briscdataset/brisc2025)

---

## Repository Structure

```
brain-tumor/
│
├── Classification/                   # Classification experiments (run on Google Colab)
│   ├── data/splits/
│   │   └── classification_split.json # Fixed train/val split (generated once)
│   ├── notebooks/
│   │   └── BrainTumorClassification.ipynb
│   ├── results/
│   │   ├── densenet121/seed_42/      # best_model.pt, history.json, summary.json
│   │   ├── efficientnet_b0/seed_42/
│   │   ├── efficientnet_b1/seed_42/
│   │   ├── efficientnet_b4/seed_42/
│   │   ├── resnet50/seed_42/
│   │   ├── vgg19/seed_42/
│   │   └── classification_comparison.csv
│   └── src/
│       ├── augmentations.py
│       ├── brisc_datasets.py
│       ├── make_splits.py
│       ├── models_classification.py
│       ├── train_classification.py
│       └── utils.py
│
├── Segmentation/                     # Segmentation experiments (run on Kaggle)
│   ├── Notebook/
│   │   └── braintumorsegmentation.ipynb
│   └── Sample Result/
│       ├── AllModels.png
│       └── SampleResult.png
│
└── .gitignore
```

---

## Experimental Setup

### Why Two Platforms?

Due to limited GPU access on a single platform, experiments were split across two environments:

- **Classification → Google Colab (T4 GPU)** — Classification models are lighter (224×224 input), making them well-suited for Colab's free GPU tier
- **Segmentation → Kaggle (T4 x2 GPU)** — Segmentation models operate on 512×512 images and are significantly more memory-intensive, requiring Kaggle's higher GPU quota (30 hrs/week free)

Both environments used identical data splits, preprocessing, augmentation, optimizer, and evaluation protocols to ensure a fair comparison.

---

## Methodology

### Fair Comparison Protocol

Every model was trained under **identical conditions** — the only variable that changed between runs was the model architecture:

| Setting | Value |
|---|---|
| Dataset | BRISC2025 |
| Seed | 42 |
| Optimizer | Adam (lr = 1e-4) |
| Max Epochs | 50 (classification), 60 (segmentation) |
| Early Stopping | Patience = 10 epochs |
| Augmentation | Flips, rotations, brightness, affine, Gaussian noise |
| Train/Val Split | 80/20 (classification), 70/15/15 (segmentation) |

### Classification

- **Architectures compared:** VGG19, ResNet50, DenseNet121, EfficientNet-B0, EfficientNet-B1, EfficientNet-B4
- **Pretrained weights:** ImageNet (via `timm` library)
- **Loss function:** Cross-Entropy Loss
- **Input size:** 224 × 224

### Segmentation

- **Architectures compared:** U-Net++, EfficientNet-B1-UNet, Attention U-Net, ResUNet
- **Loss function:** Composite Dice Loss + Focal Loss (0.7 / 0.3 weighting)
- **Input size:** 512 × 512
- **Library:** segmentation-models-pytorch

---

## Results

### Classification

| Model | Val Accuracy | Macro F1 | Macro AUC | Params |
|---|---|---|---|---|
| **EfficientNet-B0** | **99.40%** | **0.9944** | **0.9998** | 4.01M |
| VGG19 | 99.40% | 0.9938 | 0.9998 | 139.59M |
| DenseNet121 | 99.30% | 0.9931 | 0.9999 | 6.96M |
| EfficientNet-B1 | 99.20% | 0.9920 | 0.9998 | 6.52M |
| ResNet50 | 98.70% | 0.9872 | 0.9996 | 23.52M |
| EfficientNet-B4 | 98.40% | 0.9843 | 0.9993 | 17.56M |

**Best model: EfficientNet-B0** — highest accuracy with only 4.01M parameters, 35x more efficient than VGG19 at the same accuracy level.

### Segmentation

| Model | Test Dice | Test IoU | Params |
|---|---|---|---|
| **EfficientNet-B1-UNet** | **0.8837** | **0.8118** | 8.76M |
| U-Net++ + EfficientNet-B1 | 0.8834 | 0.8141 | 9.08M |
| U-Net++ | 0.8813 | 0.8110 | 26.08M |
| Attention U-Net | 0.8208 | 0.7437 | 7.85M |
| ResUNet | 0.8099 | 0.7307 | 8.12M |

**Best model: EfficientNet-B1-UNet** — highest Dice score with only 8.76M parameters, outperforming U-Net++ (26.08M parameters) in both accuracy and efficiency.

### Key Finding

In both tasks, **smaller and more efficient architectures outperformed larger ones**:
- EfficientNet-B0 (4M params) matched VGG19 (139M params) in classification accuracy
- EfficientNet-B1-UNet (8.76M params) outperformed U-Net++ (26M params) in segmentation Dice score

---

## How to Run

### Classification (Google Colab)

1. Open `Classification/notebooks/BrainTumorClassification.ipynb` in Google Colab
2. Mount Google Drive when prompted
3. Download BRISC2025 from Kaggle using the API key cell in the notebook
4. Run cells top to bottom — each phase is clearly labeled
5. To train a different model, change `model_name` in the training cell and re-run

```python
# Example — change this one line to switch models
summary = train_classification_model(
    model_name="efficientnet_b0",  # change to: vgg19, resnet50, densenet121, etc.
    ...
)
```

### Segmentation (Kaggle)

1. Open `Segmentation/Notebook/braintumorsegmentation.ipynb` in Kaggle
2. Add BRISC2025: right sidebar → **Add Input** → search `brisc2025` → add
3. Enable GPU: Settings → Accelerator → **GPU T4 x2**
4. Enable Internet: Settings → **Internet → On**
5. Run cells top to bottom
6. Change `model_name` in Phase 7 to switch between architectures

```python
# Example — change this one line to switch models
summary = train_segmentation_model(
    model_name="efficientnet_b1_unet",  # change to: unetplusplus, attention_unet, etc.
    ...
)
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| PyTorch | Deep learning framework |
| timm | Pretrained classification architectures (EfficientNet, VGG, ResNet, DenseNet) |
| segmentation-models-pytorch | Pretrained segmentation architectures (U-Net variants) |
| Albumentations | Image augmentation pipeline |
| Google Colab | Classification training (free T4 GPU) |
| Kaggle Notebooks | Segmentation training (30 hrs/week free GPU) |

---

## Sample Results

![All Models Comparison](Segmentation/Sample%20Result/AllModels.png)
![Sample Segmentation Result](Segmentation/Sample%20Result/SampleResult.png)

---

## Reference

This project is based on and extends the work from:

> Alkharaan, R., Alobaidi, J., Bakarman, J., & Alshamlan, H. (2026). *Brain Tumor Classification and Segmentation in MR Images Using EfficientNet and U-Net++ Models*. Diagnostics, 16, 1745. https://doi.org/10.3390/diagnostics16111745

---

## License

Dataset: BRISC2025 is released under CC BY 4.0 by Fateh et al. (2025).
