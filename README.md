# UAV Semantic Segmentation with DeepLabV3-ResNet50

This project implements a semantic segmentation pipeline for UAV aerial images using **DeepLabV3** with a **pretrained ResNet50 backbone** in **PyTorch**.

The project covers the full workflow from data preprocessing to inference, including data augmentation, weighted loss for class imbalance, model training and validation, test-time augmentation (TTA), RLE submission generation, training curve visualization, and qualitative error analysis.

## Project Overview

Semantic segmentation is a dense prediction task that assigns a class label to every pixel in an image. In this project, I applied DeepLabV3-ResNet50 to segment UAV aerial imagery into multiple semantic categories.

Compared with standard image classification, this task is more challenging because:
- objects appear at very different scales
- small targets are easily missed
- scene backgrounds dominate many pixels
- class imbalance is severe in aerial datasets

To address these issues, this project uses:
- **DeepLabV3** for multi-scale context modeling
- **ResNet50 pretrained on ImageNet** for transfer learning
- **Weighted CrossEntropyLoss** to reduce the impact of class imbalance
- **Test-Time Augmentation (TTA)** to improve inference robustness


## Dataset

The dataset structure is organized as follows:

```text
UAV_dataset/
├── train/
│   ├── imgs/      # 4,000 RGB training images
│   └── masks/     # 4,000 corresponding grayscale masks
└── test/
    └── imgs/      # 1,000 test images
```

### Data Format

* Training images: RGB format
* Training masks: grayscale masks with pixel values from `0` to `15`
* Number of classes: **16**
* Test set: images only, no ground-truth masks provided

## Method

### 1. Preprocessing

* Images are resized to **512 × 512**
* RGB images are normalized using ImageNet-style normalization
* Masks are resized with **nearest-neighbor interpolation** to preserve class IDs

### 2. Data Augmentation

The training pipeline includes augmentation to improve generalization:

* random horizontal flip
* random vertical flip
* color jitter

These augmentations help the model adapt to viewpoint and lighting variations in aerial scenes.

### 3. Model Architecture

This project uses **DeepLabV3 with a ResNet50 backbone**.

#### Backbone

* **ResNet50**
* initialized with pretrained weights
* strong feature extractor for transfer learning

## Segmentation Head

* based on **DeepLabV3**
* includes **ASPP (Atrous Spatial Pyramid Pooling)** for multi-scale feature extraction
* final classifier is modified to output **16 classes**

### Why DeepLabV3?

DeepLabV3 is well suited for UAV image segmentation because aerial scenes contain objects with very different spatial scales. ASPP helps capture both global context and local details, which improves segmentation quality in complex outdoor scenes.

## Training Setup

### Train / Validation Split

* **80% training**
* **20% validation**

### Hyperparameters

* Optimizer: **Adam**
* Learning rate: **1e-4**
* Scheduler: **StepLR**

  * step size: `8`
  * gamma: `0.5`
* Batch size: **8**
* Max epochs: **30**
* Early stopping patience: **5**

### Loss Function

This project uses **Weighted CrossEntropyLoss**.

Because aerial segmentation datasets often have severe class imbalance, class weights are computed from pixel statistics and applied during training. This helps the model pay more attention to underrepresented classes instead of overfitting to dominant background categories.

## Inference

During inference, the model:

* loads the best checkpoint based on validation loss
* performs prediction on test images
* applies **Test-Time Augmentation (horizontal flip)**
* resizes predictions back to original image size
* converts masks into **RLE format**
* exports the final results to `submission.csv`

## Results

### Training Curve

The training loss decreases steadily over epochs, which indicates stable convergence.

![Training Loss Curve](/training_loss_curve.png)

### Qualitative Prediction Analysis

The repository also includes qualitative visualization results:

* original RGB image
* ground-truth mask
* predicted mask
* error map

These examples help analyze model behavior beyond loss values alone.

### Example Error Analysis

![Error Analysis 1683](/error_analysis_1683.png)

![Error Analysis 2377](/error_analysis_2377.png)

## Observations

### Strengths

* DeepLabV3 captures multi-scale context effectively
* pretrained ResNet50 improves convergence and generalization
* weighted loss helps reduce class imbalance issues
* the project includes a complete pipeline from training to submission

### Limitations

* model complexity is relatively high
* small objects may still be difficult to segment accurately
* object boundaries can remain blurry in challenging regions
* performance may still be affected by limited training data

## Repository Structure

```text
.
├── train.py
├── report.pdf
├── log/
│   └── train_log.csv
├── fig/
│   ├── training_loss_curve.png
│   ├── error_analysis_1683.png
│   ├── error_analysis_2377.png
│   └── error_analysis_3202.png
├── best_model.pth
├── submission.csv
└── README.md
```

## How to Run

### 1. Prepare Dataset

The dataset is not included in this repository due to file size limitations and possible distribution restrictions.

Please prepare the dataset in the following structure before running the code:

```text
UAV_dataset/
├── train/
│   ├── imgs/
│   └── masks/
└── test/
    └── imgs/
```

### 2. Install Dependencies

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib
```

### 3. Run Training

```bash
python train.py
```

### 4. Output Files

After training and inference, the following files will be generated:

* `best_model.pth` — best model checkpoint
* `submission.csv` — test predictions in RLE format
* `log/train_log.csv` — epoch-wise training log
* `fig/training_loss_curve.png` — training loss visualization
* `fig/error_analysis_*.png` — qualitative error analysis figures

## Key Features

* DeepLabV3 + ResNet50 for semantic segmentation
* transfer learning with pretrained weights
* weighted loss for class imbalance
* test-time augmentation
* automatic submission generation
* training visualization
* qualitative error analysis

## Future Improvements

Potential directions for improvement include:

* replacing the backbone with a stronger or more efficient encoder
* using more advanced augmentation strategies
* incorporating Dice loss or focal loss
* improving boundary refinement
* adding validation metrics such as mIoU during training
* experimenting with lightweight models for faster inference


