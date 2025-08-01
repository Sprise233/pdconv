# 🧠 PDC-Net: A Parallel Decomposed Convolutional Network for Anisotropic Medical Image Segmentation

This repository provides the official implementation of **PDC-Net**, a novel 3D medical image segmentation network built upon **nnUNet v2**. PDC-Net introduces *Parallel Decomposed Convolution* to effectively model anisotropic data while maintaining high accuracy and computational efficiency. 

The framework supports full configurability via `config.yaml`, self-adapts to both **PyTorch 1.x and 2.x**, and enables model visualization and inspection via CLI.

---

## ✨ Highlights

- 🔍 **PDConv**: Decomposed 3D convolution for inter- and intra-slice learning.
- ⬇️ **PDD**: Downsampling that respects spatial anisotropy.
- 🔄 **PDFF**: Feature fusion module with voxel-level attention.
- ⚙️ **Fully configurable** via `config.yaml`.
- 🔧 **Auto-adapts** to PyTorch 1.x / 2.x environments.
- 📈 **Architecture/parameter introspection** tools provided.
- 📦 **Installation-ready** via `requirements.txt`.

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd PDC-Net
```

### 2. Create Environment & Install Dependencies

```bash
# Optional: create virtualenv
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install required packages
pip install -r requirements.txt
```

### 3. Train Your Model

```bash
python main.py
```

---

##  Configuration Example (`config.yaml`)

```yaml
# -*- coding:utf-8 -*-
dataset:
  dataset_name_or_id: "2"  # Name or ID of the dataset
  fold: 0  # Fold number for cross-validation (e.g., in K-fold cross-validation)
  use_compressed: false  # Whether to use the compressed format of the dataset

training:
  trainer: Trainer  # Trainer type defining how the model is trained {Trainer, TrainerStatic}
  plans: PDCNWithPDCDWithPDFMPlans  # Training plan defining network architecture and preprocessing {nnUNetPlans, nnUNetResEncUNetPlans, nnUnetWithPdconvPlans, PDCNPlans, PDCNWithPDCDPlans, PDCNWithPDCDWithPDFMPlans, UNETRPlans, TransBTSPlans, SwinUNETRPlans, nnFormerPlans, UX_NetPlans, MedNeXtMPlans, TransBTSPlans}
  pretrained_weights: null  # Path to pretrained weights; null if not using any pretrained model
  gpu_index: 0  # GPU device index to use
  initial_lr: 0.01  # Initial learning rate
  weight_decay: 0.00003  # Weight decay coefficient for regularization (prevents overfitting)
  oversample_foreground_percent: 0.33  # Percentage of foreground oversampling to increase foreground samples
  num_iterations_per_epoch: 250  # Number of iterations per training epoch
  num_val_iterations_per_epoch: 1  # Number of iterations per validation epoch
  num_epochs: 1000  # Total number of training epochs
  warm_up_epochs: 10  # Number of warm-up epochs
  current_epoch: 0  # Current epoch index (set > 0 for resuming from checkpoint)
  enable_deep_supervision: true  # Whether to enable deep supervision (multi-level output supervision)
  configuration: "3d_fullres"  # Configuration type, e.g., 3D full-resolution {2d, 3d_fullres, 3d_lowres, 3d_cascade_lowres}
  optimizer_type: "SGD"  # Optimizer type: SGD or AdamW
  loss_type:  # Type of loss function {}
  do_i_compile: false  # Whether to use PyTorch 2.0 compilation (torch.compile)

validation:
  npz: false  # Whether to save validation results in .npz format
  only_validation: false  # If true, only runs validation without training
  validate_with_best_checkpoint: false  # Whether to use the best checkpoint for validation

checkpointing:
  disable_checkpointing: false  # Whether to disable saving checkpoints

environment:
  device: "cuda"  # Device to use ('cuda' for GPU or 'cpu')

  nnUNet_raw: "/home/ddn/nnUnet/data"  # Path to raw data
  nnUNet_preprocessed: "/home/ddn/nnUnet/data/processed"  # Path to preprocessed data
  nnUNet_results: "/home/ddn/nnUnet/results"  # Path to save results

  nnUNet_def_n_proc: "4"  # Number of processes to use

others:
  save_file_name: 'train_4_1'  # File name used for saving logs or results

```

---

##  Benchmark Results (DSC ↑)

| Model       | ACDC      | AMOS22    | KiTS19    | MBAS      | Params    |
| ----------- | --------- | --------- | --------- | --------- | --------- |
| nnUNet      | 91.54     | 88.88     | 89.88     | 85.25     | 30.3M     |
| MedNeXt-M   | 92.42     | 89.27     | 90.78     | 84.17     | 17.6M     |
| UNETR       | 89.30     | 81.98     | 84.10     | 74.54     | 92.4M     |
| **PDC-Net** | **93.03** | **90.16** | **90.92** | **85.29** | **16.7M** |

---

##  Supported PyTorch Versions

| Version     | Compatibility     |
| ----------- | ----------------- |
| PyTorch 1.x | ✅ Fully supported |
| PyTorch 2.x | ✅ Fully supported |

Dynamic detection and compatibility are handled internally.

---

##  Citation

If you use PDC-Net in your research, please cite:

```bibtex
@article{pdcnet2025,
  title={PDC-Net: A Parallel Decomposed Convolutional Network for Anisotropic Medical Image Segmentation},
  author={Sprise, Prise and Others},
  year={2025}
}
```
