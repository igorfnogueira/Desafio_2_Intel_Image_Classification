# Intel Image Classification (Computer Vision)

Language / Idioma: **English** | [Português](README.pt-br.md)

Computer Vision project that classifies natural and urban scene images into **6 categories** (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`), using the [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset.

## What the project does

The [`intel_image_classification.ipynb`](intel_image_classification.ipynb) notebook builds a complete **PyTorch** pipeline (with GPU/CUDA support):

1. Loads and explores the data (`seg_train` / `seg_test`)
2. Normalizes images and applies **data augmentation** only on the training split
3. Trains three CNNs of increasing complexity:
   - **Baseline** — simple reference network
   - **VGG-like** — deeper, with BatchNorm
   - **AdvancedIntelCNN** — hybrid architecture (stem + VGG blocks + residual blocks)
4. Compares the models, picks the best one on validation, and evaluates it on the test set (confusion matrix and error analysis)

## Results (saved run in the notebook)

Model selected on validation: **AdvancedIntelCNN**.

| Model | Parameters | Val. acc. | Test acc. |
|-------|------------|-----------|-----------|
| **Advanced** | 14,360,390 | **87.42%** | **87.33%** |
| Baseline | 10,711,878 | 86.24% | 85.84% |
| VGG-like | 4,822,086 | 85.07% | 84.14% |

- Labeled dataset: **17,034** images (14,034 train + 3,000 test), 6 classes
- Internal train split: 11,929 train / 2,105 validation (`VAL_RATIO=0.15`, `SEED=42`)
- Best per-class F1 (test): **forest 0.971** · lowest: **glacier 0.828**
- Main confusions: `buildings→street`, `glacier↔mountain`

### Where to see it working

Open the notebook with the **outputs already saved** (charts and metrics embedded):

| Content | Where to look in the notebook |
|---------|-------------------------------|
| Class distribution and samples | Phases 1–2 |
| Data augmentation | Phase 3 |
| Loss/accuracy curves per model | Phases 4–6 |
| Comparison of the 3 models | Phase 7 |
| Confusion matrix and errors | Phase 8 |
| Quantitative summary | Phase 9 |

> Weights are not saved to disk: to reproduce training from scratch, run the cells in order (GPU recommended).

## How to download the dataset

Official source: **[Intel Image Classification — Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)**

You need **`seg_train`** and **`seg_test`**. The unlabeled **`seg_pred`** folder is optional and not used in the current notebook.

### Option 1 — Via the website

1. Go to the dataset link and log in to [Kaggle](https://www.kaggle.com/).
2. Click **Download** and extract the `.zip`.
3. Copy **`seg_train`** and **`seg_test`** to the root of this repository (same level as the notebook).

```
seg_train/
├── buildings/
├── forest/
├── glacier/
├── mountain/
├── sea/
└── street/

seg_test/
├── buildings/
├── forest/
├── glacier/
├── mountain/
├── sea/
└── street/
```

### Option 2 — Kaggle API

1. `pip install kaggle`
2. Generate a token at [Kaggle → Settings → API](https://www.kaggle.com/settings) and save `kaggle.json` at:
   - **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`
   - **Linux/macOS:** `~/.kaggle/kaggle.json`
3. From the project folder:

```powershell
kaggle datasets download -d puneet6060/intel-image-classification
```

4. Extract the zip and move `seg_train` and `seg_test` to the repo root.

> **Important:** the dataset and `kaggle.json` are **not** committed to Git (`.gitignore`). Never version your Kaggle token.

## Repository structure

```
├── intel_image_classification.ipynb   # pipeline + saved results
├── requirements.txt
├── README.md                          # this file (English)
├── README.pt-br.md                    # Portuguese version
├── seg_train/                         # local (ignored by Git)
└── seg_test/                          # local (ignored by Git)
```

## Installation

Recommended environment: **Python 3.11 or 3.12** (avoid 3.14 for deep learning projects until the ecosystem is stable).

Reference GPU for this project: **NVIDIA GeForce RTX 4060 Ti** (CUDA 12.x). Any recent CUDA-capable NVIDIA GPU also works; without a GPU, training runs on CPU (slower).

1. Open a terminal in the project folder.
2. Create the virtual environment:
   ```powershell
   py -3.12 -m venv .venv
   ```
   If you don't have 3.12: `py -3.11 -m venv .venv` or `py -3 -m venv .venv`.
3. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks scripts, for this terminal session only:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
4. Upgrade pip:
   ```powershell
   python -m pip install --upgrade pip
   ```
5. Install PyTorch with CUDA (recommended for NVIDIA GPUs), using the official index for **CUDA 12.4**:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
   CUDA 12.6 alternative (if the command above fails or NVIDIA recommends a newer driver):
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
   CPU-only install: follow the CPU options at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
6. Install the remaining dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
7. (Optional) Register the Jupyter kernel:
   ```powershell
   python -m ipykernel install --user --name=intel-image-venv --display-name "Intel Image Classification (.venv)"
   ```
8. Verify the GPU:
   ```powershell
   python -c "import torch; print('PyTorch', torch.__version__); print('CUDA', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
   ```
9. In `intel_image_classification.ipynb`, select the kernel: **Select Kernel** → **Python Environments** → `.venv (Python 3.x)` in this project's folder.

### Troubleshooting

| Problem | Action |
|---------|--------|
| `CUDA False` | Reinstall step 5 with `cu124`; update the NVIDIA driver. |
| `No module named torch` | Activate the `.venv` before `pip install`. |
| Wrong kernel in the notebook | Explicitly select this folder's `.venv`. |

Official docs: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## How to run

1. Download the dataset as described above.
2. Follow the **Installation** section to create the `.venv` and install dependencies.
3. Open `intel_image_classification.ipynb`, select the `.venv` kernel, and run the cells in order (**Run All** on the first run).

## Requirements

- Python 3.11 or 3.12
- NVIDIA GPU with CUDA (optional, but recommended for training)
