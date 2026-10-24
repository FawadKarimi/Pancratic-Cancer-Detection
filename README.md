# 3D Explainable Deep Learning for Pancreatic Cancer Detection (Research Prototype)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CT](https://img.shields.io/badge/Modality-Abdominal%20CT-0891b2?style=flat-square)](#disclaimer)

Research prototype exploring a **production-style PyTorch pipeline** for **early pancreatic cancer detection** from **3D CT volumes**: preprocessing, model training, checkpointing, logging, and **explainability metrics** (e.g., IoU-style saliency checks, pointing-game style evaluation hooks).

> **Status:** Experimental research code. **Clinical performance has not been established.** Not for diagnostic use.

## Features

- **`CTPreprocessor`** — HU clipping/normalization, resampling, crop/pad to fixed tensors  
- **Training loop** — AdamW, cosine warm restarts, structured logging under `output/logs/`  
- **XAI hooks** — interfaces for attribution-based evaluation (see module docstrings)  
- **Single-entry script** — `Pancratic Cancer Detection.py` consolidates pipeline stages  

## Repository layout

```
├── Pancratic Cancer Detection.py   # Main implementation (dataset, model, train, XAI)
└── output/logs/                    # Timestamped training logs (examples included)
```

## Quick start

```bash
git clone https://github.com/FawadKarimi/Pancratic-Cancer-Detection.git
cd Pancratic-Cancer-Detection
python -m venv .venv && source .venv/bin/activate
pip install torch numpy nibabel scipy matplotlib tqdm
python "Pancratic Cancer Detection.py"
```

You must supply **NIfTI / volumetric CT data** and adapt paths inside the script to your cohort. The included logs document prior dry-run / development sessions—they are **not** published benchmark results.

### Recommended hardware

- Linux workstation with **NVIDIA GPU** (CUDA) for 3D conv training  
- Sufficient disk for preprocessed volumes  

## Development notes

- Preprocessing targets configurable **HU bounds**, **isotropic spacing**, and **tensor shape** (defaults documented in `CTPreprocessor`).  
- Logging uses Python `logging` with per-run files (`training_YYYYMMDD_HHMMSS.log`).  
- `.idea/` metadata may appear from local PyCharm sessions; safe to ignore for runs.  

## Relationship to other work

This public prototype is separate from ongoing **private master's research** on lung-nodule detection featured on [fawadkarimi.github.io](https://fawadkarimi.github.io/). No cross-claim of validation between projects.

## Disclaimer

For **research and engineering experimentation** only. Not validated on multi-site clinical data. Always follow **IRB/ethics** and data-protection rules when handling medical imaging.

## Author

**Mohammad Fawad Karimi** — Medical AI / CT imaging research  
📧 [mfkarimi@mhs.usk.ac.id](mailto:mfkarimi@mhs.usk.ac.id) · 🌐 [Portfolio](https://fawadkarimi.github.io/)
