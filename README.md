# SegFormer-B5 Building Footprint Refinement Pipeline

An end-to-end deep learning framework utilizing **SegFormer-B5** (`nvidia/mit-b5`) for precision building footprint refinement and boundary sharpening.

---

## 1. Directory Structure Setup

Ensure your local system matches this path layout:

```text
G:\Segformer_buildings\
├── checkpoints/
├── configs/
│   └── config.py
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── validation/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── logs/
├── predictions/
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   └── visualize.py
├── dataset.py
├── engine.py
├── inference.py
├── model.py
├── predict.py
├── train.py
├── requirements.txt
└── README.md