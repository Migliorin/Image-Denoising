# Save of params and results used in train of model
---
# Content

<!-- vim-markdown-toc GitLab -->

* [1. VisionModelTransformerTorchV2 (I named it like this)](#1-visionmodeltransformertorchv2-i-named-it-like-this)
    * [1.2 Change Only Learning Rate](#12-change-only-learning-rate)

<!-- vim-markdown-toc -->

---
# 1. VisionModelTransformerTorchV2 (I named it like this)

## 1.2 Change Only Learning Rate

| Name | Version | Batch Size | Epochs | Loss | Best loss | Learning Rate | Patch Size | Token Length | Heads | Layers | Noises | Dataframe | Dataset Transformation |
|------|---------|------------|--------|------|-----------|---------------|------------|--------------|-------|--------|--------|-----------|------------------------|
| Transformer Torch V2 Normal | 2 | 32 | 64 | 185723.093750 | MSELossPatchEinops | 0.001 | 14 | 512 | 8 | 6 | Salt Pepper | `dataframe_v1.csv` | `get_transform_v2()` |
| Transformer Torch V2 Normal | 3 | 32 | 64 | 44900978688.000000 | MSELossPatchEinops | 0.01 | 14 | 512 | 8 | 6 | Salt Pepper | `dataframe_v1.csv` | `get_transform_v2()` |
| Transformer Torch V2 Normal | 4 | 32 | 64 | 172914.640625 | MSELossPatchEinops | 0.0001 | 14 | 512 | 8 | 6 | Salt Pepper | `dataframe_v1.csv` | `get_transform_v2()` |
| Transformer Torch V2 Normal | 5 | 32 | 256 | 38454.510000 | MSELossPatchEinops | 0.0001 | 14 | 512 | 8 | 6 | Salt Pepper | `dataframe_v1.csv` | `get_transform_v2()` |
| Transformer Torch V2 Normal | 6 | 32 | 256 | 132827.270000 | MSELossPatchEinops | 0.00001 | 14 | 512 | 8 | 6 | Salt Pepper | `dataframe_v1.csv` | `get_transform_v2()` |
| Transformer Torch V2 Normal | 7 | 32 | 1024 | 88106.650000 | MSELossPatchEinops | 0.0001 | 14 | 512 | 8 | 6 | Salt Pepper | `dataframe_v1.csv` | `get_transform_v2()` |
