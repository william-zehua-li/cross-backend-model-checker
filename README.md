# cross-backend-model-checker

This repository provides an open-source, reproducible framework for **testing deep learning model compatibility across devices and execution backends**.  
It supports **CPU**, **GPU**, and **torch.compile** pipelines, with configurable tolerance thresholds and structured logging.

The framework is designed as a **research artifact**, intended to complement papers on **model reproducibility, backend consistency, and operator coverage**.

Preprint available at [link] (under submission to arXiv)

---

## Paper Detail
**Authors:** Zehua Li  
**Affiliation:** Dalhousie University  
**Status:** Submitted to arXiv (under review)

📄 **PDF Download (v1.0, 2025-08-29):**  
https://github.com/william-zehua-li/cross-backend-model-checker/blob/569af53c53123bf788dbc78093fba1b53be04292/docs/Toward_Reproducible_Cross_Backend_Compatibility_for_Deep_Learning__A_Configuration_First_Framework_with_Three_Tier_Verification.pdf

---

## ✨ Features
- Batch testing of TorchVision and custom models via **YAML configuration**
- Device & backend coverage: `cpu`, `cuda`, `compiled`
- Flexible tolerance: configurable `atol` and `rtol`
- Structured logging with JSONL output for analysis
- Reproducibility: deterministic seeds and environment locking
- Modular design: extend with new loaders, comparators, and tasks

---

## 📦 Installation
```bash
git clone https://github.com/william-zehua-li/cross-backend-model-checker.git
cd cross-backend-model-checker
pip install -r requirements.txt
