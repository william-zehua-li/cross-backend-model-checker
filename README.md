# cross-backend-model-checker

This repository provides an open-source, reproducible framework for **testing deep learning model compatibility across devices and execution backends**.  
It supports **CPU**, **GPU**, and **torch.compile** pipelines, with configurable tolerance thresholds and structured logging.

The framework is designed as a **research artifact**, intended to complement papers on **model reproducibility, backend consistency, and operator coverage**.

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
git clone https://github.com/<your-username>/cross-backend-model-checker.git
cd cross-backend-model-checker
pip install -r requirements.txt
