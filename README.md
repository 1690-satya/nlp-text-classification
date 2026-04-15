# 🚀 NLP Text Classification

A polished sentiment classification pipeline with both a lightweight TF-IDF baseline and a fine-tuned DistilBERT transformer model.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-Hugging%20Face-blueviolet?logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/scikit--learn-1.2+-yellow?logo=scikit-learn" alt="scikit-learn">
</p>

## 💡 What is this project?

This repository demonstrates a real-world NLP workflow:

- **Baseline model** using TF-IDF + Logistic Regression
- **Transformer model** using DistilBERT fine-tuning
- **Automated preprocessing** and dataset conversion
- **Full evaluation** with graphs and classification reports

## 📌 Why it matters

This project is a strong showcase for a developer who wants to build practical NLP systems with both classic machine learning and modern transformer techniques. It is ideal for portfolio presentation or iterative model experimentation.

## 🚀 Features

- Clean text preprocessing for noisy IMDB reviews
- Train/test split with stratified sampling
- Reusable model classes for baseline and transformer workflows
- Automatic saving of evaluation visuals
- Lightweight configuration via `config.py`

## 🧱 Project structure

```text
NLP Text Classification/
├── main.py                  # Entry point for running the full pipeline
├── config.py                # Project settings and model parameters
├── data_loader.py           # Data ingestion and preprocessing logic
├── models.py                # Baseline and transformer models
├── visualizations.py        # Plot generation and result reporting
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── IMDB Dataset.csv         # Raw IMDB dataset input file
└── results/                 # Generated output plots and reports
```

## 🛠️ Quick Start

```bash
pip install -r requirements.txt
python main.py
```

> Note: Place `IMDB Dataset.csv` in the repository root before running. The pipeline auto-converts it into `data/dataset.csv` on first run.

## 🧠 How it works

1. `data_loader.py` loads the raw dataset, validates columns, and preprocesses reviews.
2. `main.py` orchestrates the workflow, including visualizations and evaluation.
3. `models.py` trains and evaluates both a TF-IDF baseline and a DistilBERT classifier.
4. `visualizations.py` generates publication-quality charts for model performance.

## 📊 Result outputs

The pipeline produces:

- `results/class_distribution.png`
- `results/confusion_matrices.png`
- `results/metrics_comparison.png`

## 📦 Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Matplotlib
- Seaborn

## 🌟 What makes this repo "seasoned dev" quality?

- Clear and structured documentation
- Robust input validation and preprocessing
- Modular architecture with reusable components
- Visual reports for model comparison
- Clean code and minimal duplication

## 📈 Take it further

- Add support for more datasets
- Implement cross-validation for the baseline model
- Add model checkpointing and demo prediction script
- Deploy the transformer model as a web API

## 📬 Connect

If you want, I can also help build a dedicated GitHub profile README with a developer introduction, stats cards, and featured project highlights.