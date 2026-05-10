# NLP Text Classification

A lightweight, GPU-free sentiment classification pipeline comparing **TF-IDF + Logistic Regression** and **Count Vectorizer + Naive Bayes**, both using **unigram and bigram** features.

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-yellow?logo=scikit-learn)](https://scikit-learn.org/)

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Outputs](#outputs)
- [Tech Stack](#tech-stack)

---

## Overview

This project demonstrates a complete NLP workflow in pure Python:

- Load and clean raw IMDB review text
- Vectorize with unigrams and bigrams
- Train two classic ML models side-by-side
- Compare performance with saved plots and reports

No transformers, no PyTorch, no GPU required.

---

## Models

| Model | Vectorizer | Classifier | N-grams | Key Traits |
|-------|------------|------------|---------|------------|
| **Logistic Regression** | TF-IDF | LogisticRegression | (1, 2) | Strong baseline, interpretable weights |
| **Naive Bayes** | Count Vectorizer | MultinomialNB | (1, 2) | Fast, works well with word counts |

Both models are capped at `10,000` features and remove English stop words.

---

## Project Structure

```text
nlp-text-classification/
├── main.py              # Pipeline entry point
├── config.py            # Paths, hyperparameters, random seed
├── data_loader.py       # Load IMDB CSV, clean text, split data
├── models.py            # LogisticRegressionModel & NaiveBayesModel
├── visualizations.py    # Generate and save evaluation charts
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── results/             # Auto-created output folder for plots
```

---

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your dataset**
   - Place `IMDB Dataset.csv` in the project root (columns: `review`, `sentiment`).
   - Or place a preprocessed `data/dataset.csv` (columns: `text`, `label`).

3. **Run the pipeline**
   ```bash
   python main.py
   ```

> On the first run, the script auto-converts `IMDB Dataset.csv` into `data/dataset.csv`.

---

## How It Works

| Step | File | Action |
|------|------|--------|
| 1 | `data_loader.py` | Loads CSV, validates columns, cleans text, splits 80/20 stratified |
| 2 | `main.py` | Orchestrates training, evaluation, and plotting |
| 3 | `models.py` | Fits TF-IDF + LR and Count + NB on the training set |
| 4 | `visualizations.py` | Saves class distribution, confusion matrices, and metrics comparison |

---

## Outputs

After running `main.py`, check the `results/` folder:

| File | Description |
|------|-------------|
| `class_distribution.png` | Bar chart of positive vs negative samples |
| `confusion_matrices.png` | Side-by-side confusion matrices for both models |
| `metrics_comparison.png` | Grouped bar chart of accuracy, precision, recall, F1 |

### Sample Console Output

```
============================================================
NLP TEXT CLASSIFICATION PROJECT
TF-IDF + Logistic Regression  vs.  Count Vectorizer + Naive Bayes
============================================================

[1/5] Loading data...
[2/5] Visualizing class distribution...
[3/5] Preprocessing data...
[4/5] Splitting data...
[5/5] Training models...

Logistic Regression Accuracy: 0.8901
Naive Bayes Accuracy: 0.8532

Generating visualizations...
============================================================
PIPELINE COMPLETE!
Results saved to: results/
============================================================
```

---

## Tech Stack

- **Python**
- **pandas** – data loading and manipulation
- **NumPy** – numerical operations
- **scikit-learn** – vectorization, modeling, metrics
- **Matplotlib & Seaborn** – plotting



