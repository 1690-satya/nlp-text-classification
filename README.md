# NLP Text Classification Project

End-to-end NLP classification pipeline using TF-IDF baseline and fine-tuned DistilBERT for sentiment analysis or fake news detection.

## Overview

- **Baseline**: TF-IDF + Logistic Regression
- **Advanced**: Fine-tuned DistilBERT (Hugging Face Transformers)
- **Evaluation**: Accuracy, Precision, Recall, F1-score + Confusion Matrices

## Project Structure

```
NLP Text Classification/
├── main.py                  # Entry point - run this script
├── config.py                # Configuration settings
├── data_loader.py           # Data loading and preprocessing
├── models.py                # Baseline and transformer models
├── visualizations.py        # Plotting functions
├── requirements.txt         # Dependencies
├── README.md                # Documentation
└── results/                 # Output directory (auto-created)
    ├── class_distribution.png
    ├── confusion_matrices.png
    └── metrics_comparison.png
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python main.py
```

## How It Works

1. **data_loader.py** - Loads IMDB dataset, cleans text, splits into train/test
2. **models.py** - Contains `BaselineModel` (TF-IDF + LR) and `TransformerModel` (DistilBERT)
3. **visualizations.py** - Creates 3 plots: class distribution, confusion matrices, metrics comparison
4. **config.py** - Central settings (paths, model parameters, etc.)
5. **main.py** - Orchestrates the entire pipeline

## Visualizations

1. **Class Distribution** - Bar chart showing dataset balance
2. **Confusion Matrices** - Side-by-side heatmaps for both models
3. **Metrics Comparison** - Grouped bar chart (Accuracy, Precision, Recall, F1)

All plots are automatically saved to the `results/` folder.

## Dataset

Place your `IMDB Dataset.csv` in the project folder (it will be auto-converted to the correct format).

Expected IMDB format:
- `review` column: The movie review text
- `sentiment` column: "positive" or "negative"

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- matplotlib, seaborn
- GPU recommended for transformer training

## Configuration

Edit `config.py` to customize:
- `EPOCHS`: Number of training epochs (default: 2)
- `BATCH_SIZE`: Training batch size (default: 8)
- `MAX_FEATURES`: TF-IDF features (default: 5000)

## Tips

- First run will download DistilBERT (~250MB)
- Transformer training is slow on CPU - use GPU if available
- Results are saved in `results/` directory automatically
