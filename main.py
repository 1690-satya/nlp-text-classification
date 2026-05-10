"""Main script to run the NLP Text Classification pipeline."""

import warnings
warnings.filterwarnings('ignore')

import os
import sys

from data_loader import load_data, preprocess_data, split_data
from models import LogisticRegressionModel, NaiveBayesModel
from visualizations import (
    plot_class_distribution,
    plot_confusion_matrices,
    plot_metrics_comparison
)
import config


def main():
    """Run the complete NLP classification pipeline."""

    print("=" * 60)
    print("NLP TEXT CLASSIFICATION PROJECT")
    print("TF-IDF + Logistic Regression  vs.  Count Vectorizer + Naive Bayes")
    print("=" * 60)

    try:
        # 1. Load data
        print("\n[1/5] Loading data...")
        data = load_data()
        print(f"Dataset shape: {data.shape}")
        print(f"Class distribution:\n{data['label'].value_counts().sort_index()}")

        # 2. Visualize class distribution
        print("\n[2/5] Visualizing class distribution...")
        plot_class_distribution(
            data,
            save_path=os.path.join(config.RESULTS_DIR, "class_distribution.png")
        )

        # 3. Preprocess data
        print("\n[3/5] Preprocessing data...")
        data = preprocess_data(data)

        # 4. Split data
        print("\n[4/5] Splitting data...")
        X_train, X_test, y_train, y_test = split_data(data)

        # 5. Train and evaluate models
        print("\n[5/5] Training models...")

        model1 = LogisticRegressionModel()
        model1.train(X_train, y_train)
        preds1, acc1 = model1.evaluate(X_test, y_test)

        model2 = NaiveBayesModel()
        model2.train(X_train, y_train)
        preds2, acc2 = model2.evaluate(X_test, y_test)

        # Visualize results
        print("\nGenerating visualizations...")
        plot_confusion_matrices(
            y_test, preds1, preds2, acc1, acc2,
            save_path=os.path.join(config.RESULTS_DIR, "confusion_matrices.png")
        )

        plot_metrics_comparison(
            y_test, preds1, preds2, acc1, acc2,
            save_path=os.path.join(config.RESULTS_DIR, "metrics_comparison.png")
        )

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print(f"Results saved to: {config.RESULTS_DIR}/")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
