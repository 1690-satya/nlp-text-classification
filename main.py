"""Main script to run the NLP Text Classification pipeline."""

import warnings
warnings.filterwarnings('ignore')

from data_loader import load_data, preprocess_data, split_data
from models import BaselineModel, TransformerModel
from visualizations import plot_class_distribution, plot_confusion_matrices, plot_metrics_comparison
import config


def main():
    """Run the complete NLP classification pipeline."""
    
    print("=" * 60)
    print("NLP TEXT CLASSIFICATION PROJECT")
    print("TF-IDF Baseline + DistilBERT Transformer")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    data = load_data()
    print(f"Dataset shape: {data.shape}")
    print(f"Class distribution:\n{data['label'].value_counts().sort_index()}")
    
    # 2. Visualize class distribution
    print("\n[2/6] Visualizing class distribution...")
    plot_class_distribution(data, save_path=f"{config.RESULTS_DIR}/class_distribution.png")
    
    # 3. Preprocess data
    print("\n[3/6] Preprocessing data...")
    data = preprocess_data(data)
    
    # 4. Split data
    print("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(data)
    
    # 5. Train and evaluate baseline model
    print("\n[5/6] Training baseline model...")
    baseline = BaselineModel()
    baseline.train(X_train, y_train)
    baseline_preds, baseline_acc = baseline.evaluate(X_test, y_test)
    
    # 6. Train and evaluate transformer model
    print("\n[6/6] Training transformer model...")
    transformer = TransformerModel()
    transformer.train(X_train, y_train, X_test, y_test)
    transformer_preds, transformer_acc = transformer.evaluate(X_test, y_test)
    
    # Visualize results
    print("\nGenerating visualizations...")
    plot_confusion_matrices(y_test, baseline_preds, transformer_preds,
                           baseline_acc, transformer_acc,
                           save_path=f"{config.RESULTS_DIR}/confusion_matrices.png")
    
    plot_metrics_comparison(y_test, baseline_preds, transformer_preds,
                           baseline_acc, transformer_acc,
                           save_path=f"{config.RESULTS_DIR}/metrics_comparison.png")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print(f"Results saved to: {config.RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
