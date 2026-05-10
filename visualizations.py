"""Visualization module."""

import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

import config

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_class_distribution(data, save_path=None):
    """Plot class distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    class_counts = data['label'].value_counts().sort_index()
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(
        class_counts.index, class_counts.values,
        color=colors, edgecolor='black', linewidth=1.2
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

    ax.set_xlabel('Class Label', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Negative (0)', 'Positive (1)'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close(fig)


def plot_confusion_matrices(y_test, preds1, preds2, acc1, acc2, save_path=None):
    """Plot side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm1 = confusion_matrix(y_test, preds1)
    sns.heatmap(
        cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        cbar=False
    )
    axes[0].set_title(
        f'TF-IDF + Logistic Regression\nAccuracy: {acc1:.3f}',
        fontsize=12, fontweight='bold'
    )
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    cm2 = confusion_matrix(y_test, preds2)
    sns.heatmap(
        cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1],
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        cbar=False
    )
    axes[1].set_title(
        f'Count Vectorizer + Naive Bayes\nAccuracy: {acc2:.3f}',
        fontsize=12, fontweight='bold'
    )
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close(fig)


def plot_metrics_comparison(y_test, preds1, preds2, acc1, acc2, save_path=None):
    """Plot metrics comparison bar chart."""
    models = ['TF-IDF + LR', 'Count + NB']
    accuracy_scores = [acc1, acc2]
    precision_scores = [
        precision_score(y_test, preds1, average='macro'),
        precision_score(y_test, preds2, average='macro')
    ]
    recall_scores = [
        recall_score(y_test, preds1, average='macro'),
        recall_score(y_test, preds2, average='macro')
    ]
    f1_scores = [
        f1_score(y_test, preds1, average='macro'),
        f1_score(y_test, preds2, average='macro')
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.2

    bars1 = ax.bar(
        x - 1.5 * width, accuracy_scores, width,
        label='Accuracy', color='#3498db'
    )
    bars2 = ax.bar(
        x - 0.5 * width, precision_scores, width,
        label='Precision', color='#2ecc71'
    )
    bars3 = ax.bar(
        x + 0.5 * width, recall_scores, width,
        label='Recall', color='#e74c3c'
    )
    bars4 = ax.bar(
        x + 1.5 * width, f1_scores, width,
        label='F1-Score', color='#9b59b6'
    )

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9
            )

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("SUMMARY: Model Comparison")
    print("=" * 60)
    print(f"{'Metric':<15} {'TF-IDF + LR':>15} {'Count + NB':>15}")
    print("-" * 60)
    print(f"{'Accuracy':<15} {accuracy_scores[0]:>15.4f} {accuracy_scores[1]:>15.4f}")
    print(f"{'Precision':<15} {precision_scores[0]:>15.4f} {precision_scores[1]:>15.4f}")
    print(f"{'Recall':<15} {recall_scores[0]:>15.4f} {recall_scores[1]:>15.4f}")
    print(f"{'F1-Score':<15} {f1_scores[0]:>15.4f} {f1_scores[1]:>15.4f}")
    print("=" * 60)
