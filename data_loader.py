"""Data loading and preprocessing module."""

import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

import config


def _ensure_data_dir():
    os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)


def load_data():
    """Load dataset from IMDB or preprocessed CSV."""
    if os.path.exists(config.IMDB_PATH):
        print(f"Loading IMDB dataset from {config.IMDB_PATH}...")
        data = pd.read_csv(config.IMDB_PATH)
        if 'review' not in data.columns or 'sentiment' not in data.columns:
            raise ValueError("IMDB dataset must contain 'review' and 'sentiment' columns.")
        data = data.rename(columns={'review': 'text'})
        data['label'] = data['sentiment'].map({'positive': 1, 'negative': 0})
        data = data[['text', 'label']]
        _ensure_data_dir()
        data.to_csv(config.DATA_PATH, index=False)
    elif os.path.exists(config.DATA_PATH):
        print(f"Loading dataset from {config.DATA_PATH}...")
        data = pd.read_csv(config.DATA_PATH)
        if 'text' not in data.columns or 'label' not in data.columns:
            raise ValueError("Preprocessed dataset must contain 'text' and 'label' columns.")
    else:
        raise FileNotFoundError(
            f"Dataset not found. Place '{config.IMDB_PATH}' or '{config.DATA_PATH}' in the project folder."
        )

    return data


def clean_text(text):
    """Clean and normalize text."""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_data(data):
    """Apply preprocessing to dataset."""
    print("Preprocessing text data...")
    data['clean_text'] = data['text'].apply(clean_text)
    return data


def split_data(data):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        data['clean_text'], 
        data['label'],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=data['label']
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test
