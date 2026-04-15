"""Data loading and preprocessing module."""

import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

import config


def load_data():
    """Load dataset from IMDB or preprocessed CSV."""
    if os.path.exists(config.IMDB_PATH):
        print(f"Loading IMDB dataset from {config.IMDB_PATH}...")
        data = pd.read_csv(config.IMDB_PATH)
        # Convert IMDB format
        data = data.rename(columns={'review': 'text'})
        data['label'] = data['sentiment'].map({'positive': 1, 'negative': 0})
        data = data[['text', 'label']]
        # Save for future use
        os.makedirs('data', exist_ok=True)
        data.to_csv(config.DATA_PATH, index=False)
    else:
        print(f"Loading dataset from {config.DATA_PATH}...")
        data = pd.read_csv(config.DATA_PATH)
    
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
