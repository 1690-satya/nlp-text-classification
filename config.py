"""Configuration settings for the NLP project."""

import os

# Paths
DATA_PATH = 'data/dataset.csv'
IMDB_PATH = 'IMDB Dataset.csv'
RESULTS_DIR = 'results'

# Model settings
MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Transformer settings
TRANSFORMER_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 2

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)
