"""Configuration settings for the NLP project."""

import os

# Paths
DATA_PATH = 'data/dataset.csv'
IMDB_PATH = 'IMDB Dataset.csv'
RESULTS_DIR = 'results'

# Model settings
MAX_FEATURES = 10000
TEST_SIZE = 0.2
RANDOM_STATE = 42


# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)
