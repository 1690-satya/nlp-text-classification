"""Machine learning models module."""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

import config


class BaselineModel:
    """TF-IDF + Logistic Regression baseline."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=config.MAX_FEATURES)
        self.model = LogisticRegression(max_iter=200)
    
    def train(self, X_train, y_train):
        """Train the baseline model."""
        print("Training baseline model (TF-IDF + Logistic Regression)...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)
        print("Baseline model trained!")
    
    def predict(self, X):
        """Make predictions."""
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict(X_tfidf)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nBaseline Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, 
                                   target_names=['Negative', 'Positive']))
        return predictions, accuracy


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values) if labels is not None else None

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class TransformerModel:
    """DistilBERT transformer model."""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL)
        self.model = None
        self.trainer = None
    
    def _tokenize(self, texts):
        """Tokenize texts."""
        return self.tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt"
        )
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train the transformer model."""
        print("\nTraining transformer model (DistilBERT)...")
        
        train_encodings = self._tokenize(X_train)
        test_encodings = self._tokenize(X_test)
        
        train_dataset = TextDataset(train_encodings, y_train)
        test_dataset = TextDataset(test_encodings, y_test)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.TRANSFORMER_MODEL,
            num_labels=2
        )
        
        training_args = TrainingArguments(
            output_dir=config.RESULTS_DIR,
            num_train_epochs=config.EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            evaluation_strategy="epoch",
            logging_dir=f"{config.RESULTS_DIR}/logs",
            save_strategy="no"
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        self.trainer.train()
        print("Transformer model trained!")
    
    def predict(self, X_test):
        """Make predictions."""
        test_encodings = self._tokenize(X_test)
        test_dataset = TextDataset(test_encodings)
        predictions = self.trainer.predict(test_dataset)
        return np.argmax(predictions.predictions, axis=1)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nTransformer Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions,
                                   target_names=['Negative', 'Positive']))
        return predictions, accuracy
