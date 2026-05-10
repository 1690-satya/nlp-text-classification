"""Machine learning models module."""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import config


class LogisticRegressionModel:
    """TF-IDF + Logistic Regression."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=config.MAX_FEATURES,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE
        )

    def train(self, X_train, y_train):
        print("Training TF-IDF + Logistic Regression...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)
        print("Done!")

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nLogistic Regression Accuracy: {accuracy:.4f}")
        print(classification_report(
            y_test, predictions,
            target_names=['Negative', 'Positive'],
            digits=4
        ))
        return predictions, accuracy


class NaiveBayesModel:
    """Count Vectorizer + Multinomial Naive Bayes."""

    def __init__(self):
        self.vectorizer = CountVectorizer(
            max_features=config.MAX_FEATURES,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        print("Training Count Vectorizer + Naive Bayes...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)
        print("Done!")

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nNaive Bayes Accuracy: {accuracy:.4f}")
        print(classification_report(
            y_test, predictions,
            target_names=['Negative', 'Positive'],
            digits=4
        ))
        return predictions, accuracy
