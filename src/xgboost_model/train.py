import os
import pickle
import joblib
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from src.utils import prepare_dir


class XGBoostTrainer:
    """
    A class for training a multi-label classifier using XGBoost.

    Attributes:
        tf_idf (TfidfVectorizer): The TF-IDF vectorizer.
        classifier (MultiOutputClassifier): The XGBoost classifier.
        checkpoint_dir (str): Directory for trained models.

    Methods:
        train(X_train, y_train): Trains the classifier using the provided data.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        """
        Initializes the XGBoostTrainer object.
        """
        self.tf_idf = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 3), stop_words="english"
        )
        self.classifier = MultiOutputClassifier(
            XGBClassifier(
                learning_rate=0.1,
                n_estimators=100,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
        )
        self.checkpoint_dir = checkpoint_dir

    def train(self, X_train: list, y_train: list) -> None:
        """
        Trains the classifier using the provided data.

        Args:
            X_train (list): List of training text samples.
            y_train (list): List of training labels as one-hot vectors.

        Returns:
            None
        """
        logging.info("Starting training...")

        X_train_vectorized = self.tf_idf.fit_transform(X_train)

        self.classifier.fit(X_train_vectorized, y_train)

        logging.info("Training completed.")

        prepare_dir(self.checkpoint_dir)

        with open(os.path.join(self.checkpoint_dir, "trained_model.pkl"), "wb") as f:
            pickle.dump(self.classifier, f)

        joblib.dump(
            self.tf_idf, os.path.join(self.checkpoint_dir, "trained_vectorizer.pkl")
        )

        logging.info("Model and vectorizer saved.")
