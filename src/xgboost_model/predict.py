import pickle
import numpy as np
import joblib
import logging
from src.utils import combine_labels_with_probabilities


class XGBoostPredictor:
    """
    A class for predicting new data using a trained XGBoost.

    Attributes:
        tf_idf (TfidfVectorizer): Trained vectorizer.
        model (XGBClassifier): Trained XGBoost model.

    Methods:
        predict_labels(texts, label_map): Predicts labels and score for new data.
    """

    def __init__(self, model_checkpoint_path: str, vectorizer_checkpoint_path: str) -> None:
        """
        Initializes the XGBoostPredictor class.
        """
        self.tf_idf = joblib.load(vectorizer_checkpoint_path)
        self.model = pickle.load(open(model_checkpoint_path, "rb"))

    def predict_labels(self, texts: list, label_map: dict) -> list:
        """
        Predicts labels for the given dataframe using the classifier model.

        Args:
            texts (list): List of prediction text samples.
            label_map (dict): Mapping with ids and labels.

        Returns:
            list: A list of list with predicted labels and scores.
        """
        logging.info("Starting label prediction...")
        
        X = self.tf_idf.transform(texts)
        y_predicted_probs = self.model.predict_proba(X)  # Predict the probabilities

        # Transpose the predicted probabilities
        y_predicted_probs_transpose = np.array(y_predicted_probs).transpose(1, 0, 2) 

        # Get the "positive" score for each label
        y_predicted = y_predicted_probs_transpose[:, :, 1]

        predicted_probs = [combine_labels_with_probabilities(predictions, label_map) for predictions in y_predicted]

        logging.info("Label prediction completed.")
        return predicted_probs
