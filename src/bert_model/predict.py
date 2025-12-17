import torch
import logging

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification

from src.utils import combine_labels_with_probabilities


class BertPredictor:
    """
    A class for predicting new data using a trained transformer.

    Attributes:
        tokenizer (DistilBertTokenizerFast): The Bert tokenizer.
        model (DistilBertForSequenceClassification): Trained Bert model

    Methods:
        predict_labels(data, label_map): Predicts labels and score for new data.
    """

    def __init__(self, checkpoint_path: str) -> None:
        """
        Initializes the BertPredictor class.
        """
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path)

    def predict_labels(self, summaries: list, label_map: dict) -> list:
        """
        Predicts labels for the given dataframe using the classifier model.

        Args:
            texts (list): List of prediction summary samples.
            label_map (dict): Mapping with ids and labels.

        Returns:
            list: A list of list with predicted labels and scores.
        """
        logging.info("Starting label prediction...")

        X = self.tokenizer(summaries, truncation=True, padding=True, max_length=512, return_tensors="pt")

        with torch.no_grad(): # To reduce memory consumption
            logits = self.model(**X).logits  # Predict the probabilities

        # Applying sigmoid because of multi-label classification
        y_predicted = torch.sigmoid(logits)

        # Inverse transform the predicted labels to original words
        predicted_probs = [combine_labels_with_probabilities(predictions, label_map) for predictions in y_predicted]

        logging.info("Label prediction completed.")
        return predicted_probs
