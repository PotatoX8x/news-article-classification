import logging
import torch

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

from src.bert_model.dataset import NewsDataset
from src.utils import prepare_dir


class BertTrainer:
    """
    A class for training a multi-label classifier using a transformer.

    Attributes:
        tokenizer (DistilBertTokenizerFast): The Bert tokenizer.
        checkpoint_dir (str): Directory for trained models.

    Methods:
        prepare_datasets(train_summaries, train_labels, val_summaries, val_labels): Creates train and validation datasets with tokenized text.
        train(X_train, X_val, y_train, y_val): Trains the transformer using the provided data.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        """
        Initializes the TrainBert object.
        """
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.checkpoint_dir = checkpoint_dir

    def prepare_datasets(self, train_summaries: list, val_summaries: list, train_labels: list, val_labels: list) -> tuple:
        """
        Prepares training and validation datasets by tokenizing inputs.

        Args:
            train_summaries (list): A list of input train text samples.
            val_summaries (list): A list of input validation text samples.
            train_labels (list): A list of corresponding train labels for each text.
            val_labels (list): A list of corresponding validation labels for each text.

        Returns:
            tuple: A tuple containing the training dataset and validation dataset.
        """

        # Tokenize train & validation data
        train_encodings = self.tokenizer(train_summaries, truncation=True, padding=True, max_length=512, return_tensors="pt")
        val_encodings = self.tokenizer(val_summaries, truncation=True, padding=True, max_length=512, return_tensors="pt")

        # Convert labels to tensor
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float)
        val_labels_tensor = torch.tensor(val_labels, dtype=torch.float)

        # Create dataset objects
        train_dataset = NewsDataset(train_encodings, train_labels_tensor)
        val_dataset = NewsDataset(val_encodings, val_labels_tensor)

        return train_dataset, val_dataset

    def train(self, X_train: list, X_val: list, y_train: list, y_val: list) -> None:
        """
        Trains the transformer using the provided data.

        Args:
            X_train (list): List of training summary samples.
            X_val (list): List of validation summary samples.
            y_train (list): List of training labels as one-hot vectors.
            y_val (list): List of validation labels as one-hot vectors.

        Returns:
            None
        """
        logging.info("Starting training...")

        train_dataset, val_dataset = self.prepare_datasets(X_train, X_val, y_train, y_val)

        # Model is initialized here instead of init() because it requires a number of labels
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=len(y_train[0]), 
            problem_type="multi_label_classification"
        )

        prepare_dir(self.checkpoint_dir)   

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,  
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=4,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

        logging.info("Training completed.")