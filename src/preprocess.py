import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
import logging
from collections import Counter
from bs4 import BeautifulSoup

nltk.download("stopwords")
nltk.download("punkt")

nlp = spacy.load("en_core_web_sm")


class DataPreprocessor:
    """
    Class for preprocessing data and keeping important features in the data.
    """

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocesses the given text by removing special characters, converting to lowercase,
        removing stop words, removing short words and removing duplicate sentences.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        if not isinstance(text, str):
            logging.warning("Input text is not a string.")
            return ""

        stop_words = set(stopwords.words("english"))

        text = re.sub(r'[^a-zA-Z0-9.,!?\'" ]', "", text)
        text = text.lower()
        text = " ".join([word for word in text.split() if word not in stop_words])
        text = " ".join([word for word in text.split() if len(word) > 2])
        text = ". ".join(list(dict.fromkeys(text.split(". "))))

        return text.strip()

    @staticmethod
    def strip_html_tags(text: str) -> str:
        """
        Strips HTML tags from the given body of text.

        Args:
            text (str): The input text containing HTML tags.

        Returns:
            str: The text with HTML tags removed.
        """
        text = " ".join(BeautifulSoup(text, "html.parser").stripped_strings)

        return text

    @staticmethod
    def summarize_text(text: str, num_sentences: int=3) -> str:
        """
        Select top-n most important sentences, ranked by the amount of keywords.

        Args:
            text (str): The input text to be summarized.
            num_sentences (int): Number of top sentences to select.

        Returns:
            str: Summary with most important sentences in the text.
        """
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        word_freq = Counter(word.text.lower() for word in doc if word.is_alpha)
        ranked_sentences = sorted(
            sentences,
            key=lambda s: sum(word_freq.get(w.text.lower(), 0) for w in nlp(s)),
            reverse=True,
        )
        return " ".join(ranked_sentences[:num_sentences])

    def preprocess_data(self, data: list[dict]) -> pd.DataFrame:
        """
        Preprocesses the given dataframe by stripping HTML tags, combining important features,
        and applying text preprocessing.

        Args:
            dataframe: The input dataframe to be preprocessed.

        Returns:
            pd.DataFrame: The preprocessed dataframe.
        """
        logging.info("Starting data preprocessing...")

        df = pd.DataFrame(data)
        print(df.columns)

        df_expanded = pd.json_normalize(df["content"])
        updated_data = pd.concat([df, df_expanded], axis=1).drop(columns=["content"])

        logging.info("Stripping HTML tags...")
        updated_data["fullTextHtml"] = updated_data["fullTextHtml"].apply(
            self.strip_html_tags
        )

        updated_data["text"] = (
            updated_data["title"] + " " + updated_data["fullTextHtml"]
        )
        updated_data["text"] = updated_data["text"].apply(
            lambda x: re.sub(r"\s+", " ", str(x)).strip()
        )

        logging.info("Applying text preprocessing...")
        updated_data["text"] = updated_data["text"].apply(self.preprocess_text)

        updated_data["summary"] = updated_data["text"].apply(self.summarize_text)

        if "labels" in updated_data.columns:
            logging.info("Extracting labels...")
            updated_data["label_list"] = updated_data["labels"].apply(
                lambda x: [prob[0] for prob in x if prob[1] > 0.5]
            )

        logging.info("Data preprocessing complete.")
        return (
            updated_data[["text", "summary", "labels", "label_list"]]
            if "labels" in updated_data.columns
            else updated_data[["text", "summary"]]
        )
