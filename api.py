import os
import json
import logging
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from sklearn.model_selection import train_test_split

from src.preprocess import DataPreprocessor

from src.xgboost_model.train import XGBoostTrainer
from src.xgboost_model.predict import XGBoostPredictor

from src.bert_model.train import BertTrainer
from src.bert_model.predict import BertPredictor

from src.utils import get_latest_checkpoint, labels_to_vector, prepare_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


LOADED_DATA_DIR = "./data"
XGBOOST_CHECKPOINT_DIR = "./xgboost_model/training"
BERT_CHECKPOINT_DIR = "./bert_model/training"

prepare_dir(LOADED_DATA_DIR)

data_preprocessor = DataPreprocessor()

app = FastAPI()

@app.post("/preprocess_data")
async def preprocess_data(
    dataset_file: UploadFile = File(...),
    labels_file: UploadFile = File(...)
) -> JSONResponse:
    """
    Preprocess the training data. For 7000 samples, takes arond 30 mins.
    """
    try:
        # Read the content of the uploaded dataset file
        dataset_content = await dataset_file.read()
        data = json.loads(dataset_content)

        # Read the content of the uploaded labels file
        labels_content = await labels_file.read()
        label_map = json.loads(labels_content)
        with open(os.path.join(LOADED_DATA_DIR, "label_map.json"), "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=4)

        # Preprocess the data
        df = data_preprocessor.preprocess_data(data)
        df_json = json.loads(df.to_json())
        json.dump(df_json, open(os.path.join(LOADED_DATA_DIR, "train_df.json"), "w"))

        logging.info("Data preprocessed successfully.")

        # Return the preprocessed data as JSON response
        return JSONResponse(content=df_json)
    except json.decoder.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")

@app.post("/train")
async def train_model(
    model_type: str = Query("train_model", enum=["xgboost_model", "bert_model"]),
) -> dict[str, str]:
    """
    Train the selected model on the preprocessed data.

    - xgboost_model: on 7000 samples, training takes ~10 min.
    - bert_model: on 7000 samples, training takes ~10 hours.
    """
    try:

        # Read the uploaded labels
        with open(os.path.join(LOADED_DATA_DIR, "label_map.json")) as f:
            label_map = json.load(f)

        # Read the uploaded and preprocessed training data
        with open(os.path.join(LOADED_DATA_DIR, "train_df.json")) as f:
            data = json.load(f)

        data = pd.DataFrame(data)

        # Split data into training, validation and test sets
        all_texts = data["text"].tolist()
        all_summaries = data["summary"].tolist()
        all_labels = [labels_to_vector(label_list, label_map) for label_list in data["label_list"].tolist()]

        # Text

        # Split data (80% train)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            all_texts, all_labels, test_size=0.2, random_state=42
        )
        # Split data (10% validation, 10% test
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            val_texts, val_labels, test_size=0.5, random_state=42
        )

        # Summaries

        # Split data (80% train)
        train_summaries, val_summaries, train_labels, val_labels = train_test_split(
            all_summaries, all_labels, test_size=0.2, random_state=42
        )

        # Split data (10% validation, 10% test)
        val_summaries, test_summaries, val_labels, test_labels = train_test_split(
            val_summaries, val_labels, test_size=0.5, random_state=42
        )

        train_df = pd.DataFrame({
            "text": train_texts,
            "summary": train_summaries,
            "labels": train_labels
        })

        validation_df = pd.DataFrame({
            "text": val_texts,
            "summary": val_summaries,
            "labels": val_labels
        })

        test_df = pd.DataFrame({
            "text": test_texts,
            "summary": test_summaries,
            "labels": test_labels
        })

        # Train the classifier using the preprocessed data
        if model_type == "xgboost_model":
            xgboost_trainer = XGBoostTrainer(XGBOOST_CHECKPOINT_DIR)
            xgboost_trainer.train(train_df["text"].tolist(), train_df["labels"].tolist())

        elif model_type == "bert_model":
            bert_trainer = BertTrainer(BERT_CHECKPOINT_DIR)
            bert_trainer.train(
                train_df["summary"].tolist(),
                validation_df["summary"].tolist(),
                train_df["labels"].tolist(),
                validation_df["labels"].tolist(),
            )
            
        else:
            raise HTTPException(status_code=400, detail="Invalid model type.")

        # Log the training result
        logging.info("Model trained successfully and saved to disk.")

        # Return a success message
        return {"message": "Model was trained successfully and saved to disk."}
    except json.decoder.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")

@app.post("/predict")
async def predict(
    predict_payload: UploadFile = File(...),
    model_type: str = Query("xgboost_model", enum=["xgboost_model", "bert_model"]),
) -> JSONResponse:
    """
    Predict labels of new data using the selected model. Returns the list of lists with labels and scores.
    """
    try:

        # Read the uploaded and preprocessed training data
        with open(os.path.join(LOADED_DATA_DIR, "label_map.json")) as f:
            label_map = json.load(f)

        content = await predict_payload.read()
        data = json.loads(content)

        preprocessed_data = data_preprocessor.preprocess_data(data)

        # Check the model type and predict labels
        if model_type == "xgboost_model":
            xgboost_predictor = XGBoostPredictor(
                os.path.join(XGBOOST_CHECKPOINT_DIR, "trained_model.pkl"),
                os.path.join(XGBOOST_CHECKPOINT_DIR, "trained_vectorizer.pkl")
            )
            predictions = xgboost_predictor.predict_labels(preprocessed_data["text"].tolist(), label_map)
            
        elif model_type == "bert_model":
            checkpoint_path = get_latest_checkpoint(BERT_CHECKPOINT_DIR)
            bert_predictor = BertPredictor(os.path.join(BERT_CHECKPOINT_DIR, checkpoint_path))
            predictions = bert_predictor.predict_labels(preprocessed_data["summary"].tolist(), label_map)

        else:
            raise HTTPException(status_code=400, detail="Invalid model type.")

        # Log the prediction result
        logging.info("Labels predicted successfully.")

        # Return the preprocessed dataframe and predicted labels as a JSON response
        return JSONResponse(content=predictions)
    except json.decoder.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
