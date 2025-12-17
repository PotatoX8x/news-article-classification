import os
import re
import shutil


def prepare_dir(path: str) -> None:
    """
    Creates a directory and ensures it is empty.

    Args:
        path (str): Path to the directory to be prepared.

    Returns:
        None
    """
    if os.path.exists(path):
        if len(os.listdir(path)) != 0:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def labels_to_vector(label_list: list, label_map: dict) -> list:
    """
    Convert list of predicted labels into one-hot encoding.

    Args:
        label_list (list): A list of string labels
        label_map dict: An ordered map of all labels

    Returns:
        list: One-hot encodded labels.
    """
    vector = [0] * len(label_map)  # Initialize zero vector
    for i, _ in enumerate(label_map):
        if label_map[str(i)] in label_list:
            vector[i] = 1
    return vector

def combine_labels_with_probabilities(predictions: list, label_map: dict) -> list:
    """
    Combines label with coresponding prediction score in a list of lists.

    Args:
        predictions (list): A list of probabilities.
        label_map (dict): An ordered map of all labels.

    Returns:
        list: Predicted label names and scores.
    """
    predicted_labels = [[label_map[str(i)], float(prob)] for i, prob in enumerate(predictions)]
    return predicted_labels

def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """
    Retrieves the latest checkpoint directory.

    Args:
        checkpoint_dir (str): Path to the directory containing checkpoint folders.

    Returns:
        str | None: The name of the latest checkpoint directory, or None if no checkpoints exist.
    """

    checkpoint_dirs = os.listdir(checkpoint_dir)
    if not checkpoint_dirs:
        return None

    # Extract step numbers and sort
    checkpoint_dirs.sort(key=lambda x: int(re.search(r"checkpoint-(\d+)", str(x)).group(1)), reverse=True)
    return str(checkpoint_dirs[0])