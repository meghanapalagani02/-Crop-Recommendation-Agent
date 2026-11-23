"""
Data loading and preprocessing module for Crop Recommendation Agent.
"""

import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Crop_recommendation.csv")
DATASET_SOURCE_URL = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset"

REQUIRED_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"


def ensure_dataset(data_path: str = DATA_PATH) -> str:
    """
    Ensures that the crop recommendation dataset exists locally.
    If missing, attempts to download it from the verified public repository.
    """
    if os.path.exists(data_path):
        return data_path

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    try:
        print(f"Dataset not found at {data_path}. Attempting download from {DATASET_SOURCE_URL}...")
        response = requests.get(DATASET_SOURCE_URL, timeout=20)
        response.raise_for_status()
        with open(data_path, "wb") as f:
            f.write(response.content)
        print(f"Dataset successfully downloaded ({os.path.getsize(data_path)} bytes).")
        return data_path
    except Exception as e:
        error_msg = (
            f"Failed to automatically download the dataset: {e}\n"
            f"Please download 'Crop_recommendation.csv' manually from Kaggle:\n"
            f"{KAGGLE_DATASET_URL}\n"
            f"and place it at: {data_path}"
        )
        raise FileNotFoundError(error_msg)


def load_crop_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads and validates the Crop Recommendation dataset.
    Handles null values, whitespace, and column integrity.
    """
    path = ensure_dataset(data_path)
    df = pd.read_csv(path)

    # Clean whitespace in column names
    df.columns = [c.strip() for c in df.columns]

    # Validate required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Clean text values in target
    if df[TARGET_COLUMN].dtype == "object":
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.lower()

    # Handle missing values if any
    if df.isnull().sum().sum() > 0:
        # Numeric columns impute with median
        for col in FEATURE_COLUMNS:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        if df[TARGET_COLUMN].isnull().sum() > 0:
            df = df.dropna(subset=[TARGET_COLUMN])

    return df


def get_feature_bounds(df: pd.DataFrame) -> dict:
    """
    Returns the min, max, mean, and median for all feature columns.
    Used for UI input widgets, validation, and boundary checking.
    """
    bounds = {}
    for col in FEATURE_COLUMNS:
        bounds[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std())
        }
    return bounds


def preprocess_data(df: pd.DataFrame):
    """
    Splits features X and target y, and encodes the categorical crop labels.
    Returns: X, y, encoder
    """
    X = df[FEATURE_COLUMNS].copy()
    y_raw = df[TARGET_COLUMN].copy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    return X, y, encoder


def get_train_test_split(X, y, test_size: float = 0.20, random_state: int = 42):
    """
    Splits dataset into stratified train and test sets using consistent random_state.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


if __name__ == "__main__":
    data = load_crop_data()
    print("Dataset loaded successfully!")
    print(f"Shape: {data.shape}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Number of crops: {data['label'].nunique()}")
    print("Crop list:", sorted(data["label"].unique()))
