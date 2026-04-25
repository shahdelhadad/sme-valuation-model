import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config


def load_data(filepath=config.DATA_PATH):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    return pd.read_csv(filepath)


def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), config.CATEGORICAL_FEATURES),
            ('num', RobustScaler(), config.NUMERICAL_FEATURES),
        ],
        remainder='drop'
    )
    return preprocessor


def preprocess_data(df):
    for col in config.DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    if config.TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    X_train[config.NUMERICAL_FEATURES] = X_train[config.NUMERICAL_FEATURES].astype(np.float32)
    X_test[config.NUMERICAL_FEATURES] = X_test[config.NUMERICAL_FEATURES].astype(np.float32)

    X_train[config.CATEGORICAL_FEATURES] = X_train[config.CATEGORICAL_FEATURES].astype(object)
    X_test[config.CATEGORICAL_FEATURES] = X_test[config.CATEGORICAL_FEATURES].astype(object)

    preprocessor = build_preprocessor()

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    print("Preprocessing configuration validated successfully.")
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")
    print(f"Columns: {list(X_train.columns)}")
