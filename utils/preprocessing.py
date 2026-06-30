"""
utils/preprocessing.py
Reusable preprocessing pipeline for training and inference.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

DROP_COLS = ['pkSeqID', 'saddr', 'daddr', 'subcategory']
LABEL_COLS = ['attack', 'category', 'category_encoded']


def load_and_concat(csv_paths: list) -> pd.DataFrame:
    """Load multiple Bot-IoT CSV files and concatenate."""
    return pd.concat(
        [pd.read_csv(p, low_memory=False) for p in csv_paths],
        ignore_index=True
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant columns, duplicates, inf values, and nulls."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_features(df: pd.DataFrame, fit: bool = True, encoder_path: str = None):
    """Label-encode categorical feature columns (not label columns)."""
    cat_cols = [
        c for c in df.select_dtypes(include='object').columns
        if c not in LABEL_COLS
    ]

    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        if encoder_path:
            joblib.dump(encoders, encoder_path)
        return df, encoders
    else:
        encoders = joblib.load(encoder_path)
        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        return df, encoders


def get_features(df: pd.DataFrame) -> list:
    """Return list of feature column names."""
    return [c for c in df.columns if c not in LABEL_COLS]


def scale(X_train, X_test=None, fit: bool = True, scaler_path: str = None):
    """Fit StandardScaler on train, transform both splits."""
    if fit:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        if scaler_path:
            joblib.dump(scaler, scaler_path)
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        return X_train_scaled, scaler
    else:
        scaler = joblib.load(scaler_path)
        return scaler.transform(X_train)
