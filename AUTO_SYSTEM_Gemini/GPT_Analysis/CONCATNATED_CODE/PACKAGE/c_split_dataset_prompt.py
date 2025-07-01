# split_dataset.py
# Dependencies: numpy >=1.20.0, pandas >=1.0.0
# Purpose: Split sequential (chronological) time series data for training/validation/testing without data leakage

import os
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd

# Configure basic logging for error/debug information
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from CSV or JSON file.

    Args:
        data_path (str): Path to the data file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If file at specified path is missing.
        ValueError: If file cannot be loaded or is in an unsupported format.
    """
    if not os.path.exists(data_path):
        logging.error(f"File not found at {data_path}")
        raise FileNotFoundError(f"Error: File not found at {data_path}")
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            logging.error("Unsupported file format. Please provide a CSV or JSON file.")
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")

        if df is None or df.empty:
            logging.error(f"Loaded data is empty from {data_path}")
            raise ValueError("Loaded data is empty.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise ValueError(f"Error loading data: {e}")

def split_sequential_data(
    X: np.ndarray,
    y: np.ndarray,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Dict[str, np.ndarray]:
    """
    Splits time series data chronologically into training, validation, and test sets.

    Args:
        X (np.ndarray): The input feature sequences. Shape: (n_samples, seq_len, n_features)
        y (np.ndarray): The target values. Shape: (n_samples,)
        split_ratios (Tuple[float, float, float]): Train/Val/Test split fractions (must sum to 1.0)

    Returns:
        Dict[str, np.ndarray]: Dict with keys 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'

    Raises:
        ValueError: If X and y have mismatched lengths or cannot split due to too small data.
    """
    # Shape checks
    if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
        logging.error("X and y must be numpy arrays.")
        raise ValueError("X and y must be numpy arrays.")
    if X.shape[0] != y.shape[0]:
        logging.error(f"Mismatch: X samples: {X.shape[0]}, y samples: {y.shape[0]}")
        raise ValueError("Input arrays X and y must have the same number of samples.")
    if len(split_ratios) != 3:
        logging.error("split_ratios must be a tuple with three elements (train, val, test).")
        raise ValueError("split_ratios must have exactly 3 elements: (train, val, test)")

    train_ratio, val_ratio, test_ratio = split_ratios
    # Ensure split ratios sum to 1.0 (within numerical tolerance)
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        logging.error(f"Split ratios: {split_ratios} do not sum to 1.0")
        raise ValueError("Split ratios must sum to 1.0.")

    total_samples = X.shape[0]
    train_split_idx = int(total_samples * train_ratio)
    val_split_idx = int(total_samples * (train_ratio + val_ratio))

    # Error if any split is empty or sizes do not allow at least 1 for train/val/test
    if (
        train_split_idx == 0 or
        val_split_idx - train_split_idx == 0 or
        total_samples - val_split_idx == 0 or
        val_split_idx > total_samples
    ):
        msg = (f"Insufficient data for splitting. "
               f"Total samples: {total_samples}. "
               f"Required (approx): train >=1, val >=1, test >=1 with ratios {split_ratios}.")
        logging.error(msg)
        raise ValueError(msg)

    # Split data chronologically (no shuffle)
    X_train = X[:train_split_idx].astype(np.float32)
    y_train = y[:train_split_idx].astype(np.float32)
    X_val = X[train_split_idx:val_split_idx].astype(np.float32)
    y_val = y[train_split_idx:val_split_idx].astype(np.float32)
    X_test = X[val_split_idx:].astype(np.float32)
    y_test = y[val_split_idx:].astype(np.float32)

    logging.info(
        f"Split: train {X_train.shape[0]}, val {X_val.shape[0]}, test {X_test.shape[0]}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

def create_mock_sequential_data(num_samples: int = 100, seq_len: int = 60, num_features: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates mock data of the right shape for fallback/demo/testing purposes.
    """
    np.random.seed(42)
    X_mock = np.random.rand(num_samples, seq_len, num_features).astype(np.float32)
    y_mock = np.random.rand(num_samples).astype(np.float32)
    return X_mock, y_mock

def extract_X_y_from_dataframe(df: pd.DataFrame, seq_len: int = 60, feature_prefixes=("Open", "High", "Low", "Close")) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a DataFrame, extracts X, y arrays suitable for training.
    Assumes columns have name patterns such as "Open_t-59", ..., "Close_t-0"
    and a column ("target" or similar) for the predicted value.
    """
    # Default extraction based on those columns
    feature_cols = []
    for t in range(seq_len-1, -1, -1):
        for f in feature_prefixes:
            # e.g., Open_t-59, High_t-59, ..., Close_t-0
            col_name = f"{f}_t-{t}"
            if col_name in df.columns:
                feature_cols.append(col_name)
            else:
                # Comment: Warn if columns missing, this is only a fallback
                logging.warning(f"Column {col_name} missing in DataFrame, using zeros as placeholder.")
                feature_cols.append(None)

    # Reconstruct X: shape (n_samples, seq_len, n_features)
    n_samples = df.shape[0]
    n_features = len(feature_prefixes)
    X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)

    for i, f in enumerate(feature_prefixes):
        for t in range(seq_len):
            col_idx = i * seq_len + t
            col_name = f"{f}_t-{seq_len-1-t}"
            if col_name in df.columns:
                X[:, t, i] = df[col_name].values.astype(np.float32)
            else:
                # If missing, it remains as zeros (already zeroed)
                pass

    # Target: assume "target" column (as in sample data head)
    if "target" in df.columns:
        y = df["target"].values.astype(np.float32)
    else:
        # Fallback: use last close as the target
        last_close_col = f"Close_t-0"
        if last_close_col in df.columns:
            y = df[last_close_col].values.astype(np.float32)
            logging.warning("Target column not found; using Close_t-0 as target.")
        else:
            y = np.zeros(n_samples, dtype=np.float32)
            logging.warning("No target or Close_t-0 column found; using zeros as target.")
    return X, y

def main():
    """
    Entry point for the module. Loads data, extracts X/y, and splits them chronologically
    into train, validation, and test sets. Provides error handling and mock data fallback.
    """
    # Default data path as per instruction
    data_path = r"C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated\preprocessed_data_for_training.csv"
    # seq_len may be inferred from data, but here 60 is standard as per spec/example
    seq_len = 60
    num_features = 4  # 'Open', 'High', 'Low', 'Close'
    split_ratios = (0.8, 0.1, 0.1)
    print(f"Attempting to load data from: {data_path}")

    try:
        df = load_data(data_path)
        print("Data loaded successfully. Data head:")
        print(df.head())

        X, y = extract_X_y_from_dataframe(df, seq_len=seq_len)
        if X.shape[0] < 3:
            raise ValueError(f"Dataset too small: only {X.shape[0]} samples available after extraction.")
        print(f"Extracted X shape: {X.shape}, y shape: {y.shape}")

        datasets = split_sequential_data(X, y, split_ratios=split_ratios)
        print("Split successful. Dataset keys and shapes:")
        for k, v in datasets.items():
            print(f"{k}: {v.shape}, dtype: {v.dtype}")

    except FileNotFoundError as fnf:
        print(f"\nData file not found.\n{fnf}")
        print("Using mock data for demonstration/testing...")
        X_mock, y_mock = create_mock_sequential_data(num_samples=100, seq_len=seq_len, num_features=num_features)
        datasets = split_sequential_data(X_mock, y_mock, split_ratios=split_ratios)
        print("Split on mock data. Dataset keys and shapes:")
        for k, v in datasets.items():
            print(f"{k}: {v.shape}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()