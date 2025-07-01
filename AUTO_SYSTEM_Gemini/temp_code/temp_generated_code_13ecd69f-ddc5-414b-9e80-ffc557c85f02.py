# -*- coding: utf-8 -*-
"""
Module: split_dataset
Role: Prepare the dataset for model training and evaluation by creating
      chronologically distinct subsets (training, validation, test). This
      simulates a real-world forecasting scenario, preventing data leakage
      and ensuring an unbiased assessment of the model's performance.

This module is designed based on the following model overview:
{
    "model_role": "An AI model to predict the next-day closing price of NVIDIA stock using historical price data.",
    "instructions": [
        "Load the user-provided CSV file containing daily OHLC data.",
        "Preprocess the data: handle missing values, normalize features, and create sequential data for time series forecasting.",
        "Split the dataset chronologically into training (80%), validation (10%), and test (10%) sets.",
        "Build two separate deep learning models: one using LSTM and another using a Transformer architecture.",
        "Train both models on the training set, using the validation set to monitor performance and prevent overfitting.",
        "Evaluate the trained models on the test set using RMSE and MAE as performance metrics.",
        "Visualize the prediction results by overlaying predicted values on actual values in a time series graph.",
        "Generate a plot showing the training and validation loss function transition over epochs for each model."
    ],
    ...
}
"""

# Requires: numpy >= 1.20.0
# Requires: pandas (for the demonstration in the main block)
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Configure logging to display informational messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

def split_sequential_data(X: np.ndarray, y: np.ndarray, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Dict[str, np.ndarray]:
    """
    Splits a given sequential dataset (features X and targets y) into training,
    validation, and test sets based on specified chronological proportions.

    This function adheres to the 'split_dataset_1' task description by taking
    sequential data and partitioning it chronologically.

    Args:
        X (np.ndarray): A 3D array of input sequences.
                        Shape: (num_samples, sequence_length, num_features).
        y (np.ndarray): A 1D array of corresponding target values.
                        Shape: (num_samples,).
        split_ratios (Tuple[float, float, float]): A tuple containing the
                        proportions for training, validation, and test sets.
                        Defaults to (0.8, 0.1, 0.1).

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the six split arrays:
                               'X_train', 'y_train', 'X_val', 'y_val',
                               'X_test', 'y_test'.

    Raises:
        ValueError: If X and y have mismatched lengths, if split ratios do
                    not sum to 1.0, or if the dataset is too small to
                    create non-empty splits.
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        msg = "Inputs X and y must be numpy.ndarray types."
        logging.error(msg)
        raise TypeError(msg)

    if X.shape[0] != y.shape[0]:
        msg = f"Input arrays X and y must have the same number of samples. Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}."
        logging.error(msg)
        raise ValueError(msg)

    total_samples = X.shape[0]
    train_ratio, val_ratio, test_ratio = split_ratios

    if not np.isclose(sum(split_ratios), 1.0):
        msg = f"Split ratios must sum to 1.0. Got {split_ratios} which sums to {sum(split_ratios)}."
        logging.error(msg)
        raise ValueError(msg)

    # Calculate split indices based on chronological order
    train_split_idx = int(total_samples * train_ratio)
    val_split_idx = int(total_samples * (train_ratio + val_ratio))

    # Error handling for insufficient data as per specifications
    if train_split_idx == 0 or val_split_idx == train_split_idx or val_split_idx >= total_samples:
        msg = (f"Insufficient data for splitting. Total samples available: {total_samples}, "
               f"but this is not enough to create three non-empty train/val/test sets "
               f"with ratios {split_ratios}. Required at least 3 samples for a minimal split.")
        logging.error(msg)
        raise ValueError(msg)

    logging.info(f"Splitting {total_samples} samples into: "
                 f"Train ({train_split_idx}), "
                 f"Validation ({val_split_idx - train_split_idx}), "
                 f"Test ({total_samples - val_split_idx}).")

    # Perform the chronological split
    X_train, y_train = X[:train_split_idx], y[:train_split_idx]
    X_val, y_val = X[train_split_idx:val_split_idx], y[train_split_idx:val_split_idx]
    X_test, y_test = X[val_split_idx:], y[val_split_idx:]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a specified file path.
    Supports CSV and JSON file formats.

    Args:
        data_path (str): The absolute or relative path to the data file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file format is unsupported or if there is an
                    error during file parsing.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: File not found at the specified path: {data_path}")

    try:
        if data_path.lower().endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.lower().endswith('.json'):
            return pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format for path: {data_path}. Please provide a CSV or JSON file.")
    except Exception as e:
        raise ValueError(f"An error occurred while loading data from {data_path}: {e}")


def main():
    """
    Main function to serve as the program's entry point and demonstrate module usage.
    It executes without requiring any runtime user input.
    """
    logging.info("--- Starting Demonstration of split_dataset Module ---")

    # --- Part 1: Demonstrate the core function `split_sequential_data` ---
    # The primary purpose of this module is to split pre-processed sequential
    # data (X, y). We create mock data here to demonstrate this functionality.
    logging.info("\n[Demonstration 1: Core Functionality with Mock Data]")
    try:
        # Create mock sequential data as if it came from a preprocessing step
        num_samples = 2000
        sequence_length = 60
        num_features = 4  # e.g., Open, High, Low, Close
        X_mock = np.random.rand(num_samples, sequence_length, num_features).astype('float32')
        y_mock = np.random.rand(num_samples).astype('float32')
        logging.info(f"Created mock data: X shape={X_mock.shape}, y shape={y_mock.shape}")

        # Call the splitting function
        split_datasets = split_sequential_data(X_mock, y_mock, split_ratios=(0.8, 0.1, 0.1))

        # Print the shapes of the output to verify the split
        logging.info("Data split successfully. Shapes of the resulting arrays:")
        for name, data in split_datasets.items():
            logging.info(f"  - {name}: {data.shape}")

    except (ValueError, TypeError) as e:
        logging.error(f"An error occurred during the data splitting demonstration: {e}")

    # --- Part 2: Demonstrate error handling for insufficient data ---
    logging.info("\n[Demonstration 2: Error Handling for Insufficient Data]")
    try:
        # Create a dataset that is too small to be split
        X_small = np.zeros((10, 5, 4))
        y_small = np.zeros(10)
        logging.info(f"Attempting to split a small dataset with {X_small.shape[0]} samples...")
        split_sequential_data(X_small, y_small, (0.8, 0.1, 0.1))
    except ValueError as e:
        logging.warning(f"Successfully caught expected error: {e}")


    # --- Part 3: Demonstrate data loading as per Implementation Guidelines ---
    # This part shows how the module could interact with an upstream data file.
    logging.info("\n[Demonstration 3: Data Loading Helper Function]")
    # Use the provided data path
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\generated\expected_input.csv'

    try:
        df = load_data(data_path)
        logging.info(f"Data loaded successfully from {data_path}. Head of the data:")
        print(df.head().to_string())
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
        logging.info("As a fallback, using mock DataFrame.")
        mock_df = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'Open': [100, 102, 101],
            'High': [103, 104, 102],
            'Low': [99, 101, 100],
            'Close': [102, 103, 101]
        })
        print("Using mock DataFrame:")
        print(mock_df.head().to_string())
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")

    logging.info("\n--- Demonstration Finished ---")


if __name__ == "__main__":
    main()