# Requires: numpy >= 1.20.0

import logging
from typing import Dict, Tuple

import numpy as np

# Configure logging to display INFO level messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_sequential_data(X: np.ndarray, y: np.ndarray,
                          split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Dict[str, np.ndarray]:
    """
    Splits a given sequential dataset (features X and targets y) into training,
    validation, and test sets based on specified chronological proportions.

    This function is critical for time series forecasting to prevent data leakage by
    ensuring that the model is trained on past data and evaluated on future,
    unseen data.

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
        ValueError: If X and y have mismatched lengths, if split ratios do not
                    sum to 1.0, or if the dataset is too small to be split
                    into non-empty sets.
    """
    if X.shape[0] != y.shape[0]:
        error_msg = f"Input arrays X and y must have the same number of samples. Got X: {X.shape[0]} and y: {y.shape[0]}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not np.isclose(sum(split_ratios), 1.0):
        error_msg = f"Split ratios must sum to 1.0, but got {sum(split_ratios)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    total_samples = X.shape[0]
    train_ratio, val_ratio, _ = split_ratios

    train_split_idx = int(total_samples * train_ratio)
    val_split_idx = int(total_samples * (train_ratio + val_ratio))

    # Check if any of the splits would be empty
    if train_split_idx == 0 or val_split_idx == train_split_idx or val_split_idx >= total_samples:
        error_msg = (
            f"Insufficient data for splitting. Total samples: {total_samples}. "
            f"Calculated split indices (train_end={train_split_idx}, val_end={val_split_idx}) "
            f"result in one or more empty sets with ratios {split_ratios}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    X_train, y_train = X[:train_split_idx], y[:train_split_idx]
    X_val, y_val = X[train_split_idx:val_split_idx], y[train_split_idx:val_split_idx]
    X_test, y_test = X[val_split_idx:], y[val_split_idx:]

    logger.info("Data split completed successfully.")
    logger.info(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def main():
    """
    Main function to demonstrate and test the split_sequential_data function.
    It creates mock data and attempts to split it, logging the results.
    """
    logger.info("--- Running demonstration for split_dataset module ---")

    # --- Test Case 1: Successful Split ---
    logger.info("\n[Test Case 1: Successful Split]")
    # Create mock sequential data: (num_samples, sequence_length, num_features)
    mock_X = np.random.rand(1000, 60, 4).astype(np.float32)
    mock_y = np.random.rand(1000).astype(np.float32)
    logger.info(f"Created mock data with X shape: {mock_X.shape} and y shape: {mock_y.shape}")

    try:
        datasets = split_sequential_data(mock_X, mock_y, split_ratios=(0.8, 0.1, 0.1))
        print("\n--- Split Dataset Shapes ---")
        for name, data in datasets.items():
            print(f"{name}: {data.shape}")
        print("--------------------------\n")
    except ValueError as e:
        logger.error(f"An unexpected error occurred during a valid split: {e}")

    # --- Test Case 2: Insufficient Data Error ---
    logger.info("\n[Test Case 2: Insufficient Data Error]")
    # Create mock data that is too small to split
    small_X = np.random.rand(9, 60, 4).astype(np.float32)
    small_y = np.random.rand(9).astype(np.float32)
    logger.info(f"Created small mock data with X shape: {small_X.shape} and y shape: {small_y.shape}")

    try:
        split_sequential_data(small_X, small_y, split_ratios=(0.8, 0.1, 0.1))
    except ValueError as e:
        logger.info(f"Successfully caught expected error for insufficient data: {e}")

    # --- Test Case 3: Mismatched Lengths Error ---
    logger.info("\n[Test Case 3: Mismatched Lengths Error]")
    mismatch_X = np.random.rand(100, 60, 4).astype(np.float32)
    mismatch_y = np.random.rand(99).astype(np.float32)
    logger.info(f"Created mismatched data with X shape: {mismatch_X.shape} and y shape: {mismatch_y.shape}")
    try:
        split_sequential_data(mismatch_X, mismatch_y)
    except ValueError as e:
        logger.info(f"Successfully caught expected error for mismatched lengths: {e}")

    logger.info("\n--- Demonstration complete ---")


if __name__ == "__main__":
    main()