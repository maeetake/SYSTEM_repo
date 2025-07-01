# load_user-provided_data.py

# Requires: Python >= 3.8, pandas >= 1.3.0
# Dependencies: pandas, os, logging

import os
import pandas as pd
import logging

from typing import Optional

# Configure module-level logger
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load historical OHLC stock data from a user-supplied CSV file.

    This function validates file existence, checks for all required columns,
    and returns the raw DataFrame unmodified for downstream processing.

    Args:
        file_path (str): The absolute or relative path to the user-provided CSV file.

    Returns:
        pd.DataFrame: Raw data as read from the file, with no preprocessing.

    Raises:
        FileNotFoundError: If the file at 'file_path' does not exist.
        ValueError: If the file is empty or unreadable.
        KeyError: If a required column is missing from the file.
    """
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']

    if not os.path.exists(file_path):
        logger.error(f"File not found at path: {file_path}")
        raise FileNotFoundError(f"Error: File not found at {file_path}")

    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        logger.error(
            f"Failed to parse CSV file at {file_path}. Exception: {e}"
        )
        raise ValueError(f"Error parsing CSV file at {file_path}: {e}")

    if data.empty:
        logger.error(f"CSV file at {file_path} is empty.")
        raise ValueError(f"Error: The CSV file at {file_path} is empty.")

    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        logger.error(
            f"Missing required columns in file {file_path}: {missing_cols}"
        )
        # Raise for the first missing column as per typical KeyError semantics
        raise KeyError(f"Error: Required column(s) {missing_cols} not found in the CSV file.")

    logger.info(f"CSV file '{file_path}' successfully loaded with required columns.")
    return data


def main():
    """
    Entry point for loading NVIDIA OHLC data.

    Attempts to load CSV data from the specified path. If loading fails due to a missing file,
    uses mock data as a fallback for demonstration/testing purposes.
    """
    # NOTE: Use the user's specified data path for automated operation.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\NVIDIA.csv'

    try:
        df = load_data_from_csv(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())
    except FileNotFoundError as fnfe:
        logger.error(str(fnfe))
        print("Data file not found. Please check the data path or provide valid data.")
        # Mock data fallback (for testing/demo only: does NOT satisfy downstream contracts)
        mock_data = pd.DataFrame({
            "Date": pd.date_range("2021-01-01", periods=3).strftime('%Y-%m-%d'),
            "Open": [100.0, 101.5, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low":  [99.0, 100.5, 101.3],
            "Close": [101.0, 102.7, 103.6]
        })
        print("Using mock data:")
        print(mock_data.head())
    except KeyError as ke:
        logger.error(str(ke))
        print(f"Required columns are missing: {ke}")
    except ValueError as ve:
        logger.error(str(ve))
        print(f"CSV loading or format error: {ve}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # The main entry point should not require user input and should execute automatically as per specifications.
    main()