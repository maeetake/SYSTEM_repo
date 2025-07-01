# load_user-provided_data.py

# Requires: Python >= 3.8, pandas >= 1.3.0
# Standard Libraries
import os
from typing import Optional

# Third-party Libraries
import pandas as pd

def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Loads historical stock data from a user-provided CSV file, validating its existence and required structure.
    
    Args:
        file_path (str): The absolute or relative path to the user-provided CSV file.
        
    Returns:
        pd.DataFrame: The raw, unmodified contents of the CSV as a pandas DataFrame.
        
    Raises:
        FileNotFoundError: If the file at 'file_path' does not exist.
        ValueError: If the CSV file is empty or can't be parsed.
        KeyError: If any required OHLC columns are missing.
    """
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']

    # Check file existence
    if not os.path.exists(file_path):
        err_msg = f"Error: The file was not found at {file_path}"
        print(err_msg)
        raise FileNotFoundError(err_msg)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        err_msg = f"Error reading CSV file at '{file_path}': {e}"
        print(err_msg)
        raise ValueError(err_msg)

    if df.empty:
        err_msg = f"Error: The CSV file at '{file_path}' is empty."
        print(err_msg)
        raise ValueError(err_msg)

    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        err_msg = f"Error: Missing column(s) in the CSV file at '{file_path}': {missing_cols}"
        print(err_msg)
        raise KeyError(err_msg)

    return df


def main() -> None:
    """
    Main entry point for module demonstration and validation.
    Attempts to load the data from the user-specified path. If the data cannot be loaded,
    prints an error and provides mock data as a fallback for illustration.
    """
    # Specify the data path (can update as needed)
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\NVIDIA.csv'

    try:
        df = load_data_from_csv(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())
    except FileNotFoundError as fnf_err:
        print(f"File not found: {fnf_err}")
        # Optional: Provide mock OHLC data for testing/fallback
        mock_data = pd.DataFrame({
            "Date": ["2024/01/01", "2024/01/02", "2024/01/03"],
            "Open": [500.0, 505.0, 510.0],
            "High": [505.0, 510.0, 515.0],
            "Low":  [495.0, 500.0, 505.0],
            "Close": [503.0, 508.0, 512.0]
        })
        print("Using mock data:")
        print(mock_data.head())
    except (ValueError, KeyError) as data_err:
        print(f"An error occurred while loading the data: {data_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Ensures main() only runs when this file is executed (not when imported)
if __name__ == "__main__":
    main()