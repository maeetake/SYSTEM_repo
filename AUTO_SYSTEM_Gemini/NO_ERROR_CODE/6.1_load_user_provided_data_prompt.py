# load_user-provided_data.py

# ==============================================================================
# Module: load_user-provided_data
#
# Role and Purpose:
# This module is the sole data ingestion point for the system. Its purpose is to
# abstract the file loading mechanism and provide a standardized raw data object
# (a pandas DataFrame) to downstream modules. It ensures a consistent starting
# point for all subsequent data preprocessing and modeling tasks by loading
# user-provided data from a CSV file and performing initial validation.
#
# Dependencies:
# - Python 3.8+
# - pandas >= 1.3.0
# ==============================================================================

import os
import pandas as pd
import logging

# Configure basic logging
# In case of an error, this will log details to the console.
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Loads historical stock data from a user-provided CSV file.

    This function reads a CSV file from the specified path, validates its
    existence and structure (presence of required columns), and returns the
    data as a pandas DataFrame.

    Args:
        file_path (str): The absolute or relative path to the user-provided CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the raw, unmodified data read
                      from the CSV file. The returned DataFrame is guaranteed
                      to contain the required columns.

    Raises:
        FileNotFoundError: If the file at 'file_path' does not exist.
        ValueError: If the CSV file is empty or cannot be parsed.
        KeyError: If one of the required columns ('Date', 'Open', 'High', 'Low', 'Close')
                  is missing in the CSV file.
    """
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']

    # 1. Validate that the file exists at the given 'file_path'.
    if not os.path.exists(file_path):
        error_msg = f"File not found at the specified path: {file_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    # 2. Use the pandas library to read the CSV file into a DataFrame.
    try:
        raw_dataframe = pd.read_csv(file_path)
    # Ambiguity Note: The specification mentions a generic exception for parsing.
    # We catch pd.errors.ParserError for specific CSV format issues and a generic
    # Exception for other unexpected file reading problems (e.g., permissions).
    # This provides more specific error handling while adhering to the spec.
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing CSV file at '{file_path}'. The file may be corrupted or malformed. Details: {e}"
        logging.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"An unexpected error occurred while reading the file '{file_path}'. Details: {e}"
        logging.error(error_msg)
        raise ValueError(error_msg) from e

    # Additional validation: Check if the loaded DataFrame is empty.
    if raw_dataframe.empty:
        error_msg = f"The CSV file at '{file_path}' is empty or contains only headers."
        logging.error(error_msg)
        raise ValueError(error_msg)

    # 3. Check if the loaded DataFrame contains the required columns.
    missing_columns = [col for col in required_columns if col not in raw_dataframe.columns]
    if missing_columns:
        # Joining for a comprehensive error message if multiple columns are missing.
        error_msg = f"Missing required column(s) in the CSV file '{file_path}': {', '.join(missing_columns)}"
        logging.error(error_msg)
        # Raise KeyError as specified for missing columns.
        raise KeyError(error_msg)

    # 4. Return the DataFrame without any modifications.
    print(f"Successfully loaded data from: {file_path}")
    return raw_dataframe


def main():
    """
    Main function to demonstrate the usage of the load_data_from_csv module.

    This entry point automatically loads data from a predefined path,
    simulating the start of a data processing pipeline. It includes error
    handling and a fallback to mock data for demonstration purposes, ensuring
    the script can run even if the primary data source is unavailable.
    """
    # Per "Implementation Guidelines", use a predefined data path.
    # Using a raw string (r'...') to handle backslashes in Windows paths correctly.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\NVIDIA.csv'

    print("--- Data Loading Demonstration ---")
    try:
        # Attempt to load the primary dataset
        df = load_data_from_csv(data_path)
        print("\nData loaded successfully. Raw DataFrame head:")
        print(df.head())

    except (FileNotFoundError, ValueError, KeyError) as e:
        # Per "main Function Instructions", handle errors with informative messages
        # and provide mock data as a fallback.
        print(f"\nAn error occurred while loading the primary data: {e}")
        print("---")
        print("Proceeding with mock data for demonstration purposes.")

        # Create a mock DataFrame that mimics the expected structure
        mock_data = {
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'Open': [150.0, 152.5, 151.0, 155.0, 154.5],
            'High': [153.0, 153.5, 155.5, 156.0, 157.0],
            'Low': [149.5, 150.5, 150.0, 153.0, 154.0],
            'Close': [152.0, 151.5, 155.0, 154.0, 156.5],
            'Volume': [1000000, 1200000, 1100000, 1300000, 1250000] # Extra column to show it's preserved
        }
        mock_df = pd.DataFrame(mock_data)

        print("\nMock data generated. DataFrame head:")
        print(mock_df.head())
        # In a real pipeline, you would return or use this mock_df
        # for subsequent processing steps.

    except Exception as e:
        print(f"\nAn unexpected critical error occurred: {e}")
        print("Execution halted.")


if __name__ == '__main__':
    # This block allows the script to be run directly to test the module's functionality.
    main()