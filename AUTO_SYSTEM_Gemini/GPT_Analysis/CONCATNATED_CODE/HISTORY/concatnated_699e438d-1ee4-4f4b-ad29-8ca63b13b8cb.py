from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
# main.py
# This script serves as an executable entry point that utilizes the
# 'a_load_user_provided_data_prompt' and 'b_preprocess_data_prompt' modules.

# Third-party Libraries
# pandas is required here because the main function's exception handling
# creates a mock DataFrame as a fallback.
import pandas as pd
import numpy as np # Required for handling numpy arrays from the preprocessor


def main() -> None:
    """
    Main entry point for the integrated data loading and preprocessing pipeline.
    This function first loads the data using 'load_data_from_csv' and then
    processes it using the 'DataPreprocessor' class.
    """
    # Specify the data path (can update as needed)
    # This path is identical to the one in the original script to ensure
    # the same behavior.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\NVIDIA.csv'

    df = None  # Initialize df to None to ensure it's defined

    try:
        # Step 1: Load the data using the imported function
        df = load_data_from_csv(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())

    except FileNotFoundError as fnf_err:
        # This exception is raised by the data loading module but handled here.
        print(f"File not found: {fnf_err}")
        # Optional: Provide mock OHLC data for testing/fallback.
        # This logic is preserved from the original 'concatnated_1.py' script.
        # Note: This mock data has too few rows for the default preprocessor (which needs 61),
        # so the preprocessor will correctly raise a ValueError, demonstrating
        # the integrated error handling.
        df = pd.DataFrame({
            "Date": ["2024/01/01", "2024/01/02", "2024/01/03"],
            "Open": [500.0, 505.0, 510.0],
            "High": [505.0, 510.0, 515.0],
            "Low":  [495.0, 500.0, 505.0],
            "Close": [503.0, 508.0, 512.0]
        })
        print("\nUsing mock data for demonstration:")
        print(df.head())

    except (ValueError, KeyError) as data_err:
        # These exceptions are raised by the data loading module but handled here.
        print(f"An error occurred while loading the data: {data_err}")
        # Exit if data loading fails, as preprocessing is not possible.
        return

    except Exception as e:
        # A general catch-all for any other unexpected errors during data loading.
        print(f"An unexpected error occurred during data loading: {e}")
        # Exit if data loading fails.
        return

    # If df was successfully loaded (either from file or mock data), proceed to preprocessing.
    if df is not None:
        print("\n--- Starting Data Preprocessing ---")
        # Instantiate the preprocessor with a sequence length of 60
        preprocessor = DataPreprocessor(sequence_length=60)
        try:
            # Step 2: Process the loaded dataframe using the imported preprocessor
            processed_data = preprocessor.process(df)
            print("Data preprocessing successful.")
            print(f"X_train shape: {processed_data['X_train'].shape}")
            print(f"y_train shape: {processed_data['y_train'].shape}")
            print(f"X_val shape: {processed_data['X_val'].shape}")
            print(f"y_val shape: {processed_data['y_val'].shape}")
            print(f"X_test shape: {processed_data['X_test'].shape}")
            print(f"y_test shape: {processed_data['y_test'].shape}")
            print(f"Scaler object: {processed_data['scaler']}")

        except ValueError as ve:
            # Handle errors from the preprocessing step (e.g., not enough data, missing columns)
            print(f"Preprocessing Error: {ve}")
        except Exception as e:
            # A general catch-all for any other unexpected errors during preprocessing.
            print(f"An unexpected error occurred during preprocessing: {e}")


# Ensures main() only runs when this file is executed directly.
if __name__ == "__main__":
    main()