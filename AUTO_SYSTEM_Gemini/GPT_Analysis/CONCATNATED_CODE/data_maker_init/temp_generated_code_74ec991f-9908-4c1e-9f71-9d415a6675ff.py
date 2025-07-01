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

# Standard Libraries
import os
from typing import Dict


# ADDED: Helper function to save processed data to a CSV file.
def _save_processed_data_as_csv(processed_data: Dict, file_path: str) -> None:
    """
    Transforms and saves the preprocessed data splits into a single CSV file.

    The input for the next module (training) consists of multiple NumPy arrays
    (X_train, y_train, etc.). To save this as a single, well-structured CSV,
    this function performs the following steps:
    1.  Flattens the 3D sequence data (X) into a 2D format.
    2.  Creates descriptive headers for the flattened features (e.g., 'Open_t-59', 'Close_t-0').
    3.  Combines the features (X) and targets (y) for each data split (train, val, test).
    4.  Adds a 'split' column to distinguish between the different datasets.
    5.  Concatenates all splits into a single pandas DataFrame.
    6.  Saves the final DataFrame to the specified CSV file path.

    Args:
        processed_data (Dict): The dictionary output from the preprocessor,
                               containing data splits like 'X_train', 'y_train', etc.
        file_path (str): The full path where the CSV file will be saved.
    """
    try:
        # Define feature names based on the expected OHLC input
        feature_names = ['Open', 'High', 'Low', 'Close']
        sequence_length = processed_data['X_train'].shape[1]

        # Generate descriptive column headers for the flattened sequence data
        # e.g., Open_t-59, High_t-59, ..., Close_t-0
        time_step_columns = [
            f'{feat}_t-{t}'
            for t in range(sequence_length - 1, -1, -1)
            for feat in feature_names
        ]

        all_dfs = []
        # Process each data split (train, val, test)
        for split_name in ['train', 'val', 'test']:
            X_data = processed_data[f'X_{split_name}']
            y_data = processed_data[f'y_{split_name}']

            # Reshape 3D (samples, timesteps, features) to 2D (samples, timesteps*features)
            num_samples = X_data.shape[0]
            num_features = X_data.shape[1] * X_data.shape[2]
            X_reshaped = X_data.reshape(num_samples, num_features)

            # Create a DataFrame for the current split
            df_split = pd.DataFrame(X_reshaped, columns=time_step_columns)
            df_split['target'] = y_data
            df_split['split'] = split_name
            all_dfs.append(df_split)

        # Concatenate all split DataFrames into one
        final_df = pd.concat(all_dfs, ignore_index=True)

        # Reorder columns to have 'split' and 'target' first for better readability
        final_df = final_df[
            ['split', 'target'] + time_step_columns
        ]

        # Save the final DataFrame to a CSV file, overwriting if it exists
        final_df.to_csv(file_path, index=False)

    except KeyError as ke:
        raise ValueError(f"The 'processed_data' dictionary is missing an expected key: {ke}")
    except Exception as e:
        # Catch any other unexpected errors during the process
        raise IOError(f"An error occurred while preparing or saving the CSV file: {e}")


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

            # ADDED: Save the processed data for consumption by the next module (e.g., training).
            # This block executes only if preprocessing is successful.
            try:
                # Define hardcoded output directory and file name as per requirements
                output_dir = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated'
                file_name = 'preprocessed_data_for_training.csv'
                full_path = os.path.join(output_dir, file_name)

                # Ensure the target directory exists
                os.makedirs(output_dir, exist_ok=True)
                print(f"\n--- Saving Processed Data ---")
                print(f"Target directory: {output_dir}")

                # Call the helper function to format and save the data
                _save_processed_data_as_csv(processed_data, full_path)
                print(f"Successfully saved the input data for the next module to: {full_path}")

            except (IOError, OSError, ValueError) as save_err:
                # Handle potential errors during file/directory creation or saving
                print(f"Error: Could not save the processed data as CSV. Reason: {save_err}")

        except ValueError as ve:
            # Handle errors from the preprocessing step (e.g., not enough data, missing columns)
            print(f"Preprocessing Error: {ve}")
        except Exception as e:
            # A general catch-all for any other unexpected errors during preprocessing.
            print(f"An unexpected error occurred during preprocessing: {e}")


# Ensures main() only runs when this file is executed directly.
if __name__ == "__main__":
    main()