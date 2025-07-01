from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
from PACKAGE.c_split_dataset_prompt import split_sequential_data
# main.py
# This script serves as an executable entry point that utilizes the
# 'a_load_user_provided_data_prompt', 'b_preprocess_data_prompt',
# and 'c_split_dataset_prompt' modules.

# Third-party Libraries
import pandas as pd
import numpy as np # Required for handling numpy arrays from preprocessor and for concatenation


def main() -> None:
    """
    Main entry point for the integrated data loading, preprocessing, and splitting pipeline.
    This function first loads data, processes it using the 'DataPreprocessor' class
    (which includes an internal split), and then demonstrates the standalone splitting
    functionality from the 'c_split_dataset_prompt' module on the same data.
    """
    # Specify the data path (can update as needed)
    # This path is identical to the one in the original script to ensure
    # the same behavior.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\NVIDIA.csv'

    df = None  # Initialize df to None to ensure it's defined
    processed_data = None # Initialize processed_data to None

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
            # This step normalizes, creates sequences, and splits the data.
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

    # If preprocessing was successful, demonstrate the imported splitter function.
    if processed_data:
        print("\n--- Demonstrating Standalone Data Splitting from 'c_split_dataset_prompt' ---")
        try:
            # Reconstruct the full, unsplit dataset from the preprocessor's output
            # to provide as input to the standalone split function.
            X_full = np.concatenate([
                processed_data['X_train'],
                processed_data['X_val'],
                processed_data['X_test']
            ], axis=0)
            y_full = np.concatenate([
                processed_data['y_train'],
                processed_data['y_val'],
                processed_data['y_test']
            ], axis=0)
            print(f"Reconstructed full dataset for demonstration: X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")

            # Step 3: Use the imported function from 'c_split_dataset_prompt' to split the data.
            # This demonstrates its functionality on the same data processed earlier.
            split_ratios = (0.8, 0.1, 0.1)
            split_datasets_demo = split_sequential_data(X_full, y_full, split_ratios=split_ratios)

            print("\nData splitting using imported 'split_sequential_data' function successful.")
            print("Shapes of the new splits:")
            for key, value in split_datasets_demo.items():
                print(f"  {key}: {value.shape}")

        except ValueError as ve_split:
            print(f"Error during standalone splitting demonstration: {ve_split}")
        except Exception as e_split:
            print(f"An unexpected error occurred during standalone splitting demonstration: {e_split}")


# Ensures main() only runs when this file is executed directly.
if __name__ == "__main__":
    main()