from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
from PACKAGE.c_split_dataset_prompt import split_sequential_data
# main.py
# This script serves as an executable entry point that utilizes the
# 'a_load_user_provided_data_prompt', 'b_preprocess_data_prompt',
# and 'c_split_dataset_prompt' modules.

# Standard Libraries
import os

# Third-party Libraries
import pandas as pd
import numpy as np # Required for handling numpy arrays from preprocessor and for concatenation


# --- ADDED CODE START ---
def save_expected_input_for_training(
    processed_data: dict,
    output_dir: str,
    filename: str
) -> None:
    """
    Saves the split dataset (train, validation, test) into a single CSV file.

    This function reshapes the 3D feature arrays (X) into a 2D format,
    combines them with the target arrays (y) and a 'split' indicator,
    and saves the result as a CSV file. This format is expected by the
    subsequent training module.

    Args:
        processed_data (dict): A dictionary containing the split datasets:
                               'X_train', 'y_train', 'X_val', 'y_val',
                               'X_test', 'y_test'.
        output_dir (str): The directory where the CSV file will be saved.
        filename (str): The name of the CSV file.

    Returns:
        None
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define feature names and extract sequence length from the data
        # This assumes the standard OHLC feature order used in preprocessing.
        feature_names = ['Open', 'High', 'Low', 'Close']
        if not processed_data['X_train'].shape[0] > 0:
            print("Warning: Training data is empty. Skipping file save.")
            return
            
        sequence_length = processed_data['X_train'].shape[1]

        # Generate column headers based on sequence length and feature names
        # e.g., 'Open_t-59', 'High_t-59', ..., 'Close_t-0'
        columns = [
            f'{feature}_t-{sequence_length - 1 - i}'
            for i in range(sequence_length)
            for feature in feature_names
        ]

        # Process each data split (train, validation, test)
        all_dfs = []
        for split_name in ['train', 'val', 'test']:
            X = processed_data[f'X_{split_name}']
            y = processed_data[f'y_{split_name}']

            # Skip if the split is empty
            if X.shape[0] == 0:
                continue

            # Reshape 3D X array to 2D for CSV format and create DataFrame
            X_reshaped = X.reshape(X.shape[0], -1)
            df = pd.DataFrame(X_reshaped, columns=columns)

            # Add target and split columns
            df['target'] = y
            df['split'] = 'validation' if split_name == 'val' else split_name
            all_dfs.append(df)

        # Concatenate all splits into a single DataFrame
        if not all_dfs:
            print("Warning: No data to save. All splits are empty.")
            return
            
        final_df = pd.concat(all_dfs, ignore_index=True)

        # Reorder columns to match the specification: ['split', 'target', features...]
        final_df = final_df[['split', 'target'] + columns]

        # Construct the full file path and save to CSV
        file_path = os.path.join(output_dir, filename)
        final_df.to_csv(file_path, index=False)
        print(f"Successfully saved the expected input for the next module to: {file_path}")

    except (OSError, IOError) as e:
        print(f"Error: Could not create directory or write to file at '{output_dir}'. Reason: {e}")
    except KeyError as e:
        print(f"Error: The 'processed_data' dictionary is missing a required key: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the file saving process: {e}")
# --- ADDED CODE END ---


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

    # --- ADDED CODE START ---
    # Hardcoded directory and filename for saving the expected input for the next module
    SAVE_DIR = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated'
    SAVE_FILENAME = 'expected_input_for_training.csv'
    # --- ADDED CODE END ---

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
        # --- ADDED CODE START ---
        # Save the successfully processed data to CSV for the next module
        print("\n--- Saving Processed Data for Next Module ---")
        save_expected_input_for_training(processed_data, SAVE_DIR, SAVE_FILENAME)
        # --- ADDED CODE END ---

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