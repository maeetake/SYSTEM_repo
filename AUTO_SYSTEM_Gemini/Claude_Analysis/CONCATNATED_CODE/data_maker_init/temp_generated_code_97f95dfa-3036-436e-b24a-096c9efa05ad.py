from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main
from PACKAGE.c_split_dataset_prompt import DataPreprocessor
import sys
import os
import logging
import numpy as np
import pandas as pd

# Configure logging to match the style of the provided modules
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ADDED CODE START ---
def _save_input_for_next_module(processed_data: dict):
    """
    Saves the reconstructed, pre-split dataset to a CSV file.

    This function takes the split data, reconstructs the full sequenced dataset,
    formats it into a pandas DataFrame based on the specification, and saves it 
    to a predefined location. This saved file represents the expected input for 
    the subsequent training module.

    Args:
        processed_data (dict): A dictionary containing the split data arrays
                               ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test').
    """
    # Hardcoded save directory and filename as per requirements.
    SAVE_DIR = r"C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Claude\UNITTEST_DATA\generated"
    FILE_NAME = "expected_input_for_training_module.csv"
    full_path = os.path.join(SAVE_DIR, FILE_NAME)

    try:
        logger.info(f"Attempting to save the input data for the next module to {full_path}")

        # Verify that the expected input data can be determined from the processed output.
        required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
        if not all(key in processed_data and isinstance(processed_data[key], np.ndarray) for key in required_keys):
            logger.warning("Could not determine the expected input data. 'processed_data' dictionary is missing required keys or contains invalid data. Skipping file save.")
            return

        # Reconstruct the full dataset by concatenating the chronological splits.
        X_full = np.concatenate(
            [processed_data['X_train'], processed_data['X_val'], processed_data['X_test']],
            axis=0
        )
        y_full = np.concatenate(
            [processed_data['y_train'], processed_data['y_val'], processed_data['y_test']],
            axis=0
        )

        # Determine dimensions. The specification implies 4 features (OHLC).
        num_samples, sequence_length, num_features = X_full.shape
        if num_features != 4:
            logger.warning(f"Expected 4 features (OHLC) based on spec, but found {num_features}. Cannot generate correct headers. Skipping file save.")
            return

        # Generate column headers as described in the specification's "Data Head (example)".
        feature_names = ['Open', 'High', 'Low', 'Close']
        headers = []
        for i in range(sequence_length - 1, -1, -1):
            for feature in feature_names:
                headers.append(f'{feature}_t-{i}')
        
        # Reshape the 3D X array into a 2D array for the DataFrame.
        X_reshaped = X_full.reshape(num_samples, -1)
        
        df_to_save = pd.DataFrame(X_reshaped, columns=headers)
        
        # Add the target column as specified: 'target_Close_t+1'.
        df_to_save['target_Close_t+1'] = y_full

        # Create the directory if it does not exist.
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # Save to CSV, overwriting any existing file. Do not include the index.
        df_to_save.to_csv(full_path, index=False)
        
        logger.info(f"Successfully saved expected input for the next module ({df_to_save.shape[0]} rows) to {full_path}")

    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"File system error occurred while trying to save to {full_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the file saving process: {e}")
# --- ADDED CODE END ---

def main():
    """
    Main entry point for the integrated script.
    
    This function orchestrates the data loading and preprocessing pipeline by:
    1. Calling the data loading module to get the user-provided stock data.
    2. Passing the loaded data to the data preprocessing and splitting module.
    3. Reporting the results of the preprocessing and splitting steps.
    """
    logger.info("Starting the data processing pipeline.")

    # --- Step 1: Load Data ---
    # The load_data_main function from a_load_user_provided_data_prompt is expected
    # to handle user interaction, file loading, and return a pandas DataFrame.
    # We will assume it returns None or raises an exception on failure.
    try:
        logger.info("Attempting to load data using the user-provided data loader.")
        # This function is assumed to be part of the PACKAGE and correctly loaded.
        # It handles the user prompt and returns a DataFrame.
        df = load_data_main()

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Data loading failed or returned an empty DataFrame. Aborting.")
            return

        logger.info("Data loaded successfully.")

    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        return

    # --- Step 2: Preprocess and Split Data ---
    # This section integrates the functionality from c_split_dataset_prompt.py.
    # It uses the loaded DataFrame to create normalized sequences and split them.
    try:
        logger.info("Initializing data preprocessor.")
        # Sequence length of 60 is based on the specifications.
        # The DataPreprocessor class from c_split_dataset_prompt handles all steps:
        # cleaning, normalizing, sequencing, and chronological splitting.
        preprocessor = DataPreprocessor(sequence_length=60)
        
        logger.info("Processing data (normalize, create sequences, and split)...")
        # The process method returns a dictionary with train, validation, and test sets.
        processed_data = preprocessor.process(df)
        logger.info("Data preprocessing and splitting completed successfully.")

        # --- ADDED CODE START ---
        # Save the processed data as CSV, formatted as the input for the next module.
        _save_input_for_next_module(processed_data)
        # --- ADDED CODE END ---

        # --- Step 3: Display Results ---
        # Print the shapes of the processed data arrays to verify the output.
        # This confirms that the train/val/test split was successful.
        print("\n--- Preprocessing and Splitting Results ---")
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                print(f"{key} shape: {value.shape}")
        # The 'scaler' object is also in the dictionary but will be skipped by the check.
        print("-----------------------------------------\n")

    except Exception as e:
        # The preprocessor's internal logging will have already logged the specifics.
        # This catch is for any unexpected errors during the instantiation or call.
        logger.error(f"An error occurred during data preprocessing: {e}")

if __name__ == "__main__":
    # This block ensures that the main function is called only when the script
    # is executed directly. It also handles potential import errors if the
    # package structure is not set up correctly.
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure that this script is run from a directory containing the 'PACKAGE' folder,", file=sys.stderr)
        print("and that 'PACKAGE' contains '__init__.py', 'a_load_user_provided_data_prompt.py', and 'c_split_dataset_prompt.py'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution block: {e}")
        sys.exit(1)