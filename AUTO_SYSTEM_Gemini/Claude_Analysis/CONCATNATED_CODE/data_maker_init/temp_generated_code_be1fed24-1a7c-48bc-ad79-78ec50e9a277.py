from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main
from PACKAGE.c_split_dataset_prompt import DataPreprocessor
from PACKAGE.d_build_lstm_and_transformer_models_prompt import validate_data_format
import sys
import os
import logging
import numpy as np
import pandas as pd

# Configure logging to match the style of the provided modules
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Start of Added Code ---

def save_processed_data_as_csv(processed_data: dict, sequence_length: int):
    """
    Saves the processed and sequenced data into a single CSV file.

    This function reshapes the 3D sequence arrays (X_train, X_val, X_test) into a 2D format,
    concatenates all data splits, and saves it to a CSV file. The format matches the
    specification for the subsequent training module.

    Args:
        processed_data (dict): A dictionary containing the split data ('X_train', 'y_train', etc.).
        sequence_length (int): The length of the input sequences (e.g., 60 days).
    """
    # Define the hardcoded save directory and filename as per requirements.
    save_dir = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Claude\UNITTEST_DATA\generated'
    file_name = 'expected_input_for_training_module.csv'
    file_path = os.path.join(save_dir, file_name)

    try:
        # Create the directory if it does not exist.
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {save_dir}: {e}")
        # If directory creation fails, we cannot save the file.
        return

    # Check if the required data arrays exist in the dictionary.
    required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
    if not all(key in processed_data for key in required_keys):
        logger.warning("Could not determine the expected input data from 'processed_data' dictionary. Skipping CSV save.")
        return

    try:
        logger.info("Preparing data for CSV export.")
        # Consolidate train, validation, and test sets into single arrays.
        X_all = np.concatenate((
            processed_data['X_train'],
            processed_data['X_val'],
            processed_data['X_test']
        ), axis=0)

        y_all = np.concatenate((
            processed_data['y_train'],
            processed_data['y_val'],
            processed_data['y_test']
        ), axis=0)

        # Reshape X from 3D (samples, timesteps, features) to 2D (samples, timesteps * features)
        num_samples, seq_len, num_features = X_all.shape
        if seq_len != sequence_length:
            logger.error(f"Data sequence length mismatch. Expected {sequence_length}, but found {seq_len}. Aborting save.")
            return

        X_reshaped = X_all.reshape(num_samples, -1)

        # Generate column headers based on the specification document.
        # e.g., 'Open_t-59', 'High_t-59', ..., 'Close_t-0'
        features = ['Open', 'High', 'Low', 'Close']
        if num_features != len(features):
             logger.warning(f"Expected {len(features)} features, but data has {num_features}. Column names might not align perfectly with data.")

        x_columns = [f'{feat}_t-{i}' for i in range(sequence_length - 1, -1, -1) for feat in features]
        target_column = ['target_Close_t+1']

        # Create a pandas DataFrame from the consolidated data.
        df_x = pd.DataFrame(X_reshaped, columns=x_columns)
        df_y = pd.DataFrame(y_all, columns=target_column)
        final_df = pd.concat([df_x, df_y], axis=1)

        # Save the DataFrame to a CSV file, overwriting if it exists.
        logger.info(f"Saving the consolidated data to {file_path}")
        final_df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved expected input data for the next module to {file_path}")

    except Exception as e:
        logger.error(f"An error occurred while preparing or saving the CSV file: {e}")

# --- End of Added Code ---


def main():
    """
    Main entry point for the integrated script.
    
    This function orchestrates the data loading, validation, and preprocessing pipeline by:
    1. Calling the data loading module to get the user-provided stock data.
    2. Validating the format of the loaded DataFrame.
    3. Passing the loaded data to the data preprocessing and splitting module.
    4. Reporting the results of the preprocessing and splitting steps.
    5. Acknowledging the unimplemented model training and evaluation steps.
    """
    logger.info("Starting the data processing pipeline.")

    # --- Step 1: Load Data ---
    # The load_data_main function from a_load_user_provided_data_prompt is expected
    # to handle user interaction, file loading, and return a pandas DataFrame.
    try:
        logger.info("Attempting to load data using the user-provided data loader.")
        df = load_data_main()

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Data loading failed or returned an empty DataFrame. Aborting.")
            return

        logger.info("Data loaded successfully.")

    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        return

    # --- Step 2: Validate Data Format ---
    # This step uses the validation function from the model-building module to ensure
    # the DataFrame has the required columns before further processing.
    try:
        logger.info("Validating loaded data format.")
        # The `validate_data_format` function is from the model building module.
        # It ensures the necessary columns are present before proceeding.
        validate_data_format(df)
        logger.info("Data format validated successfully.")
    except ValueError as e:
        logger.error(f"Data validation failed: {e}. Aborting.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during data validation: {e}")
        return

    # --- Step 3: Preprocess and Split Data ---
    # This section integrates the functionality from c_split_dataset_prompt.py.
    # It uses the loaded DataFrame to create normalized sequences and split them.
    try:
        logger.info("Initializing data preprocessor.")
        # Sequence length of 60 is based on the specifications.
        # The DataPreprocessor class from c_split_dataset_prompt handles all steps:
        # cleaning, normalizing, sequencing, and chronological splitting.
        sequence_length = 60
        preprocessor = DataPreprocessor(sequence_length=sequence_length)
        
        logger.info("Processing data (normalize, create sequences, and split)...")
        # The process method returns a dictionary with train, validation, and test sets.
        processed_data = preprocessor.process(df)
        logger.info("Data preprocessing and splitting completed successfully.")

        # --- Start of Added Code ---
        # Save the processed data to a CSV file as specified.
        save_processed_data_as_csv(processed_data, sequence_length)
        # --- End of Added Code ---
        
        # --- Step 4: Display Results ---
        # Print the shapes of the processed data arrays to verify the output.
        # This confirms that the train/val/test split was successful.
        print("\n--- Preprocessing and Splitting Results ---")
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                print(f"{key} shape: {value.shape}")
        # The 'scaler' object is also in the dictionary but will be skipped by this check.
        print("-----------------------------------------\n")

    except Exception as e:
        # The preprocessor's internal logging will have already logged the specifics.
        # This catch is for any unexpected errors during the instantiation or call.
        logger.error(f"An error occurred during data preprocessing: {e}")
        return

    # --- Step 5: Build, Train, and Evaluate Models (Placeholder) ---
    # According to the specifications, this is where the LSTM and Transformer
    # models would be built, trained, and evaluated on the preprocessed data.
    # However, the provided 'd_build_lstm_and_transformer_models_prompt.py'
    # does not contain the implementation for these functions.
    # Adhering to the constraint of not adding unimplemented features,
    # this step is currently a placeholder.
    #
    # Example of how it would be called if model-building functions were implemented:
    #
    # from PACKAGE.d_build_lstm_and_transformer_models_prompt import run_model_pipeline
    #
    # results = run_model_pipeline(processed_data)
    # logger.info("Model training and evaluation finished.")
    #
    
    logger.warning("Model building, training, and evaluation steps are skipped as they are not implemented in the provided source modules.")
    logger.info("Pipeline finished.")

if __name__ == "__main__":
    # This block ensures that the main function is called only when the script
    # is executed directly. It also handles potential import errors if the
    # package structure is not set up correctly.
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure that this script is run from a directory containing the 'PACKAGE' folder,", file=sys.stderr)
        print("and that 'PACKAGE' contains the necessary modules like '__init__.py', 'a_load_user_provided_data_prompt.py', etc.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution block: {e}")
        sys.exit(1)