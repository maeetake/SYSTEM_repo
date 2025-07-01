from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
import sys
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict

# Configure logging to match the style of the provided modules
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ADDED: Function to save processed data ---
def save_sequenced_data_to_csv(processed_data: Dict[str, any], file_path: str):
    """
    Combines, flattens, and saves the sequenced data into a single CSV file.

    This function takes the output dictionary from the preprocessor, which is the
    expected input for the next module (training). It transforms the 3D sequence
    data (samples, timesteps, features) into a 2D format where each row
    represents one sample and saves it to a CSV file.

    Args:
        processed_data (dict): Dictionary from the preprocessor containing
                               'X_train', 'y_train', 'X_val', 'y_val',
                               'X_test', and 'y_test' numpy arrays.
        file_path (str): The full path where the CSV file will be saved.
    """
    logger.info(f"Attempting to save the input data for the next module to CSV.")

    try:
        # 1. Validate that the processed_data dictionary can be correctly determined
        required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
        if not isinstance(processed_data, dict) or not all(key in processed_data for key in required_keys):
            logger.warning("Could not determine the expected input data. The 'processed_data' dictionary is missing required keys. Skipping file save.")
            return

        # 2. Concatenate the data splits to form a complete dataset
        X_all = np.concatenate((processed_data['X_train'], processed_data['X_val'], processed_data['X_test']), axis=0)
        y_all = np.concatenate((processed_data['y_train'], processed_data['y_val'], processed_data['y_test']), axis=0)

        if X_all.shape[0] == 0:
            logger.warning("The processed data is empty. Skipping file save.")
            return

        # 3. Reshape the 3D X data into 2D for CSV compatibility
        n_samples, sequence_length, n_features = X_all.shape
        X_flat = X_all.reshape(n_samples, -1)

        # 4. Generate descriptive headers based on the specification
        # The spec indicates 4 features (OHLC) and a sequence length of 60
        feature_names = ['Open', 'High', 'Low', 'Close']
        if n_features != len(feature_names):
            logger.warning(f"Data has {n_features} features, but expected {len(feature_names)}. Headers may be incorrect. Skipping file save.")
            return

        headers = []
        # Create headers like 'Open_t-59', 'High_t-59', ..., 'Close_t-0'
        for i in range(sequence_length):
            time_step_label = f"t-{sequence_length - 1 - i}"
            for feature in feature_names:
                headers.append(f"{feature}_{time_step_label}")
        # Add the header for the target variable
        headers.append('target_Close_t+1')

        # 5. Combine the flattened features and the target variable
        # Ensure y_all is a column vector for horizontal stacking
        combined_data = np.hstack((X_flat, y_all.reshape(-1, 1)))

        # 6. Create a DataFrame and save it to a CSV file
        df_to_save = pd.DataFrame(combined_data, columns=headers)

        # 7. Ensure the save directory exists
        save_dir = os.path.dirname(file_path)
        os.makedirs(save_dir, exist_ok=True)

        # 8. Save the DataFrame, overwriting any existing file
        df_to_save.to_csv(file_path, index=False)
        logger.info(f"Successfully saved expected input data for the next module to: {file_path}")

    except (KeyError, ValueError) as e:
        logger.error(f"Error preparing data for saving: {e}. File not saved.")
    except (IOError, OSError) as e:
        logger.error(f"Error writing file to '{file_path}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the CSV saving process: {e}")
# --- END ADDED SECTION ---

def main():
    """
    Main entry point for the integrated script.
    
    This function orchestrates the data loading and preprocessing pipeline by:
    1. Calling the data loading module to get the user-provided stock data.
    2. Passing the loaded data to the data preprocessing module.
    3. Reporting the results of the preprocessing step.
    4. Saving the preprocessed data (input for the next module) to a CSV file.
    """
    logger.info("Starting the data processing pipeline.")

    # --- Step 1: Load Data ---
    # The load_data_main function from a_load_user_provided_data_prompt is expected
    # to handle user interaction, file loading, and return a pandas DataFrame.
    # We will assume it returns None or raises an exception on failure.
    try:
        logger.info("Attempting to load data using the user-provided data loader.")
        # Note: The original 'a_load_user_provided_data_prompt.py' is not provided,
        # but based on its name and the project specs, we assume its main function
        # loads and returns a DataFrame.
        # For this integration to be testable, we'll need to modify the original
        # main function to return the DataFrame instead of just running.
        # This is a common refactoring step when turning a script into a module.
        df = load_data_main()

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Data loading failed or returned an empty DataFrame. Aborting.")
            return

        logger.info("Data loaded successfully.")

    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        return

    # --- Step 2: Preprocess Data ---
    # This section integrates the functionality from b_preprocess_data_prompt.py.
    # It uses the loaded DataFrame to create sequences for the models.
    try:
        logger.info("Initializing data preprocessor.")
        # Sequence length of 60 is based on the specifications.
        preprocessor = DataPreprocessor(sequence_length=60)
        
        logger.info("Processing data...")
        processed_data = preprocessor.process(df)
        logger.info("Data preprocessing completed successfully.")

        # --- Step 3: Display Results ---
        # Print the shapes of the processed data arrays to verify the output.
        print("\n--- Preprocessing Results ---")
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                print(f"{key} shape: {value.shape}")
        print("---------------------------\n")

        # --- ADDED: Save the input data for the next module as a CSV file ---
        # This data corresponds to the 'Expected Output Format' of the 'preprocess_data' module.
        save_directory = r"C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Claude\UNITTEST_DATA\generated"
        output_filename = "expected_input_for_training_module.csv"
        full_save_path = os.path.join(save_directory, output_filename)
        
        # Call the dedicated function to handle the saving logic.
        save_sequenced_data_to_csv(processed_data, full_save_path)
        # --- END ADDED SECTION ---

    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")
        # The preprocessor's internal logging will have already logged the specifics.

if __name__ == "__main__":
    # This block ensures that the main function is called only when the script
    # is executed directly. It also handles potential import errors if the
    # package structure is not set up correctly.
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure that this script is run from a directory containing the 'PACKAGE' folder,", file=sys.stderr)
        print("and that 'PACKAGE' contains '__init__.py', 'a_load_user_provided_data_prompt.py', and 'b_preprocess_data_prompt.py'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution block: {e}")
        sys.exit(1)