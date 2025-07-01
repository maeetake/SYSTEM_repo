from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
import sys
import os
import logging
import numpy as np
import pandas as pd

# Configure logging to match the style of the provided modules
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the integrated script.
    
    This function orchestrates the data loading and preprocessing pipeline by:
    1. Calling the data loading module to get the user-provided stock data.
    2. Passing the loaded data to the data preprocessing module.
    3. Reporting the results of the preprocessing step.
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