# Revised Integrated Script
from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main
from PACKAGE.c_split_dataset_prompt import DataPreprocessor
from PACKAGE.d_build_lstm_and_transformer_models_prompt import build_lstm_model, build_transformer_model
import sys
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from io import StringIO

# Configure logging to match the style of the provided modules
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the integrated script.
    
    This function orchestrates the data loading, preprocessing, and model building pipeline by:
    1. Calling the data loading module to get the user-provided stock data.
    2. Passing the loaded data to the data preprocessing and splitting module.
    3. Reporting the results of the preprocessing and splitting steps.
    4. Building the LSTM and Transformer models using the preprocessed data.
    5. Acknowledging the unimplemented model training and evaluation steps.
    """
    logger.info("Starting the full data processing and model building pipeline.")

    # --- Step 1: Load Data ---
    # The load_data_main function from a_load_user_provided_data_prompt is expected
    # to handle user interaction, file loading, and return a pandas DataFrame.
    # This module already performs the necessary validation for ['Date', 'Open', 'High', 'Low', 'Close'].
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

    # --- (Validation Step Removed) ---
    # The previous call to `validate_data_format` from the 'd' module has been removed.
    # That validation incorrectly required a 'Volume' column, contradicting the specification
    # that only OHLC data should be used. The validation in `a_load_user_provided_data_prompt`
    # is sufficient and correctly aligned with the requirements.

    # --- Step 2: Preprocess and Split Data ---
    # This section integrates the functionality from c_split_dataset_prompt.py.
    # It uses the loaded DataFrame to create normalized sequences and split them.
    processed_data = None
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

        # --- Step 3: Display Preprocessing Results ---
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

    # --- Step 4: Build LSTM and Transformer Models ---
    # This step uses the preprocessed data to construct the model architectures.
    # Per the feedback, the model-building functions are correctly sourced from
    # d_build_lstm_and_transformer_models_prompt.py.
    try:
        logger.info("Starting model building phase.")
        
        # Determine input shape from the preprocessed training data.
        # Shape is (samples, sequence_length, features), model input is (sequence_length, features).
        if 'X_train' not in processed_data or processed_data['X_train'].ndim != 3:
            logger.error("X_train not found or has incorrect dimensions in processed_data. Aborting model building.")
            return
            
        input_shape = (processed_data['X_train'].shape[1], processed_data['X_train'].shape[2])
        logger.info(f"Determined model input shape: {input_shape}")

        # Build LSTM Model
        lstm_model = build_lstm_model(input_shape=input_shape)

        # Build Transformer Model
        transformer_model = build_transformer_model(input_shape=input_shape)

        # Capture and display model summaries
        print("\n--- Model Architectures ---")
        
        # Capture LSTM summary
        string_buffer_lstm = StringIO()
        lstm_model.summary(print_fn=lambda x: string_buffer_lstm.write(x + '\n'))
        print(string_buffer_lstm.getvalue())

        # Capture Transformer summary
        string_buffer_transformer = StringIO()
        transformer_model.summary(print_fn=lambda x: string_buffer_transformer.write(x + '\n'))
        print(string_buffer_transformer.getvalue())
        
        print("---------------------------\n")
        logger.info("LSTM and Transformer models built successfully.")

    except Exception as e:
        logger.error(f"An error occurred during model building: {e}")
        return

    # --- Step 5: Train and Evaluate Models (Placeholder) ---
    # The models have been built, but the provided modules do not contain
    # the implementation for training or evaluation logic.
    # Adhering to the constraint of not adding unimplemented features,
    # these steps are currently skipped.
    logger.warning("Model training, evaluation, and visualization steps are skipped as they are not implemented in the provided source modules.")
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
        print("and that 'PACKAGE' contains the necessary modules like '__init__.py' and all required prompts.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution block: {e}")
        sys.exit(1)