from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main
from PACKAGE.c_split_dataset_prompt import DataPreprocessor
from PACKAGE.d_build_lstm_and_transformer_models_prompt import validate_data_format
from PACKAGE.e_train_models_prompt import build_lstm_model, build_transformer_model
import sys
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

# Configure logging to match the style of the provided modules
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the integrated script.
    
    This function orchestrates the data loading, validation, preprocessing, and model building pipeline by:
    1. Calling the data loading module to get the user-provided stock data.
    2. Validating the format of the loaded DataFrame.
    3. Passing the loaded data to the data preprocessing and splitting module.
    4. Building the LSTM and Transformer model architectures using the preprocessed data.
    5. Acknowledging the unimplemented model training and evaluation steps.
    """
    logger.info("Starting the data processing and model building pipeline.")

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
        preprocessor = DataPreprocessor(sequence_length=60)
        
        logger.info("Processing data (normalize, create sequences, and split)...")
        # The process method returns a dictionary with train, validation, and test sets.
        processed_data = preprocessor.process(df)
        logger.info("Data preprocessing and splitting completed successfully.")

        # --- Step 4: Display Preprocessing Results ---
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

    # --- Step 5: Build Deep Learning Models ---
    # This step integrates the model-building functions from e_train_models_prompt.py
    try:
        logger.info("Building LSTM and Transformer models.")
        
        # Determine input shape from the preprocessed data
        # Shape is (samples, sequence_length, features) -> Model input is (sequence_length, features)
        input_shape = (processed_data['X_train'].shape[1], processed_data['X_train'].shape[2])
        logger.info(f"Using input shape for models: {input_shape}")

        # Build LSTM model using the imported function
        lstm_model = build_lstm_model(input_shape)
        print("\n--- LSTM Model Summary ---")
        lstm_model.summary()
        print("--------------------------\n")

        # Build Transformer model using the imported function
        transformer_model = build_transformer_model(input_shape)
        print("\n--- Transformer Model Summary ---")
        transformer_model.summary()
        print("-------------------------------\n")

        logger.info("Successfully built both LSTM and Transformer models.")

    except Exception as e:
        logger.error(f"An error occurred during model building: {e}")
        return

    # --- Step 6: Train, Evaluate, and Visualize Models (Placeholder) ---
    # The models have been successfully built based on the provided e_train_models_prompt.py.
    # However, the provided scripts do not include implementation for training the models,
    # evaluating their performance, or visualizing the results.
    #
    # Adhering to the constraint of not adding unimplemented features,
    # these subsequent steps are currently placeholders.
    #
    # Example of how it would be called if model-training functions were implemented:
    #
    # from PACKAGE.f_evaluate_models_prompt import run_training_and_evaluation
    #
    # results = run_training_and_evaluation(lstm_model, transformer_model, processed_data)
    # logger.info("Model training and evaluation finished.")
    
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
        print("and that 'PACKAGE' contains the necessary modules like '__init__.py', 'a_load_user_provided_data_prompt.py', etc.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution block: {e}")
        sys.exit(1)