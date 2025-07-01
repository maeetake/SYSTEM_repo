# Revised Integrated Script
from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main
from PACKAGE.c_split_dataset_prompt import DataPreprocessor
from PACKAGE.f_evaluate_model_performance_prompt import build_lstm_model, build_transformer_model, train_model
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
    
    This function orchestrates the data loading, preprocessing, model building,
    and training pipeline by:
    1. Calling the data loading module to get the user-provided stock data.
    2. Passing the loaded data to the data preprocessing and splitting module.
    3. Building the LSTM and Transformer model architectures.
    4. Training both models on the preprocessed data.
    5. Acknowledging the unimplemented model evaluation and visualization steps.
    """
    logger.info("Starting the data processing and model building pipeline.")

    # --- Step 1: Load and Validate Data ---
    # The load_data_main function from a_load_user_provided_data_prompt is expected
    # to handle user interaction, file loading, and return a pandas DataFrame.
    # It also contains the necessary initial validation for OHLC and Date columns.
    try:
        logger.info("Attempting to load data using the user-provided data loader.")
        df = load_data_main()

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Data loading failed or returned an empty DataFrame. Aborting.")
            return

        logger.info("Data loaded and initially validated successfully.")

    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        return

    # --- Step 2: Preprocess and Split Data ---
    # This section integrates the functionality from c_split_dataset_prompt.py.
    # It uses the loaded DataFrame to create normalized sequences and split them.
    # The redundant and incorrect validation step for a 'Volume' column has been removed
    # as per the evaluation feedback. The validation in `load_data_main` is sufficient.
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

        # --- Display Preprocessing Results ---
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

    # --- Step 3: Build Deep Learning Models ---
    # This step integrates the model-building functions from f_evaluate_model_performance_prompt.py
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

    # --- Step 4: Train Models ---
    # This step integrates the training functionality from f_evaluate_model_performance_prompt.py.
    try:
        logger.info("Preparing to train models.")
        
        training_params = {
            'epochs': 50,
            'batch_size': 32,
            'patience': 10  # For EarlyStopping
        }
        
        # Train LSTM Model
        logger.info("Compiling LSTM model.")
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        
        trained_lstm_model, lstm_history = train_model(
            model=lstm_model,
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            X_val=processed_data['X_val'],
            y_val=processed_data['y_val'],
            training_params=training_params
        )
        logger.info("LSTM model training complete.")
        print(f"LSTM Training History Keys: {lstm_history.keys()}")

        # Train Transformer Model
        logger.info("Compiling Transformer model.")
        transformer_model.compile(optimizer='adam', loss='mean_squared_error')

        trained_transformer_model, transformer_history = train_model(
            model=transformer_model,
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            X_val=processed_data['X_val'],
            y_val=processed_data['y_val'],
            training_params=training_params
        )
        logger.info("Transformer model training complete.")
        print(f"Transformer Training History Keys: {transformer_history.keys()}")

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        return

    # --- Step 5: Evaluate and Visualize Models (Placeholder) ---
    # The models have been successfully trained using the `train_model` function.
    # However, the provided scripts do not include implementation for evaluating
    # performance (RMSE, MAE) on the test set or visualizing the results.
    #
    # Adhering to the constraint of not adding unimplemented features,
    # these subsequent steps are currently placeholders.
    
    logger.warning("Model evaluation and visualization steps are skipped as they are not implemented in the provided source modules.")
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