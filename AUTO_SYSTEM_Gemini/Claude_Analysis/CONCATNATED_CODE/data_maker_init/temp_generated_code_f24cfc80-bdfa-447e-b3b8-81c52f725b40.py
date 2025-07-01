from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main
from PACKAGE.c_split_dataset_prompt import DataPreprocessor
from PACKAGE.d_build_lstm_and_transformer_models_prompt import validate_data_format
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

# --- Start of Added Code ---

def save_data_for_evaluation_module(X_test: np.ndarray, y_test: np.ndarray):
    """
    Saves the test data (X_test, y_test) into a CSV format that is expected
    by the 'evaluate_model_performance' module, based on the specification.

    Args:
        X_test (np.ndarray): The test features with shape (n_samples, sequence_length, n_features).
        y_test (np.ndarray): The test targets with shape (n_samples, 1).
    """
    SAVE_DIR = r"C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Claude\UNITTEST_DATA\generated"
    FILE_NAME = "input_for_evaluate_model_performance.csv"
    
    # Determine if the input data can be correctly determined and saved.
    # The specification implies 4 features (OHLC) and a sequence length of 60.
    if X_test is None or y_test is None or X_test.ndim != 3 or X_test.shape[2] != 4:
        logger.warning(
            "Expected input data for the next module could not be determined or is in the wrong format. "
            f"Skipping CSV file generation. X_test shape: {X_test.shape if X_test is not None else 'None'}"
        )
        return

    try:
        logger.info(f"Preparing to save input data for the evaluation module to {os.path.join(SAVE_DIR, FILE_NAME)}.")

        # Create the target directory if it does not exist.
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # Reshape X_test from (samples, sequence_length, features) to (samples, sequence_length * features).
        n_samples, sequence_length, n_features = X_test.shape
        X_test_reshaped = X_test.reshape(n_samples, sequence_length * n_features)

        # Create column headers as specified in the "Data Head (example)".
        # The features are 'Open', 'High', 'Low', 'Close'.
        features = ['Open', 'High', 'Low', 'Close']
        x_columns = [
            f'{feature}_t-{i}' 
            for i in range(sequence_length - 1, -1, -1) 
            for feature in features
        ]
        
        # Create a DataFrame for the features (X_test).
        df_x = pd.DataFrame(X_test_reshaped, columns=x_columns)
        
        # Create a DataFrame for the target (y_test).
        df_y = pd.DataFrame(y_test, columns=['target_Close_t+1'])
        
        # Combine features and target into a single DataFrame.
        final_df = pd.concat([df_x, df_y], axis=1)

        # Save the combined DataFrame to a CSV file, overwriting any existing file.
        file_path = os.path.join(SAVE_DIR, FILE_NAME)
        final_df.to_csv(file_path, index=False)
        
        logger.info(f"Successfully saved evaluation input data to {file_path}")

    except (IOError, PermissionError) as e:
        logger.error(f"Failed to write to directory {SAVE_DIR}. Please check permissions. Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the data for the next module: {e}")

# --- End of Added Code ---

def main():
    """
    Main entry point for the integrated script.
    
    This function orchestrates the data loading, preprocessing, model building,
    and training pipeline by:
    1. Calling the data loading module to get the user-provided stock data.
    2. Validating the format of the loaded DataFrame.
    3. Passing the loaded data to the data preprocessing and splitting module.
    4. Building the LSTM and Transformer model architectures.
    5. Training both models on the preprocessed data.
    6. Acknowledging the unimplemented model evaluation and visualization steps.
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

    # --- Step 6: Train Models ---
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

    # --- Start of Added Code ---
    # Save the input data for the next module ('evaluate_model_performance').
    # This is done after data processing and model training are complete,
    # ensuring that the required data ('X_test', 'y_test') is available from the
    # 'processed_data' dictionary.
    save_data_for_evaluation_module(
        processed_data.get('X_test'),
        processed_data.get('y_test')
    )
    # --- End of Added Code ---

    # --- Step 7: Evaluate and Visualize Models (Placeholder) ---
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