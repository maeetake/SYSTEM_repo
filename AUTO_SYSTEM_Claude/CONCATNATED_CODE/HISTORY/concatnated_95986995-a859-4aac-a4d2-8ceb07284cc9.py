# Revised Integrated Script
import sys
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Module Imports (Corrected based on feedback) ---
# The following imports have been restructured to correctly integrate functionality
# from the source modules, resolving previous issues with code duplication and
# incorrect sourcing.

# 1. Data Loading: Sourced from 'a'.
from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main

# 2. Data Preprocessing: Sourced from 'c'.
# 'c_split_dataset_prompt' provides the consolidated DataPreprocessor.
from PACKAGE.c_split_dataset_prompt import DataPreprocessor

# 3. Model Building and Training: Sourced from 'e' and 'f'.
# Model architecture functions ('build_*') are sourced from 'e_train_models_prompt'.
from PACKAGE.e_train_models_prompt import build_lstm_model, build_transformer_model
# The 'train_model' function is correctly sourced from 'f_evaluate_model_performance_prompt',
# resolving the previous code duplication and incorrect local implementation.
from PACKAGE.f_evaluate_model_performance_prompt import train_model

# 4. Evaluation Helpers: Sourced from 'g'.
# Helper functions for making predictions and processing results.
from PACKAGE.g_visualize_prediction_results_prompt import make_predictions, inverse_transform_values, calculate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model_performance(model_name: str, model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, scaler: object):
    """
    Evaluates a trained model by making predictions and calculating performance metrics.

    This helper function encapsulates the evaluation pipeline for a single model:
    1. Generates predictions on the test set.
    2. Inverse transforms predictions and actual values to their original scale.
    3. Calculates and prints RMSE and MAE metrics.

    Args:
        model_name (str): The name of the model being evaluated (e.g., "LSTM").
        model (tf.keras.Model): The trained Keras model.
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The normalized test target values.
        scaler (object): The fitted scaler object used for normalization.
    """
    logger.info(f"--- Starting evaluation for {model_name} ---")
    
    try:
        # Step 1: Generate predictions on the test set
        predictions_normalized = make_predictions(model, X_test)

        # Step 2: Inverse transform predictions and actual values to the original scale
        transformed_values = inverse_transform_values(
            predictions_normalized=predictions_normalized,
            y_test_normalized=y_test,
            scaler=scaler
        )
        y_pred_actual = transformed_values['predictions_actual_scale']
        y_test_actual = transformed_values['y_test_actual_scale']

        # Step 3: Calculate performance metrics (RMSE, MAE)
        metrics = calculate_metrics(y_test_actual, y_pred_actual)

        # Step 4: Display the results
        print(f"\n--- {model_name} Evaluation Results ---")
        print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
        print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
        print("-------------------------------------------\n")
        logger.info(f"{model_name} evaluation complete. RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")

    except Exception as e:
        logger.error(f"An error occurred during {model_name} evaluation: {e}")
        raise

def main():
    """
    Main entry point for the integrated script.
    
    Orchestrates the data loading, preprocessing, model building, training,
    and evaluation pipeline based on a corrected, modular structure.
    """
    logger.info("Starting the data processing and model building pipeline.")

    # --- Step 1: Load and Validate Data ---
    try:
        logger.info("Attempting to load data using the data loading module.")
        # Uses 'a_load_user_provided_data_prompt'
        df = load_data_main()

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Data loading failed or returned an empty DataFrame. Aborting.")
            return

        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        return

    # --- Step 2: Preprocess and Split Data ---
    # This step now uses the consolidated DataPreprocessor from 'c_split_dataset_prompt'.
    try:
        logger.info("Initializing data preprocessor.")
        # Uses 'c_split_dataset_prompt'
        preprocessor = DataPreprocessor(sequence_length=60)
        
        logger.info("Processing data (normalize, create sequences, and split)...")
        # The preprocessor correctly uses only OHLC features as required.
        processed_data = preprocessor.process(df)
        logger.info("Data preprocessing and splitting completed successfully.")

        print("\n--- Preprocessing and Splitting Results ---")
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                print(f"{key} shape: {value.shape}")
        print("-----------------------------------------\n")
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")
        return

    # --- Step 3: Build Deep Learning Models ---
    # Model building functions are sourced from 'e_train_models_prompt'.
    try:
        logger.info("Building LSTM and Transformer models.")
        input_shape = (processed_data['X_train'].shape[1], processed_data['X_train'].shape[2])
        logger.info(f"Using input shape for models: {input_shape}")

        # Uses 'e_train_models_prompt'
        lstm_model = build_lstm_model(input_shape)
        print("\n--- LSTM Model Summary ---")
        lstm_model.summary()
        print("--------------------------\n")

        # Uses 'e_train_models_prompt'
        transformer_model = build_transformer_model(input_shape)
        print("\n--- Transformer Model Summary ---")
        transformer_model.summary()
        print("-------------------------------\n")

        logger.info("Successfully built both LSTM and Transformer models.")
    except Exception as e:
        logger.error(f"An error occurred during model building: {e}")
        return

    # --- Step 4: Train Models ---
    # The 'train_model' function is now correctly imported from 'f_evaluate_model_performance_prompt',
    # ensuring proper modular integration.
    try:
        logger.info("Preparing to train models.")
        training_params = {'epochs': 50, 'batch_size': 32, 'patience': 10}
        
        logger.info("Compiling and training LSTM model.")
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        # Uses 'train_model' from module 'f'
        trained_lstm_model, lstm_history = train_model(
            model=lstm_model,
            X_train=processed_data['X_train'], y_train=processed_data['y_train'],
            X_val=processed_data['X_val'], y_val=processed_data['y_val'],
            training_params=training_params
        )
        logger.info("LSTM model training complete.")

        logger.info("Compiling and training Transformer model.")
        transformer_model.compile(optimizer='adam', loss='mean_squared_error')
        # Uses 'train_model' from module 'f'
        trained_transformer_model, transformer_history = train_model(
            model=transformer_model,
            X_train=processed_data['X_train'], y_train=processed_data['y_train'],
            X_val=processed_data['X_val'], y_val=processed_data['y_val'],
            training_params=training_params
        )
        logger.info("Transformer model training complete.")
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        return

    # --- Step 5: Evaluate Models ---
    # The evaluation logic uses helper functions imported from 'g_visualize_prediction_results_prompt'.
    try:
        logger.info("Starting model evaluation on the test set.")
        
        # Evaluate the trained LSTM model
        evaluate_model_performance(
            model_name="LSTM Model",
            model=trained_lstm_model,
            X_test=processed_data['X_test'],
            y_test=processed_data['y_test'],
            scaler=processed_data['scaler']
        )
        
        # Evaluate the trained Transformer model
        evaluate_model_performance(
            model_name="Transformer Model",
            model=trained_transformer_model,
            X_test=processed_data['X_test'],
            y_test=processed_data['y_test'],
            scaler=processed_data['scaler']
        )
    except Exception as e:
        logger.error(f"An error occurred during the evaluation phase: {e}")
        return

    # --- Step 6: Acknowledge Unimplemented Visualizations ---
    # Visualization plots for predictions and loss curves were not implemented in the
    # source modules and are thus not included here.
    logger.warning("Visualization of prediction results and loss curves is not implemented in the provided source modules.")
    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure that this script is run from a directory containing the 'PACKAGE' folder,", file=sys.stderr)
        print("and that 'PACKAGE' contains the necessary, correctly organized modules.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution block: {e}")
        sys.exit(1)