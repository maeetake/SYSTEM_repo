# Revised Integrated Script
from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main
from PACKAGE.c_split_dataset_prompt import DataPreprocessor
from PACKAGE.f_evaluate_model_performance_prompt import build_lstm_model, build_transformer_model, train_model
from PACKAGE.g_visualize_prediction_results_prompt import make_predictions, inverse_transform_values, calculate_metrics
import sys
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

# Configure logging to match the style of the provided modules
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
        # Re-raise the exception to be caught by the main loop if necessary
        raise

def main():
    """
    Main entry point for the integrated script.
    
    This function orchestrates the data loading, preprocessing, model building,
    training, and evaluation pipeline by:
    1. Calling the data loading module to get the user-provided stock data.
    2. Passing the loaded data to the data preprocessing and splitting module.
    3. Building the LSTM and Transformer model architectures.
    4. Training both models on the preprocessed data.
    5. Evaluating both trained models on the test set using RMSE and MAE.
    6. Acknowledging the unimplemented visualization steps.
    """
    logger.info("Starting the data processing and model building pipeline.")

    # --- Step 1: Load and Validate Data ---
    # The load_data_main function from a_load_user_provided_data_prompt
    # handles loading and initial validation (checking for required columns).
    try:
        logger.info("Attempting to load and validate data using the user-provided data loader.")
        df = load_data_main()

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Data loading failed or returned an empty DataFrame. Aborting.")
            return

        logger.info("Data loaded and initially validated successfully.")
    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        return

    # --- Step 2: Preprocess and Split Data ---
    try:
        logger.info("Initializing data preprocessor.")
        preprocessor = DataPreprocessor(sequence_length=60)
        
        logger.info("Processing data (normalize, create sequences, and split)...")
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
    try:
        logger.info("Building LSTM and Transformer models.")
        input_shape = (processed_data['X_train'].shape[1], processed_data['X_train'].shape[2])
        logger.info(f"Using input shape for models: {input_shape}")

        lstm_model = build_lstm_model(input_shape)
        print("\n--- LSTM Model Summary ---")
        lstm_model.summary()
        print("--------------------------\n")

        transformer_model = build_transformer_model(input_shape)
        print("\n--- Transformer Model Summary ---")
        transformer_model.summary()
        print("-------------------------------\n")

        logger.info("Successfully built both LSTM and Transformer models.")
    except Exception as e:
        logger.error(f"An error occurred during model building: {e}")
        return

    # --- Step 4: Train Models ---
    try:
        logger.info("Preparing to train models.")
        training_params = {'epochs': 50, 'batch_size': 32, 'patience': 10}
        
        logger.info("Compiling and training LSTM model.")
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        trained_lstm_model, lstm_history = train_model(
            model=lstm_model,
            X_train=processed_data['X_train'], y_train=processed_data['y_train'],
            X_val=processed_data['X_val'], y_val=processed_data['y_val'],
            training_params=training_params
        )
        logger.info("LSTM model training complete.")

        logger.info("Compiling and training Transformer model.")
        transformer_model.compile(optimizer='adam', loss='mean_squared_error')
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
    # The provided scripts do not include implementation for visualizing
    # the prediction results or the training/validation loss curves.
    # Adhering to the constraint of not adding unimplemented features,
    # this step is a final acknowledgment.
    logger.warning("Visualization of prediction results and loss curves is not implemented in the provided source modules.")
    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
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