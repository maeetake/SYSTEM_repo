from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main
from PACKAGE.c_split_dataset_prompt import DataPreprocessor
from PACKAGE.d_build_lstm_and_transformer_models_prompt import validate_data_format
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

# ADDED: Function to save input data for the next module as per specification.
def save_data_for_next_module(dates: pd.Series, actual_prices: np.ndarray, lstm_predictions: np.ndarray, transformer_predictions: np.ndarray):
    """
    Saves the data required for the visualization module to a CSV file.

    Based on the specification for the 'visualize_prediction_results' module, this
    function constructs a DataFrame containing the actual prices, LSTM predictions,
    Transformer predictions, and corresponding dates, then saves it as a CSV.

    Args:
        dates (pd.Series): The dates for the test set.
        actual_prices (np.ndarray): The actual closing prices (original scale).
        lstm_predictions (np.ndarray): The predicted prices from the LSTM model (original scale).
        transformer_predictions (np.ndarray): The predicted prices from the Transformer model (original scale).
    """
    # Specification: Hardcode the file save directory.
    save_dir = r"C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Claude\UNITTEST_DATA\generated"
    
    # Specification: The file name and extension must be appropriately determined by the LLM.
    file_name = "input_for_visualize_prediction_results.csv"
    output_path = os.path.join(save_dir, file_name)

    logger.info(f"Preparing to save input data for the next module to {output_path}")

    try:
        # Specification: Save the file only if the expected input data can be correctly determined.
        # Check if all data components are valid and have the same length.
        if not all(isinstance(data, (pd.Series, np.ndarray)) and len(data) > 0 for data in [dates, actual_prices, lstm_predictions, transformer_predictions]):
            logger.error("One or more data components for saving are invalid or empty. Skipping file save.")
            return
        
        if not (len(dates) == len(actual_prices) == len(lstm_predictions) == len(transformer_predictions)):
            logger.error(f"Mismatched lengths in data components for saving. "
                         f"Dates: {len(dates)}, Actuals: {len(actual_prices)}, "
                         f"LSTM: {len(lstm_predictions)}, Transformer: {len(transformer_predictions)}. "
                         f"Skipping file save.")
            return

        # Create a DataFrame from the provided data. This format is easily consumable
        # by the next module, which expects pandas Series and numpy arrays.
        df_to_save = pd.DataFrame({
            'Date': dates,
            'Actual_Price': actual_prices.flatten(),
            'LSTM_Prediction': lstm_predictions.flatten(),
            'Transformer_Prediction': transformer_predictions.flatten()
        })
        
        # Specification: Add a process to create the directory if necessary.
        os.makedirs(save_dir, exist_ok=True)
        
        # Specification: Save the data as a properly formatted CSV file, overwriting if it exists.
        df_to_save.to_csv(output_path, index=False)
        
        logger.info(f"Successfully saved data for visualization module to {output_path}")

    except (IOError, OSError, PermissionError) as e:
        logger.error(f"Failed to save data to {output_path}. Check permissions. Reason: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving data for the next module: {e}")

def main():
    """
    Main entry point for the integrated script.
    
    This function orchestrates the data loading, preprocessing, model building,
    training, and evaluation pipeline by:
    1. Calling the data loading module to get the user-provided stock data.
    2. Validating the format of the loaded DataFrame.
    3. Passing the loaded data to the data preprocessing and splitting module.
    4. Building the LSTM and Transformer model architectures.
    5. Training both models on the preprocessed data.
    6. Evaluating both trained models on the test set using RMSE and MAE.
    7. Saving the required inputs for the next module (visualize_prediction_results) to a CSV file.
    8. Acknowledging the unimplemented visualization steps.
    """
    logger.info("Starting the data processing and model building pipeline.")

    # --- Step 1: Load Data ---
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
    try:
        logger.info("Validating loaded data format.")
        validate_data_format(df)
        logger.info("Data format validated successfully.")
    except ValueError as e:
        logger.error(f"Data validation failed: {e}. Aborting.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during data validation: {e}")
        return

    # --- Step 3: Preprocess and Split Data ---
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

    # --- Step 4: Build Deep Learning Models ---
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

    # --- Step 5: Train Models ---
    trained_lstm_model, trained_transformer_model = None, None
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

    # --- Step 6: Evaluate Models ---
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

    # --- ADDED: Step 6.5: Save input data for the next module (visualize_prediction_results) ---
    try:
        # Ensure models were trained successfully before proceeding
        if trained_lstm_model and trained_transformer_model:
            logger.info("Preparing data for the visualization module.")

            # Generate predictions to get all necessary arrays
            lstm_preds_normalized = make_predictions(trained_lstm_model, processed_data['X_test'])
            transformer_preds_normalized = make_predictions(trained_transformer_model, processed_data['X_test'])

            # Inverse transform predictions and actual values to the original scale
            transformed_values = inverse_transform_values(
                predictions_normalized=lstm_preds_normalized,
                y_test_normalized=processed_data['y_test'],
                scaler=processed_data['scaler']
            )
            y_test_actual = transformed_values['y_test_actual_scale']
            lstm_preds_actual = transformed_values['predictions_actual_scale']
            
            # Since y_test is the same, we only need to inverse transform the transformer predictions
            transformer_preds_actual = inverse_transform_values(
                predictions_normalized=transformer_preds_normalized,
                y_test_normalized=processed_data['y_test'],
                scaler=processed_data['scaler']
            )['predictions_actual_scale']

            # Get the corresponding dates for the test set
            num_test_samples = len(processed_data['y_test'])
            test_dates = df['Date'].iloc[-num_test_samples:].reset_index(drop=True)

            # Call the new function to save the data
            save_data_for_next_module(
                dates=test_dates,
                actual_prices=y_test_actual,
                lstm_predictions=lstm_preds_actual,
                transformer_predictions=transformer_preds_actual
            )
        else:
            logger.warning("Models were not trained successfully. Skipping data save for visualization.")
    except Exception as e:
        logger.error(f"Failed to generate or save data for the visualization module: {e}")

    # --- Step 7: Acknowledge Unimplemented Visualizations ---
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