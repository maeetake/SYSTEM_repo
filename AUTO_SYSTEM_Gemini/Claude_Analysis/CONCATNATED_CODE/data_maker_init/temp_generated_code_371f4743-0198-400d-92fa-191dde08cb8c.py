from PACKAGE.a_load_user_provided_data_prompt import main as load_data_main
from PACKAGE.c_split_dataset_prompt import DataPreprocessor
from PACKAGE.d_build_lstm_and_transformer_models_prompt import validate_data_format
from PACKAGE.f_evaluate_model_performance_prompt import build_lstm_model, build_transformer_model, train_model
from PACKAGE.g_visualize_prediction_results_prompt import make_predictions, inverse_transform_values, calculate_metrics
from PACKAGE.h_visualize_training_history_prompt import plot_predictions
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
    training, evaluation, and visualization pipeline by:
    1. Calling the data loading module to get the user-provided stock data.
    2. Validating the format of the loaded DataFrame.
    3. Passing the loaded data to the data preprocessing and splitting module.
    4. Building the LSTM and Transformer model architectures.
    5. Training both models on the preprocessed data.
    6. Evaluating both trained models on the test set using RMSE and MAE.
    7. Visualizing the prediction results against the actual values.
    8. Acknowledging unimplemented visualization for loss curves.
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
        sequence_length = 60
        preprocessor = DataPreprocessor(sequence_length=sequence_length)
        
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

    # --- BEGIN MODIFICATION: Save training history for the next module ---
    try:
        # Based on the specification for 'visualize_training_history', the next module
        # expects a dictionary with 'loss' and 'val_loss' keys. We save this data for both models.
        if (lstm_history and hasattr(lstm_history, 'history') and 'loss' in lstm_history.history and 'val_loss' in lstm_history.history and
            transformer_history and hasattr(transformer_history, 'history') and 'loss' in transformer_history.history and 'val_loss' in transformer_history.history):
            
            logger.info("Preparing to save model training history for the next module.")
            
            # Use the shorter history if early stopping results in different epoch counts
            num_epochs = min(len(lstm_history.history['loss']), len(transformer_history.history['loss']))
            
            # Create a dictionary to build the DataFrame
            history_data = {
                'epoch': range(1, num_epochs + 1),
                'lstm_loss': lstm_history.history['loss'][:num_epochs],
                'lstm_val_loss': lstm_history.history['val_loss'][:num_epochs],
                'transformer_loss': transformer_history.history['loss'][:num_epochs],
                'transformer_val_loss': transformer_history.history['val_loss'][:num_epochs]
            }
            
            history_df = pd.DataFrame(history_data)

            # Hardcode the save directory and filename as specified
            save_dir = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Claude\UNITTEST_DATA\generated'
            file_name = 'input_for_visualize_training_history.csv'
            output_file_path = os.path.join(save_dir, file_name)

            # Create the directory if it doesn't exist, handling potential errors
            os.makedirs(save_dir, exist_ok=True)
            
            # Save the DataFrame to a CSV file, overwriting if it exists
            history_df.to_csv(output_file_path, index=False)
            
            logger.info(f"Successfully saved model training history to {output_file_path}")
            
        else:
            logger.warning("Could not save training history: History objects or required keys ('loss', 'val_loss') are missing.")

    except (IOError, OSError) as e:
        logger.error(f"A file system error occurred while saving the model training history: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the model training history: {e}")
    # --- END MODIFICATION ---

    # --- Step 6: Evaluate Models & Generate Predictions for Visualization ---
    try:
        logger.info("Starting model evaluation and prediction generation for visualization.")

        # Generate predictions for both models
        lstm_predictions_norm = make_predictions(trained_lstm_model, processed_data['X_test'])
        transformer_predictions_norm = make_predictions(trained_transformer_model, processed_data['X_test'])

        # Inverse transform all values to their original scale
        transformed_lstm = inverse_transform_values(
            predictions_normalized=lstm_predictions_norm,
            y_test_normalized=processed_data['y_test'],
            scaler=processed_data['scaler']
        )
        y_test_actual = transformed_lstm['y_test_actual_scale']
        lstm_predictions_actual = transformed_lstm['predictions_actual_scale']
        
        # We only need to inverse transform the predictions for the second model
        # as y_test_actual is the same.
        transformed_transformer = inverse_transform_values(
            predictions_normalized=transformer_predictions_norm,
            y_test_normalized=processed_data['y_test'],
            scaler=processed_data['scaler']
        )
        transformer_predictions_actual = transformed_transformer['predictions_actual_scale']

        # Calculate and print metrics for LSTM model
        lstm_metrics = calculate_metrics(y_test_actual, lstm_predictions_actual)
        print("\n--- LSTM Model Evaluation Results ---")
        print(f"Root Mean Squared Error (RMSE): {lstm_metrics['RMSE']:.4f}")
        print(f"Mean Absolute Error (MAE): {lstm_metrics['MAE']:.4f}")
        print("-------------------------------------")
        logger.info(f"LSTM evaluation complete. RMSE: {lstm_metrics['RMSE']:.4f}, MAE: {lstm_metrics['MAE']:.4f}")

        # Calculate and print metrics for Transformer model
        transformer_metrics = calculate_metrics(y_test_actual, transformer_predictions_actual)
        print("\n--- Transformer Model Evaluation Results ---")
        print(f"Root Mean Squared Error (RMSE): {transformer_metrics['RMSE']:.4f}")
        print(f"Mean Absolute Error (MAE): {transformer_metrics['MAE']:.4f}")
        print("--------------------------------------------\n")
        logger.info(f"Transformer evaluation complete. RMSE: {transformer_metrics['RMSE']:.4f}, MAE: {transformer_metrics['MAE']:.4f}")

    except Exception as e:
        logger.error(f"An error occurred during the evaluation phase: {e}")
        return
    
    # --- Step 7: Visualize Prediction Results ---
    try:
        logger.info("Generating prediction visualization plot.")
        
        # Determine the date range for the test set
        num_sequences = len(df) - sequence_length
        train_size = int(num_sequences * 0.8)
        val_size = int(num_sequences * 0.1)
        test_start_index = train_size + val_size + sequence_length
        
        # Ensure the date slice has the same length as the test set predictions
        test_dates = df['Date'].iloc[test_start_index:test_start_index + len(y_test_actual)]
        test_dates = test_dates.reset_index(drop=True)

        # Create a pandas Series for actual prices for plotting
        actual_prices = pd.Series(y_test_actual.flatten(), name='Actual Price')

        # Call the plotting function
        output_plot_path = "prediction_comparison_plot.png"
        saved_path = plot_predictions(
            actual_prices=actual_prices,
            predicted_prices_lstm=lstm_predictions_actual,
            predicted_prices_transformer=transformer_predictions_actual,
            dates=test_dates,
            output_path=output_plot_path
        )
        logger.info(f"Prediction plot saved successfully to: {saved_path}")

    except Exception as e:
        logger.error(f"Failed to generate or save prediction plot: {e}")

    # --- Step 8: Acknowledge Unimplemented Visualizations ---
    # The provided scripts do not include implementation for visualizing
    # the training/validation loss curves.
    logger.warning("Visualization of training and validation loss function transition is not implemented.")
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