from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
from PACKAGE.c_split_dataset_prompt import split_sequential_data
from PACKAGE.d_build_lstm_and_transformer_models_prompt import build_lstm_model, build_transformer_model
from PACKAGE.e_train_models_prompt import train_model
from PACKAGE.f_evaluate_model_performance_prompt import evaluate_model_performance
from PACKAGE.g_visualize_prediction_results_prompt import plot_predictions

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Any, Dict, Optional
import os

def main() -> None:
    """
    Main entry point for the integrated data loading, preprocessing, splitting,
    model building, training, evaluation, and visualization pipeline. This function
    loads data, processes it, builds LSTM and Transformer models, trains them,
    evaluates their performance, and visualizes the results.
    """
    # Specify the data path (can update as needed)
    data_path: str = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\NVIDIA.csv'

    df: Optional[pd.DataFrame] = None  # Initialize df to None to ensure it's defined
    processed_data: Optional[Dict[str, np.ndarray]] = None  # Initialize processed_data to None
    preprocessor: Optional[DataPreprocessor] = None # Initialize preprocessor to None

    try:
        # Step 1: Load the data using the imported function
        df = load_data_from_csv(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())

    except FileNotFoundError as fnf_err:
        print(f"File not found: {fnf_err}")
        # Using mock data for demonstration purposes if file is not found.
        df = pd.DataFrame({
            "Date": pd.to_datetime([f"2024-01-{i:02d}" for i in range(1, 91)]),
            "Open": np.linspace(500, 550, 90),
            "High": np.linspace(505, 555, 90),
            "Low": np.linspace(495, 545, 90),
            "Close": np.linspace(503, 553, 90)
        })
        print("\nUsing mock data for demonstration:")
        print(df.head())

    except (ValueError, KeyError) as data_err:
        print(f"An error occurred while loading the data: {data_err}")
        return

    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    # If df was successfully loaded (either from file or mock data), proceed to preprocessing.
    if df is not None:
        print("\n--- Starting Data Preprocessing ---")
        preprocessor = DataPreprocessor(sequence_length=60)
        try:
            # Step 2: Process the loaded dataframe using the imported preprocessor
            processed_data = preprocessor.process(df)
            print("Data preprocessing successful.")

            # Data shape assertions to ensure compatibility
            x_shapes = {
                'X_train': processed_data['X_train'],
                'X_val':   processed_data['X_val'],
                'X_test':  processed_data['X_test'],
            }
            y_shapes = {
                'y_train': processed_data['y_train'],
                'y_val':   processed_data['y_val'],
                'y_test':  processed_data['y_test'],
            }
            for xk, xv in x_shapes.items():
                if len(xv.shape) != 3:
                    raise ValueError(f"{xk} must be 3D (samples, sequence_length, features), but got {xv.shape}")
            for yk, yv in y_shapes.items():
                if len(yv.shape) != 1:
                    raise ValueError(f"{yk} must be 1D (samples,), but got {yv.shape}")
            n_x = [x.shape[0] for x in x_shapes.values()]
            n_y = [y.shape[0] for y in y_shapes.values()]
            if n_x != n_y:
                raise ValueError(f"Mismatch in number of samples: X splits: {n_x}, y splits: {n_y}")

            print(f"X_train shape: {processed_data['X_train'].shape}")
            print(f"y_train shape: {processed_data['y_train'].shape}")
            print(f"X_val shape: {processed_data['X_val'].shape}")
            print(f"y_val shape: {processed_data['y_val'].shape}")
            print(f"X_test shape: {processed_data['X_test'].shape}")
            print(f"y_test shape: {processed_data['y_test'].shape}")

        except ValueError as ve:
            print(f"Preprocessing Error: {ve}")
            return
        except Exception as e:
            print(f"An unexpected error occurred during preprocessing: {e}")
            return

    # If preprocessing was successful, build, train models, and then evaluate.
    if processed_data and preprocessor:
        lstm_model: Optional[tf.keras.Model] = None
        transformer_model: Optional[tf.keras.Model] = None

        # Step 3: Build the deep learning models using the imported functions
        print("\n--- Starting Model Building ---")
        try:
            input_shape = processed_data['X_train'].shape[1:]
            print(f"Building models with input shape: {input_shape}")

            # Build models
            lstm_model = build_lstm_model(input_shape)
            print("\nLSTM model built successfully. Summary:")
            lstm_model.summary()

            transformer_model = build_transformer_model(input_shape)
            print("\nTransformer model built successfully. Summary:")
            transformer_model.summary()

            # ------------ Model Compilation Step ------------
            lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            transformer_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            # -------------------------------------------------

        except ValueError as ve_model:
            print(f"Model Building Error: {ve_model}")
            return
        except Exception as e_model:
            print(f"An unexpected error occurred during model building: {e_model}")
            return

        # Step 4: Train the built models
        trained_lstm_model = None
        trained_transformer_model = None
        if lstm_model is not None and transformer_model is not None:
            print("\n--- Starting Model Training ---")

            training_params: Dict[str, Any] = {
                'epochs': 50,
                'batch_size': 32,
                'early_stopping_patience': 10
            }
            lstm_history, transformer_history = {}, {}

            try:
                # Train LSTM Model
                print("\nTraining LSTM Model...")
                lstm_params = training_params.copy()
                lstm_params['checkpoint_path'] = 'best_LSTM_model.h5'
                trained_lstm_model, lstm_history = train_model(
                    model=lstm_model, X_train=processed_data['X_train'], y_train=processed_data['y_train'],
                    X_val=processed_data['X_val'], y_val=processed_data['y_val'], training_params=lstm_params
                )
                print("LSTM Model training complete.")
                if lstm_history.get('loss'): print(f"Final LSTM Training Loss: {lstm_history['loss'][-1]:.4f}")
                if lstm_history.get('val_loss'): print(f"Final LSTM Validation Loss: {lstm_history['val_loss'][-1]:.4f}")

                # Train Transformer Model
                print("\nTraining Transformer Model...")
                transformer_params = training_params.copy()
                transformer_params['checkpoint_path'] = 'best_Transformer_model.h5'
                trained_transformer_model, transformer_history = train_model(
                    model=transformer_model, X_train=processed_data['X_train'], y_train=processed_data['y_train'],
                    X_val=processed_data['X_val'], y_val=processed_data['y_val'], training_params=transformer_params
                )
                print("Transformer Model training complete.")
                if transformer_history.get('loss'): print(f"Final Transformer Training Loss: {transformer_history['loss'][-1]:.4f}")
                if transformer_history.get('val_loss'): print(f"Final Transformer Validation Loss: {transformer_history['val_loss'][-1]:.4f}")

            except Exception as e_train:
                print(f"An unexpected error occurred during model training: {e_train}")
        
        # Step 5: Evaluate the trained models on the test set and visualize results
        if trained_lstm_model and trained_transformer_model:
            print("\n--- Starting Model Evaluation ---")
            try:
                models_to_evaluate = {"LSTM": trained_lstm_model, "Transformer": trained_transformer_model}
                scaler = preprocessor.scaler
                close_feature_index = preprocessor.feature_cols.index('Close')

                evaluation_results = evaluate_model_performance(
                    models=models_to_evaluate, X_test=processed_data['X_test'], y_test=processed_data['y_test'],
                    scaler=scaler, close_index=close_feature_index
                )

                print("\nEvaluation Metrics (RMSE, MAE):")
                metrics_report = evaluation_results.get('metrics_report', {})
                for model_name, metrics in metrics_report.items():
                    print(f"  {model_name}:\n    RMSE: {metrics.get('RMSE', 'N/A'):.4f}\n    MAE: {metrics.get('MAE', 'N/A'):.4f}")

                # --- Visualize Prediction Results (Integration of g_visualize_prediction_results_prompt) ---
                print("\n--- Visualizing Prediction Results ---")
                try:
                    predictions_data = evaluation_results.get('predictions')
                    if not predictions_data: raise ValueError("Evaluation results missing 'predictions' data for plotting.")

                    y_true_inv = predictions_data['LSTM']['y_true_inv']
                    y_pred_lstm_inv = predictions_data['LSTM']['y_pred_inv']
                    y_pred_transformer_inv = predictions_data['Transformer']['y_pred_inv']

                    n_train = len(processed_data['X_train'])
                    n_val = len(processed_data['X_val'])
                    test_start_date_idx = n_train + n_val + preprocessor.sequence_length
                    test_end_date_idx = test_start_date_idx + len(processed_data['X_test'])
                    test_dates = df['Date'].iloc[test_start_date_idx:test_end_date_idx].reset_index(drop=True)

                    actual_prices_series = pd.Series(y_true_inv.flatten(), name='Actual')
                    predicted_lstm_array = np.array(y_pred_lstm_inv).flatten()
                    predicted_transformer_array = np.array(y_pred_transformer_inv).flatten()
                    
                    plot_path = os.path.join('results', 'prediction_comparison_plot.png')
                    saved_plot_path = plot_predictions(
                        actual_prices=actual_prices_series, predicted_prices_lstm=predicted_lstm_array,
                        predicted_prices_transformer=predicted_transformer_array, dates=test_dates, output_path=plot_path
                    )
                    print(f"Prediction comparison plot saved successfully to: {saved_plot_path}")

                except Exception as e_viz:
                    print(f"An error occurred during prediction results visualization: {e_viz}")
                
                # --- Visualize Training History ---
                # Note: The specification requires plotting training/validation loss curves. However, the provided
                # 'g_visualize_prediction_results_prompt.py' script does not contain a function for this purpose.
                # As per the constraint to "limit integration to already implemented parts", this feature is omitted.
                print("\n--- Training History Visualization ---")
                print("Skipping loss curve plotting as the function is not in the provided visualization module.")
                
            except Exception as e_eval:
                print(f"An unexpected error occurred during model evaluation: {e_eval}")

        # Step 6: Demonstrate the standalone data splitting function
        print("\n--- Demonstrating Standalone Data Splitting from 'c_split_dataset_prompt' ---")
        try:
            X_full = np.concatenate([processed_data['X_train'], processed_data['X_val'], processed_data['X_test']], axis=0)
            y_full = np.concatenate([processed_data['y_train'], processed_data['y_val'], processed_data['y_test']], axis=0)
            print(f"Reconstructed full dataset for demonstration: X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")

            split_datasets_demo = split_sequential_data(X_full, y_full, split_ratios=(0.8, 0.1, 0.1))

            print("\nData splitting using imported 'split_sequential_data' function successful.")
            print("Shapes of the new splits:")
            for key, value in split_datasets_demo.items():
                print(f"  {key}: {value.shape}")

        except Exception as e_split:
            print(f"An unexpected error occurred during standalone splitting demonstration: {e_split}")

if __name__ == "__main__":
    main()