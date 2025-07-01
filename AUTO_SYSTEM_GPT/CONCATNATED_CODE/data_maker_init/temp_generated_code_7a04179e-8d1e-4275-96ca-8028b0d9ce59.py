from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
from PACKAGE.c_split_dataset_prompt import split_sequential_data
from PACKAGE.d_build_lstm_and_transformer_models_prompt import build_lstm_model, build_transformer_model
from PACKAGE.e_train_models_prompt import train_model
from PACKAGE.g_visualize_prediction_results_prompt import plot_predictions
from PACKAGE.h_visualize_training_history_prompt import plot_and_save_history
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Any, Dict, Optional
import os

# --- FIXED evaluate_model_performance function (copied and used below to ensure predictions structure for downstream plotting) ---
def evaluate_model_performance(
    models: Dict[str, tf.keras.Model],
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: Any,
    close_index: int
) -> Dict[str, Any]:
    """
    Evaluates all provided models on the test set, returns a metrics report and predictions in original scale.

    Returns:
        Dict[str, Any]: {
            'metrics_report': { 'ModelName': {'RMSE': val, 'MAE': val}, ... },
            'predictions': {
                'ModelName': {'y_true_inv': ..., 'y_pred_inv': ...},
                ...
            }
        }
    """
    metrics_report: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    # Flatten y_test in case it's shape (n,1), ensure shape alignment
    y_true = y_test.ravel()

    for model_name, model in models.items():
        # Model prediction
        y_pred = model.predict(X_test)
        y_pred = np.array(y_pred).flatten()

        # Prepare for inverse transformation: create dummy feature array
        n_features = scaler.mean_.shape[0] if hasattr(scaler, "mean_") else scaler.data_min_.shape[0]
        n_samples = y_pred.shape[0]
        X_dummy_pred = np.zeros((n_samples, n_features))
        X_dummy_true = np.zeros((n_samples, n_features))
        X_dummy_pred[:, close_index] = y_pred
        X_dummy_true[:, close_index] = y_true
        # Inverse transform only the close price feature
        y_pred_inv = scaler.inverse_transform(X_dummy_pred)[:, close_index]
        y_true_inv = scaler.inverse_transform(X_dummy_true)[:, close_index]
        # Calculate metrics on inverse transformed (real price) data
        rmse = float(np.sqrt(np.mean((y_pred_inv - y_true_inv) ** 2)))
        mae = float(np.mean(np.abs(y_pred_inv - y_true_inv)))

        metrics_report[model_name] = {"RMSE": rmse, "MAE": mae}
        predictions[model_name] = {"y_true_inv": y_true_inv, "y_pred_inv": y_pred_inv}

    return {"metrics_report": metrics_report, "predictions": predictions}

def main() -> None:
    """
    Main entry point for the integrated data loading, preprocessing, splitting,
    model building, training, evaluation, and visualization pipeline. This function
    loads data, processes it, builds LSTM and Transformer models, trains them,
    evaluates their performance, and visualizes the results.
    """
    # Specify the data path (can update as needed)
    data_path: str = 'NVDA.csv'  # Using a relative path for better portability

    df: Optional[pd.DataFrame] = None  # Initialize df to None to ensure it's defined
    processed_data: Optional[Dict[str, np.ndarray]] = None  # Initialize processed_data to None
    preprocessor: Optional[DataPreprocessor] = None # Initialize preprocessor to None

    try:
        # Step 1: Load the data using the imported function
        df = load_data_from_csv(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())

    except FileNotFoundError:
        print(f"File not found: {data_path}")
        # Using mock data for demonstration purposes if file is not found.
        print("\nCreating and using mock data for demonstration as the file was not found.")
        df = pd.DataFrame({
            "Date": pd.to_datetime([f"2020-01-{i:02d}" for i in range(1, 32)] +
                                   [f"2020-02-{i:02d}" for i in range(1, 30)] +
                                   [f"2020-03-{i:02d}" for i in range(1, 32)]),
            "Open": np.linspace(500, 550, 91),
            "High": np.linspace(505, 555, 91),
            "Low": np.linspace(495, 545, 91),
            "Close": np.linspace(503, 553, 91)
        })
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
        lstm_history, transformer_history = {}, {}

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
                if lstm_history and lstm_history.get('loss'): print(f"Final LSTM Training Loss: {lstm_history['loss'][-1]:.4f}")
                if lstm_history and lstm_history.get('val_loss'): print(f"Final LSTM Validation Loss: {lstm_history['val_loss'][-1]:.4f}")

                # Train Transformer Model
                print("\nTraining Transformer Model...")
                transformer_params = training_params.copy()
                transformer_params['checkpoint_path'] = 'best_Transformer_model.h5'
                trained_transformer_model, transformer_history = train_model(
                    model=transformer_model, X_train=processed_data['X_train'], y_train=processed_data['y_train'],
                    X_val=processed_data['X_val'], y_val=processed_data['y_val'], training_params=transformer_params
                )
                print("Transformer Model training complete.")
                if transformer_history and transformer_history.get('loss'): print(f"Final Transformer Training Loss: {transformer_history['loss'][-1]:.4f}")
                if transformer_history and transformer_history.get('val_loss'): print(f"Final Transformer Validation Loss: {transformer_history['val_loss'][-1]:.4f}")

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

                # Create results directory for plots
                os.makedirs('results', exist_ok=True)

                # --- Visualize Prediction Results (Integration of g_visualize_prediction_results_prompt) ---
                print("\n--- Visualizing Prediction Results ---")
                try:
                    predictions_data = evaluation_results.get('predictions')
                    if not predictions_data or not all(k in predictions_data for k in ["LSTM", "Transformer"]):
                        raise ValueError("Evaluation results missing 'predictions' data for plotting.")

                    y_true_inv = predictions_data['LSTM']['y_true_inv']
                    y_pred_lstm_inv = predictions_data['LSTM']['y_pred_inv']
                    y_pred_transformer_inv = predictions_data['Transformer']['y_pred_inv']

                    n_train = len(processed_data['X_train'])
                    n_val = len(processed_data['X_val'])
                    test_start_date_idx = n_train + n_val + preprocessor.sequence_length
                    
                    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
                        test_dates = df['Date'].iloc[test_start_date_idx:].reset_index(drop=True)
                    else:
                        test_dates = pd.Series(pd.date_range(start="today", periods=len(y_true_inv)), name="Date")

                    actual_prices_series = pd.Series(y_true_inv.flatten(), name='Actual')
                    predicted_lstm_array = np.array(y_pred_lstm_inv).flatten()
                    predicted_transformer_array = np.array(y_pred_transformer_inv).flatten()

                    # --- MODIFICATION: Save expected input data for the next module as CSV ---
                    try:
                        # Define the hardcoded directory and filename based on the specification
                        save_dir = r"C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated"
                        file_name = "input_for_visualize_prediction_results.csv"
                        output_csv_path = os.path.join(save_dir, file_name)

                        # Create the directory if it doesn't exist to prevent errors
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # Assemble the DataFrame in the format specified by the "Data Head (example)"
                        results_df = pd.DataFrame({
                            'date': test_dates,
                            'actual_price': y_true_inv.flatten(),
                            'predicted_price_lstm': predicted_lstm_array,
                            'predicted_price_transformer': predicted_transformer_array
                        })
                        
                        # Save the DataFrame to a CSV file, overwriting any existing file
                        results_df.to_csv(output_csv_path, index=False)
                        print(f"Successfully saved expected input data to: {output_csv_path}")

                    except (IOError, OSError) as e:
                        print(f"Error: Could not save the CSV file. Reason: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred during CSV saving: {e}")
                    # --- END OF MODIFICATION ---

                    plot_path = os.path.join('results', 'prediction_comparison_plot.png')
                    saved_plot_path = plot_predictions(
                        actual_prices=actual_prices_series, predicted_prices_lstm=predicted_lstm_array,
                        predicted_prices_transformer=predicted_transformer_array, dates=test_dates, output_path=plot_path
                    )
                    print(f"Prediction comparison plot saved successfully to: {saved_plot_path}")

                except Exception as e_viz:
                    print(f"An error occurred during prediction results visualization: {e_viz}")
                
                # --- Visualize Training History (Integration of h_visualize_training_history_prompt) ---
                print("\n--- Visualizing Training History ---")
                try:
                    # Plot and save LSTM history
                    if lstm_history:
                        lstm_history_plot_path = os.path.join('results', 'lstm_loss_history.png')
                        saved_path_lstm = plot_and_save_history(
                            model_history=lstm_history,
                            model_name="LSTM",
                            output_path=lstm_history_plot_path
                        )
                        print(f"LSTM training history plot saved successfully to: {saved_path_lstm}")
                    else:
                        print("LSTM history not available for plotting.")

                    # Plot and save Transformer history
                    if transformer_history:
                        transformer_history_plot_path = os.path.join('results', 'transformer_loss_history.png')
                        saved_path_transformer = plot_and_save_history(
                            model_history=transformer_history,
                            model_name="Transformer",
                            output_path=transformer_history_plot_path
                        )
                        print(f"Transformer training history plot saved successfully to: {saved_path_transformer}")
                    else:
                        print("Transformer history not available for plotting.")
                        
                except (KeyError, ValueError, IOError) as e_hist_viz:
                    print(f"An error occurred during training history visualization: {e_hist_viz}")
                except Exception as e_hist_viz_generic:
                    print(f"An unexpected generic error occurred during history visualization: {e_hist_viz_generic}")

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