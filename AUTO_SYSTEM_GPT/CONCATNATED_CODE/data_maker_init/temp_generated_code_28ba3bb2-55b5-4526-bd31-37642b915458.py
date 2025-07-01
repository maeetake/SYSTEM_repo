from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
from PACKAGE.c_split_dataset_prompt import split_sequential_data
from PACKAGE.d_build_lstm_and_transformer_models_prompt import build_lstm_model, build_transformer_model
from PACKAGE.e_train_models_prompt import train_model
from PACKAGE.f_evaluate_model_performance_prompt import evaluate_model_performance

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Any, Dict, Optional, List
import os

# ADDED: Function to save the expected input for the evaluation module as a CSV file.
def save_evaluation_input_as_csv(X_test: np.ndarray, y_test: np.ndarray, sequence_length: int, feature_cols: List[str], save_path: str):
    """
    Saves the input data for the evaluation module (X_test, y_test) to a CSV file,
    formatting it based on the specification's example data structure.

    Args:
        X_test (np.ndarray): The test features (n_samples, seq_len, n_features).
        y_test (np.ndarray): The test targets (n_samples,).
        sequence_length (int): The length of the input sequences.
        feature_cols (List[str]): The names of the features (e.g., ['Open', 'High', 'Low', 'Close']).
        save_path (str): The full path where the CSV file will be saved.
    """
    try:
        # Ensure the input data is valid before attempting to save.
        if not all([
            isinstance(X_test, np.ndarray),
            isinstance(y_test, np.ndarray),
            X_test.ndim == 3,
            y_test.ndim == 1,
            X_test.shape[0] == y_test.shape[0],
            X_test.size > 0
        ]):
            print("Warning: Skipping CSV save. Input data for evaluation is not valid, correctly shaped, or is empty.")
            return

        # Ensure the save directory exists.
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        # Reshape the 3D X_test array into a 2D array for the DataFrame.
        n_samples = X_test.shape[0]
        n_features = len(feature_cols)
        X_test_reshaped = X_test.reshape(n_samples, sequence_length * n_features)

        # Create column headers as seen in the specification example (e.g., 'Open_t-59', 'Close_t-0').
        # The timesteps are ordered from the most distant past (t-59) to the most recent (t-0).
        headers = []
        for i in range(sequence_length - 1, -1, -1):
            for col_name in feature_cols:
                headers.append(f"{col_name}_t-{i}")

        # Create the DataFrame from the reshaped test data and the generated headers.
        df_to_save = pd.DataFrame(X_test_reshaped, columns=headers)

        # Insert the 'target' (y_test) and 'split' columns at the beginning, as per the specification's format.
        df_to_save.insert(0, 'target', y_test)
        df_to_save.insert(0, 'split', 'test')

        # Save the DataFrame to a CSV file, overwriting it if it already exists.
        df_to_save.to_csv(save_path, index=False)
        print(f"Successfully saved the expected input for the evaluation module to: {save_path}")

    except (OSError, IOError) as e:
        print(f"Error: Could not write file to disk at {save_path}. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the evaluation input CSV: {e}")
# END of ADDED section

def main() -> None:
    """
    Main entry point for the integrated data loading, preprocessing, splitting,
    model building, training, and evaluation pipeline. This function loads data,
    processes it, builds LSTM and Transformer models, trains them, evaluates their
    performance, and finally demonstrates the standalone splitting functionality.
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
        # ADDED: Save the input for the evaluation module as a CSV
        # The specification for 'evaluate_model_performance' indicates its inputs are X_test and y_test.
        # This section saves that data to a CSV in the specified format and location.
        save_dir = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated'
        file_name = 'expected_input_for_evaluation.csv'
        save_path = os.path.join(save_dir, file_name)
        
        save_evaluation_input_as_csv(
            X_test=processed_data['X_test'],
            y_test=processed_data['y_test'],
            sequence_length=preprocessor.sequence_length,
            feature_cols=preprocessor.feature_cols,
            save_path=save_path
        )
        # END of ADDED section

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
            # Compile LSTM model
            lstm_model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )

            # Compile Transformer model
            transformer_model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
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

            try:
                # Train LSTM Model
                print("\nTraining LSTM Model...")
                lstm_params = training_params.copy()
                lstm_params['checkpoint_path'] = 'best_LSTM_model.h5'

                assert processed_data['X_train'].shape[0] == processed_data['y_train'].shape[0], "Mismatch in LSTM train data shapes"
                assert processed_data['X_val'].shape[0] == processed_data['y_val'].shape[0], "Mismatch in LSTM val data shapes"

                trained_lstm_model, lstm_history = train_model(
                    model=lstm_model,
                    X_train=processed_data['X_train'],
                    y_train=processed_data['y_train'],
                    X_val=processed_data['X_val'],
                    y_val=processed_data['y_val'],
                    training_params=lstm_params
                )
                print("LSTM Model training complete.")
                if lstm_history.get('loss'):
                    print(f"Final LSTM Training Loss: {lstm_history['loss'][-1]:.4f}")
                if lstm_history.get('val_loss'):
                    print(f"Final LSTM Validation Loss: {lstm_history['val_loss'][-1]:.4f}")

                # Train Transformer Model
                print("\nTraining Transformer Model...")
                transformer_params = training_params.copy()
                transformer_params['checkpoint_path'] = 'best_Transformer_model.h5'

                assert processed_data['X_train'].shape[0] == processed_data['y_train'].shape[0], "Mismatch in Transformer train data shapes"
                assert processed_data['X_val'].shape[0] == processed_data['y_val'].shape[0], "Mismatch in Transformer val data shapes"

                trained_transformer_model, transformer_history = train_model(
                    model=transformer_model,
                    X_train=processed_data['X_train'],
                    y_train=processed_data['y_train'],
                    X_val=processed_data['X_val'],
                    y_val=processed_data['y_val'],
                    training_params=transformer_params
                )
                print("Transformer Model training complete.")
                if transformer_history.get('loss'):
                    print(f"Final Transformer Training Loss: {transformer_history['loss'][-1]:.4f}")
                if transformer_history.get('val_loss'):
                    print(f"Final Transformer Validation Loss: {transformer_history['val_loss'][-1]:.4f}")

            except Exception as e_train:
                print(f"An unexpected error occurred during model training: {e_train}")
        
        # Step 5: Evaluate the trained models on the test set
        if trained_lstm_model and trained_transformer_model:
            print("\n--- Starting Model Evaluation ---")
            try:
                models_to_evaluate = {
                    "LSTM": trained_lstm_model,
                    "Transformer": trained_transformer_model
                }
                
                # Get the scaler from the preprocessor instance
                scaler = preprocessor.scaler
                # Determine the index of the 'Close' column for inverse transformation
                close_feature_index = preprocessor.feature_cols.index('Close')

                # Call the evaluation function from the imported module
                evaluation_results = evaluate_model_performance(
                    models=models_to_evaluate,
                    X_test=processed_data['X_test'],
                    y_test=processed_data['y_test'],
                    scaler=scaler,
                    close_index=close_feature_index
                )

                # Print the evaluation report
                print("\nEvaluation Metrics (RMSE, MAE):")
                metrics_report = evaluation_results.get('metrics_report', {})
                for model_name, metrics in metrics_report.items():
                    print(f"  {model_name}:")
                    print(f"    RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
                    print(f"    MAE: {metrics.get('MAE', 'N/A'):.4f}")

                # Note: Visualization is specified but not implemented in the provided scripts.
                # Per constraints, unimplemented features are omitted.
                
            except Exception as e_eval:
                print(f"An unexpected error occurred during model evaluation: {e_eval}")

        # Step 6: Demonstrate the standalone data splitting function
        print("\n--- Demonstrating Standalone Data Splitting from 'c_split_dataset_prompt' ---")
        try:
            # Reconstruct the full, unsplit dataset from the preprocessor's output
            X_full = np.concatenate([
                processed_data['X_train'],
                processed_data['X_val'],
                processed_data['X_test']
            ], axis=0)
            y_full = np.concatenate([
                processed_data['y_train'],
                processed_data['y_val'],
                processed_data['y_test']
            ], axis=0)

            assert X_full.shape[0] == y_full.shape[0], "Mismatch in full dataset for splitting demonstration"

            print(f"Reconstructed full dataset for demonstration: X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")

            split_ratios = (0.8, 0.1, 0.1)
            split_datasets_demo = split_sequential_data(X_full, y_full, split_ratios=split_ratios)

            print("\nData splitting using imported 'split_sequential_data' function successful.")
            print("Shapes of the new splits:")
            for key, value in split_datasets_demo.items():
                print(f"  {key}: {value.shape}")

        except ValueError as ve_split:
            print(f"Error during standalone splitting demonstration: {ve_split}")
        except Exception as e_split:
            print(f"An unexpected error occurred during standalone splitting demonstration: {e_split}")


if __name__ == "__main__":
    main()