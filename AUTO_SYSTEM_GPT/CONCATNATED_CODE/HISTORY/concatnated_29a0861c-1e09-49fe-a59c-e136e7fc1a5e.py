from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
from PACKAGE.c_split_dataset_prompt import split_sequential_data
from PACKAGE.d_build_lstm_and_transformer_models_prompt import build_lstm_model, build_transformer_model
# main.py
# This script serves as an executable entry point that utilizes the
# 'a_load_user_provided_data_prompt', 'b_preprocess_data_prompt',
# 'c_split_dataset_prompt', and 'd_build_lstm_and_transformer_models_prompt' modules.

# Third-party Libraries
import pandas as pd
import numpy as np # Required for handling numpy arrays and concatenation
import tensorflow as tf # Required for building and summarizing models

def main() -> None:
    """
    Main entry point for the integrated data loading, preprocessing, splitting,
    and model building pipeline. This function loads data, processes it, builds
    LSTM and Transformer models based on the processed data shape, and finally
    demonstrates the standalone splitting functionality.
    """
    # Specify the data path (can update as needed)
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\NVIDIA.csv'

    df = None  # Initialize df to None to ensure it's defined
    processed_data = None # Initialize processed_data to None

    try:
        # Step 1: Load the data using the imported function
        df = load_data_from_csv(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())

    except FileNotFoundError as fnf_err:
        # This exception is raised by the data loading module but handled here.
        print(f"File not found: {fnf_err}")
        # Using mock data for demonstration purposes if file is not found.
        df = pd.DataFrame({
            "Date": pd.to_datetime([f"2024-01-{i:02d}" for i in range(1, 91)]),
            "Open": np.linspace(500, 550, 90),
            "High": np.linspace(505, 555, 90),
            "Low":  np.linspace(495, 545, 90),
            "Close": np.linspace(503, 553, 90)
        })
        print("\nUsing mock data for demonstration:")
        print(df.head())

    except (ValueError, KeyError) as data_err:
        # These exceptions are raised by the data loading module but handled here.
        print(f"An error occurred while loading the data: {data_err}")
        return

    except Exception as e:
        # A general catch-all for any other unexpected errors during data loading.
        print(f"An unexpected error occurred during data loading: {e}")
        return

    # If df was successfully loaded (either from file or mock data), proceed to preprocessing.
    if df is not None:
        print("\n--- Starting Data Preprocessing ---")
        # Instantiate the preprocessor with a sequence length of 60
        preprocessor = DataPreprocessor(sequence_length=60)
        try:
            # Step 2: Process the loaded dataframe using the imported preprocessor
            # This step normalizes, creates sequences, and splits the data.
            processed_data = preprocessor.process(df)
            print("Data preprocessing successful.")
            print(f"X_train shape: {processed_data['X_train'].shape}")
            print(f"y_train shape: {processed_data['y_train'].shape}")
            print(f"X_val shape: {processed_data['X_val'].shape}")
            print(f"y_val shape: {processed_data['y_val'].shape}")
            print(f"X_test shape: {processed_data['X_test'].shape}")
            print(f"y_test shape: {processed_data['y_test'].shape}")

        except ValueError as ve:
            # Handle errors from the preprocessing step (e.g., not enough data, missing columns)
            print(f"Preprocessing Error: {ve}")
        except Exception as e:
            # A general catch-all for any other unexpected errors during preprocessing.
            print(f"An unexpected error occurred during preprocessing: {e}")

    # If preprocessing was successful, build models and then demonstrate the splitter.
    if processed_data:
        # Step 3: Build the deep learning models using the imported functions
        print("\n--- Starting Model Building ---")
        try:
            # Derive input_shape from the preprocessed training data.
            # The shape of X_train is (samples, sequence_length, features),
            # but the model needs an input_shape of (sequence_length, features).
            input_shape = processed_data['X_train'].shape[1:]
            print(f"Building models with input shape: {input_shape}")

            # Build the LSTM model
            lstm_model = build_lstm_model(input_shape)
            print("\nLSTM model built successfully. Summary:")
            lstm_model.summary()

            # Build the Transformer model
            transformer_model = build_transformer_model(input_shape)
            print("\nTransformer model built successfully. Summary:")
            transformer_model.summary()

        except ValueError as ve_model:
            print(f"Model Building Error: {ve_model}")
        except Exception as e_model:
            print(f"An unexpected error occurred during model building: {e_model}")


        # Step 4: Demonstrate the standalone data splitting function
        print("\n--- Demonstrating Standalone Data Splitting from 'c_split_dataset_prompt' ---")
        try:
            # Reconstruct the full, unsplit dataset from the preprocessor's output
            # to provide as input to the standalone split function.
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
            print(f"Reconstructed full dataset for demonstration: X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")

            # Use the imported function from 'c_split_dataset_prompt' to split the data.
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


# Ensures main() only runs when this file is executed directly.
if __name__ == "__main__":
    main()