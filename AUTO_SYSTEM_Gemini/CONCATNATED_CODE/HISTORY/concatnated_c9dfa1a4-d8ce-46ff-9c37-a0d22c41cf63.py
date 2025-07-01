from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
from PACKAGE.d_build_lstm_and_transformer_models_prompt import build_lstm_model, build_transformer_model
# main.py
# This script serves as the executable entry point for the full data preparation and model building pipeline.
# It integrates data loading, preprocessing, dataset splitting, and model architecture definition.

import pandas as pd
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the data loading, preprocessing, splitting, and model building pipeline.

    This entry point first loads data, then preprocesses it into sequences,
    splits the sequences into training, validation, and test sets, and finally
    builds the LSTM and Transformer models based on the data shape. It includes
    error handling for all stages and uses a fallback to mock data for the
    loading stage to ensure the script can demonstrate its flow even if the
    primary data source is unavailable.
    """
    # Per "Implementation Guidelines", use a predefined data path.
    # Using a raw string (r'...') to handle backslashes in Windows paths correctly.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\NVIDIA.csv'
    df = None  # Initialize df to ensure it's in scope

    print("--- Step 1: Data Loading ---")
    try:
        # Attempt to load the primary dataset by calling the imported function
        df = load_data_from_csv(data_path)
        print("\nData loaded successfully. Raw DataFrame head:")
        print(df.head())

    except (FileNotFoundError, ValueError, KeyError) as e:
        # Per "main Function Instructions", handle errors with informative messages
        # and provide mock data as a fallback.
        print(f"\nAn error occurred while loading the primary data: {e}")
        print("---")
        print("Proceeding with mock data for demonstration purposes.")

        # Create a mock DataFrame that mimics the expected structure.
        mock_data = {
            'Date': pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 91)]),
            'Open': np.linspace(150, 200, 90),
            'High': np.linspace(152, 205, 90),
            'Low': np.linspace(148, 198, 90),
            'Close': np.linspace(151, 202, 90),
            'Volume': np.linspace(1000000, 1500000, 90, dtype=int)
        }
        df = pd.DataFrame(mock_data)

        print("\nMock data generated. DataFrame head:")
        print(df.head())

    except Exception as e:
        print(f"\nAn unexpected critical error occurred during data loading: {e}")
        print("Execution halted.")
        return # Halt if loading fails critically

    # This section proceeds if the 'df' DataFrame was successfully created.
    if df is not None:
        final_datasets = None  # Initialize to ensure it's in scope
        # The DataPreprocessor module handles both preprocessing and splitting.
        print("\n--- Step 2: Data Preprocessing and Splitting ---")
        try:
            # Instantiate the preprocessor from the imported module
            preprocessor = DataPreprocessor(sequence_length=60)
            
            # Run the full preprocessing pipeline.
            # The preprocessor returns a dictionary with already-split datasets.
            final_datasets = preprocessor.process(df)
            
            print("\nPreprocessing and splitting complete. Verifying output...")
            
            # Display the shapes and types of the final dataset arrays
            print("\n" + "="*20 + " Final Datasets Output " + "="*20)
            for key, value in final_datasets.items():
                if isinstance(value, np.ndarray):
                    print(f"Key: {key:<10} | Shape: {str(value.shape):<20} | DType: {value.dtype}")
                else:
                    print(f"Key: {key:<10} | Type: {type(value)}")
            print("="*64)

        except (ValueError, Exception) as e:
            print(f"\nAn error occurred during data preprocessing and splitting: {e}")
            print("Pipeline halted.")
            final_datasets = None # Ensure it is None on failure

        # --- Step 3: Model Building ---
        # This step proceeds only if preprocessing was successful.
        if final_datasets:
            print("\n--- Step 3: Model Building ---")
            try:
                # Determine the input shape from the preprocessed training data.
                # Shape is (sequence_length, num_features).
                input_shape = final_datasets['X_train'].shape[1:]
                print(f"Determined model input shape: {input_shape}")

                # Build the LSTM model
                print("\nBuilding LSTM Model...")
                lstm_model = build_lstm_model(input_shape)
                print("LSTM Model Summary:")
                lstm_model.summary()

                # Build the Transformer model
                print("\nBuilding Transformer Model...")
                transformer_model = build_transformer_model(input_shape)
                print("Transformer Model Summary:")
                transformer_model.summary()

            except (ValueError, KeyError, Exception) as e:
                print(f"\nAn error occurred during model building: {e}")
                print("Model building halted.")
        else:
            print("\nSkipping model building due to preprocessing failure.")
    
    else:
        # This case occurs if data loading (Step 1) failed and the mock data
        # fallback also failed.
        print("\nData loading failed. Cannot proceed with preprocessing and model building.")


if __name__ == '__main__':
    # This block allows the script to be run directly to execute the main pipeline logic.
    main()