from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
from PACKAGE.c_split_dataset_prompt import split_sequential_data
# main.py
# This script serves as the executable entry point for the full data preparation pipeline.
# It integrates data loading, preprocessing, and dataset splitting.

import pandas as pd
import numpy as np

def main():
    """
    Main function to demonstrate the data loading, preprocessing, and splitting pipeline.

    This entry point first loads data, then preprocesses it into sequences,
    and finally splits the sequences into training, validation, and test sets.
    It includes error handling for all stages and uses a fallback to mock
    data for the loading stage to ensure the script can demonstrate its flow
    even if the primary data source is unavailable.
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
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'Open': [150.0, 152.5, 151.0, 155.0, 154.5],
            'High': [153.0, 153.5, 155.5, 156.0, 157.0],
            'Low': [149.5, 150.5, 150.0, 153.0, 154.0],
            'Close': [152.0, 151.5, 155.0, 154.0, 156.5],
            'Volume': [1000000, 1200000, 1100000, 1300000, 1250000]
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
        processed_data = None  # Initialize to ensure it's in scope
        print("\n--- Step 2: Data Preprocessing ---")
        try:
            # Instantiate the preprocessor from the imported module
            preprocessor = DataPreprocessor(sequence_length=60)
            
            # Run the full preprocessing pipeline.
            # Assuming the preprocessor returns a dictionary with 'X' and 'y' keys.
            processed_data = preprocessor.process(df)
            
            print("\nPreprocessing complete. Verifying output...")
            
            # Display the shapes and types of the preprocessed data arrays
            print("\n" + "="*20 + " Preprocessing Output " + "="*20)
            for key, value in processed_data.items():
                if isinstance(value, np.ndarray):
                    print(f"Key: {key:<10} | Shape: {str(value.shape):<20} | DType: {value.dtype}")
                else:
                    print(f"Key: {key:<10} | Type: {type(value)}")
            print("="*62)

        except (ValueError, Exception) as e:
            print(f"\nAn error occurred during data preprocessing: {e}")
            print("Preprocessing halted.")
    
    # --- Step 3: Data Splitting ---
    # This step proceeds if 'processed_data' was successfully created.
    if processed_data and 'X' in processed_data and 'y' in processed_data:
        print("\n--- Step 3: Data Splitting ---")
        try:
            # Extract feature (X) and target (y) arrays from the preprocessed dictionary
            X, y = processed_data['X'], processed_data['y']

            # Call the imported splitting function
            split_datasets = split_sequential_data(X, y, split_ratios=(0.8, 0.1, 0.1))

            print("\nData splitting complete. Verifying output...")
            print("\n" + "="*20 + " Splitting Output " + "="*20)
            for name, data in split_datasets.items():
                if isinstance(data, np.ndarray):
                    print(f"Key: {name:<10} | Shape: {str(data.shape):<20} | DType: {data.dtype}")
            print("="*58)

        except (ValueError, TypeError) as e:
            # This block will catch errors raised by the splitter, for example,
            # if the preprocessed data doesn't have enough samples to split.
            print(f"\nAn error occurred during data splitting: {e}")
            print("Data splitting halted.")
    else:
        # This case occurs if preprocessing failed or did not produce the expected output.
        print("\nSkipping data splitting as preprocessed data is not available or invalid.")


if __name__ == '__main__':
    # This block allows the script to be run directly to execute the main pipeline logic.
    main()