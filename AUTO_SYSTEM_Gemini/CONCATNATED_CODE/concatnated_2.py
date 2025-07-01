from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
# main.py
# This script serves as the executable entry point for the data loading and preprocessing pipeline.
# It integrates the data loading functionality from 'a_load_user_provided_data_prompt'
# and the preprocessing logic from 'b_preprocess_data_prompt'.

import pandas as pd
import numpy as np

def main():
    """
    Main function to demonstrate the data loading and preprocessing pipeline.

    This entry point first loads data from a predefined path. If successful,
    it passes the data to the preprocessing module. It includes error
    handling for both stages and uses a fallback to mock data for the loading
    stage to ensure the script can demonstrate its flow even if the primary
    data source is unavailable.
    """
    # Per "Implementation Guidelines", use a predefined data path.
    # Using a raw string (r'...') to handle backslashes in Windows paths correctly.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\NVIDIA.csv'
    df = None  # Initialize df to ensure it's in scope

    print("--- Step 1: Data Loading Demonstration ---")
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
        # Note: This mock data has insufficient rows for the preprocessor's
        # default sequence length (60), which will demonstrate the
        # integrated error handling in the next step.
        mock_data = {
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'Open': [150.0, 152.5, 151.0, 155.0, 154.5],
            'High': [153.0, 153.5, 155.5, 156.0, 157.0],
            'Low': [149.5, 150.5, 150.0, 153.0, 154.0],
            'Close': [152.0, 151.5, 155.0, 154.0, 156.5],
            'Volume': [1000000, 1200000, 1100000, 1300000, 1250000] # Extra column to show it's preserved
        }
        df = pd.DataFrame(mock_data)

        print("\nMock data generated. DataFrame head:")
        print(df.head())

    except Exception as e:
        print(f"\nAn unexpected critical error occurred during data loading: {e}")
        print("Execution halted.")
        return # Halt if loading fails critically

    # --- Step 2: Data Preprocessing ---
    # This step proceeds if the 'df' DataFrame was successfully created,
    # either from the real file or the mock data.
    if df is not None:
        print("\n--- Step 2: Data Preprocessing Demonstration ---")
        try:
            # Instantiate the preprocessor from the imported module
            preprocessor = DataPreprocessor(sequence_length=60)
            
            # Run the full preprocessing pipeline on the loaded data
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
            # This block will catch errors raised by the preprocessor,
            # for example, if the mock data doesn't have enough rows to
            # create sequences. The preprocessor's internal logging will also fire.
            print(f"\nAn error occurred during data preprocessing: {e}")
            print("Preprocessing halted.")
    else:
        # This case would occur if a critical loading error happened but didn't exit the program.
        print("\nSkipping preprocessing as no data was successfully loaded.")


if __name__ == '__main__':
    # This block allows the script to be run directly to execute the main demonstration logic.
    main()