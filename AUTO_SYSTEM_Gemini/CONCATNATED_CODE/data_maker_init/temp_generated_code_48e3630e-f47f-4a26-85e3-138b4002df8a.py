from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
# from PACKAGE.c_split_dataset_prompt import split_sequential_data # This is no longer needed as the preprocessor handles splitting.
# main.py
# This script serves as the executable entry point for the full data preparation pipeline.
# It integrates data loading, preprocessing, and dataset splitting.

import pandas as pd
import numpy as np
import os
from typing import Optional

# --- ADDED CODE START ---

def save_expected_input_for_next_module(df: Optional[pd.DataFrame]):
    """
    Saves the input DataFrame to a CSV file as expected by the next module.

    Based on the specification document, the next module expects a CSV file.
    This function saves the provided DataFrame to a hardcoded path and filename
    derived from the specification.

    - The save directory is hardcoded as required.
    - The filename 'expected_input.csv' is chosen based on the 'Data Path'
      in the specification.
    - The directory is created if it doesn't exist.
    - Errors during file writing are handled.
    - The file is only saved if the DataFrame is valid.

    Args:
        df (Optional[pd.DataFrame]): The DataFrame to be saved. If None or empty,
                                     the function will not save anything.
    """
    # Hardcode the file save directory as specified.
    save_dir = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\generated'
    # Determine the file name and extension based on the specification document.
    file_name = 'expected_input.csv'
    full_path = os.path.join(save_dir, file_name)

    # Save the file only if the expected input data can be correctly determined.
    if df is None or df.empty:
        print("\n[Save Feature] Input data is missing or empty. Skipping CSV file saving.")
        return

    try:
        # Add a process to create the directory if necessary.
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the DataFrame to a CSV file.
        # Use index=False as the example data in the specification does not include an index column.
        # This will overwrite any existing file, ensuring only one file per execution.
        df.to_csv(full_path, index=False)
        print(f"\n[Save Feature] Successfully saved the expected input for the next module to: {full_path}")

    except (OSError, PermissionError) as e:
        # Implement appropriate error handling for file processing.
        print(f"\n[Save Feature] Error: Could not create directory or write to file at {full_path}. Reason: {e}")
    except Exception as e:
        print(f"\n[Save Feature] An unexpected error occurred while saving the CSV file: {e}")

# --- ADDED CODE END ---


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
        # --- ADDED CODE START ---
        # Save the loaded DataFrame as a CSV file, which is the expected input
        # format for the next module as described in the specification.
        save_expected_input_for_next_module(df)
        # --- ADDED CODE END ---

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

        # --- Step 3: Data Splitting (Removed) ---
        # This step is no longer necessary as the `DataPreprocessor` module
        # now handles the data splitting internally and returns the final
        # training, validation, and test sets. The final datasets have already
        # been displayed in the output of Step 2.
    
    else:
        # This case occurs if data loading (Step 1) failed and the mock data
        # fallback also failed.
        print("\nData loading failed. Cannot proceed with preprocessing and splitting.")


if __name__ == '__main__':
    # This block allows the script to be run directly to execute the main pipeline logic.
    main()