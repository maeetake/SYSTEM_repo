# main.py
# This script serves as the executable entry point for the data loading process.
# It imports the core data loading functionality from the 'a_load_user_provided_data_prompt'
# module and uses it to perform a data loading demonstration.

import pandas as pd
import os
# As per the requirements, the original script is treated as a module within a 'PACKAGE' directory.
from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv

# --- Start of Added Code ---

# Hardcoded directory and filename for the output CSV file as per specifications.
SAVE_DIRECTORY = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\generated'
SAVE_FILENAME = 'expected_input.csv'

def save_expected_input_for_next_module(df: pd.DataFrame):
    """
    Saves the data expected by the next module to a CSV file.

    Based on the specification, the next module in the pipeline requires
    the 'Date', 'Open', 'High', 'Low', and 'Close' columns for time-series
    modeling. This function filters the DataFrame for these columns and saves
    it to a predefined location.

    Args:
        df (pd.DataFrame): The raw DataFrame loaded from the source file.
    """
    # Determine the columns expected by the next module from the specification.
    # The constraint is: "Only Open, High, Low, Close (OHLC) data from the
    # file should be used as input features."
    expected_columns = ['Date', 'Open', 'High', 'Low', 'Close']

    # Verify that all expected columns are present in the loaded DataFrame.
    if not all(col in df.columns for col in expected_columns):
        print(f"\nWarning: Could not save expected input file. The loaded data is missing one or more required columns: {expected_columns}.")
        return

    try:
        # Create the target directory if it does not exist.
        os.makedirs(SAVE_DIRECTORY, exist_ok=True)

        # Define the full path for the output file.
        output_path = os.path.join(SAVE_DIRECTORY, SAVE_FILENAME)

        # Filter the DataFrame to include only the necessary columns.
        filtered_df = df[expected_columns]

        # Save the filtered DataFrame to a CSV file.
        # index=False prevents pandas from writing row indices into the file.
        # The file will be overwritten if it already exists.
        filtered_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved the expected input data for the next module to:\n{output_path}")

    except (IOError, PermissionError) as e:
        print(f"\nError: Failed to save the expected input file. Reason: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the file saving process: {e}")

# --- End of Added Code ---


def main():
    """
    Main function to demonstrate the usage of the load_data_from_csv module.

    This entry point automatically loads data from a predefined path,
    simulating the start of a data processing pipeline. It includes error
    handling and a fallback to mock data for demonstration purposes, ensuring
    the script can run even if the primary data source is unavailable.
    """
    # Per "Implementation Guidelines", use a predefined data path.
    # Using a raw string (r'...') to handle backslashes in Windows paths correctly.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\NVIDIA.csv'

    print("--- Data Loading Demonstration ---")
    try:
        # Attempt to load the primary dataset by calling the imported function
        df = load_data_from_csv(data_path)
        print("\nData loaded successfully. Raw DataFrame head:")
        print(df.head())

        # --- Start of Added Code ---
        # Call the new function to save the data required by the next module.
        # This is only executed if the data loading is successful.
        save_expected_input_for_next_module(df)
        # --- End of Added Code ---

    except (FileNotFoundError, ValueError, KeyError) as e:
        # Per "main Function Instructions", handle errors with informative messages
        # and provide mock data as a fallback.
        print(f"\nAn error occurred while loading the primary data: {e}")
        print("---")
        print("Proceeding with mock data for demonstration purposes.")

        # Create a mock DataFrame that mimics the expected structure
        mock_data = {
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'Open': [150.0, 152.5, 151.0, 155.0, 154.5],
            'High': [153.0, 153.5, 155.5, 156.0, 157.0],
            'Low': [149.5, 150.5, 150.0, 153.0, 154.0],
            'Close': [152.0, 151.5, 155.0, 154.0, 156.5],
            'Volume': [1000000, 1200000, 1100000, 1300000, 1250000] # Extra column to show it's preserved
        }
        mock_df = pd.DataFrame(mock_data)

        print("\nMock data generated. DataFrame head:")
        print(mock_df.head())
        # In a real pipeline, you would return or use this mock_df
        # for subsequent processing steps.
        # The saving feature is not triggered for mock data.

    except Exception as e:
        print(f"\nAn unexpected critical error occurred: {e}")
        print("Execution halted.")


if __name__ == '__main__':
    # This block allows the script to be run directly to execute the main demonstration logic.
    main()