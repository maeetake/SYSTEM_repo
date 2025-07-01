# main.py
# This script serves as an executable entry point that utilizes the
# 'a_load_user_provided_data_prompt' module.

# Standard library imports
import os # Added for file path and directory operations

# Third-party Libraries
# pandas is required here because the main function's exception handling
# creates a mock DataFrame as a fallback.
import pandas as pd

# Assuming 'a_load_user_provided_data_prompt.py' is in a 'PACKAGE' directory,
# we import the necessary function and classes.
# Note: The original script's exception types (FileNotFoundError, ValueError, KeyError)
# are built-in, so they don't need to be imported from the module.
from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv

# --- Start of Added Code ---

def _save_expected_input_for_next_module(df: pd.DataFrame) -> None:
    """
    Saves the data expected by the next module (e.g., preprocessing) to a CSV file.

    Based on the specification document, the subsequent modules are constrained
    to use only the 'Date', 'Open', 'High', 'Low', and 'Close' columns.
    This function filters the input DataFrame to include only these columns and
    saves the result to a predefined, hardcoded location.

    Args:
        df (pd.DataFrame): The raw DataFrame loaded from the source file.
    """
    # Hardcode the file save directory as specified in the requirements
    save_dir = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated'
    # Determine and hardcode an appropriate file name and extension
    file_name = 'expected_input_for_preprocessing.csv'
    output_path = os.path.join(save_dir, file_name)

    # Determine the required columns from the specification for the next module.
    # The 'constraints' and 'data_preprocessing' sections indicate that only
    # OHLC data plus the 'Date' for context are used.
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']

    # Defensive check: ensure the loaded data contains all necessary columns.
    # Although 'load_data_from_csv' already validates this, this check ensures
    # this function's logic is self-contained and robust.
    if not all(col in df.columns for col in required_columns):
        print(f"Warning: Could not save expected input. One or more required columns "
              f"({required_columns}) not found in the loaded data. No file will be saved.")
        return

    # Create a new DataFrame containing only the data expected by the next module.
    expected_input_df = df[required_columns]

    try:
        # Create the target directory if it does not exist, ensuring no errors are raised if it does.
        os.makedirs(save_dir, exist_ok=True)

        # Save the DataFrame to a CSV file.
        # This will overwrite the file if it already exists.
        # index=False prevents pandas from writing the DataFrame index as a column.
        expected_input_df.to_csv(output_path, index=False)
        print(f"Successfully saved the expected input for the next module to: {output_path}")

    except (IOError, OSError) as e:
        # Handle potential file system errors during directory creation or file write.
        print(f"Error: Failed to save the file to '{output_path}'. Reason: {e}")
    except Exception as e:
        # Catch any other unexpected errors during the saving process.
        print(f"An unexpected error occurred during the file saving process: {e}")

# --- End of Added Code ---

def main() -> None:
    """
    Main entry point for module demonstration and validation.
    This function replicates the original script's main logic by calling
    the imported 'load_data_from_csv' function.
    """
    # Specify the data path (can update as needed)
    # This path is identical to the one in the original script to ensure
    # the same behavior.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\NVIDIA.csv'

    try:
        # Call the imported function to load the data
        df = load_data_from_csv(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())

        # --- Start of Added Code ---
        # After successfully loading the data, save the expected input for the next module.
        _save_expected_input_for_next_module(df)
        # --- End of Added Code ---

    except FileNotFoundError as fnf_err:
        # This exception is raised by the imported module but handled here.
        print(f"File not found: {fnf_err}")
        # Optional: Provide mock OHLC data for testing/fallback
        # This logic is preserved from the original script.
        mock_data = pd.DataFrame({
            "Date": ["2024/01/01", "2024/01/02", "2024/01/03"],
            "Open": [500.0, 505.0, 510.0],
            "High": [505.0, 510.0, 515.0],
            "Low":  [495.0, 500.0, 505.0],
            "Close": [503.0, 508.0, 512.0]
        })
        print("Using mock data:")
        print(mock_data.head())
    except (ValueError, KeyError) as data_err:
        # These exceptions are raised by the imported module but handled here.
        print(f"An error occurred while loading the data: {data_err}")
    except Exception as e:
        # A general catch-all for any other unexpected errors.
        print(f"An unexpected error occurred: {e}")

# Ensures main() only runs when this file is executed directly.
if __name__ == "__main__":
    main()