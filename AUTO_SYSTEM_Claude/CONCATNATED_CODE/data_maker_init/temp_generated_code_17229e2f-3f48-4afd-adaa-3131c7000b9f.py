# main.py
# Executable Python script that replicates the functionality of a_load_user_provided_data_prompt.py

import sys
import os
# ADDED: Import pandas to handle the DataFrame for the new save feature.
import pandas as pd

# To make the import work, we need to add the parent directory of 'PACKAGE' to the Python path.
# This assumes the script is run from a location where 'PACKAGE' is a subdirectory.
# For a more robust solution, the package would be installed, but for this self-contained example,
# modifying the path is a common approach.
# We'll assume the structure is something like:
# /some/path/
# ├── main.py
# └── PACKAGE/
#     └── a_load_user_provided_data_prompt.py
#     └── __init__.py
# In this case, the current directory '.' is already in sys.path.
# If the structure is different, this path adjustment might be necessary.
# For example, if PACKAGE is in a 'src' folder: sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    # MODIFICATION: Import the specific data loading function instead of the main entry point.
    # This allows us to get the DataFrame object and then add the saving functionality.
    # The function 'load_data_from_csv' is derived from the specification document's
    # 'Module Definition' section.
    from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
except ImportError as e:
    print(f"Error: {e}")
    # MODIFIED: Updated error message to reflect the new import.
    print("Please ensure that 'load_data_from_csv' is defined in 'PACKAGE/a_load_user_provided_data_prompt.py',")
    print("and that the script is run from a directory containing the 'PACKAGE' folder.")
    sys.exit(1)

# ADDED: A new function to handle saving the data to CSV.
def save_expected_input_for_next_module(df: pd.DataFrame):
    """
    Saves the provided DataFrame to a CSV file.

    This function saves the data that is expected as input by the next module
    in the pipeline (e.g., a preprocessing or training module). The output
    directory and filename are hardcoded as per the requirements.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
    """
    # Hardcode the save directory as specified.
    output_dir = r"C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Claude\UNITTEST_DATA\generated"
    # Determine and hardcode an appropriate file name. Based on the spec, this raw data
    # is the input for the preprocessing/training module.
    file_name = "expected_input_for_preprocessing.csv"
    output_path = os.path.join(output_dir, file_name)

    try:
        # Create the target directory if it doesn't exist to prevent errors.
        os.makedirs(output_dir, exist_ok=True)

        # Save the DataFrame to a CSV file. The file will be overwritten if it exists.
        # The index is not saved as it's typically not part of the expected data schema.
        df.to_csv(output_path, index=False)
        print(f"Successfully saved the expected input for the next module to: {output_path}")

    except (OSError, IOError) as e:
        print(f"Error: Could not save the file to {output_path}. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during file saving: {e}")

def main():
    """
    Main entry point for the executable script.
    
    This function has been modified to directly call the data loading function,
    replicate the original script's output (printing the head of the data),
    and then add the new functionality of saving the loaded data to a CSV file.
    """
    # The specification document provides this hardcoded path in its "main Function Instructions".
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Claude\UNITTEST_DATA\NVIDIA.csv'
    
    try:
        # Step 1: Load the data using the imported function.
        # This replicates the core action of the original script's main function.
        df = load_data_from_csv(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())

        # Step 2: ADDED - Save the loaded DataFrame.
        # This new feature saves the data that the next module in the pipeline
        # is expected to receive. This action is only performed if the data
        # is loaded successfully.
        save_expected_input_for_next_module(df)

    except (FileNotFoundError, ValueError, KeyError) as e:
        # Replicating the error handling from the original script's logic.
        print(f"An error occurred during data loading: {e}")
        print("Data loading failed. No file will be saved.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Execute the main function when the script is run directly.
    main()