# main.py
# Executable Python script that replicates the functionality of a_load_user_provided_data_prompt.py

import sys
import os

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
    # Import the main function from the module and alias it to avoid name conflicts.
    from PACKAGE.a_load_user_provided_data_prompt import main as process_data_main
except ImportError as e:
    print(f"Error: {e}")
    print("Please ensure that the script is run from a directory containing the 'PACKAGE' folder,")
    print("and that 'PACKAGE' contains 'a_load_user_provided_data_prompt.py' and an '__init__.py' file.")
    sys.exit(1)

def main():
    """
    Main entry point for the executable script.
    
    This function calls the main logic from the imported module, ensuring that the
    original script's functionality is replicated exactly.
    """
    # Call the main function from the imported module. This will execute the
    # entire data loading and validation process, including all printouts
    # and logging from the original script.
    process_data_main()

if __name__ == "__main__":
    # Execute the main function when the script is run directly.
    main()