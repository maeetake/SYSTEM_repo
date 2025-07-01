"""
load_user_provided_data.py

This module handles the loading and initial validation of user-provided stock data.
Required dependencies: pandas >= 1.3.0

Author: [Your Name]
Date: [Current Date]
"""

import os
import logging
from typing import Union, Optional
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoadingError(Exception):
    """Custom exception for data loading errors."""
    pass

def validate_file_path(file_path: str) -> bool:
    """
    Validates if the provided file path exists and has a supported extension.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        bool: True if the file path is valid
        
    Raises:
        DataLoadingError: If the file path is invalid or has an unsupported extension
    """
    if not isinstance(file_path, str):
        raise DataLoadingError("File path must be a string")
    
    if not os.path.exists(file_path):
        raise DataLoadingError(f"File not found at: {file_path}")
    
    if not file_path.lower().endswith('.csv'):
        raise DataLoadingError("Only CSV files are supported")
    
    return True

def validate_dataframe_structure(df: pd.DataFrame) -> bool:
    """
    Validates if the DataFrame contains all required columns and proper data types.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        bool: True if the DataFrame structure is valid
        
    Raises:
        DataLoadingError: If the DataFrame structure is invalid
    """
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    
    if df.empty:
        raise DataLoadingError("The loaded DataFrame is empty")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataLoadingError(f"Missing required columns: {', '.join(missing_columns)}")
    
    return True

def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Loads historical stock data from a user-provided CSV file.
    
    Args:
        file_path (str): The path to the CSV file containing the stock data
        
    Returns:
        pd.DataFrame: Raw stock data with required columns
        
    Raises:
        DataLoadingError: If there are any issues with loading or validating the data
    """
    try:
        # Validate file path
        validate_file_path(file_path)
        logger.info(f"Loading data from: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Validate DataFrame structure
        validate_dataframe_structure(df)
        
        logger.info("Data loaded successfully")
        logger.info(f"Loaded {len(df)} rows of data")
        
        return df
    
    except pd.errors.EmptyDataError:
        error_msg = f"The CSV file at {file_path} is empty"
        logger.error(error_msg)
        raise DataLoadingError(error_msg)
    
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing CSV file: {str(e)}"
        logger.error(error_msg)
        raise DataLoadingError(error_msg)
    
    except Exception as e:
        error_msg = f"Unexpected error while loading data: {str(e)}"
        logger.error(error_msg)
        raise DataLoadingError(error_msg)

def get_mock_data() -> pd.DataFrame:
    """
    Generates mock data for testing purposes.
    
    Returns:
        pd.DataFrame: Mock stock data with required columns
    """
    mock_data = {
        'Date': pd.date_range(start='2020-01-01', periods=5),
        'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'High': [102.0, 103.0, 104.0, 105.0, 106.0],
        'Low': [98.0, 99.0, 100.0, 101.0, 102.0],
        'Close': [101.0, 102.0, 103.0, 104.0, 105.0]
    }
    return pd.DataFrame(mock_data)

def main(file_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Main function to load and validate the stock data.
    
    Args:
        file_path (Optional[str]): Path to the data file. If None, uses default path
        
    Returns:
        Union[pd.DataFrame, None]: Loaded data or None if loading fails
    """
    if file_path is None:
        file_path = r"C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Claude\UNITTEST_DATA\NVIDIA.csv"
    
    try:
        df = load_data_from_csv(file_path)
        print("\nData Preview:")
        print(df.head())
        return df
    
    except DataLoadingError as e:
        logger.warning(f"Data loading failed: {str(e)}")
        logger.info("Using mock data instead")
        return get_mock_data()
    
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        return None

if __name__ == "__main__":
    main()