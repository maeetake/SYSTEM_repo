#
# Module: preprocess_data
#
# Description: This module is responsible for converting raw stock data into a clean,
#              normalized, and structured format required for time-series forecasting
#              with deep learning models. It handles missing values, normalizes features,
#              creates sequential data windows, and splits the data chronologically.
#

# ==============================================================================
# 1. DEPENDENCIES
# ==============================================================================

# Requires:
# pandas >= 1.3.0
# numpy >= 1.20.0
# scikit-learn >= 1.0.0
# Python 3.8+

import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# 2. LOGGING CONFIGURATION
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 3. DATA PREPROCESSING CLASS
# ==============================================================================

class DataPreprocessor:
    """
    A class to preprocess time-series data for deep learning models.

    This class encapsulates the entire preprocessing pipeline, including handling
    missing values, scaling features, creating sequences, and splitting the
    dataset into training, validation, and test sets.
    """
    def __init__(self, sequence_length: int = 60):
        """
        Initializes the DataPreprocessor.

        Args:
            sequence_length (int): The number of past time steps to use as input
                                   features for predicting the next time step.
                                   Defaults to 60 as per specification.
        """
        # Adherence to constraint: sequence length must be fixed at 60.
        if sequence_length != 60:
            logger.warning(
                f"Specified sequence_length is {sequence_length}, but model "
                f"specifications require 60. Using sequence_length=60."
            )
            self.sequence_length = 60
        else:
            self.sequence_length = sequence_length

        # Adherence to constraint: Use MinMaxScaler with feature_range=(0, 1).
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # Adherence to constraint: Only OHLC data should be used as input features.
        self.feature_cols = ['Open', 'High', 'Low', 'Close']

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks for and fills any missing values in the OHLC columns.

        It uses the forward fill ('ffill') method. If any NaNs remain at the
        beginning of the DataFrame, those rows are dropped.

        Args:
            df (pd.DataFrame): The input DataFrame containing OHLC data.

        Returns:
            pd.DataFrame: A DataFrame with missing values handled.
        """
        logger.info("Handling missing values...")
        df_filled = df.copy()
        initial_nans = df_filled[self.feature_cols].isnull().sum().sum()
        
        if initial_nans > 0:
            df_filled[self.feature_cols] = df_filled[self.feature_cols].ffill()
            
            # Check for remaining NaNs at the beginning of the dataframe
            remaining_nans = df_filled[self.feature_cols].isnull().sum().sum()
            if remaining_nans > 0:
                # This can happen if the first row(s) had NaN values.
                rows_before_drop = len(df_filled)
                df_filled.dropna(inplace=True)
                rows_after_drop = len(df_filled)
                logger.warning(
                    f"Dropped {rows_before_drop - rows_after_drop} rows from the "
                    "beginning of the dataset due to unfillable NaN values."
                )
        
        logger.info("Missing value handling complete.")
        return df_filled

    def _normalize_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Normalizes the OHLC feature columns to a range of [0, 1].

        This method uses the instance's MinMaxScaler to fit and transform the
        data. The fitted scaler is stored for later inverse transformation.

        Args:
            df (pd.DataFrame): The DataFrame with OHLC data.

        Returns:
            np.ndarray: A NumPy array of shape (n_samples, n_features) with
                        values scaled between 0 and 1.
        """
        logger.info("Normalizing data using MinMaxScaler...")
        feature_data = df[self.feature_cols].values
        scaled_data = self.scaler.fit_transform(feature_data)
        logger.info("Data normalization complete.")
        return scaled_data

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts time-series data into input sequences and target values.

        It uses a sliding window of `self.sequence_length` days of OHLC data (X)
        to predict the closing price of the next day (y).

        Args:
            data (np.ndarray): NumPy array of scaled OHLC data.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X: NumPy array of shape (n_samples, sequence_length, n_features)
                     containing the input sequences.
                - y: NumPy array of shape (n_samples,) containing the target
                     closing prices.
        """
        logger.info(f"Creating sequences with length {self.sequence_length}...")
        X, y = [], []
        
        # The number of data points must be at least sequence_length + 1
        # to create at least one sequence.
        if len(data) <= self.sequence_length:
            raise ValueError(
                f"Not enough data to create sequences. Need at least "
                f"{self.sequence_length + 1} data points, but got {len(data)}."
            )
            
        for i in range(self.sequence_length, len(data)):
            # Input sequence: past 'sequence_length' days of OHLC data
            X.append(data[i-self.sequence_length:i, :])
            # Target value: 'Close' price of the current day (index 3)
            y.append(data[i, 3])
            
        logger.info(f"Successfully created {len(X)} sequences.")
        return np.array(X), np.array(y)

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Splits the sequential data chronologically into training (80%),
        validation (10%), and test (10%) sets.

        Args:
            X (np.ndarray): NumPy array of input sequences.
            y (np.ndarray): NumPy array of target values.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the six split arrays:
                                   'X_train', 'y_train', 'X_val', 'y_val',
                                   'X_test', 'y_test'.
        """
        logger.info("Splitting data into training (80%), validation (10%), and test (10%) sets...")
        total_samples = len(X)
        
        # Calculate split indices
        train_split_idx = int(total_samples * 0.8)
        val_split_idx = int(total_samples * 0.9)
        
        # Perform chronological split
        splits = {
            'X_train': X[:train_split_idx],
            'y_train': y[:train_split_idx],
            'X_val': X[train_split_idx:val_split_idx],
            'y_val': y[train_split_idx:val_split_idx],
            'X_test': X[val_split_idx:],
            'y_test': y[val_split_idx:],
        }
        
        logger.info(f"Data split complete. Shapes: "
                    f"Train={splits['X_train'].shape}, "
                    f"Validation={splits['X_val'].shape}, "
                    f"Test={splits['X_test'].shape}")
        return splits

    def process(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Executes the full preprocessing pipeline on the input DataFrame.

        Args:
            df (pd.DataFrame): The raw input DataFrame with 'Open', 'High',
                               'Low', 'Close' columns.

        Returns:
            Dict[str, any]: A dictionary containing the preprocessed and split
                            data arrays (as float32) and the fitted scaler object.
                            Keys: 'X_train', 'y_train', 'X_val', 'y_val',
                                  'X_test', 'y_test', 'scaler'.
        """
        logger.info("Starting data preprocessing pipeline...")
        
        # 1. Validate input DataFrame
        if not all(col in df.columns for col in self.feature_cols):
            missing = set(self.feature_cols) - set(df.columns)
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        # 2. Handle missing values
        df_clean = self._handle_missing_values(df)

        # 3. Normalize data
        scaled_data = self._normalize_data(df_clean)

        # 4. Create sequences
        try:
            X, y = self._create_sequences(scaled_data)
        except ValueError as e:
            logger.error(f"Failed to create sequences: {e}")
            raise

        # 5. Split data
        split_data = self._split_data(X, y)
        
        # 6. Convert to float32 for DL framework compatibility and assemble final output
        final_output = {
            key: arr.astype(np.float32) for key, arr in split_data.items()
        }
        final_output['scaler'] = self.scaler
        
        logger.info("Data preprocessing pipeline finished successfully.")
        return final_output


# ==============================================================================
# 4. DATA LOADING HELPER FUNCTION
# ==============================================================================
def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a specified file path.

    Checks for file existence and supports CSV format as per the project spec.

    Args:
        data_path (str): The path to the data file.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file format is not supported or if there's an error
                    during loading.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: File not found at {data_path}")
    try:
        if data_path.lower().endswith('.csv'):
            return pd.read_csv(data_path)
        else:
            # Although specs mention only CSV, this provides a clear error for other types.
            raise ValueError("Unsupported file format. Please provide a CSV file.")
    except Exception as e:
        raise ValueError(f"Error loading or reading data from {data_path}: {e}")


# ==============================================================================
# 5. MAIN EXECUTION BLOCK (DEMONSTRATION)
# ==============================================================================

def main():
    """
    Main function to demonstrate the data preprocessing module.

    It loads data, runs the preprocessing pipeline, and prints the shapes of
    the resulting datasets.
    """
    # Use the data path provided in the specification.
    # Using a raw string (r'...') or forward slashes to handle Windows paths.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\generated\expected_input.csv'
    
    # --- Load Data ---
    try:
        raw_df = load_data(data_path)
        logger.info(f"Data loaded successfully from {data_path}.")
        print("Head of the raw data:")
        print(raw_df.head())
        print("-" * 50)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("Terminating demonstration due to data loading failure.")
        # As per instructions, mock data could be used here for testing purposes.
        # For this demonstration, we will exit if the primary data is not found.
        return

    # --- Preprocess Data ---
    try:
        preprocessor = DataPreprocessor(sequence_length=60)
        processed_data = preprocessor.process(raw_df)
        
        logger.info("Preprocessing complete. Verifying output...")
        
        # --- Print Output Shapes and Types ---
        print("\n" + "="*20 + " Preprocessing Output " + "="*20)
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                print(f"Key: {key:<10} | Shape: {value.shape:<20} | DType: {value.dtype}")
            else:
                print(f"Key: {key:<10} | Type: {type(value)}")
        print("="*62)

    except (ValueError, Exception) as e:
        logger.error(f"An error occurred during data preprocessing: {e}")

if __name__ == "__main__":
    main()