"""
preprocess_data.py - Module for preprocessing stock market OHLC data for deep learning models.

Required dependencies:
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0

This module handles data preprocessing tasks including:
- Missing value handling
- Feature normalization
- Sequence creation
- Train/validation/test splitting
"""

from typing import Dict, Tuple, Any
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, sequence_length: int = 60):
        """
        Initialize the DataPreprocessor with specified sequence length.
        
        Args:
            sequence_length (int): Number of time steps to use for sequence creation.
                                 Defaults to 60 days as per specifications.
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_cols = ['Open', 'High', 'Low', 'Close']

    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate that input DataFrame contains required columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = set(self.feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame using forward fill method.
        
        Args:
            df (pd.DataFrame): Input DataFrame with possible missing values
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        df_filled = df.copy()
        
        # Check for missing values
        missing_count = df_filled[self.feature_cols].isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values. Applying forward fill.")
            df_filled[self.feature_cols] = df_filled[self.feature_cols].ffill()
            
            # Drop any remaining NaN rows at the start
            remaining_nans = df_filled[self.feature_cols].isnull().any(axis=1)
            if remaining_nans.any():
                logger.warning(f"Dropping {remaining_nans.sum()} rows with NaN values at start of dataset")
                df_filled = df_filled[~remaining_nans]
                
        return df_filled

    def _normalize_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Normalize OHLC features using MinMaxScaler.
        
        Args:
            df (pd.DataFrame): Input DataFrame with OHLC data
            
        Returns:
            np.ndarray: Normalized feature array
        """
        try:
            normalized_data = self.scaler.fit_transform(df[self.feature_cols])
            return normalized_data.astype(np.float32)
        except Exception as e:
            logger.error(f"Error during data normalization: {str(e)}")
            raise

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction using sliding window approach.
        
        Args:
            data (np.ndarray): Normalized feature array
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Input sequences (X) and target values (y)
        """
        if len(data) < self.sequence_length + 1:
            raise ValueError(f"Not enough data points to create sequences. "
                           f"Need at least {self.sequence_length + 1}, got {len(data)}")

        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 3])  # Index 3 corresponds to Close price
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split data chronologically into train, validation, and test sets.
        
        Args:
            X (np.ndarray): Input sequences
            y (np.ndarray): Target values
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing split datasets
        """
        n_samples = len(X)
        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)
        
        # Perform splits
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }

    def process(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Raw input DataFrame with OHLC data
            
        Returns:
            Dict[str, Any]: Preprocessed data splits and fitted scaler
        """
        try:
            # Validate input
            self._validate_input(df)
            
            # Handle missing values
            df_cleaned = self._handle_missing_values(df)
            
            # Normalize data
            normalized_data = self._normalize_data(df_cleaned)
            
            # Create sequences
            X, y = self._create_sequences(normalized_data)
            
            # Split data
            splits = self._split_data(X, y)
            
            # Add scaler to output dictionary
            splits['scaler'] = self.scaler
            
            return splits
            
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            raise

def main():
    """Main function for testing the preprocessing module."""
    try:
        # Example usage
        data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Claude\UNITTEST_DATA\generated\expected_input_for_preprocessing.csv'
        df = pd.read_csv(data_path)
        
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process(df)
        
        # Print shapes of processed data
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                print(f"{key} shape: {value.shape}")
                
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()