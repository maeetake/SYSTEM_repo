# preprocess_data.py
# Dependencies:
#   Requires: Python >= 3.8
#   Requires: pandas >= 1.3.0, numpy >= 1.20.0, scikit-learn >= 1.0.0
#   Purpose: Preprocess OHLC stock dataset for deep learning models (LSTM, Transformer)
#   Authorship: Auto-generated. Edit carefully.

from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

class DataPreprocessor:
    """
    DataPreprocessor encapsulates the data cleaning, normalization, preparation,
    and splitting pipeline for OHLC time series stock data.
    """
    def __init__(self, sequence_length: int = 60):
        """
        Initializes the preprocessor with the sequence length for time series windows.
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_cols = ['Open', 'High', 'Low', 'Close']

    def _validate_columns(self, df: pd.DataFrame):
        """Check for required OHLC columns, raise ValueError if missing."""
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            msg = f"Input DataFrame is missing required columns: {', '.join(missing)}"
            raise ValueError(msg)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in OHLC columns using forward fill and drops any
        remaining NaNs (at the start of the DataFrame).
        Logs a warning if rows are dropped after forward fill.
        """
        df_filled = df.copy()
        # Forward fill only the feature columns (not 'Date' etc.)
        df_filled[self.feature_cols] = df_filled[self.feature_cols].ffill()
        # Check for any NaNs remain (will be at top, if at all)
        remaining_nans = df_filled[self.feature_cols].isna().any(axis=1)
        if remaining_nans.sum() > 0:
            warnings.warn(
                f"{remaining_nans.sum()} row(s) with missing values remain after forward fill "
                f"and will be dropped.",
                UserWarning
            )
            df_filled = df_filled.loc[~remaining_nans].reset_index(drop=True)
        return df_filled

    def _normalize_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Scales the OHLC feature columns to [0, 1] with MinMaxScaler.
        Returns the scaled array and the fitted scaler.
        """
        try:
            # Ensure correct dtype for normalization
            arr = df[self.feature_cols].astype(float).to_numpy()
        except Exception as e:
            raise ValueError(f"Could not convert OHLC columns to numeric: {e}")
        scaled_data = self.scaler.fit_transform(arr)
        return scaled_data, self.scaler

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts scaled OHLC data into input (X) sequences and target (y) closing prices.
        Each X is a window of [sequence_length] rows of OHLC data;
        each y is the next day's (61st) closing price (scaled).
        """
        num_samples = data.shape[0] - self.sequence_length
        if num_samples < 1:
            raise ValueError(
                f"Not enough data to create sequences: "
                f"required at least {self.sequence_length+1}, got {data.shape[0]}"
            )

        X_list = []
        y_list = []
        for i in range(num_samples):
            X_seq = data[i:i+self.sequence_length, :]
            y_target = data[i+self.sequence_length, 3]  # Close column (index 3)
            X_list.append(X_seq)
            y_list.append(y_target)
        # Convert to float32 for deep learning compatibility
        X = np.stack(X_list).astype(np.float32)    # shape: (num_samples, seq_len, 4)
        y = np.array(y_list, dtype=np.float32)     # shape: (num_samples,)
        return X, y

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Splits sequential data chronologically:
            - First 80%: training
            - Next 10%: validation
            - Last 10%: test
        """
        n_total = X.shape[0]
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        # The remainder goes to test set
        n_test = n_total - n_train - n_val
        # Chronological split
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
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
        Executes the full preprocessing pipeline as specified.

        Args:
            df (pd.DataFrame): The raw input DataFrame with OHLC data.

        Returns:
            Dict with split datasets (X_train, y_train, X_val, y_val, X_test, y_test),
            and the fitted scaler object.
        """
        # 1. Check presence/validity of required columns
        try:
            self._validate_columns(df)
        except Exception as e:
            raise ValueError(f"Validation Error: {e}")

        # 2. Handle missing values
        df_filled = self._handle_missing_values(df)

        # 3. Normalize data (scaler stored for inverse transform)
        scaled_data, scaler = self._normalize_data(df_filled)

        # 4. Create input sequences (X) and targets (y)
        try:
            X, y = self._create_sequences(scaled_data)
        except ValueError as e:
            raise ValueError(f"Sequence Creation Error: {e}")

        # 5. Split into train/val/test
        splits = self._split_data(X, y)
        # 6. Compose output dictionary
        output = dict(splits)
        output['scaler'] = scaler
        return output


def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from the specified data path.
    Supported formats: CSV, JSON (per specs).
    Raises FileNotFoundError if file is missing.
    Raises ValueError for unsupported file types or loading errors.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: File not found at {data_path}")
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            return df
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
            return df
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")


def main():
    """
    Main entry point: runs the preprocessing pipeline from data loading to processed splits.
    Handles file errors, uses mock data if file fails to load.
    """
    # NOTE: Windows backslash in file path, use raw string or double slashes.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated\expected_input_for_preprocessing.csv'
    try:
        df = load_data(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())
    except FileNotFoundError:
        print(f"Data file not found at:\n{data_path}")
        print("Please check the data path or provide valid data.")
        # As fallback: mock data with required columns
        mock_data = pd.DataFrame({
            "Open": np.random.rand(100) * 100,
            "High": np.random.rand(100) * 110,
            "Low": np.random.rand(100) * 90,
            "Close": np.random.rand(100) * 100,
        })
        print("Using mock generated data for demonstration:")
        print(mock_data.head())
        df = mock_data
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    preprocessor = DataPreprocessor(sequence_length=60)
    try:
        processed = preprocessor.process(df)
        print("Data preprocessing successful.")
        print(f"X_train shape: {processed['X_train'].shape}")
        print(f"y_train shape: {processed['y_train'].shape}")
        print(f"X_val shape: {processed['X_val'].shape}")
        print(f"y_val shape: {processed['y_val'].shape}")
        print(f"X_test shape: {processed['X_test'].shape}")
        print(f"y_test shape: {processed['y_test'].shape}")
        print(f"Scaler object: {processed['scaler']}")
    except ValueError as ve:
        print(f"Preprocessing Error: {ve}")

# For module testing or standalone run
if __name__ == "__main__":
    main()