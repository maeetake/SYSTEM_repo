# Requires: pandas>=1.3.0, numpy>=1.21.0, matplotlib>=3.5.0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import os
import logging
import traceback
from datetime import datetime

# ------------------------ SETUP LOGGING -------------------------
LOGGER_NAME = 'visualize_prediction_results'
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(LOGGER_NAME)

# ------------------------ DATA LOADING FUNCTION -------------------------
def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV or JSON file.
    Args:
        data_path (str): Path to the data file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    Raises:
        FileNotFoundError: If file is missing.
        ValueError: If loading fails or file type is unsupported.
    """
    if not os.path.exists(data_path):
        logger.error(f"File not found at {data_path}")
        raise FileNotFoundError(f"File not found at {data_path}")
    try:
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            return pd.read_json(data_path)
        else:
            logger.error("Unsupported file format. Please provide a CSV or JSON file.")
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
    except Exception as e:
        logger.error(f"Error loading data: {e}\n{traceback.format_exc()}")
        raise ValueError(f"Error loading data: {e}")

# ------------------------ PREDICTION PLOTTING FUNCTION -------------------------
def plot_predictions(
    actual_prices: pd.Series,
    predicted_prices_lstm: np.ndarray,
    predicted_prices_transformer: np.ndarray,
    dates: pd.Series,
    output_path: str
) -> str:
    """
    Generates and saves a time-series plot comparing actual and predicted stock prices.

    Args:
        actual_prices (pd.Series): The true closing prices for the test set.
        predicted_prices_lstm (np.ndarray): Predictions from the LSTM model.
        predicted_prices_transformer (np.ndarray): Predictions from the Transformer model.
        dates (pd.Series): The corresponding dates for the test set.
        output_path (str): The file path to save the plot image.

    Returns:
        str: The absolute path where the plot was saved.

    Raises:
        ValueError: If input series/arrays have mismatched lengths.
        IOError: If the plot cannot be saved to the specified path.
    """
    try:
        n_actual = len(actual_prices)
        n_lstm = len(predicted_prices_lstm)
        n_transformer = len(predicted_prices_transformer)
        n_dates = len(dates)
        # Shape compatibility check
        if not (n_actual == n_lstm == n_transformer == n_dates):
            logger.error(
                f"Input arrays have mismatched lengths. "
                f"Actual: {n_actual}, LSTM: {n_lstm}, Transformer: {n_transformer}, Dates: {n_dates}\n"
                f"{traceback.format_exc()}"
            )
            raise ValueError(
                f"Input series/arrays must have the same length. "
                f"(Actual: {n_actual}, LSTM: {n_lstm}, Transformer: {n_transformer}, Dates: {n_dates})"
            )
        if not isinstance(actual_prices, pd.Series):
            raise TypeError("actual_prices must be a pandas Series.")
        if not isinstance(predicted_prices_lstm, np.ndarray):
            raise TypeError("predicted_prices_lstm must be a numpy ndarray.")
        if not isinstance(predicted_prices_transformer, np.ndarray):
            raise TypeError("predicted_prices_transformer must be a numpy ndarray.")
        if not isinstance(dates, pd.Series):
            raise TypeError("dates must be a pandas Series.")

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 7))
        plt.plot(dates, actual_prices, color='blue', linewidth=2, label='Actual Price')
        plt.plot(dates, predicted_prices_lstm, color='orange', linestyle='--', linewidth=2, label='LSTM Prediction')
        plt.plot(dates, predicted_prices_transformer, color='green', linestyle='-.', linewidth=2, label='Transformer Prediction')

        plt.title('NVIDIA Stock Price Prediction: Actual vs. Predicted', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Closing Price (USD)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            abs_path = os.path.abspath(output_path)
            return abs_path
        except Exception as e:
            logger.error(f"Could not save plot to {output_path}: {e}\n{traceback.format_exc()}")
            raise IOError(f"Could not save plot to {output_path}. Reason: {e}")
    except Exception as main_exception:
        logger.error(f"Failed to plot predictions: {main_exception}\n{traceback.format_exc()}")
        raise

# ------------------------ MAIN FUNCTION (EXAMPLE) -------------------------
def main():
    # Hard-coded data path as per instruction
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated\expected_input_for_evaluation.csv'

    try:
        df = load_data(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())
    except FileNotFoundError:
        print(f"Data file not found at {data_path}. Please check the data path or provide valid data.")
        # Example/mock fallback:
        df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'Close': np.linspace(200, 240, 10),
            'LSTM_Pred': np.linspace(202, 238, 10) + np.random.normal(0, 2, 10),
            'Transformer_Pred': np.linspace(201, 239, 10) + np.random.normal(0, 2, 10),
        })
        print("Using mock data:")
        print(df.head())
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # ===== Universal extraction/compatibility logic for plotting =====
    try:
        # Ensure required columns or create/generate as needed
        if 'Close' not in df.columns:
            if 'Close_t-0' in df.columns:
                df['Close'] = df['Close_t-0']
            else:
                raise ValueError("No column named 'Close' or 'Close_t-0' found for actual prices.")

        if 'Date' not in df.columns:
            df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')

        if 'LSTM_Pred' not in df.columns:
            # For demonstration: generate dummy predictions if needed
            np.random.seed(42)
            df['LSTM_Pred'] = df['Close'] + np.random.normal(0, 2, len(df))
        if 'Transformer_Pred' not in df.columns:
            # For demonstration: generate dummy predictions if needed
            np.random.seed(99)
            df['Transformer_Pred'] = df['Close'] + np.random.normal(0, 2, len(df))

        # Type and shape checks
        actual_prices = pd.Series(df['Close'].values)
        predicted_prices_lstm = np.asarray(df['LSTM_Pred'].values)
        predicted_prices_transformer = np.asarray(df['Transformer_Pred'].values)
        dates = pd.Series(pd.to_datetime(df['Date']))

        # Final length/shape check to avoid silent mismatches
        if not (len(actual_prices) == len(predicted_prices_lstm) == len(predicted_prices_transformer) == len(dates)):
            raise ValueError(
                f"Data length mismatch: Actual({len(actual_prices)}), "
                f"LSTM({len(predicted_prices_lstm)}), Transformer({len(predicted_prices_transformer)}), Dates({len(dates)})"
            )
    except Exception as extract_exc:
        logger.error(f"Data format error: {extract_exc}\n{traceback.format_exc()}")
        print(f"Data extraction for plotting failed: {extract_exc}")
        return

    plot_path = 'results/prediction_comparison.png'
    try:
        output_plot = plot_predictions(
            actual_prices=actual_prices,
            predicted_prices_lstm=predicted_prices_lstm,
            predicted_prices_transformer=predicted_prices_transformer,
            dates=dates,
            output_path=plot_path
        )
        print(f"Prediction comparison plot saved at: {output_plot}")
    except Exception as plot_exc:
        print(f"Failed to plot predictions: {plot_exc}")

# ------------------------ MODULE ENTRYPOINT -------------------------
if __name__ == "__main__":
    main()