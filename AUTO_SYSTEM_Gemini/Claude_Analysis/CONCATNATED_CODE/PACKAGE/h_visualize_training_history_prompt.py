# Requires: pandas >= 1.3.0, numpy >= 1.21.0, matplotlib >= 3.5.0
import logging
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging to include a timestamp and module name.
# This basic configuration will log to the console.
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s:%(name)s: %(message)s'
)
logger = logging.getLogger('visualize_prediction_results')

def plot_predictions(
    actual_prices: pd.Series,
    predicted_prices_lstm: np.ndarray,
    predicted_prices_transformer: np.ndarray,
    dates: pd.Series,
    output_path: str
) -> str:
    """
    Creates and saves a time-series plot comparing actual test set prices with the
    predicted prices from both the LSTM and Transformer models.

    Args:
        actual_prices (pd.Series): A pandas Series containing the true closing
                                   prices for the test period.
        predicted_prices_lstm (np.ndarray): An array of predicted closing prices
                                            from the LSTM model for the test period.
        predicted_prices_transformer (np.ndarray): An array of predicted closing prices
                                                   from the Transformer model for the
                                                   test period.
        dates (pd.Series): A pandas Series of datetime objects corresponding to the
                           test period, to be used as the x-axis.
        output_path (str): The file path (including filename and extension,
                           e.g., 'results/prediction_comparison.png') where the
                           generated plot will be saved.

    Returns:
        str: Returns the absolute path to the saved image file upon successful
             generation.

    Raises:
        ValueError: If input series/arrays have mismatched lengths.
        IOError: If the plot cannot be saved to the specified path.
    """
    # 1. Input validation
    if not (len(actual_prices) == len(predicted_prices_lstm) == len(predicted_prices_transformer) == len(dates)):
        msg = (f"Input arrays have mismatched lengths. Actual: {len(actual_prices)}, "
               f"LSTM: {len(predicted_prices_lstm)}, Transformer: {len(predicted_prices_transformer)}, "
               f"Dates: {len(dates)}.")
        logger.error(msg, exc_info=True)
        raise ValueError(msg)

    try:
        # 2. Plotting logic
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 7))

        ax.plot(dates, actual_prices, color='blue', label='Actual Price', linewidth=2)
        ax.plot(dates, predicted_prices_lstm, color='orange', linestyle='--', label='LSTM Prediction')
        ax.plot(dates, predicted_prices_transformer, color='green', linestyle='-.', label='Transformer Prediction')

        # 3. Formatting
        ax.set_title('Stock Price Prediction: Actual vs. Predicted', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Closing Price', fontsize=12)
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate() # Improve date formatting on x-axis

        # 4. Save and return path
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        absolute_path = os.path.abspath(output_path)
        return absolute_path

    except Exception as e:
        msg = f"Could not generate or save plot to {output_path}. Reason: {e}"
        # Log the error with a stack trace
        logger.error(msg, exc_info=True)
        raise IOError(msg) from e
    finally:
        # Ensure the plot is closed to free memory, even if errors occur
        plt.close()