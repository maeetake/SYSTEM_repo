# -*- coding: utf-8 -*-
"""
Module: visualize_prediction_results

This module provides functionality to visualize the performance of time-series
forecasting models by plotting their predictions against the actual values.
It is designed to be used in the final stage of a model evaluation pipeline,
offering a clear visual comparison that complements quantitative metrics.
"""

# ----------------------------------
#  Dependencies
# ----------------------------------
# Requires:
#   - pandas>=1.3.0
#   - numpy>=1.21.0
#   - matplotlib>=3.5.0

import logging
import os
import sys
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------
#  Module-level Configuration
# ----------------------------------

# Configure logging to display timestamp, level, module name, and message.
# This setup ensures that any errors are logged in the specified format.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ----------------------------------
#  Core Functionality
# ----------------------------------

def plot_predictions(
    actual_prices: pd.Series,
    predicted_prices_lstm: np.ndarray,
    predicted_prices_transformer: np.ndarray,
    dates: pd.Series,
    output_path: str
) -> str:
    """
    Creates and saves a time-series plot comparing actual test set prices
    with the predicted prices from both the LSTM and Transformer models.

    Args:
        actual_prices (pd.Series): A pandas Series containing the true closing
            prices for the test period.
        predicted_prices_lstm (np.ndarray): An array of predicted closing prices
            from the LSTM model for the test period.
        predicted_prices_transformer (np.ndarray): An array of predicted closing
            prices from the Transformer model for the test period.
        dates (pd.Series): A pandas Series of datetime objects corresponding to
            the test period, to be used as the x-axis.
        output_path (str): The file path (including filename and extension,
            e.g., 'results/prediction_comparison.png') where the generated plot
            will be saved.

    Returns:
        str: The absolute path to the saved image file upon successful generation.

    Raises:
        ValueError: If the input Series/arrays have mismatched lengths.
        IOError: If the plot cannot be saved to the specified path due to
                 permissions or other file system issues.
    """
    # 1. Input Validation
    # Ensure all data inputs have the same number of data points.
    if not (len(actual_prices) == len(predicted_prices_lstm) ==
            len(predicted_prices_transformer) == len(dates)):
        msg = (
            "Input arrays have mismatched lengths. "
            f"Actual: {len(actual_prices)}, LSTM: {len(predicted_prices_lstm)}, "
            f"Transformer: {len(predicted_prices_transformer)}, Dates: {len(dates)}"
        )
        # As per requirements, log the error before raising the exception.
        logger.error(msg, exc_info=False)
        raise ValueError(msg)

    # Ensure the output directory exists. If not, create it.
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            # Handle potential race condition or permission errors during creation.
            err_msg = f"Could not create directory at {output_dir}. Reason: {e}"
            logger.error(err_msg, exc_info=True)
            raise IOError(err_msg) from e

    # 2. Plotting Logic
    try:
        # Use a professional and clear plot style as required.
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot the three series with distinct styles for clarity.
        ax.plot(dates, actual_prices, color='royalblue', label='Actual Price', linewidth=2)
        ax.plot(dates, predicted_prices_lstm, color='darkorange', linestyle='--', label='LSTM Prediction')
        ax.plot(dates, predicted_prices_transformer, color='forestgreen', linestyle='-.', label='Transformer Prediction')

        # 3. Formatting
        ax.set_title('Stock Price Prediction: Actual vs. Predicted', fontsize=18, weight='bold')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Closing Price', fontsize=14)

        # Improve readability of date labels on the x-axis.
        fig.autofmt_xdate()

        ax.legend(fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # 4. Save Plot
        # Save the plot with high resolution and tight bounding box.
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction plot successfully saved to: {output_path}")

    except Exception as e:
        # Catch any other unexpected errors during plotting or saving.
        err_msg = f"Failed to generate or save plot. Reason: {e}"
        logger.error(err_msg, exc_info=True)
        raise IOError(err_msg) from e
    finally:
        # Ensure the plot is closed to free up memory, regardless of success.
        plt.close()

    # Return the absolute path to the created file.
    return os.path.abspath(output_path)


# ----------------------------------
#  Example Usage (main block)
# ----------------------------------

def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from the specified file path.

    This function is provided as a helper for the main execution block
    as per the implementation guidelines.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: File not found at {data_path}")
    try:
        # The prompt specifies CSV, but adding basic logic for other types is robust.
        if data_path.lower().endswith('.csv'):
            df = pd.read_csv(data_path, parse_dates=['Date'])
            return df
        else:
            raise ValueError("Unsupported file format. Please provide a CSV file.")
    except Exception as e:
        # This catches errors like malformed CSVs.
        raise IOError(f"Error loading or parsing data from {data_path}: {e}")

def create_mock_data(df: pd.DataFrame, test_size: int = 50) -> dict:
    """
    Creates mock prediction data based on the actual data for demonstration.
    """
    if len(df) < test_size:
        raise ValueError("Not enough data to create a mock test set.")

    test_df = df.tail(test_size).copy()
    actuals = test_df['Close']
    dates = test_df['Date']

    # Generate plausible mock predictions by adding some noise to actuals.
    noise_lstm = np.random.normal(0, actuals.std() * 0.05, size=test_size)
    preds_lstm = actuals + noise_lstm

    noise_transformer = np.random.normal(0, actuals.std() * 0.03, size=test_size)
    preds_transformer = actuals + noise_transformer

    return {
        "actual_prices": actuals,
        "predicted_prices_lstm": preds_lstm.values,
        "predicted_prices_transformer": preds_transformer.values,
        "dates": dates
    }


def main():
    """
    Main function to demonstrate the usage of the plot_predictions function.
    It loads sample data, creates mock predictions, and generates a plot.
    """
    logger.info("--- Starting Visualization Demo ---")

    # The data_path is specified in the prompt.
    # Note: Using raw string or forward slashes for cross-platform compatibility.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\generated\expected_input.csv'
    output_plot_path = os.path.join("results", "prediction_comparison.png")

    try:
        # 1. Load data
        stock_df = load_data(data_path)
        logger.info(f"Data loaded successfully from {data_path}. Shape: {stock_df.shape}")
        logger.info("Data Head:\n" + stock_df.head().to_string())

        # 2. Create mock data for plotting
        mock_data = create_mock_data(stock_df, test_size=100)
        logger.info("Generated mock data for demonstration.")

        # 3. Call the plotting function
        saved_path = plot_predictions(
            actual_prices=mock_data["actual_prices"],
            predicted_prices_lstm=mock_data["predicted_prices_lstm"],
            predicted_prices_transformer=mock_data["predicted_prices_transformer"],
            dates=mock_data["dates"],
            output_path=output_plot_path
        )
        logger.info(f"SUCCESS: Plot saved at absolute path: {saved_path}")

    except (FileNotFoundError, IOError, ValueError) as e:
        logger.error(f"Demo failed. Reason: {e}", exc_info=False)
        # As per prompt, demonstrate mock data usage on failure.
        logger.info("Proceeding with internal mock data as fallback.")
        try:
            # Create a fully synthetic DataFrame if loading fails.
            mock_dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100))
            mock_actuals = pd.Series(100 + np.cumsum(np.random.randn(100)), index=mock_dates)
            mock_lstm = mock_actuals + np.random.randn(100) * 2
            mock_transformer = mock_actuals + np.random.randn(100) * 1.5

            fallback_path = os.path.join("results", "fallback_prediction_plot.png")
            saved_path = plot_predictions(
                actual_prices=mock_actuals,
                predicted_prices_lstm=mock_lstm.values,
                predicted_prices_transformer=mock_transformer.values,
                dates=mock_actuals.index.to_series(),
                output_path=fallback_path
            )
            logger.info(f"SUCCESS: Fallback plot saved at: {saved_path}")
        except Exception as fallback_e:
            logger.error(f"Fallback demo also failed. Reason: {fallback_e}", exc_info=True)

    logger.info("--- Visualization Demo Finished ---")


if __name__ == "__main__":
    main()