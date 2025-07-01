# visualize_training_history.py

# ======================== #
#  Module: visualize_training_history
#  Purpose: Plot and save training/validation loss curves for model history.
#  Dependencies: matplotlib>=3.5.0, pandas>=1.0.0, numpy>=1.18.0
# ======================== #

import os
import sys
from typing import Dict, List, Any
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

# ========================
# Error logging utility
# ========================

def log_io_error(error_msg: str):
    """
    Logs an I/O error with a timestamp.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sys.stderr.write(f"[{timestamp}] I/O Error: {error_msg}\n")


# =========================
# Core Function: Plot and Save Training/Validation Loss History
# =========================

def plot_and_save_history(
    model_history: Dict[str, List[float]],
    model_name: str,
    output_path: str
) -> str:
    """
    Plots the training and validation loss from a model's training history and saves it to a PNG file.
    
    Args:
        model_history (Dict[str, List[float]]): Dictionary or Keras History object containing at least 'loss' and 'val_loss' lists.
        model_name (str): Name of the model used in the plot title (e.g. 'LSTM').
        output_path (str): Absolute or relative path where the plot image will be saved (.png).

    Returns:
        str: The output_path where the plot was saved.

    Raises:
        KeyError: If 'loss' or 'val_loss' keys are missing.
        ValueError: If 'model_name' or 'output_path' are invalid or empty.
        IOError: If the plot cannot be saved to the disk.
    """
    # --- Input validation ---
    # Accept Keras History object as well as dict-like objects
    if hasattr(model_history, 'history') and isinstance(model_history.history, dict):
        history_dict = model_history.history
    elif isinstance(model_history, dict):
        history_dict = model_history
    else:
        raise ValueError("model_history must be a dict-like object or a Keras History object with a 'history' attribute.")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("output_path must be a valid non-empty string.")

    # Ensure required keys are present
    if 'loss' not in history_dict:
        raise KeyError("Missing 'loss' key in model_history object.")
    if 'val_loss' not in history_dict:
        raise KeyError("Missing 'val_loss' key in model_history object.")

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    # Basic type checks
    if not (isinstance(loss, (list, tuple)) and isinstance(val_loss, (list, tuple))):
        raise ValueError("'loss' and 'val_loss' must be sequences of floats.")

    if len(loss) != len(val_loss):
        raise ValueError(f"Length mismatch: 'loss' ({len(loss)}) and 'val_loss' ({len(val_loss)}) must have same length.")

    # --- Plot configuration ---
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss', linewidth=2, color='royalblue')
    plt.plot(val_loss, label='Validation Loss', linewidth=2, color='tomato')
    plt.title(f'{model_name} Model - Training & Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Ensure directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            log_io_error(f"Unable to create output directory '{output_dir}': {e}")
            raise IOError(f"Unable to create output directory '{output_dir}': {e}")

    # --- Save file ---
    try:
        # Save as PNG with at least 300 DPI, as per requirements
        plt.savefig(output_path, dpi=300, format='png')
        plt.close()
    except Exception as e:
        log_io_error(f"Failed to save plot to {output_path}: {e}")
        raise IOError(f"Failed to save plot to {output_path}: {e}")

    return output_path


# =========================
# Data loading utility (for demonstration/extension, not directly used in plotting)
# =========================

import pandas as pd

def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads dataset from CSV or JSON file at the specified path.
    Args:
        data_path (str): Path to a CSV or JSON file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported or is corrupted.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: File not found at {data_path}")
    try:
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            return pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")


# =========================
# Main function for demonstration & standalone usage
# =========================

def main():
    """
    Entry point for module demonstration.
    Loads data, simulates model history, and plots training/validation loss.
    Does NOT require user input at runtime.
    """
    # Example usage data_path (adjust as needed):
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated\input_for_visualize_prediction_results.csv'

    # Try loading the data file as demonstration, not used in plotting directly.
    try:
        df = load_data(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())
    except FileNotFoundError:
        print("Data file not found. Please check the data path or provide valid data.")
        # Provide minimal mock data for demonstration
        mock_data = pd.DataFrame({
            'date': pd.date_range(start='2022-01-01', periods=5, freq='D'),
            'actual_price': [150, 151.2, 149.3, 148.7, 152.6],
            'predicted_price_lstm': [149.8, 150.7, 149.1, 148.5, 152.0],
            'predicted_price_transformer': [150.1, 151.0, 149.5, 148.9, 152.2]
        })
        print("Using mock data:")
        print(mock_data.head())
    except Exception as e:
        print(f"An error occurred: {e}")

    # --- For demonstration, generate or supply a mock training history ---
    # Real use: model.fit(..., validation_data=...).history

    mock_history = {
        'loss':       [1.2, 0.9, 0.7, 0.5, 0.45, 0.43, 0.42],
        'val_loss':   [1.3, 1.1, 0.8, 0.6,  0.5, 0.49, 0.48],
    }
    model_name = "LSTM"
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'plots',
        'lstm_loss_history.png'
    )

    try:
        saved_plot_path = plot_and_save_history(mock_history, model_name, output_path)
        print(f"Training/validation loss plot successfully saved to: {saved_plot_path}")
    except Exception as e:
        print(f"Failed to save loss history plot: {e}")


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    main()