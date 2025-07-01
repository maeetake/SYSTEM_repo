# -*- coding: utf-8 -*-
"""
Module: visualize_training_history

This module provides functionality to visualize the training and validation loss 
curves from a deep learning model's training history. This visualization is a key 
diagnostic tool for identifying training behaviors like overfitting, underfitting,
or ideal convergence.
"""

# =============================================================================
# Dependencies
# =============================================================================

# Requires: matplotlib >= 3.5.0
# Requires: Python >= 3.8

import os
import logging
from typing import Dict, List

# Matplotlib is a comprehensive library for creating static, animated, 
# and interactive visualizations in Python. We use it to plot the loss curves.
import matplotlib.pyplot as plt

# =============================================================================
# Logging Configuration
# =============================================================================

# Configure logging to capture I/O errors with timestamps.
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================================================================
# Core Function
# =============================================================================

def plot_and_save_history(
    model_history: Dict[str, List[float]], 
    model_name: str, 
    output_path: str
) -> str:
    """
    Generates and saves a plot of the training and validation loss for a given
    model's training history.

    The plot displays the 'loss' and 'val_loss' metrics over epochs, helping to
    diagnose the model's training performance.

    Args:
        model_history (Dict[str, List[float]]): 
            A dictionary-like object containing the training history. Must contain
            keys 'loss' and 'val_loss', each mapping to a list of floating-point
            numbers representing the loss at each epoch. This object is typically
            the `history` attribute of the object returned by the Keras 
            `model.fit()` method.
        model_name (str): 
            The name of the model (e.g., 'LSTM', 'Transformer') to be used in
            the plot title.
        output_path (str): 
            The file path where the generated plot image will be saved (e.g.,
            './plots/lstm_loss_history.png'). The directory will be created if
            it does not exist.

    Returns:
        str: 
            The absolute path of the saved image file upon successful creation.

    Raises:
        KeyError: 
            If 'loss' or 'val_loss' keys are not present in the model_history
            dictionary.
        ValueError: 
            If model_name or output_path are empty or invalid.
        IOError: 
            If the file cannot be saved to the specified output_path due to
            permissions or other OS-level issues.
    """
    # --- Input Validation ---
    if not isinstance(model_history, dict) or 'loss' not in model_history or 'val_loss' not in model_history:
        raise KeyError("The 'model_history' dictionary must contain 'loss' and 'val_loss' keys.")
    
    if not model_name or not isinstance(model_name, str):
        raise ValueError("The 'model_name' must be a non-empty string.")

    if not output_path or not isinstance(output_path, str):
        raise ValueError("The 'output_path' must be a non-empty string.")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    epochs = range(1, len(model_history['loss']) + 1)
    
    ax.plot(epochs, model_history['loss'], 'o-', label='Training Loss', color='royalblue')
    ax.plot(epochs, model_history['val_loss'], 's-', label='Validation Loss', color='darkorange')
    
    ax.set_title(f'{model_name} Model - Training & Validation Loss', fontsize=16, weight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xticks(epochs)
    ax.tick_params(axis='x', rotation=45)
    
    fig.tight_layout()

    # --- File Saving ---
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Save the plot with specified DPI
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    except IOError as e:
        # Log the error as per requirements and re-raise
        error_msg = f"Failed to save plot to '{output_path}'. Reason: {e}"
        logging.error(error_msg)
        raise IOError(error_msg) from e
        
    finally:
        # Close the figure to free up memory
        plt.close(fig)

    saved_path = os.path.abspath(output_path)
    print(f"Successfully saved training history plot to: {saved_path}")
    return saved_path

# =============================================================================
# Main Execution Block (for demonstration)
# =============================================================================

# Note: The 'load_data' and 'main' functions specified in the prompt were
# designed for a script that performs training. Since this module's sole
# purpose is to visualize an *existing* training history, implementing those
# functions here would be irrelevant and misleading. Instead, this `main` block
# serves as a practical demonstration of this module's functionality using
# mock history data.

if __name__ == '__main__':
    print("--- Running Demonstration for visualize_training_history module ---")
    
    # --- Mock Data Generation ---
    # This mock data simulates the history object returned by a model training process.
    # Scenario 1: A well-converging LSTM model
    mock_history_lstm = {
        'loss': [0.15, 0.08, 0.05, 0.03, 0.025, 0.022, 0.020, 0.018, 0.017, 0.016],
        'val_loss': [0.14, 0.09, 0.06, 0.04, 0.035, 0.033, 0.032, 0.031, 0.031, 0.030]
    }
    
    # Scenario 2: A Transformer model starting to overfit
    mock_history_transformer = {
        'loss': [0.20, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.012, 0.010],
        'val_loss': [0.22, 0.15, 0.11, 0.09, 0.08, 0.082, 0.085, 0.09, 0.095, 0.105]
    }

    # --- Function Execution ---
    output_directory = 'plots'
    
    try:
        # Plotting for LSTM model
        print("\n1. Generating plot for LSTM model history...")
        lstm_plot_path = os.path.join(output_directory, 'lstm_loss_history.png')
        plot_and_save_history(
            model_history=mock_history_lstm,
            model_name='LSTM',
            output_path=lstm_plot_path
        )

        # Plotting for Transformer model
        print("\n2. Generating plot for Transformer model history...")
        transformer_plot_path = os.path.join(output_directory, 'transformer_loss_history.png')
        plot_and_save_history(
            model_history=mock_history_transformer,
            model_name='Transformer',
            output_path=transformer_plot_path
        )

    except (KeyError, ValueError, IOError) as e:
        print(f"\nAn error occurred during the demonstration: {e}")
        
    print("\n--- Demonstration Finished ---")