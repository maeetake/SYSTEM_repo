# Requires: matplotlib >= 3.5.0
import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt

# Configure logging for error reporting.
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s:%(name)s: %(message)s'
)
# Use a logger named after the module's purpose.
logger = logging.getLogger('visualize_training_history')

def plot_and_save_history(model_history: Dict[str, List[float]], model_name: str, output_path: str) -> str:
    """
    Generates and saves a plot of the training and validation loss for a given
    model's training history.

    Args:
        model_history (Dict[str, List[float]]): A dictionary-like object, typically
                                                 from Keras history. It must contain
                                                 'loss' and 'val_loss' keys, each
                                                 mapping to a list of floating-point
                                                 numbers representing loss at each epoch.
        model_name (str): The name of the model (e.g., 'LSTM', 'Transformer') to be
                          used in the plot title.
        output_path (str): The file path where the generated plot image will be
                           saved (e.g., './plots/lstm_loss_history.png').

    Returns:
        str: Returns the path of the saved image file upon successful creation.

    Raises:
        KeyError: If 'loss' or 'val_loss' keys are not found in the model_history
                  dictionary.
        IOError: If the plot cannot be saved to the specified path due to an I/O issue.
    """
    # 1. Input validation
    if 'loss' not in model_history or 'val_loss' not in model_history:
        msg = "The 'model_history' dictionary must contain 'loss' and 'val_loss' keys."
        logger.error(msg)
        raise KeyError(msg)

    try:
        # 2. Plotting logic
        plt.figure(figsize=(10, 6))
        plt.plot(model_history['loss'], label='Training Loss')
        plt.plot(model_history['val_loss'], label='Validation Loss')

        # 3. Formatting
        plt.title(f'{model_name} Model - Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 4. Save the plot
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return output_path

    except IOError as e:
        msg = f"Could not save plot to {output_path}. Reason: {e}"
        # Log the I/O error with a stack trace
        logger.error(msg, exc_info=True)
        raise IOError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while generating the plot: {e}"
        logger.error(msg, exc_info=True)
        raise
    finally:
        # Ensure the plot is closed to free up memory, even if errors occur
        plt.close()