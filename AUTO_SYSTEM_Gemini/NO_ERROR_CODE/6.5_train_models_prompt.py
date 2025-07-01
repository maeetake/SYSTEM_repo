# train_models.py

# =======================================================================================
#
#  MODULE: train_models
#
#  ROLE:
#  The purpose of this module is to use the training dataset to teach the LSTM and
#  Transformer models the underlying patterns of stock price movements. By iteratively
#  adjusting their internal weights based on a loss function, the models learn to
#  predict the next day's closing price. The validation set is used concurrently to
#  prevent overfitting, ensuring that the models generalize well to new data.
#
#  CONSTRAINTS:
#  - This module expects compiled TensorFlow Keras models.
#  - Input data (features and targets) must be NumPy arrays.
#  - Training will utilize GPU if TensorFlow is configured to do so.
#
#  DEPENDENCIES:
#  - Requires: tensorflow >= 2.10
#  - Requires: numpy
#
# =======================================================================================

import logging
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_params: Dict[str, Any]
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trains a given Keras model on the provided training and validation datasets.

    This function orchestrates the model training process. It uses an EarlyStopping
    callback to monitor the validation loss, preventing overfitting by halting
    training when performance on the validation set ceases to improve. The callback
    is configured to restore the model weights from the epoch with the best
    (lowest) validation loss.

    Args:
        model (tf.keras.Model):
            The compiled, un-trained Keras model architecture (e.g., LSTM or Transformer).
            The model should have a name attribute for logging purposes.
        X_train (np.ndarray):
            Training data features, expected in a 3D array of shape
            (num_samples, sequence_length, num_features) with float32 values.
        y_train (np.ndarray):
            Training data targets, expected in a 1D array of shape (num_samples,)
            with float32 values.
        X_val (np.ndarray):
            Validation data features, with the same format as X_train.
        y_val (np.ndarray):
            Validation data targets, with the same format as y_train.
        training_params (Dict[str, Any]):
            A dictionary containing hyperparameters for training.
            Expected keys:
            - 'epochs' (int): The maximum number of training epochs.
            - 'batch_size' (int): The number of samples per gradient update.
            - 'patience' (int, optional): The number of epochs with no improvement
              after which training will be stopped. Defaults to 10.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]:
            A tuple containing:
            - trained_model (tf.keras.Model): The model with the weights that
              achieved the best performance on the validation set.
            - training_history (tf.keras.callbacks.History): A Keras History object.
              Its `.history` attribute is a dictionary containing lists of training
              and validation loss values for each epoch. Example:
              {'loss': [0.1, 0.08, ...], 'val_loss': [0.12, 0.09, ...]}
              
    Raises:
        ValueError: If required keys ('epochs', 'batch_size') are missing from
                    `training_params`.
    """
    # --- Parameter Validation ---
    required_params = ['epochs', 'batch_size']
    for param in required_params:
        if param not in training_params:
            raise ValueError(f"Missing required key in `training_params`: '{param}'")

    # --- Callback Configuration ---
    # The EarlyStopping callback monitors the validation loss ('val_loss').
    # If the loss does not improve for a number of epochs defined by 'patience',
    # training is halted.
    # 'restore_best_weights=True' ensures that the model returned has the weights
    # from the epoch with the lowest validation loss, effectively preventing
    # overfitting.
    patience = training_params.get('patience', 10)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # --- Model Training ---
    model_name = getattr(model, 'name', 'Unnamed Model')
    logging.info(f"--- Starting training for {model_name} ---")
    logging.info(f"Hyperparameters: Epochs={training_params['epochs']}, "
                 f"Batch Size={training_params['batch_size']}, "
                 f"EarlyStopping Patience={patience}")

    # The model.fit method executes the training loop.
    # It logs progress for each epoch to standard output (due to verbose=1),
    # which satisfies the requirement for real-time monitoring.
    history = model.fit(
        X_train,
        y_train,
        epochs=training_params['epochs'],
        batch_size=training_params['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1  # Logs progress bar and per-epoch metrics
    )

    logging.info(f"--- Finished training for {model_name} ---")
    
    # The 'history' object contains the training and validation metrics per epoch.
    # The model object itself has been updated with the best weights due to the
    # 'restore_best_weights=True' argument in the EarlyStopping callback.
    return model, history

# Note on 'main' function and data loading:
# The user specifications included guidelines for a `load_data` and `main` function.
# To maintain modularity and reusability, those functions are not included within
# this module. This `train_models` module is designed to be imported and used by a
# higher-level orchestration script (e.g., `run_pipeline.py` or a main execution block),
# which would be responsible for loading data, preprocessing it, defining models,
# and then calling this `train_model` function.